package tui

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	tea "charm.land/bubbletea/v2"

	"github.com/spa-skyson/pi-rate/internal/permission"
	"github.com/spa-skyson/pi-rate/internal/tools"
)

// Ctrl+B detach/attach: a running turn can be sent to the background. The
// agent loop keeps running — same process, same session, same transcript —
// but its events stop rendering and start buffering, and the prompt frees up.
// Pressing Ctrl+B again replays the buffer and, if the run is still alive,
// returns it to the foreground.
//
// The load-bearing invariant (AGENTS.md, "Turn-loop invariants"): at most one
// live main agent loop on the *Agent, ever. A backgrounded run IS that live
// loop, so while one exists every path that would start a second one —
// startNextPrompt, /agent, /model, /plan, skill activation — refuses or
// queues instead. RebuildWithSession and RebuildWithModel mutate the *Agent
// in place; running a rebuild under a live loop is what the guards prevent.
//
// Channel ownership never changes hands. The TUI keeps exactly one parked
// reader on the agent channel — the waitForAgent command, re-armed after
// every delivery. Detach and attach only flip where updateAgentStream routes
// the message it received: into the buffer, or through the ordinary handlers.
// A second reader goroutine would race the parked command for messages; not
// having one closes that race by construction.

// backgroundBufferCap bounds the replay buffer. Oldest events are evicted
// past it, with the count kept so the replay can say what was lost.
const backgroundBufferCap = 512

// backgroundRun is the detached run's state: the buffered events, what the
// buffer dropped, and — while the run is alive — an input request that
// arrived detached and waits for its dialog until attach.
type backgroundRun struct {
	buf      []agentMsg // events since detach, in arrival order
	dropped  int        // events evicted from buf by the cap
	seen     int        // everything observed: len(buf) + dropped
	started  time.Time
	done     bool
	doneErr  error
	finished time.Time

	// ch is the run's agent channel, captured at detach. After a synthetic
	// finalization (cancel while detached) it marks whose late events are
	// orphans: they arrive wrapped in bgEnvelope and die in handleBgEnvelope.
	ch chan agentMsg

	// A permission/question request that arrived while detached. The agent
	// loop is blocked until it is answered, so the stash is the run's most
	// important state: attach opens its dialog, a dead run answers it denied.
	approval *permission.ApprovalRequest
	question *tools.QuestionRequest
}

// bgEnvelope wraps an event read from a backgrounded run's channel, carrying
// the run it belongs to. Update cannot tell which channel a message was read
// from, and after a synthetic cancel the dead run's parked reader keeps
// delivering — the envelope is what keeps those late events from being
// mistaken for the next turn's events.
type bgEnvelope struct {
	bg *backgroundRun
	ev agentMsg // nil: the channel closed — the run is over
}

// waitForRunAgent reads the run's channel into an envelope. The envelope is
// what lets handleBgEnvelope tell a late close from the next turn's done
// without guessing at error values.
func waitForBgAgent(ch chan agentMsg, bg *backgroundRun) tea.Cmd {
	if ch == nil {
		return nil
	}
	return func() tea.Msg {
		ev, ok := <-ch
		return bgEnvelope{bg: bg, ev: ev2nil(ok, ev)}
	}
}

// ev2nil maps a closed-channel read to a nil event.
func ev2nil(ok bool, ev agentMsg) agentMsg {
	if !ok {
		return nil
	}
	return ev
}

// record buffers one detached event, evicting the oldest past the cap.
func (b *backgroundRun) record(msg agentMsg) {
	b.seen++
	if len(b.buf) >= backgroundBufferCap {
		b.buf = append(b.buf[:0], b.buf[1:]...)
		b.dropped++
	}
	b.buf = append(b.buf, msg)
}

// bgAlive reports whether a detached run is still working. A finished one
// stays until attach (its buffer is the replay), but it no longer holds the
// agent: queued prompts may start, agent and model may switch.
func (m *model) bgAlive() bool {
	return m.bg != nil && !m.bg.done
}

// backgroundTickMsg repaints the status bar while a detached run is alive —
// its elapsed-time field is the only thing on screen that moves.
type backgroundTickMsg struct{}

func armBackgroundTick() tea.Cmd {
	return tea.Tick(time.Second, func(time.Time) tea.Msg { return backgroundTickMsg{} })
}

// toggleBackground is Ctrl+B: detach the running turn, or attach the
// backgrounded one back. Idle, there is nothing to background; a /run cycle
// cannot be backgrounded at all — its events bypass the agent channel, so a
// detached entry would never see the cycle's end and wedge the queue.
func (m *model) toggleBackground() (tea.Model, tea.Cmd) {
	switch {
	case m.bg != nil:
		return m.attachBackground()
	case m.running && m.runCycleActive():
		return m, m.setFlash("/run cycles cannot be backgrounded — wait for the cycle to finish")
	case m.running:
		return m.detachRun()
	default:
		return m, m.setFlash("Nothing to background")
	}
}

// runCycleActive reports whether a /run spec cycle is mid-flight. Terminal
// phases stay in m.run for their summary, so the field alone is not a yes.
func (m *model) runCycleActive() bool {
	return m.run != nil && m.run.phase != "done" && m.run.phase != "failed"
}

// detachRun sends the running turn to the background: the loop keeps
// running, its events buffer instead of rendering, the input frees up.
func (m *model) detachRun() (tea.Model, tea.Cmd) {
	m.bg = &backgroundRun{started: time.Now(), ch: m.agentCh}
	m.running = false
	m.escCancelCount = 0
	m.statusModel.ActiveTool = ""
	m.statusModel.ActiveTools = nil
	m.chatModel.Streaming = ""
	m.chatModel.Thinking = ""
	if m.face != nil {
		m.face.SetMood(MoodIdle)
	}
	m.chatModel.AppendNotice("Run sent to background — Ctrl+B to attach; new prompts wait in the queue.")
	return m, armBackgroundTick()
}

// attachBackground replays a detached run's buffer into the transcript and,
// if the run is still alive, returns it to the foreground.
//
// The replay routes every buffered event through its ordinary handler and
// discards the re-arm each handler returns: the parked reader on the channel
// survived detach, so ownership never moved and no re-arm is needed. When
// another turn went live meanwhile (the background finished and a queued
// prompt started), the replay degrades to a digest — text deltas appended
// now would land in the new turn's open message.
func (m *model) attachBackground() (tea.Model, tea.Cmd) {
	bg := m.bg
	m.bg = nil

	if bg.dropped > 0 {
		m.chatModel.AppendNotice(fmt.Sprintf("… %d background events skipped", bg.dropped))
	}
	if m.running {
		m.replayBackgroundDigest(bg)
		return m, m.setFlash("Attached")
	}
	for _, ev := range bg.buf {
		_, _ = m.replayAgentEvent(ev)
	}
	if !bg.done {
		m.running = true
		// A request that arrived detached opens its dialog only now. The
		// channel readers were re-armed at stash time — opening the dialog
		// here must not arm a second one. A dialog left over from before the
		// detach is answered denied first: a loop can only block on one
		// request, so the old one was abandoned by construction.
		if bg.approval != nil {
			if m.approval != nil {
				m.approval.Reply <- permission.ApprovalResult{}
			}
			m.approval = bg.approval
			bg.approval = nil
			m.openOverlay(overlayApproval)
		}
		if bg.question != nil {
			if m.question != nil {
				m.question.req.Reply <- tools.QuestionAnswer{Selected: "canceled"}
			}
			m.question = &questionDialog{req: *bg.question}
			bg.question = nil
			m.openOverlay(overlayQuestion)
		}
	} else {
		// The run is over: replay may have painted an unpaired tool call,
		// and nothing will deliver its result. Leave no stale tool status.
		m.statusModel.ActiveTool = ""
		m.statusModel.ActiveTools = nil
		if m.face != nil {
			m.face.SetMood(MoodIdle)
		}
	}
	return m, m.setFlash("Attached")
}

// replayAgentEvent runs one buffered event through the handler the live path
// would have used. Callers drop the returned command: it is a waitForAgent
// re-arm, and the one-parked-reader discipline forbids stacking them.
func (m *model) replayAgentEvent(msg agentMsg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case agentThinkingMsg:
		return m.handleAgentThinking(msg)
	case agentTextMsg:
		return m.handleAgentText(msg)
	case agentToolCallMsg:
		return m.handleAgentToolCall(msg)
	case agentToolResultMsg:
		return m.handleAgentToolResult(msg)
	case agentSubEventMsg:
		return m.handleAgentSubEvent(msg)
	case agentWarningMsg:
		return m.handleAgentWarning(msg)
	case agentUsageMsg:
		return m.handleAgentUsage(msg)
	default:
		// agentDoneMsg never lands in the buffer — it finalizes the run.
		return m, nil
	}
}

// replayBackgroundDigest appends the buffered events as transcript notes
// instead of replaying them structurally. Used at attach when another turn is
// already live: appenders are safe mid-stream (they close the open block,
// which the live handlers expect), but a delta replay would append the old
// run's text into the new turn's open message.
func (m *model) replayBackgroundDigest(bg *backgroundRun) {
	var text strings.Builder
	toolCounts := map[string]int{}
	var toolOrder []string
	for _, ev := range bg.buf {
		switch v := ev.(type) {
		case agentTextMsg:
			text.WriteString(v.text)
		case agentToolCallMsg:
			if toolCounts[v.name] == 0 {
				toolOrder = append(toolOrder, v.name)
			}
			toolCounts[v.name]++
		case agentWarningMsg:
			m.chatModel.AppendWarning(v.text)
		}
	}
	if s := strings.TrimSpace(text.String()); s != "" {
		m.chatModel.AppendNotice("Background run output:\n" + truncateLabel(s, 2000))
	}
	if len(toolOrder) > 0 {
		names := make([]string, 0, len(toolOrder))
		for _, n := range toolOrder {
			names = append(names, fmt.Sprintf("%s×%d", n, toolCounts[n]))
		}
		m.chatModel.AppendMeta("Background tools: " + strings.Join(names, ", "))
	}
}

// handleBackgroundAgentMsg is where updateAgentStream routes a run's events
// while it is detached: everything buffers, and the re-arm follows the
// message's source. agentSubEventMsg rides the session-wide AgentEventCh —
// the subagent monitor's reader — so it re-arms waitForSubEvent; re-arming
// agentCh for it would park a second reader on the run channel (a second
// synthetic done at close, a doubled finalization) and freeze the monitor.
// The run channel's reader re-arms into an envelope so late events stay
// attributable after a synthetic cancel.
func (m *model) handleBackgroundAgentMsg(msg tea.Msg) (tea.Model, tea.Cmd) {
	if done, ok := msg.(agentDoneMsg); ok {
		return m.finishBackgroundRun(done)
	}
	ev, ok := msg.(agentMsg)
	if !ok {
		return m, nil
	}
	m.bg.record(ev)
	if _, ok := msg.(agentSubEventMsg); ok {
		return m, waitForSubEvent(m.cfg.AgentEventCh)
	}
	return m, waitForBgAgent(m.agentCh, m.bg)
}

// handleBgEnvelope delivers an event read from a run's channel. The envelope
// names the run, which is the whole point: after a synthetic cancel the dead
// run's parked reader keeps waking, and its late events must never be
// mistaken for the next turn's — handleAgentDone run mid-turn would drop the
// live turn's running flag.
func (m *model) handleBgEnvelope(env bgEnvelope) (tea.Model, tea.Cmd) {
	bg := env.bg
	if bg.done {
		// Finalized synthetically (cancel while detached): the loop is
		// dying and dribbles into a channel nobody replays anymore. Swallow
		// without re-arm — the parked reader exits at the close, the late
		// events die in the channel buffer. No drain goroutine here: it
		// would race the reader this command replaced.
		return m, nil
	}
	if env.ev == nil {
		if m.bg == bg {
			// The run's channel closed while still detached: the real done.
			return m.finishBackgroundRun(agentDoneMsg{})
		}
		// The channel closed after attach (m.bg moved on): an ordinary
		// foreground completion.
		return m.handleAgentDone(agentDoneMsg{})
	}
	if m.bg == bg {
		// Still detached: buffer, and keep the envelope discipline.
		bg.record(env.ev)
		return m, waitForBgAgent(m.agentCh, bg)
	}
	// Detach/attach happened since this reader was armed: the run is
	// foreground again. Ordinary handlers, ordinary re-arms — the envelope
	// lineage ends here.
	model, cmd := m.replayAgentEvent(env.ev)
	return model, cmd
}

// finishBackgroundRun finalizes a run that finished on its own, detached.
// The buffer stays until attach; the notice tells the user it is there.
// handleAgentDone then does the shared close-out — it dismisses dialogs left
// open (the dead-turn rule), renders an error, finalizes the plan worktree,
// restarts the prompt queue, and rings the done attention against
// turnStarted, which the detach never touched. After it returns the agent is
// free: startNextPrompt may start a queued turn even though nothing was
// attached — the transcript is shared, the replay catches up on attach.
func (m *model) finishBackgroundRun(done agentDoneMsg) (tea.Model, tea.Cmd) {
	bg := m.bg
	bg.done = true
	bg.doneErr = done.err
	bg.finished = time.Now()
	m.agentCh = nil
	// A request stashed for attach dies with the run: answer it denied so
	// nothing waits on a Reply nobody will send (the same contract as the
	// dead-turn dialog dismissal in handleAgentDone).
	if bg.approval != nil {
		bg.approval.Reply <- permission.ApprovalResult{}
		bg.approval = nil
	}
	if bg.question != nil {
		bg.question.Reply <- tools.QuestionAnswer{Selected: "canceled"}
		bg.question = nil
	}
	label := "finished"
	if errors.Is(done.err, context.Canceled) {
		label = "canceled"
	}
	skipped := ""
	if bg.dropped > 0 {
		skipped = fmt.Sprintf(", %d older events evicted", bg.dropped)
	}
	m.chatModel.AppendNotice(fmt.Sprintf("Background run %s — %d events buffered%s. Ctrl+B to review.", label, bg.seen, skipped))
	return m.handleAgentDone(done)
}

// cancelBackgroundRun stops a detached run the way Esc cancels a foreground
// one. The loop's context is canceled, stashed input requests are denied by
// finishBackgroundRun, and the run finalizes synthetically: a real done
// would have to outrun the queue this reopens, and a loop wedged on a
// context-deaf call would never deliver one. Late events from the dying
// loop arrive in envelopes and die in handleBgEnvelope. cancelAgent's drain
// goroutine is deliberately not used — a second reader on the channel would
// race the parked one.
func (m *model) cancelBackgroundRun() tea.Cmd {
	if m.agentCancel != nil {
		m.agentCancel()
		m.agentCancel = nil
	}
	m.escCancelCount = 0
	_, cmd := m.finishBackgroundRun(agentDoneMsg{err: context.Canceled})
	return cmd
}

// handleBackgroundApproval stashes a permission request that arrived while
// the run is detached. The dialog is the foreground UI; detached, the loop
// blocked on the request gets a notice and attention instead, and the dialog
// opens on attach. Re-arms the channel reader — exactly one, always.
func (m *model) handleBackgroundApproval(msg approvalRequestMsg) (tea.Model, tea.Cmd) {
	if m.bg.approval != nil {
		// Requests are sequential inside the loop: a second one means the
		// first was abandoned. Fail it closed, like the foreground dialog.
		m.bg.approval.Reply <- permission.ApprovalResult{}
	}
	m.bg.approval = &msg.req
	m.chatModel.AppendNotice("Background run needs your input — Ctrl+B to attach: approval " + msg.req.Tool)
	return m, tea.Batch(m.attentionCmd("☠ Pi-rate", "Approval required: "+msg.req.Tool),
		waitForApproval(m.cfg.ApprovalCh))
}

// handleBackgroundQuestion is the question tool's twin of
// handleBackgroundApproval.
func (m *model) handleBackgroundQuestion(msg questionRequestMsg) (tea.Model, tea.Cmd) {
	if m.bg.question != nil {
		m.bg.question.Reply <- tools.QuestionAnswer{Selected: "canceled"}
	}
	m.bg.question = &msg.req
	m.chatModel.AppendNotice("Background run needs your input — Ctrl+B to attach: " + truncateLabel(msg.req.Question, 60))
	return m, tea.Batch(m.attentionCmd("☠ Pi-rate", "Question: "+truncateLabel(msg.req.Question, 60)),
		waitForQuestion(m.cfg.QuestionCh))
}

// BackgroundStatus is the detached run's slice of the status bar input.
type BackgroundStatus struct {
	Elapsed time.Duration
	Events  int
	Dropped int
	Done    bool
}

// backgroundStatus snapshots the detached run for the status bar. nil when
// nothing is backgrounded.
func (m *model) backgroundStatus() *BackgroundStatus {
	if m.bg == nil {
		return nil
	}
	elapsed := time.Since(m.bg.started)
	if m.bg.done {
		elapsed = m.bg.finished.Sub(m.bg.started)
	}
	return &BackgroundStatus{
		Elapsed: elapsed,
		Events:  m.bg.seen,
		Dropped: m.bg.dropped,
		Done:    m.bg.done,
	}
}
