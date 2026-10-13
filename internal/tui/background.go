package tui

import (
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

	// A permission/question request that arrived while detached. The agent
	// loop is blocked until it is answered, so the stash is the run's most
	// important state: attach opens its dialog, a dead run answers it denied.
	approval *permission.ApprovalRequest
	question *tools.QuestionRequest
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
// backgrounded one back. Idle, there is nothing to background.
func (m *model) toggleBackground() (tea.Model, tea.Cmd) {
	switch {
	case m.bg != nil:
		return m.attachBackground()
	case m.running:
		return m.detachRun()
	default:
		return m, m.setFlash("Nothing to background")
	}
}

// detachRun sends the running turn to the background: the loop keeps
// running, its events buffer instead of rendering, the input frees up.
func (m *model) detachRun() (tea.Model, tea.Cmd) {
	m.bg = &backgroundRun{started: time.Now()}
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
// discards the waitForAgent re-arms each handler returns: the parked reader
// on the channel survived detach, so ownership never moved and no re-arm is
// needed. When another turn went live meanwhile (the background finished and
// a queued prompt started), the replay degrades to a digest — text deltas
// appended now would land in the new turn's open message.
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
		// here must not arm a second one.
		if bg.approval != nil {
			m.approval = bg.approval
			bg.approval = nil
			m.openOverlay(overlayApproval)
		}
		if bg.question != nil {
			m.question = &questionDialog{req: *bg.question}
			bg.question = nil
			m.openOverlay(overlayQuestion)
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
// while it is detached: everything buffers, and the parked reader re-arms on
// the same channel it always read.
func (m *model) handleBackgroundAgentMsg(msg tea.Msg) (tea.Model, tea.Cmd) {
	if done, ok := msg.(agentDoneMsg); ok {
		return m.finishBackgroundRun(done)
	}
	if ev, ok := msg.(agentMsg); ok {
		m.bg.record(ev)
	}
	return m, waitForAgent(m.agentCh)
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
	// A request stashed for attach dies with the loop: answer it denied so
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
	skipped := ""
	if bg.dropped > 0 {
		skipped = fmt.Sprintf(", %d older events evicted", bg.dropped)
	}
	m.chatModel.AppendNotice(fmt.Sprintf("Background run finished — %d events buffered%s. Ctrl+B to review.", bg.seen, skipped))
	return m.handleAgentDone(done)
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
