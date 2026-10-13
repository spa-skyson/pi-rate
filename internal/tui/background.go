package tui

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	tea "charm.land/bubbletea/v2"

	"github.com/spa-skyson/pi-rate/internal/agent"
	"github.com/spa-skyson/pi-rate/internal/permission"
	"github.com/spa-skyson/pi-rate/internal/tools"
)

// Ctrl+B fork-on-detach (issue #74, MR1): the running session moves WHOLE
// into a background slot — its agent, its transcript, its inbox, its channel
// and its loop — and the active slot opens a brand-new session through the
// factory (Config.NewSessionAgent). Nothing migrates mid-run: the loop keeps
// executing on the agent it started with, in the session it started with,
// and its durable transcript was never the TUI's business — the run writes
// events.jsonl through the SessionService directly, independent of any
// rendering. What detach changes is only who renders: the slot's events
// buffer instead of painting, and a prompt typed to the fresh session starts
// immediately instead of queueing.
//
// The load-bearing invariant (AGENTS.md) is per-slot now: at most one live
// main loop per session slot, and an agent is rebuilt only while no live
// loop runs in ITS slot. The active slot's agent (m.cfg.Agent) is rebuilt
// freely — the background slot's agent is frozen while its loop lives, and
// cross-slot mutation never happens.
//
// Channel ownership never changes hands. Each run channel keeps exactly one
// parked reader — the waitForSlotAgent command, re-armed after every
// delivery — from arm time, before any detach can happen. Detach and attach
// only flip where updateAgentStream routes the message it received, and the
// routing decision is the channel the envelope names, never a guess about
// who is running. A second reader goroutine would race the parked command
// for messages; not having one closes that race by construction.

// backgroundBufferCap bounds the replay buffer. Oldest events are evicted
// past it, with the count kept so the replay can say what was lost.
const backgroundBufferCap = 512

// backgroundRun is the background slot: the detached session's state. It
// owns the session whole — agent, transcript, inbox — plus the v1 replay
// buffer, and — while the run is alive — an input request that arrived
// detached and waits for its dialog until attach.
type backgroundRun struct {
	buf      []agentMsg // events since detach, in arrival order
	dropped  int        // events evicted from buf by the cap
	seen     int        // everything observed: len(buf) + dropped
	started  time.Time
	done     bool
	doneErr  error
	finished time.Time

	// ch is the slot's run channel. Captured at detach, replaced when the
	// slot's inbox restarts a run: the envelope reader re-arms on the
	// channel the message names, never on m.agentCh, which belongs to the
	// active slot from the moment detach hands it back.
	ch chan agentMsg

	// cancel stops the slot's live loop (Esc-Esc while detached).
	cancel context.CancelFunc

	// The session the slot owns. agent is frozen while the slot's loop
	// lives; messages is the session's transcript, swapped back on attach;
	// inbox holds prompts queued before the detach — they belonged to this
	// conversation, and restart here when the slot's run finishes.
	sessionID string
	agent     *agent.Agent
	messages  []message
	inbox     []queuedPrompt

	// A permission/question request that arrived while detached. The agent
	// loop is blocked until it is answered, so the stash is the run's most
	// important state: attach opens its dialog, a dead run answers it denied.
	approval *permission.ApprovalRequest
	question *tools.QuestionRequest
}

// slotEnvelope wraps an event read from a run channel, carrying the channel
// itself. Update cannot tell which channel a message was read from; the
// envelope can, and classification is by CHANNEL — never by pointer, never
// by guessing at who is running. After a synthetic cancel the dead run's
// parked reader keeps delivering, and a second detach can wrap the same
// channel in a new slot: matching the channel is what keeps those late
// events the real done of the slot the model holds now instead of the next
// turn's events.
type slotEnvelope struct {
	ch chan agentMsg
	ev agentMsg // nil: the channel closed — the run is over
}

// waitForSlotAgent (agent_loop.go) reads a run channel into a slotEnvelope:
// every run-channel reader is armed as an envelope from the start — the
// active loop's too — because a reader armed raw before a detach would
// deliver raw forever, and a raw event is indistinguishable from the active
// run's once two loops run side by side.

// record buffers one detached event, evicting the oldest past the cap.
func (b *backgroundRun) record(msg agentMsg) {
	b.seen++
	if len(b.buf) >= backgroundBufferCap {
		b.buf = append(b.buf[:0], b.buf[1:]...)
		b.dropped++
	}
	b.buf = append(b.buf, msg)
}

// bgAlive reports whether the background slot is still working. A finished
// one stays until attach (its buffer is the replay), but it holds nothing
// the active slot needs.
func (m *model) bgAlive() bool {
	return m.bg != nil && !m.bg.done
}

// backgroundTickMsg repaints the status bar while a detached run is alive —
// its elapsed-time field is the only thing on screen that moves.
type backgroundTickMsg struct{}

func armBackgroundTick() tea.Cmd {
	return tea.Tick(time.Second, func(time.Time) tea.Msg { return backgroundTickMsg{} })
}

// toggleBackground is Ctrl+B: detach the running session into its own
// background slot, or attach that slot back. Idle, there is nothing to
// background; a /run cycle cannot be backgrounded at all — its events bypass
// the agent channel, so a detached entry would never see the cycle's end and
// wedge the queue.
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

// shortSessionID trims a session ID to its leading time-sortable prefix —
// "260913-1745" — the least a notice can name without eating a transcript
// line.
func shortSessionID(id string) string {
	if i := strings.IndexByte(id, '-'); i > 0 {
		if j := strings.IndexByte(id[i+1:], '-'); j > 0 {
			return id[:i+1+j]
		}
	}
	if len(id) > 10 {
		return id[:10]
	}
	return id
}

// detachRun forks: the running session moves whole into a background slot,
// and the factory opens a fresh session in the active slot. The loop keeps
// running on the agent and in the session it started with — only the
// rendering detaches. Ownership moves, it never copies (AGENTS.md): the
// transcript, the inbox, the agent, the channel and the cancel func are the
// slot's now; the active model gets fresh fields from the factory.
func (m *model) detachRun() (tea.Model, tea.Cmd) {
	if m.mode == "plan" {
		// Plan state (mode, planWorktree, phases) is still global model
		// state; finalizing a detached plan run must never write it under
		// the active slot. Not slotted yet — refuse, naming the reason
		// (documented exception, design §7).
		return m, m.setFlash("Plan sessions cannot be backgrounded yet — wait for the plan to finish")
	}
	if m.cfg.NewSessionAgent == nil {
		return m, m.setFlash("Cannot background: no session factory in this context")
	}
	ag, sid, title, err := m.cfg.NewSessionAgent(m.ctx, m.cfg.LLM)
	if err != nil {
		m.chatModel.AppendError(fmt.Sprintf("Backgrounding failed: %v", err))
		return m, nil
	}
	m.bg = &backgroundRun{
		started:   time.Now(),
		ch:        m.agentCh,
		cancel:    m.agentCancel,
		sessionID: m.cfg.SessionID,
		agent:     m.cfg.Agent,
		messages:  m.chatModel.Messages,
		inbox:     m.pendingPrompts,
	}
	m.pendingPrompts = nil
	m.agentCh, m.agentCancel = nil, nil
	// Fresh active session: empty transcript, fresh retry state, and the
	// factory's default title for the window/tab.
	m.cfg.Agent, m.cfg.SessionID = ag, sid
	m.chatModel.Messages = make([]message, 0, 16)
	m.chatModel.Streaming, m.chatModel.Thinking = "", ""
	m.lastPrompt, m.lastMentions, m.lastPromptFailed = "", nil, false
	if title != "" {
		m.sessionTitle = title
	}
	m.running = false
	m.escCancelCount = 0
	m.statusModel.ActiveTool = ""
	m.statusModel.ActiveTools = nil
	if m.face != nil {
		m.face.SetMood(MoodIdle)
	}
	m.chatModel.AppendNotice(fmt.Sprintf(
		"Session forked — the previous session runs in the background (Ctrl+B to attach); this is a fresh session (%s).",
		shortSessionID(sid)))
	// The forked session comes up on the base prompt; re-apply the agent the
	// user was talking to, so a prompt typed now answers in the same voice
	// the backgrounded run left behind. AgentSwitcher rebuilds only
	// m.cfg.Agent — the background slot's agent is untouched, its loop safe.
	var extra tea.Cmd
	if m.activeAgent != "" && m.cfg.AgentSwitcher != nil {
		_, extra = m.applyAgent(m.activeAgent)
	}
	return m, tea.Batch(armBackgroundTick(), extra)
}

// attachBackground swaps the slot's session back into the active view: its
// transcript, its inbox, its agent and — if still alive — its run. The
// buffer replays structurally through the ordinary handlers before the swap
// finishes, so the transcript catches up exactly where detach cut it. The
// displaced active session goes nowhere in MR1 (one slot): its transcript
// stays on disk in events.jsonl, resumable. A busy active session cannot
// take the swap — refuse, and keep the slot.
func (m *model) attachBackground() (tea.Model, tea.Cmd) {
	bg := m.bg
	if m.running {
		return m, m.setFlash("The active session is running — attach when it finishes or cancel it first")
	}
	m.bg = nil

	// Swap ownership back. What was active is dropped from the model, named
	// in the notice so /resume can find it.
	m.cfg.Agent, bg.agent = bg.agent, m.cfg.Agent
	m.cfg.SessionID, bg.sessionID = bg.sessionID, m.cfg.SessionID
	m.chatModel.Messages, bg.messages = bg.messages, m.chatModel.Messages
	m.pendingPrompts, bg.inbox = bg.inbox, nil
	m.agentCh, m.agentCancel = bg.ch, bg.cancel
	displaced := bg.sessionID
	// The swapped-in transcript may end in an open assistant block — the
	// deltas that streamed before the detach. Its content IS the streaming
	// accumulator the live handlers extend; leaving the field empty (the
	// fork reset it for the fresh session) would make the first replayed
	// delta overwrite the block instead of appending to it.
	m.chatModel.Streaming = ""
	if n := len(m.chatModel.Messages); n > 0 && m.chatModel.Messages[n-1].role == "assistant" && !m.chatModel.Messages[n-1].closed() {
		m.chatModel.Streaming = m.chatModel.Messages[n-1].content
	}

	if bg.dropped > 0 {
		m.chatModel.AppendNotice(fmt.Sprintf("… %d background events skipped", bg.dropped))
	}
	for _, ev := range bg.buf {
		_, _ = m.replayAgentEvent(ev)
	}
	bg.buf = nil
	if !bg.done {
		m.running = true
		// A request that arrived detached opens its dialog only now. The
		// channel readers were re-armed at stash time — opening the dialog
		// here must not arm a second one. A dialog left over from the
		// displaced session is answered denied first: a loop can only block
		// on one request, so the old one was abandoned by construction.
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
		// and nothing will deliver its result. Leave no stale tool status;
		// the slot's channel is closed, so the active slot holds none.
		m.agentCh, m.agentCancel = nil, nil
		m.statusModel.ActiveTool = ""
		m.statusModel.ActiveTools = nil
		if m.face != nil {
			m.face.SetMood(MoodIdle)
		}
	}
	m.chatModel.AppendNotice("Attached — background session returned." +
		fmt.Sprintf(" The previous session (%s) stays on disk: /resume %s.", shortSessionID(displaced), displaced))
	return m, m.setFlash("Attached")
}

// replayAgentEvent runs one buffered event through the handler the live path
// would have used. Callers drop the returned command: it is a waitForSlotAgent
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

// handleSlotEnvelope delivers an event read from a run channel. The envelope
// names the channel; the channel names the run. The background slot's
// events buffer (deltas render only for the active slot); the active run's
// go through the ordinary handlers; anything else is an orphan reader on a
// channel neither run owns anymore — canceled and drained — and dies
// without a re-arm.
func (m *model) handleSlotEnvelope(env slotEnvelope) (tea.Model, tea.Cmd) {
	if m.bg != nil && env.ch == m.bg.ch {
		if m.bg.done {
			// Finalized synthetically (cancel while detached): the loop is
			// dying and dribbles into a channel nobody replays anymore.
			// Swallow without re-arm — the parked reader exits at the close
			// and the late events die in the channel buffer. A loop deaf to
			// its context never reaches that close: its reader stays parked,
			// one goroutine per cancel, bounded by the channel buffer — a
			// debt cancelBackgroundRun accepts by finalizing synthetically.
			return m, nil
		}
		if env.ev == nil {
			// The slot's channel closed while detached: the real done.
			return m.finishSlotRun(m.bg, agentDoneMsg{})
		}
		if done, ok := env.ev.(agentDoneMsg); ok {
			return m.finishSlotRun(m.bg, done)
		}
		m.bg.record(env.ev)
		// Subagent events are NOT handled here — they never ride a run
		// channel (waitForSubEvent delivers them raw from AgentEventCh,
		// and updateAgentStream routes them to the monitor before this
		// code runs). Re-arming waitForSubEvent from an envelope would
		// park a second reader on AgentEventCh and freeze the monitor.
		return m, waitForSlotAgent(m.bg.ch)
	}
	if env.ch == m.agentCh {
		if env.ev == nil {
			return m.handleAgentDone(agentDoneMsg{})
		}
		if done, ok := env.ev.(agentDoneMsg); ok {
			return m.handleAgentDone(done)
		}
		// Ordinary handlers, ordinary re-arm — the envelope lineage ends
		// here for the active run.
		model, cmd := m.replayAgentEvent(env.ev)
		return model, cmd
	}
	// Orphan: a reader on a channel neither run owns anymore. Swallow
	// without re-arm — the reader exits at the close.
	return m, nil
}

// finishSlotRun finalizes a run that finished on its own, detached. It is
// the slot-side close-out, and the ONLY path a detached run's final takes
// (AGENTS.md: a run's final writes only to its own slot): done flags,
// stashed requests answered denied, a notice plus attention into the visible
// transcript, the slot's own inbox restarted. handleAgentDone — the ACTIVE
// slot's finalizer — is deliberately not called: it would clobber the
// active slot's m.running, dialogs, plan state and prompt queue mid-turn.
func (m *model) finishSlotRun(bg *backgroundRun, done agentDoneMsg) (tea.Model, tea.Cmd) {
	bg.done = true
	bg.doneErr = done.err
	bg.finished = time.Now()
	bg.cancel = nil
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
	m.chatModel.AppendNotice(fmt.Sprintf(
		"Background session %s: run %s — %d events buffered%s. Ctrl+B to review.",
		shortSessionID(bg.sessionID), label, bg.seen, skipped))
	// External notifiers learn about the completion; the active slot's own
	// lifecycle state is untouched.
	m.runLifecycleHooks("turn_complete", map[string]any{
		"error":      done.err != nil,
		"background": true,
	})
	cmds := []tea.Cmd{m.attentionCmd("☠ Pi-rate", "Background run "+label)}
	if next := m.startSlotInbox(bg); next != nil {
		cmds = append(cmds, next)
	}
	return m, tea.Batch(cmds...)
}

// startSlotInbox runs the slot's oldest queued prompt on the slot's own
// agent — a finished detached run restarts ITS OWN queue, per-session, the
// same restart v1 performed globally. The slot becomes busy again: done
// clears, a fresh channel carries the loop, and the envelope reader follows
// the channel the message names.
func (m *model) startSlotInbox(bg *backgroundRun) tea.Cmd {
	if bg == nil || len(bg.inbox) == 0 {
		return nil
	}
	next := bg.inbox[0]
	bg.inbox = bg.inbox[1:]
	bg.done = false
	bg.doneErr = nil
	// The prompt belongs to the slot's conversation: its transcript carries
	// the pair, exactly as submitPrompt does for the active one.
	bg.messages = append(bg.messages,
		message{role: "user", content: next.text},
		message{role: "assistant", content: ""})
	ch := make(chan agentMsg, 64)
	bg.ch = ch
	agentCtx, cancel := context.WithCancel(m.ctx)
	bg.cancel = cancel
	go m.runAgentLoop(agentCtx, wrapMentions(next.text, next.mentions), ch, agentRunConfig{
		agent:     bg.agent,
		sessionID: bg.sessionID,
		sessions:  m.cfg.SessionService,
		logger:    m.cfg.Logger,
	})
	return waitForSlotAgent(ch)
}

// cancelBackgroundRun stops a detached run the way Esc cancels a foreground
// one. The loop's context is canceled, stashed input requests are denied by
// finishSlotRun, and the run finalizes synthetically: a real done would have
// to outrun the slot's own queue, and a loop wedged on a context-deaf call
// would never deliver one. Late events from the dying loop arrive in
// envelopes and die in handleSlotEnvelope. cancelAgent's drain goroutine is
// deliberately not used — a second reader on the channel would race the
// parked one.
func (m *model) cancelBackgroundRun() tea.Cmd {
	if m.bg == nil {
		return nil
	}
	if m.bg.cancel != nil {
		m.bg.cancel()
		m.bg.cancel = nil
	}
	m.escCancelCount = 0
	_, cmd := m.finishSlotRun(m.bg, agentDoneMsg{err: context.Canceled})
	return cmd
}

// handleBackgroundApproval stashes a permission request that arrived from
// the background session. The dialog is the foreground UI; detached, the
// loop blocked on the request gets a notice and attention instead, and the
// dialog opens on attach. Re-arms the channel reader — exactly one, always.
func (m *model) handleBackgroundApproval(msg approvalRequestMsg) (tea.Model, tea.Cmd) {
	if m.bg.approval != nil {
		// Requests are sequential inside the loop: a second one means the
		// first was abandoned. Fail it closed, like the foreground dialog.
		m.bg.approval.Reply <- permission.ApprovalResult{}
	}
	req := msg.sa.Req
	m.bg.approval = &req
	m.chatModel.AppendNotice("Background run needs your input — Ctrl+B to attach: approval " + req.Tool)
	return m, tea.Batch(m.attentionCmd("☠ Pi-rate", "Approval required: "+req.Tool),
		waitForApproval(m.cfg.ApprovalCh))
}

// handleBackgroundQuestion is the question tool's twin of
// handleBackgroundApproval.
func (m *model) handleBackgroundQuestion(msg questionRequestMsg) (tea.Model, tea.Cmd) {
	if m.bg.question != nil {
		m.bg.question.Reply <- tools.QuestionAnswer{Selected: "canceled"}
	}
	req := msg.sq.Req
	m.bg.question = &req
	m.chatModel.AppendNotice("Background run needs your input — Ctrl+B to attach: " + truncateLabel(req.Question, 60))
	return m, tea.Batch(m.attentionCmd("☠ Pi-rate", "Question: "+truncateLabel(req.Question, 60)),
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
