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

	// Window title and retry state follow the transcript, not the screen:
	// the title names the session on screen, and Ctrl+R replays the prompt
	// of the session that actually failed — never a stranger's.
	title            string
	lastPrompt       string
	lastMentions     []string
	lastPromptFailed bool

	// A permission/question request that arrived while detached. The agent
	// loop is blocked until it is answered, so the stash is the run's most
	// important state: attach opens its dialog, a dead run answers it denied.
	approval *permission.ApprovalRequest
	question *tools.QuestionRequest

	// parked marks a session with no run of its own — a /new parking or an
	// attach's displaced session. It renders "idle", not "✓": nothing
	// happened in it since it left the screen.
	parked bool
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

// bgAlive reports whether any slot in the registry still runs a loop. A
// finished or parked slot stays until attach (its buffer is the replay, its
// transcript the conversation), but it holds nothing that gates the active
// slot.
func (m *model) bgAlive() bool {
	return len(m.liveSlots()) > 0
}

// --- the slot registry ------------------------------------------------------

// addSlot appends a slot to the registry. Insertion order is the popup's
// order; nothing reorders the list, so a slot keeps its row for its whole
// life.
func (m *model) addSlot(s *backgroundRun) {
	m.slots = append(m.slots, s)
}

// removeSlot drops a slot from the registry. Used by dismissal only: attach
// keeps the slot, re-parked with the displaced session.
func (m *model) removeSlot(s *backgroundRun) bool {
	for i, cur := range m.slots {
		if cur == s {
			m.slots = append(m.slots[:i], m.slots[i+1:]...)
			return true
		}
	}
	return false
}

// slotByChannel finds the slot whose run channel the envelope names. The
// whole routing rests on this: a channel belongs to exactly one run — the
// active one is never in the registry, a registry slot's is never m.agentCh
// — so the match is unambiguous, and a second detach of a later turn lands
// on a fresh channel.
func (m *model) slotByChannel(ch chan agentMsg) *backgroundRun {
	if ch == nil {
		return nil
	}
	for _, s := range m.slots {
		if s.ch == ch {
			return s
		}
	}
	return nil
}

// slotBySession finds the slot owning a session id — how a tagged
// approval/question request finds its addressee.
func (m *model) slotBySession(id string) *backgroundRun {
	if id == "" {
		return nil
	}
	for _, s := range m.slots {
		if s.sessionID == id {
			return s
		}
	}
	return nil
}

// liveSlots lists the slots with a running loop.
func (m *model) liveSlots() []*backgroundRun {
	var out []*backgroundRun
	for _, s := range m.slots {
		if !s.done {
			out = append(out, s)
		}
	}
	return out
}

// newestLiveSlot is the Esc-Esc cancel target: the most recently detached
// run still working — stack discipline, the thing the input was freed from
// last.
func (m *model) newestLiveSlot() *backgroundRun {
	live := m.liveSlots()
	if len(live) == 0 {
		return nil
	}
	return live[len(live)-1]
}

// backgroundTickMsg repaints the status bar while a detached run is alive —
// its elapsed-time field is the only thing on screen that moves.
type backgroundTickMsg struct{}

func armBackgroundTick() tea.Cmd {
	return tea.Tick(time.Second, func(time.Time) tea.Msg { return backgroundTickMsg{} })
}

// toggleBackground is Ctrl+B. Running, it detaches — every detach adds one
// slot, which is how the registry grows to N (sequential run+detach cycles).
// Idle, it attaches the single slot back, opens the registry popup when
// several are parked, or explains itself; a /run cycle cannot be backgrounded
// at all — its events bypass the agent channel, so a detached entry would
// never see the cycle's end and wedge the queue.
func (m *model) toggleBackground() (tea.Model, tea.Cmd) {
	switch {
	case m.running && m.runCycleActive():
		return m, m.setFlash("/run cycles cannot be backgrounded — wait for the cycle to finish")
	case m.running:
		return m.detachRun()
	case len(m.slots) == 0:
		return m, m.setFlash("Nothing to background")
	case len(m.slots) == 1:
		return m.attachBackground(m.slots[0])
	default:
		m.newSlotsPopup()
		return m, nil
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

// appendNotice adds a notice to the slot's transcript. The slot's close-out
// lines belong to ITS conversation — the active session must not grow text
// for a run it knows nothing about. Like ChatModel.AppendNotice, the notice
// is a closed block, so the next delta of the slot's next run starts a fresh
// block instead of absorbing it.
func (b *backgroundRun) appendNotice(text string) {
	b.messages = append(b.messages, message{
		role:     "assistant",
		content:  text,
		isNotice: true,
	})
}

// detachRun forks: the running session moves whole into a background slot,
// and the factory opens a fresh session in the active slot. The loop keeps
// running on the agent and in the session it started with — only the
// rendering detaches. Every detach adds one slot; N background sessions come
// from N sequential run+detach cycles.
func (m *model) detachRun() (tea.Model, tea.Cmd) {
	model, cmd, ok := m.parkActiveSession(false)
	if !ok {
		return model, nil
	}
	return model, cmd
}

// parkActiveSession moves the active session's state into a new slot and
// swaps the factory's fresh session in. It serves both detach (idle=false:
// the slot inherits the live loop's channel and cancel, and the loop keeps
// running in the slot) and /new (idle=true: the slot parks with no loop at
// all — done from birth). Ownership moves, it never copies (AGENTS.md): the
// transcript, the inbox, the agent, the channel and the cancel func are the
// slot's; the active model gets fresh fields from the factory. Reports
// whether the park happened.
func (m *model) parkActiveSession(idle bool) (tea.Model, tea.Cmd, bool) {
	if m.mode == "plan" {
		// Plan state (mode, planWorktree, phases) is still global model
		// state; finalizing a detached plan run must never write it under
		// the active slot. Not slotted yet — refuse, naming the reason
		// (documented exception, design §7).
		return m, m.setFlash("Plan sessions cannot be backgrounded yet — wait for the plan to finish"), false
	}
	if m.cfg.NewSessionAgent == nil {
		return m, m.setFlash("Cannot background: no session factory in this context"), false
	}
	ag, sid, title, err := m.cfg.NewSessionAgent(m.ctx, m.cfg.LLM)
	if err != nil {
		m.chatModel.AppendError(fmt.Sprintf("Backgrounding failed: %v", err))
		return m, nil, false
	}
	slot := &backgroundRun{
		started:          time.Now(),
		done:             idle, // an idle session parks with no loop to wait for
		finished:         time.Now(),
		sessionID:        m.cfg.SessionID,
		agent:            m.cfg.Agent,
		messages:         m.chatModel.Messages,
		inbox:            m.pendingPrompts,
		title:            m.sessionTitle,
		lastPrompt:       m.lastPrompt,
		lastMentions:     m.lastMentions,
		lastPromptFailed: m.lastPromptFailed,
		parked:           idle,
	}
	if !idle {
		slot.ch = m.agentCh
		slot.cancel = m.agentCancel
	}
	m.addSlot(slot)
	m.pendingPrompts = nil
	m.agentCh, m.agentCancel = nil, nil
	// Fresh active session: empty transcript, empty retry state, and the
	// factory's default title for the window/tab. Ctrl+R must never replay
	// the parked session's prompt into the fresh one, and the todo plan
	// belonged to the session that just left (per-session todos are MR2).
	m.cfg.Agent, m.cfg.SessionID = ag, sid
	m.chatModel.Messages = make([]message, 0, 16)
	m.chatModel.Streaming, m.chatModel.Thinking = "", ""
	m.lastPrompt, m.lastMentions, m.lastPromptFailed = "", nil, false
	m.todoState = nil
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
	if idle {
		m.chatModel.AppendNotice(fmt.Sprintf(
			"New session — the previous one (%s) moved to the background registry (/slots to view, Ctrl+B to attach).",
			shortSessionID(slot.sessionID)))
	} else {
		m.chatModel.AppendNotice(fmt.Sprintf(
			"Session forked — the previous session runs in the background (Ctrl+B to attach); this is a fresh session (%s).",
			shortSessionID(sid)))
	}
	// The forked session comes up on the base prompt; re-apply the agent the
	// user was talking to, so a prompt typed now answers in the same voice
	// the parked run left behind. AgentSwitcher rebuilds only
	// m.cfg.Agent — the parked slot's agent is untouched, its loop safe.
	var extra tea.Cmd
	if m.activeAgent != "" && m.cfg.AgentSwitcher != nil {
		_, extra = m.applyAgent(m.activeAgent)
	}
	cmd := extra
	if !idle {
		// The status tick repaints the bar while a loop's elapsed time
		// moves; a parked session has nothing that moves.
		cmd = tea.Batch(armBackgroundTick(), extra)
	}
	return m, cmd, true
}

// attachBackground swaps the chosen slot's session back into the active
// view: its transcript, its inbox, its agent and — if still alive — its
// run. The swap is symmetric: the displaced active session stays in the
// registry, parked in the same slot struct it swapped with — switching
// never loses a session (design §1: a slot is born from Ctrl+B or /new,
// and from every swap). The buffer replays structurally through the
// ordinary handlers before the swap finishes, so the transcript catches up
// exactly where detach cut it. A busy active session cannot take the swap —
// refuse, and keep the slot (the v1 guard; the design's park-the-active
// attach is a later increment).
func (m *model) attachBackground(bg *backgroundRun) (tea.Model, tea.Cmd) {
	if bg == nil {
		return m, nil
	}
	if m.running {
		return m, m.setFlash("The active session is running — attach when it finishes or cancel it first")
	}
	if m.mode == "plan" && m.planWorktree != nil {
		// Symmetric with the detach refusal: plan finalization writes
		// global model state, and swapping the session under a pending
		// plan would finalize it into the slot's session.
		return m, m.setFlash("Cannot attach while a plan session is active — wait for the plan to finish")
	}

	// Swap ownership. The slot struct ends up holding the displaced active
	// session, parked: idle, channelless, and still in the registry.
	wasLive := !bg.done // read before the parking writes below
	m.cfg.Agent, bg.agent = bg.agent, m.cfg.Agent
	m.cfg.SessionID, bg.sessionID = bg.sessionID, m.cfg.SessionID
	m.chatModel.Messages, bg.messages = bg.messages, m.chatModel.Messages
	m.pendingPrompts, bg.inbox = bg.inbox, nil
	m.agentCh, bg.ch = bg.ch, nil
	m.agentCancel, bg.cancel = bg.cancel, nil
	displaced := bg.sessionID
	now := time.Now()
	bg.done = true   // the displaced session is idle: nothing runs in it
	bg.parked = true // ...and it left the screen: "idle", not "✓"
	bg.finished = now
	if bg.started.IsZero() {
		bg.started = now
	}
	// Retry state and window title follow the transcript back: Ctrl+R
	// replays THIS session's failed prompt, the title names the session now
	// on screen, and the displaced session keeps its own for its attach.
	m.lastPrompt, bg.lastPrompt = bg.lastPrompt, m.lastPrompt
	m.lastMentions, bg.lastMentions = bg.lastMentions, m.lastMentions
	m.lastPromptFailed, bg.lastPromptFailed = bg.lastPromptFailed, m.lastPromptFailed
	m.sessionTitle, bg.title = bg.title, m.sessionTitle
	// Todos are not slotted yet (MR2): nothing to restore for this session,
	// and the displaced plan must not show.
	m.todoState = nil
	// The swapped-in transcript may end in an open assistant or thinking
	// block — the deltas that streamed before the detach. Their contents
	// ARE the accumulators the live handlers extend; leaving the fields
	// empty (the fork reset them for the fresh session) would make the
	// first replayed delta overwrite the block instead of appending to it.
	m.chatModel.Streaming, m.chatModel.Thinking = "", ""
	if n := len(m.chatModel.Messages); n > 0 {
		last := m.chatModel.Messages[n-1]
		if last.role == "assistant" && !last.closed() {
			m.chatModel.Streaming = last.content
		}
		if last.role == "thinking" {
			m.chatModel.Thinking = last.content
		}
	}

	if bg.dropped > 0 {
		m.chatModel.AppendNotice(fmt.Sprintf("… %d background events skipped", bg.dropped))
	}
	for _, ev := range bg.buf {
		_, _ = m.replayAgentEvent(ev)
	}
	bg.buf = nil
	if wasLive {
		// The run came back to the foreground — it lives on m.agentCh now,
		// NOT in this slot: the slot struct holds the displaced idle
		// session after the swap, so its done flag stays true.
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
	m.popOverlay(overlaySessions)
	m.chatModel.AppendNotice(fmt.Sprintf(
		"Attached — background session returned. The previous session (%s) is parked in the registry: Ctrl+B or /slots.",
		shortSessionID(displaced)))
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
// names the channel; slotByChannel matches it against the registry — with N
// slots each channel still names exactly one run. The matched slot's events
// buffer (deltas render only for the active slot); the active run's go
// through the ordinary handlers; anything else is an orphan reader on a
// channel neither the registry nor the active slot owns anymore — canceled
// and drained — and dies without a re-arm.
func (m *model) handleSlotEnvelope(env slotEnvelope) (tea.Model, tea.Cmd) {
	if s := m.slotByChannel(env.ch); s != nil {
		if s.done {
			// Finalized synthetically (cancel while detached): the loop is
			// dying and dribbles into a channel nobody replays anymore.
			// Swallow without re-arm — the parked reader exits at the close
			// and the late events die in the channel buffer. A loop deaf to
			// its context never reaches that close: its reader stays parked,
			// one goroutine per cancel, bounded by the channel buffer — a
			// debt cancelBackgroundSlot accepts by finalizing synthetically.
			return m, nil
		}
		if env.ev == nil {
			// The slot's channel closed while detached: the real done.
			return m.finishSlotRun(s, agentDoneMsg{})
		}
		if done, ok := env.ev.(agentDoneMsg); ok {
			return m.finishSlotRun(s, done)
		}
		s.record(env.ev)
		// Subagent events are NOT handled here — they never ride a run
		// channel (waitForSubEvent delivers them raw from AgentEventCh,
		// and updateAgentStream routes them to the monitor before this
		// code runs). Re-arming waitForSubEvent from an envelope would
		// park a second reader on AgentEventCh and freeze the monitor.
		return m, waitForSlotAgent(s.ch)
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
	// The close-out line lands in the SLOT's transcript — the conversation
	// that actually finished. Writing it into the active transcript would
	// grow text for a run that session knows nothing about; the active user
	// learns of it through the status bar's done marker and the attention
	// bell below, both slot-agnostic by design.
	bg.appendNotice(fmt.Sprintf(
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
	// Same snapshot rule as agentRun: the slot's loop reads its own copy —
	// Update is free to swap m.cfg fields while it runs.
	go m.runAgentLoop(agentCtx, wrapMentions(next.text, next.mentions), ch, agentRunConfig{
		agent:           bg.agent,
		sessionID:       bg.sessionID,
		sessions:        m.cfg.SessionService,
		logger:          m.cfg.Logger,
		providerName:    m.cfg.ProviderName,
		midTurnAttempts: m.cfg.MidTurnAttempts,
	})
	return waitForSlotAgent(ch)
}

// cancelBackgroundSlot stops a slot's detached run the way Esc cancels a
// foreground one. The loop's context is canceled, stashed input requests are
// denied by finishSlotRun, and the run finalizes synthetically: a real done
// would have to outrun the slot's own queue, and a loop wedged on a
// context-deaf call would never deliver one. Late events from the dying loop
// arrive in envelopes and die in handleSlotEnvelope. cancelAgent's drain
// goroutine is deliberately not used — a second reader on the channel would
// race the parked one.
func (m *model) cancelBackgroundSlot(bg *backgroundRun) tea.Cmd {
	if bg == nil || bg.done {
		return nil
	}
	if bg.cancel != nil {
		bg.cancel()
		bg.cancel = nil
	}
	m.escCancelCount = 0
	_, cmd := m.finishSlotRun(bg, agentDoneMsg{err: context.Canceled})
	return cmd
}

// cancelAllBackgroundSlots stops every live slot — the Ctrl+C quit path:
// nothing survives into quit with a loop still writing.
func (m *model) cancelAllBackgroundSlots() tea.Cmd {
	var cmds []tea.Cmd
	for _, s := range m.liveSlots() {
		if c := m.cancelBackgroundSlot(s); c != nil {
			cmds = append(cmds, c)
		}
	}
	if len(cmds) == 0 {
		return nil
	}
	return tea.Batch(cmds...)
}

// handleBackgroundApproval stashes a permission request that arrived from
// the named slot's session. The dialog is the foreground UI; detached, the
// loop blocked on the request gets a notice and attention instead, and the
// dialog opens on attach. Re-arms the channel reader — exactly one, always.
func (m *model) handleBackgroundApproval(bg *backgroundRun, msg approvalRequestMsg) (tea.Model, tea.Cmd) {
	if bg.approval != nil {
		// Requests are sequential inside the loop: a second one means the
		// first was abandoned. Fail it closed, like the foreground dialog.
		bg.approval.Reply <- permission.ApprovalResult{}
	}
	req := msg.sa.Req
	bg.approval = &req
	m.chatModel.AppendNotice(fmt.Sprintf(
		"Background session %s needs your input — Ctrl+B or /slots to attach: approval %s",
		shortSessionID(bg.sessionID), req.Tool))
	return m, tea.Batch(m.attentionCmd("☠ Pi-rate", "Approval required: "+req.Tool),
		waitForApproval(m.cfg.ApprovalCh))
}

// handleBackgroundQuestion is the question tool's twin of
// handleBackgroundApproval.
func (m *model) handleBackgroundQuestion(bg *backgroundRun, msg questionRequestMsg) (tea.Model, tea.Cmd) {
	if bg.question != nil {
		bg.question.Reply <- tools.QuestionAnswer{Selected: "canceled"}
	}
	req := msg.sq.Req
	bg.question = &req
	m.chatModel.AppendNotice(fmt.Sprintf(
		"Background session %s needs your input — Ctrl+B or /slots to attach: %s",
		shortSessionID(bg.sessionID), truncateLabel(req.Question, 60)))
	return m, tea.Batch(m.attentionCmd("☠ Pi-rate", "Question: "+truncateLabel(req.Question, 60)),
		waitForQuestion(m.cfg.QuestionCh))
}

// BackgroundStatus is one slot's slice of the registry views (popup row).
type BackgroundStatus struct {
	Elapsed time.Duration
	Events  int
	Dropped int
	Done    bool
	Parked  bool // no run of its own (a /new parking, an attach's displaced session)
	Await   bool // holds a stashed approval/question — the ⚑ badge
}

// Idle reports whether the slot has nothing running and nothing to review:
// a parked session waiting only for its attach.
func (s *BackgroundStatus) Idle() bool { return s.Parked || (s.Done && s.Events == 0) }

// backgroundStatus snapshots one slot for the popup rows.
func backgroundStatus(bg *backgroundRun) *BackgroundStatus {
	elapsed := time.Since(bg.started)
	if bg.done {
		elapsed = bg.finished.Sub(bg.started)
	}
	return &BackgroundStatus{
		Elapsed: elapsed,
		Events:  bg.seen,
		Dropped: bg.dropped,
		Done:    bg.done,
		Parked:  bg.parked,
		Await:   bg.approval != nil || bg.question != nil,
	}
}

// BackgroundSummary aggregates the slot registry for the status bar: the
// count answers "how much runs in the background", Awaiting is the ⚑ badge
// for slots holding a stashed input request. nil when the registry is empty.
type BackgroundSummary struct {
	Slots    int // registry size, any state
	Live     int // slots with a live loop
	Awaiting int // slots holding a stashed approval/question
	Events   int // buffered events across slots
}

// backgroundSummary snapshots the registry for the status bar. nil when
// nothing is backgrounded.
func (m *model) backgroundSummary() *BackgroundSummary {
	if len(m.slots) == 0 {
		return nil
	}
	sum := &BackgroundSummary{Slots: len(m.slots)}
	for _, s := range m.slots {
		if !s.done {
			sum.Live++
		}
		if s.approval != nil || s.question != nil {
			sum.Awaiting++
		}
		sum.Events += s.seen
	}
	return sum
}
