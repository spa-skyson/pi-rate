package tui

import (
	"context"
	"fmt"
	"iter"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	llmmodel "google.golang.org/adk/v2/model"

	tea "charm.land/bubbletea/v2"

	"github.com/spa-skyson/pi-rate/internal/agent"
	"github.com/spa-skyson/pi-rate/internal/permission"
	pisession "github.com/spa-skyson/pi-rate/internal/session"
	"github.com/spa-skyson/pi-rate/internal/tools"
)

// The Ctrl+B fork-on-detach state machine (issue #74, MR1): running → detach
// → attach → done. Detach moves the running session WHOLE into a background
// slot — its agent, its transcript, its inbox, its channel — and the factory
// opens a fresh session in the active slot. The per-slot invariant under test:
// a slot's run never writes to the OTHER slot's state, so a prompt typed to
// the fresh session starts immediately (the complaint the issue closes), and
// a background finish mid-active-turn leaves the active turn alone.
// The channel-ownership rule is unchanged from v1: the parked
// waitForSlotAgent command stays the only reader on each run channel through
// detach and attach; routing, never a second goroutine.

// twoTextLLM streams two text deltas per model call, so a real runAgentLoop
// emits two agentTextMsg events per turn — enough to detach between them.
type twoTextLLM struct{ name string }

func (l *twoTextLLM) Name() string { return l.name }

func (l *twoTextLLM) GenerateContent(_ context.Context, _ *llmmodel.LLMRequest, _ bool) iter.Seq2[*llmmodel.LLMResponse, error] {
	return func(yield func(*llmmodel.LLMResponse, error) bool) {
		if !yield(textResp("part one "), nil) {
			return
		}
		yield(textResp("part two"), nil)
	}
}

// gatedLLM holds the stream open between its two deltas until gate closes —
// a run blocked mid-stream is alive but silent, which is exactly the state
// detach tests need. Closing the gate lets every call finish.
type gatedLLM struct {
	name string
	gate chan struct{}
}

func (l *gatedLLM) Name() string { return l.name }

func (l *gatedLLM) GenerateContent(_ context.Context, _ *llmmodel.LLMRequest, _ bool) iter.Seq2[*llmmodel.LLMResponse, error] {
	return func(yield func(*llmmodel.LLMResponse, error) bool) {
		if !yield(textResp("part one "), nil) {
			return
		}
		<-l.gate
		yield(textResp("part two"), nil)
	}
}

// heldLLM blocks every model call until release closes — a run stays alive
// with nothing on its channel, so a test can detach and then act on the
// active session while the slot runs.
type heldLLM struct {
	name    string
	release chan struct{}
}

func (l *heldLLM) Name() string { return l.name }

func (l *heldLLM) GenerateContent(_ context.Context, _ *llmmodel.LLMRequest, _ bool) iter.Seq2[*llmmodel.LLMResponse, error] {
	return func(yield func(*llmmodel.LLMResponse, error) bool) {
		<-l.release
		yield(textResp("done"), nil)
	}
}

// bgFactory is the stub session factory detach forks through: a fresh real
// agent (same llm) and a fresh session id per call, the shape
// newSessionAgent in the CLI produces. Only Update calls it — single-threaded.
type bgFactory struct {
	t      *testing.T
	llm    llmmodel.LLM
	called int
}

func (f *bgFactory) New(_ context.Context, _ llmmodel.LLM) (*agent.Agent, string, string, error) {
	f.called++
	a, sid := newRunTestAgent(f.t, f.llm)
	return a, sid, fmt.Sprintf("fresh session %d", f.called), nil
}

// newBgTestModel builds a model wired to a real agent over llm, with the
// session factory set — the minimum detachRun needs to fork.
func newBgTestModel(t *testing.T, llm llmmodel.LLM) *model {
	t.Helper()
	a, sid := newRunTestAgent(t, llm)
	m := newTestModel(t)
	m.cfg.Agent = a
	m.cfg.SessionID = sid
	m.cfg.LLM = llm
	f := &bgFactory{t: t, llm: llm}
	m.cfg.NewSessionAgent = f.New
	return m
}

// newPersistedTestAgent builds a real Agent whose sessions persist through
// svc — the events.jsonl assertions need a FileService both the starting
// session and the forked one are registered in.
func newPersistedTestAgent(t *testing.T, llm llmmodel.LLM, svc *pisession.FileService) *agent.Agent {
	t.Helper()
	sb, err := tools.NewSandbox(t.TempDir())
	if err != nil {
		t.Fatalf("NewSandbox: %v", err)
	}
	t.Cleanup(func() { _ = sb.Close() })
	coreTools, err := tools.CoreTools(sb)
	if err != nil {
		t.Fatalf("CoreTools: %v", err)
	}
	a, err := agent.New(agent.Config{Model: llm, Tools: coreTools, Instruction: "test", SessionService: svc})
	if err != nil {
		t.Fatalf("agent.New: %v", err)
	}
	return a
}

// transcriptText joins the rendered assistant text of the transcript, so
// assertions can ask whether a reply has been painted yet.
func transcriptText(m *model) string {
	var b strings.Builder
	for _, msg := range m.chatModel.Messages {
		if msg.role == "assistant" && !msg.isNotice && !msg.isWarning && !msg.isMeta {
			b.WriteString(msg.content)
		}
	}
	return b.String()
}

// noticesText joins the transcript's notices — the fork/finish/attach
// notices all land there.
func noticesText(m *model) string {
	var b strings.Builder
	for _, msg := range m.chatModel.Messages {
		if msg.isNotice {
			b.WriteString(msg.content + "\n")
		}
	}
	return b.String()
}

// ctrlB is the background hotkey as the terminal delivers it.
func ctrlB() tea.Key { return tea.Key{Code: 'b', Mod: tea.ModCtrl} }

// readWithTimeout runs a parked command with a deadline. A reader parked on
// the wrong channel blocks forever — that hang IS the bug, so the deadline
// turns it into a fast failure instead of a test timeout.
func readWithTimeout(t *testing.T, cmd tea.Cmd) tea.Msg {
	t.Helper()
	done := make(chan tea.Msg, 1)
	go func() { done <- cmd() }()
	select {
	case msg := <-done:
		return msg
	case <-time.After(2 * time.Second):
		t.Fatal("a parked reader blocked — it is waiting on the wrong channel")
		return nil
	}
}

// waitForCond polls cond until it holds or the deadline passes. The loops
// under test run on their own goroutines, so "the run has written its event"
// is only observable by waiting for it.
func waitForCond(t *testing.T, d time.Duration, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(d)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatal("condition not met in time: " + what)
}

// drainRun pumps a run channel to its close, so a released gate cannot leave
// a loop writing behind the test's back. Bounded: a healthy turn finishes in
// a handful of envelopes; a run that never closes would hang the reader, and
// the budget turns that into a failure naming the leak.
func drainRun(t *testing.T, m *model, cmd tea.Cmd) {
	t.Helper()
	for i := 0; i < 200; i++ {
		if cmd == nil {
			return
		}
		msg := cmd()
		env, ok := msg.(slotEnvelope)
		if !ok {
			t.Fatalf("drain: reader delivered %T, want slotEnvelope", msg)
		}
		_, cmd = m.Update(env)
		if env.ev == nil {
			return // the close: done was handled
		}
	}
	t.Fatal("run did not drain within the envelope budget")
}

// TestBackground_DetachForksFreshSession walks the core cycle over a real
// agent: detach mid-stream moves the session whole into the slot and opens a
// fresh one; attach hands the slot back with the run still alive; the rest
// of the turn then renders in the foreground exactly once.
func TestBackground_DetachForksFreshSession(t *testing.T) {
	gate := make(chan struct{})
	m := newBgTestModel(t, &gatedLLM{name: "fork", gate: gate})
	origAgent, origSID := m.cfg.Agent, m.cfg.SessionID

	m.beginTurn()
	cmd := m.startAgentLoop("long task")
	if cmd == nil {
		t.Fatal("startAgentLoop returned no command")
	}

	// The first delta renders in the foreground, before the detach.
	env := firstEnvelope(t, cmd)
	if _, ok := env.ev.(agentTextMsg); !ok {
		t.Fatalf("first event = %T, want agentTextMsg", env.ev)
	}
	_, cmd = m.Update(env)
	if !strings.Contains(transcriptText(m), "part one") {
		t.Fatal("first delta did not render before detach")
	}

	// Ctrl+B: forked. The slot owns the old session whole; the active slot
	// is a fresh, idle session.
	newM, tick := m.handleKey(tea.KeyPressMsg(ctrlB()))
	m = newM.(*model)
	if m.running {
		t.Fatal("detach left m.running set")
	}
	if m.bg == nil {
		t.Fatal("detach did not arm the background slot")
	}
	if tick == nil {
		t.Fatal("detach did not arm the status tick")
	}
	if m.bg.sessionID != origSID || m.bg.agent != origAgent {
		t.Fatal("the slot did not take over the original session and agent")
	}
	if m.cfg.SessionID == origSID || m.cfg.Agent == origAgent {
		t.Fatal("the active slot kept the original session — the fork did not happen")
	}
	for _, msg := range m.chatModel.Messages {
		if !msg.isNotice {
			t.Fatalf("the fresh session inherited transcript content: %q", msg.content)
		}
	}
	if len(m.bg.messages) == 0 {
		t.Fatal("the slot lost the original transcript")
	}
	if !strings.Contains(noticesText(m), "Session forked") {
		t.Fatal("the fork notice is missing")
	}

	// Ctrl+B again: attach returns the LIVE slot — running, its channel
	// parked as the active one.
	newM, _ = m.handleKey(tea.KeyPressMsg(ctrlB()))
	m = newM.(*model)
	if m.bg != nil {
		t.Fatal("attach left the background slot armed")
	}
	if !m.running {
		t.Fatal("attach did not return the live run to the foreground")
	}
	if m.cfg.SessionID != origSID {
		t.Fatal("attach did not restore the slot's session id")
	}

	// Release the gate: the rest of the turn renders live, exactly once.
	close(gate)
	sawTwo := false
	for i := 0; i < 50; i++ {
		msg := cmd()
		env, ok := msg.(slotEnvelope)
		if !ok {
			t.Fatalf("reader delivered %T, want slotEnvelope", msg)
		}
		if text, ok := env.ev.(agentTextMsg); ok && strings.Contains(text.text, "part two") {
			if sawTwo {
				t.Fatal("part two rendered twice")
			}
			sawTwo = true
		}
		_, cmd = m.Update(env)
		if env.ev == nil {
			break
		}
	}
	if !sawTwo || strings.Contains(transcriptText(m), "part twopart two") {
		t.Fatalf("the post-attach delta rendered wrong: %q", transcriptText(m))
	}
	if !strings.Contains(transcriptText(m), "part one") || !strings.Contains(transcriptText(m), "part two") {
		t.Fatalf("transcript incomplete after attach: %q", transcriptText(m))
	}
	if strings.Contains(noticesText(m), "Background session") {
		t.Fatal("a live attach finalized the run as a background finish")
	}
	if m.running {
		t.Fatal("the turn did not finish after the gate opened")
	}
}

// firstEnvelope reads the first envelope from a parked reader.
func firstEnvelope(t *testing.T, cmd tea.Cmd) slotEnvelope {
	t.Helper()
	msg := cmd()
	env, ok := msg.(slotEnvelope)
	if !ok {
		t.Fatalf("reader delivered %T, want slotEnvelope", msg)
	}
	return env
}

// TestBackground_DetachedFinishReplaysOnAttach drives a run to completion
// while detached, then attaches: the buffer replays structurally and the
// finished slot releases the input without inventing a run.
func TestBackground_DetachedFinishReplaysOnAttach(t *testing.T) {
	m := newBgTestModel(t, &twoTextLLM{name: "smoke"})
	m.beginTurn()
	cmd := m.startAgentLoop("smoke run")
	if cmd == nil {
		t.Fatal("startAgentLoop returned no command")
	}

	// Pump until the first delta renders, then detach between deltas.
	env := firstEnvelope(t, cmd)
	if _, ok := env.ev.(agentTextMsg); !ok {
		t.Fatalf("first event = %T, want agentTextMsg", env.ev)
	}
	_, cmd = m.Update(env)
	newM, _ := m.handleKey(tea.KeyPressMsg(ctrlB()))
	m = newM.(*model)
	if m.bg == nil || m.bg.done {
		t.Fatal("detach failed mid-turn")
	}

	// Pump the rest detached: events buffer unrendered, the close finalizes
	// the slot on its own.
	sawTwo := false
	for i := 0; i < 50; i++ {
		msg := cmd()
		e, ok := msg.(slotEnvelope)
		if !ok {
			t.Fatalf("reader delivered %T, want slotEnvelope", msg)
		}
		if _, ok := e.ev.(agentTextMsg); ok && strings.Contains(fmt.Sprint(e.ev), "part two") {
			sawTwo = true
		}
		_, cmd = m.Update(e)
		if e.ev == nil {
			break
		}
	}
	if !sawTwo {
		t.Fatal("the second delta never arrived detached")
	}
	if m.bg == nil || !m.bg.done {
		t.Fatal("the detached run did not finalize on close")
	}
	if strings.Contains(transcriptText(m), "part two") {
		t.Fatal("a detached delta rendered before attach")
	}
	if !strings.Contains(noticesText(m), "Background session") {
		t.Fatal("the slot finish notice is missing")
	}

	// Attach after the fact: faithful replay, input idle again.
	newM, _ = m.handleKey(tea.KeyPressMsg(ctrlB()))
	m = newM.(*model)
	if m.bg != nil {
		t.Fatal("attach after finish left the background slot armed")
	}
	if m.running {
		t.Fatal("attaching a finished run marked the model as running")
	}
	body := transcriptText(m)
	if !strings.Contains(body, "part one") || !strings.Contains(body, "part two") {
		t.Fatalf("replay incomplete: %q", body)
	}
	if strings.Count(body, "part two") != 1 {
		t.Fatalf("replay duplicated the buffered delta: %q", body)
	}
	if !strings.Contains(noticesText(m), "Attached") {
		t.Fatal("the attach notice is missing")
	}
}

// TestBackground_DetachedPromptStartsImmediately is THE acceptance test of
// issue #74: with a slot running in the background, a prompt typed to the
// fresh session starts a turn NOW — not queued behind the background run.
func TestBackground_DetachedPromptStartsImmediately(t *testing.T) {
	release := make(chan struct{})
	t.Cleanup(func() { close(release) })
	m := newBgTestModel(t, &heldLLM{name: "issue74", release: release})
	m.beginTurn()
	_ = m.startAgentLoop("long background task")

	newM, _ := m.handleKey(tea.KeyPressMsg(ctrlB()))
	m = newM.(*model)
	bgSID := m.bg.sessionID

	// A prompt typed while detached starts on the fresh session at once.
	model2, _, _ := m.handleInputSubmit(InputSubmitMsg{Text: "fresh question"})
	m = model2.(*model)
	if !m.running {
		t.Fatal("the prompt did not start immediately — it queued (the issue #74 complaint)")
	}
	if len(m.pendingPrompts) != 0 {
		t.Fatalf("the prompt was queued (%d pending) instead of starting", len(m.pendingPrompts))
	}
	if m.flash == "Queued" {
		t.Fatal("the prompt reported itself queued")
	}
	if m.cfg.SessionID == bgSID {
		t.Fatal("the prompt started on the background session instead of the fresh one")
	}
	found := false
	for _, msg := range m.chatModel.Messages {
		if msg.role == "user" && msg.content == "fresh question" {
			found = true
		}
	}
	if !found {
		t.Fatal("the prompt did not land in the fresh transcript")
	}

	// Release and drain so the loop does not outlive the test.
	m.agentCancel()
}

// TestBackground_BackgroundSessionWritesEventsAfterDetach pins durability:
// the detached run keeps persisting to ITS events.jsonl through the
// SessionService, regardless of what the TUI renders (AGENTS.md: durability
// is the run's, not the render's).
func TestBackground_BackgroundSessionWritesEventsAfterDetach(t *testing.T) {
	dir := t.TempDir()
	svc, err := pisession.NewFileService(dir)
	if err != nil {
		t.Fatal(err)
	}
	gate := make(chan struct{})
	llm := &gatedLLM{name: "durable", gate: gate}

	a := newPersistedTestAgent(t, llm, svc)
	sid, _, err := a.CreateSession(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	m := newTestModel(t)
	m.cfg.Agent = a
	m.cfg.SessionID = sid
	m.cfg.LLM = llm
	m.cfg.SessionService = svc
	m.cfg.NewSessionAgent = func(ctx context.Context, _ llmmodel.LLM) (*agent.Agent, string, string, error) {
		ag := newPersistedTestAgent(t, llm, svc)
		freshSID, title, err := ag.CreateSessionWithID(ctx, pisession.GenerateSessionID())
		return ag, freshSID, title, err
	}

	m.beginTurn()
	cmd := m.startAgentLoop("write durable events")
	env := firstEnvelope(t, cmd)
	_, _ = m.Update(env)

	newM, _ := m.handleKey(tea.KeyPressMsg(ctrlB()))
	m = newM.(*model)

	eventsPath := filepath.Join(dir, sid, "events.jsonl")
	waitForCond(t, 5*time.Second, "bg session events.jsonl to appear", func() bool {
		_, err := os.Stat(eventsPath)
		return err == nil
	})
	before, err := os.Stat(eventsPath)
	if err != nil {
		t.Fatal(err)
	}
	sizeBefore := before.Size()

	// The rest of the turn completes detached: the run — not the renderer —
	// must persist it.
	close(gate)
	waitForCond(t, 5*time.Second, "events.jsonl to grow after the detached turn completed", func() bool {
		after, err := os.Stat(eventsPath)
		return err == nil && after.Size() > sizeBefore
	})
	// The fresh session has no turn, so its events.jsonl — which
	// FileService.Create makes empty — must have stayed empty: only the
	// slot's file grew.
	freshPath := filepath.Join(dir, m.cfg.SessionID, "events.jsonl")
	if st, err := os.Stat(freshPath); err == nil && st.Size() > 0 {
		t.Fatalf("the fresh session's events.jsonl grew without a turn: %d bytes", st.Size())
	}
}

// TestBackground_AttachRefusesWhileActiveRuns: the swap needs an idle active
// session — attaching over a running turn would clobber it mid-flight.
func TestBackground_AttachRefusesWhileActiveRuns(t *testing.T) {
	m := newTestModel(t)
	m.running = true
	m.bg = &backgroundRun{started: time.Now(), sessionID: "sess-bg"}

	newM, _ := m.handleKey(tea.KeyPressMsg(ctrlB()))
	m = newM.(*model)
	if m.bg == nil {
		t.Fatal("attach over a running turn displaced the slot")
	}
	if !strings.Contains(m.flash, "active session is running") {
		t.Fatalf("flash = %q, want the busy-active refusal", m.flash)
	}
}

// TestBackground_PlanCannotDetach pins the documented exception: plan state
// is still global model state, so a detached plan run would finalize into
// whatever slot is active. The refusal names the reason.
func TestBackground_PlanCannotDetach(t *testing.T) {
	m := newBgTestModel(t, &heldLLM{name: "plan", release: make(chan struct{})})
	m.beginTurn()
	m.mode = "plan"
	_ = m.startAgentLoop("plan task")

	newM, _ := m.handleKey(tea.KeyPressMsg(ctrlB()))
	m = newM.(*model)
	if m.bg != nil {
		t.Fatal("a plan session was backgrounded")
	}
	if !strings.Contains(m.flash, "Plan sessions cannot be backgrounded") {
		t.Fatalf("flash = %q, want the plan refusal", m.flash)
	}
}

// TestBackground_ApprovalDetachedStashOpensOnAttach: a permission request
// from the background session must not open the dialog — it stashes with its
// session tag, notices, and attach opens it.
func TestBackground_ApprovalDetachedStashOpensOnAttach(t *testing.T) {
	m := newBgTestModel(t, &heldLLM{name: "bg-appr", release: make(chan struct{})})
	m.beginTurn()
	_ = m.startAgentLoop("long task")
	newM, _ := m.handleKey(tea.KeyPressMsg(ctrlB()))
	m = newM.(*model)

	approvalCh := make(chan SessionApproval, 4)
	m.cfg.ApprovalCh = approvalCh
	m.bg.sessionID = "sess-bg"
	req := permission.ApprovalRequest{Tool: "bash", Command: "rm -rf /", Reply: make(chan permission.ApprovalResult, 1)}

	m.Update(approvalRequestMsg{sa: SessionApproval{Req: req, SessionID: "sess-bg"}})
	if m.approval != nil {
		t.Fatal("the approval dialog opened while the run was detached")
	}
	if m.bg == nil || m.bg.approval == nil {
		t.Fatal("the request was not stashed for attach")
	}
	if !strings.Contains(noticesText(m), "Background run needs your input") {
		t.Fatal("the needs-input notice is missing")
	}

	// Attach: the dialog opens from the stash.
	newM, _ = m.handleKey(tea.KeyPressMsg(ctrlB()))
	m = newM.(*model)
	if m.approval == nil || m.approval.Tool != "bash" {
		t.Fatalf("attach did not open the stashed dialog: %+v", m.approval)
	}
	if m.bg != nil && m.bg.approval != nil {
		t.Fatal("the stash survived attach")
	}
}

// TestBackground_QuestionDetachedStashOpensOnAttach mirrors the approval
// stash for the question tool.
func TestBackground_QuestionDetachedStashOpensOnAttach(t *testing.T) {
	m := newBgTestModel(t, &heldLLM{name: "bg-q", release: make(chan struct{})})
	m.beginTurn()
	_ = m.startAgentLoop("long task")
	newM, _ := m.handleKey(tea.KeyPressMsg(ctrlB()))
	m = newM.(*model)

	questionCh := make(chan SessionQuestion, 4)
	m.cfg.QuestionCh = questionCh
	m.bg.sessionID = "sess-bg"
	req := tools.QuestionRequest{
		Question: "Proceed?",
		Options:  []tools.QuestionOption{{Label: "yes"}, {Label: "no"}},
		Reply:    make(chan tools.QuestionAnswer, 1),
	}

	m.Update(questionRequestMsg{sq: SessionQuestion{Req: req, SessionID: "sess-bg"}})
	if m.question != nil {
		t.Fatal("the question dialog opened while the run was detached")
	}
	if m.bg == nil || m.bg.question == nil {
		t.Fatal("the question was not stashed for attach")
	}

	newM, _ = m.handleKey(tea.KeyPressMsg(ctrlB()))
	m = newM.(*model)
	if m.question == nil || m.question.req.Question != "Proceed?" {
		t.Fatalf("attach did not open the stashed question: %+v", m.question)
	}
}

// TestBackground_TwoRunsApprovalsRouteByTag is the design's top-risk
// acceptance test (§11.1): two live runs, and every approval goes to the run
// that raised it — by its session tag, never by guessing who is running.
// An untagged or unknown-tagged request is denied, not routed.
func TestBackground_TwoRunsApprovalsRouteByTag(t *testing.T) {
	release := make(chan struct{})
	llm := &heldLLM{name: "two-runs", release: release}
	m := newBgTestModel(t, llm)
	m.beginTurn()
	arm := m.startAgentLoop("background task")

	// Fork: the fresh session goes active, then starts ITS OWN live run.
	newM, _ := m.handleKey(tea.KeyPressMsg(ctrlB()))
	m = newM.(*model)
	_, _, _ = m.handleInputSubmit(InputSubmitMsg{Text: "active task"})
	if !m.running {
		t.Fatal("the active run did not start beside the background one")
	}
	// handleInputSubmit hands back a Batch; its reader closure never ran in
	// the test, so arming the one reader the runtime would have parked is
	// exactly one reader, not a second.
	activeCmd := waitForSlotAgent(m.agentCh)

	approvalCh := make(chan SessionApproval, 8)
	m.cfg.ApprovalCh = approvalCh
	bgReply := make(chan permission.ApprovalResult, 1)
	activeReply := make(chan permission.ApprovalResult, 1)
	strayReply := make(chan permission.ApprovalResult, 1)

	// 1. Background-tagged: stashes behind a notice, dialog stays closed.
	m.Update(approvalRequestMsg{sa: SessionApproval{
		Req:       permission.ApprovalRequest{Tool: "bg-tool", Reply: bgReply},
		SessionID: m.bg.sessionID,
	}})
	if m.approval != nil {
		t.Fatal("a background-tagged request opened the active dialog")
	}
	if m.bg == nil || m.bg.approval == nil || m.bg.approval.Tool != "bg-tool" {
		t.Fatal("the background-tagged request was not stashed")
	}

	// 2. Active-tagged: opens the dialog.
	m.Update(approvalRequestMsg{sa: SessionApproval{
		Req:       permission.ApprovalRequest{Tool: "active-tool", Reply: activeReply},
		SessionID: m.cfg.SessionID,
	}})
	if m.approval == nil || m.approval.Tool != "active-tool" {
		t.Fatalf("the active-tagged request did not open its dialog: %+v", m.approval)
	}

	// 3. Untagged: a factory bug — denied, never routed by guesswork.
	m.Update(approvalRequestMsg{sa: SessionApproval{
		Req: permission.ApprovalRequest{Tool: "untagged", Reply: strayReply},
	}})
	select {
	case res := <-strayReply:
		if res.Allowed {
			t.Fatal("the untagged request was allowed")
		}
	default:
		t.Fatal("the untagged request was never answered")
	}
	if m.approval == nil || m.approval.Tool != "active-tool" {
		t.Fatal("the untagged request displaced the active dialog")
	}

	// 4. Unknown tag (a dead slot's late arrival): denied too.
	m.Update(approvalRequestMsg{sa: SessionApproval{
		Req:       permission.ApprovalRequest{Tool: "stray", Reply: strayReply},
		SessionID: "sess-gone",
	}})
	select {
	case res := <-strayReply:
		if res.Allowed {
			t.Fatal("the unknown-tagged request was allowed")
		}
	default:
		t.Fatal("the unknown-tagged request was never answered")
	}

	// Cleanup: release both runs. The background slot's final denies its
	// stashed request — a dead run must not leave a loop waiting forever.
	close(release)
	drainRun(t, m, activeCmd)
	m.Update(slotEnvelope{ch: m.bg.ch, ev: nil}) // the slot's real close
	select {
	case res := <-bgReply:
		if res.Allowed {
			t.Fatal("the stashed background approval survived a dead run as allowed")
		}
	default:
		t.Fatal("the slot close never answered the stashed approval")
	}
	_ = arm // the slot's first reader; its channel is closed now
}

// TestBackground_FinishMidActiveRunSparesActiveSlot: a background run's
// final must not write to the active slot's state (AGENTS.md: a run's final
// writes only to its own slot). The active turn keeps running untouched.
func TestBackground_FinishMidActiveRunSparesActiveSlot(t *testing.T) {
	release := make(chan struct{})
	llm := &heldLLM{name: "mid-finish", release: release}
	m := newBgTestModel(t, llm)
	m.beginTurn()
	_ = m.startAgentLoop("background task")

	newM, _ := m.handleKey(tea.KeyPressMsg(ctrlB()))
	m = newM.(*model)
	_, _, _ = m.handleInputSubmit(InputSubmitMsg{Text: "active task"})
	activeSID, activeCh := m.cfg.SessionID, m.agentCh
	if !m.running {
		t.Fatal("the active run did not start")
	}
	// The one reader the runtime would have parked (see the TwoRuns test).
	activeCmd := waitForSlotAgent(m.agentCh)

	// The background run dies mid-active-turn.
	m.Update(slotEnvelope{ch: m.bg.ch, ev: nil})

	if !m.running {
		t.Fatal("the background finish clobbered the active turn's running flag")
	}
	if m.cfg.SessionID != activeSID {
		t.Fatal("the background finish swapped the active session")
	}
	if m.agentCh != activeCh {
		t.Fatal("the background finish displaced the active run's channel")
	}
	if m.bg == nil || !m.bg.done {
		t.Fatal("the background run was not finalized")
	}
	if !strings.Contains(noticesText(m), "Background session") {
		t.Fatal("the background finish notice is missing")
	}
	if m.approval != nil || m.question != nil {
		t.Fatal("the background finish touched the active slot's dialogs")
	}

	// Cleanup: drain the active turn.
	close(release)
	drainRun(t, m, activeCmd)
}

// TestBackground_EscEscCancelsBackgroundRun: Esc-Esc cancels the slot's run
// through its own cancel func — the run finalizes synthetically, a stashed
// approval is denied, and the slot restarts ITS OWN inbox. The active slot
// stays idle: the queue is the slot's, not the global one.
func TestBackground_EscEscCancelsBackgroundRun(t *testing.T) {
	release := make(chan struct{})
	t.Cleanup(func() { close(release) })
	m := newBgTestModel(t, &heldLLM{name: "bg-cancel", release: release})
	m.beginTurn()
	_ = m.startAgentLoop("long task")
	newM, _ := m.handleKey(tea.KeyPressMsg(ctrlB()))
	m = newM.(*model)

	// A stashed approval and a queued prompt wait on the slot's run.
	slotAgent, slotSID := m.bg.agent, m.bg.sessionID
	oldCh := m.bg.ch
	canceled := false
	m.bg.cancel = func() { canceled = true }
	m.bg.approval = &permission.ApprovalRequest{Tool: "bash", Reply: make(chan permission.ApprovalResult, 1)}
	m.bg.inbox = []queuedPrompt{{text: "waiting in the slot"}}
	esc := tea.KeyPressMsg(tea.Key{Code: tea.KeyEsc}).Key()

	newM, _, _ = m.handleInterruptKey(esc)
	m = newM.(*model)
	if m.escCancelCount != 1 {
		t.Fatalf("first Esc: escCancelCount = %d, want 1", m.escCancelCount)
	}
	if m.bg == nil || m.bg.done {
		t.Fatal("the first Esc canceled the run outright")
	}

	newM, _, _ = m.handleInterruptKey(esc)
	m = newM.(*model)
	if !canceled {
		t.Fatal("Esc-Esc did not call the slot's cancel func")
	}
	if m.bg == nil || m.bg.done {
		t.Fatal("Esc-Esc did not finalize the background run")
	}
	if m.bg.ch == oldCh {
		t.Fatal("the slot's inbox restart did not give the slot a fresh channel")
	}
	if m.bg.agent != slotAgent || m.bg.sessionID != slotSID {
		t.Fatal("the inbox restart left the slot's session")
	}
	if len(m.bg.inbox) != 0 {
		t.Fatalf("the slot still holds %d inbox prompts after the restart", len(m.bg.inbox))
	}
	if m.running {
		t.Fatal("the slot's restart leaked into the active slot's state")
	}
	if m.bg.cancel == nil {
		t.Fatal("the restarted slot run has no cancel func")
	}
	m.bg.cancel() // stop the restarted run; release unblocks its model call
}

// TestBackground_LateEventsAfterCancelDie: after a synthetic cancel the dead
// loop may still deliver through its parked envelope reader. Those events
// must be swallowed — no second finalization, no rendering — while the
// subagent monitor keeps its own reader.
func TestBackground_LateEventsAfterCancelDie(t *testing.T) {
	m := newTestModel(t)
	m.agentCh = make(chan agentMsg, 8)
	bgCh := make(chan agentMsg, 8)
	subCh := make(chan AgentSubEvent, 8)
	m.cfg.AgentEventCh = subCh
	m.beginTurn()
	m.running = false
	m.bg = &backgroundRun{started: time.Now(), ch: bgCh}

	_, cancelCmd, _ := m.handleInterruptKey(tea.KeyPressMsg(tea.Key{Code: tea.KeyEsc}).Key())
	_, _, _ = m.handleInterruptKey(tea.KeyPressMsg(tea.Key{Code: tea.KeyEsc}).Key())
	_ = cancelCmd
	if !m.bg.done {
		t.Fatal("the cancel did not finalize the run")
	}
	noticesAfterCancel := noticesText(m)

	// Late events from the dying loop: swallowed, not rendered, not
	// finalized again.
	m.Update(slotEnvelope{ch: bgCh, ev: agentTextMsg{text: "late"}})
	if strings.Contains(transcriptText(m), "late") {
		t.Fatal("a late event from a canceled run rendered")
	}
	m.Update(slotEnvelope{ch: bgCh, ev: nil})
	if n := strings.Count(noticesText(m), "Background session"); n != strings.Count(noticesAfterCancel, "Background session") {
		t.Fatalf("late delivery re-finalized the run: notices grew from %d to %d",
			strings.Count(noticesAfterCancel, "Background session"), strings.Count(noticesText(m), "Background session"))
	}
	// Subagent events ride the session channel: never swallowed.
	_, subCmd := m.Update(agentSubEventMsg{agentID: "a", kind: "text", content: "s"})
	if subCmd == nil {
		t.Fatal("a subagent event after cancel killed the monitor's reader")
	}
}

// TestBackground_CycleWorksWithBackgroundSlot pins the flipped guard: the
// slot freezes ITS OWN agent, so Shift+Tab rebuilds the ACTIVE agent freely
// while a background run lives. Under the v1 global guard this refused.
func TestBackground_CycleWorksWithBackgroundSlot(t *testing.T) {
	var asked []string
	m := newSwitchTestModel(t, &asked)
	m.bg = &backgroundRun{started: time.Now(), sessionID: "sess-bg"}

	shiftTab := (tea.KeyPressMsg{Code: tea.KeyTab, Mod: tea.ModShift}).Key()
	newM, _, _ := m.handleToggleKey(shiftTab)
	m = newM.(*model)
	if m.activeAgent != "build" {
		t.Fatalf("agent did not cycle under a live background slot: %q", m.activeAgent)
	}
	if len(asked) != 1 {
		t.Fatalf("the switcher ran %d times under a live slot, want 1", len(asked))
	}
	if strings.Contains(m.flash, "Background run active") {
		t.Fatalf("flash = %q, the v1 refusal must be gone", m.flash)
	}
}

// TestBackground_SlashCommandsWorkWithBackgroundSlot: a background slot owns
// its own session, so slash commands act on the active session immediately.
// Under the v1 global guard they were refused.
func TestBackground_SlashCommandsWorkWithBackgroundSlot(t *testing.T) {
	m := newBgTestModel(t, &heldLLM{name: "slash", release: make(chan struct{})})
	m.bg = &backgroundRun{started: time.Now(), sessionID: "sess-bg"}

	submitted, _, _ := m.handleInputSubmit(InputSubmitMsg{Text: "/session"})
	m = submitted.(*model)
	if strings.Contains(noticesText(m), "Slash commands are unavailable") {
		t.Fatal("slash commands are still refused while a slot is backgrounded")
	}
	if !strings.Contains(transcriptText(m), "Session: `"+m.cfg.SessionID+"`") {
		t.Fatalf("/session did not run on the active session: %q", transcriptText(m))
	}
}

// TestBackground_QueuedPromptStartsWithSlotAlive: the prompt queue is the
// ACTIVE session's inbox; a background slot never blocks it from starting.
func TestBackground_QueuedPromptStartsWithSlotAlive(t *testing.T) {
	release := make(chan struct{})
	t.Cleanup(func() { close(release) })
	m := newBgTestModel(t, &heldLLM{name: "queue", release: release})
	// Detached state: the input is free (running=false), the slot runs.
	m.bg = &backgroundRun{started: time.Now(), sessionID: "sess-bg"}
	m.pendingPrompts = []queuedPrompt{{text: "waiting"}}

	started, cmd := m.startNextPrompt()
	m = started.(*model)
	if cmd == nil {
		t.Fatal("startNextPrompt refused to start under a live background slot")
	}
	if !m.running || len(m.pendingPrompts) != 0 {
		t.Fatal("the queued prompt did not start")
	}
	m.agentCancel()
}

// TestBackground_BufferEvictsOldest: the ring keeps the newest 512 events and
// counts what it dropped, so the replay can say what was lost.
func TestBackground_BufferEvictsOldest(t *testing.T) {
	bg := &backgroundRun{}
	for i := 0; i < backgroundBufferCap+3; i++ {
		bg.record(agentTextMsg{text: fmt.Sprint(i)})
	}
	if len(bg.buf) != backgroundBufferCap {
		t.Fatalf("buffer holds %d events, want %d", len(bg.buf), backgroundBufferCap)
	}
	if bg.dropped != 3 {
		t.Fatalf("dropped = %d, want 3", bg.dropped)
	}
	if bg.seen != backgroundBufferCap+3 {
		t.Fatalf("seen = %d, want %d", bg.seen, backgroundBufferCap+3)
	}
	first := bg.buf[0].(agentTextMsg).text
	if first != "3" {
		t.Fatalf("oldest buffered event = %q, want \"3\" (eviction dropped 0..2)", first)
	}
}

// TestBackground_PumpRace is the ownership proof: one producer floods the
// channel while the test pumps it the way the TUI does — envelope → Update →
// re-arm — and every event lands exactly once. Run with -race.
func TestBackground_PumpRace(t *testing.T) {
	const total = 1200 // past the buffer cap: eviction runs under load too

	m := newTestModel(t)
	ch := make(chan agentMsg, 64)
	m.bg = &backgroundRun{started: time.Now(), ch: ch}

	go func() {
		for i := 0; i < total; i++ {
			ch <- agentTextMsg{text: fmt.Sprint(i)}
		}
		close(ch)
	}()

	seen := map[string]int{}
	parked := waitForSlotAgent(ch)
	for {
		if parked == nil {
			t.Fatal("no reader parked before the channel closed")
		}
		msg := parked()
		env, ok := msg.(slotEnvelope)
		if !ok {
			t.Fatalf("reader delivered %T, want slotEnvelope", msg)
		}
		if env.ev == nil {
			// The close: finalize through Update and stop.
			m.Update(env)
			break
		}
		text, ok := env.ev.(agentTextMsg)
		if !ok {
			t.Fatalf("producer delivered %T, want agentTextMsg", env.ev)
		}
		_, parked = m.Update(env)
		seen[text.text]++
	}

	if m.bg == nil || !m.bg.done {
		t.Fatal("the detached run did not finalize")
	}
	if m.bg.seen != total {
		t.Fatalf("seen = %d, want %d", m.bg.seen, total)
	}
	if m.bg.dropped != total-backgroundBufferCap {
		t.Fatalf("dropped = %d, want %d", m.bg.dropped, total-backgroundBufferCap)
	}
	for _, ev := range m.bg.buf {
		text := ev.(agentTextMsg).text
		if seen[text] != 1 {
			t.Fatalf("event %q delivered %d times, want exactly 1", text, seen[text])
		}
	}
}

// TestBackground_IdleKeyIsAFlash: Ctrl+B with nothing running and nothing
// backgrounded explains itself instead of swallowing the chord.
func TestBackground_IdleKeyIsAFlash(t *testing.T) {
	m := newTestModel(t)
	newM, _ := m.handleKey(tea.KeyPressMsg(ctrlB()))
	m = newM.(*model)
	if m.flash != "Nothing to background" {
		t.Fatalf("flash = %q, want \"Nothing to background\"", m.flash)
	}
	if m.bg != nil || m.running {
		t.Fatal("an idle Ctrl+B changed the run state")
	}
}

// TestBackground_RunCycleCannotDetach: /run events bypass the agent channel,
// so a detached entry would never see the cycle's end and wedge the queue.
func TestBackground_RunCycleCannotDetach(t *testing.T) {
	release := make(chan struct{})
	t.Cleanup(func() { close(release) })
	m := newBgTestModel(t, &heldLLM{name: "run-cycle", release: release})
	m.beginTurn()
	_ = m.startAgentLoop("cycle task")
	m.run = &runState{phase: "running"}
	newM, _ := m.handleKey(tea.KeyPressMsg(ctrlB()))
	m = newM.(*model)
	if m.bg != nil {
		t.Fatal("a /run cycle was backgrounded")
	}
	if !strings.Contains(m.flash, "/run cycles cannot be backgrounded") {
		t.Fatalf("flash = %q, want the /run refusal", m.flash)
	}

	// A finished cycle is no longer a cycle: detach works.
	m.run.phase = "done"
	newM, _ = m.handleKey(tea.KeyPressMsg(ctrlB()))
	m = newM.(*model)
	if m.bg == nil {
		t.Fatal("detach refused for a terminal /run state")
	}
	m.bg.cancel()
}

// TestBackground_StatusFieldRenders: the bar names the detached run while it
// works and marks it done once it finished unattached; no run, no field.
func TestBackground_StatusFieldRenders(t *testing.T) {
	p := paletteOrDark(Palette{})
	if got := statusBackgroundField(StatusRenderInput{}, p); got != nil {
		t.Fatalf("idle status field = %v, want nil", got)
	}
	alive := statusBackgroundField(StatusRenderInput{Background: &BackgroundStatus{
		Elapsed: 2*time.Minute + 13*time.Second, Events: 34,
	}}, p)
	if len(alive) != 1 || !strings.Contains(alive[0], "bg 2m 13s") || !strings.Contains(alive[0], "34 ev") {
		t.Fatalf("alive field = %v, want elapsed and event count", alive)
	}
	done := statusBackgroundField(StatusRenderInput{Background: &BackgroundStatus{Events: 34, Done: true}}, p)
	if len(done) != 1 || !strings.Contains(done[0], "✓") || !strings.Contains(done[0], "34 ev") {
		t.Fatalf("done field = %v, want the done marker and event count", done)
	}
}

// TestBackground_StatusDoneShowsDropped: a finished run's evicted events are
// gone for good — the done field carries the count.
func TestBackground_StatusDoneShowsDropped(t *testing.T) {
	p := paletteOrDark(Palette{})
	got := statusBackgroundField(StatusRenderInput{Background: &BackgroundStatus{
		Events: 34, Dropped: 12, Done: true,
	}}, p)
	if len(got) != 1 || !strings.Contains(got[0], "12 dropped") {
		t.Fatalf("done field = %v, want the dropped count", got)
	}
}

// TestBackground_AttachFailClosesAPriorDialog: a dialog left open from
// before the detach is answered denied when the stash opens its own — a
// loop blocks on one request, the old one was abandoned by construction.
func TestBackground_AttachFailClosesAPriorDialog(t *testing.T) {
	m := newTestModel(t)
	oldReply := make(chan permission.ApprovalResult, 1)
	m.approval = &permission.ApprovalRequest{Tool: "old", Reply: oldReply}
	m.bg = &backgroundRun{
		started:   time.Now(),
		sessionID: "sess-bg",
		approval:  &permission.ApprovalRequest{Tool: "stashed", Reply: make(chan permission.ApprovalResult, 1)},
	}
	newM, _ := m.handleKey(tea.KeyPressMsg(ctrlB()))
	m = newM.(*model)
	if m.approval == nil || m.approval.Tool != "stashed" {
		t.Fatalf("attach opened %+v, want the stashed dialog", m.approval)
	}
	select {
	case res := <-oldReply:
		if res.Allowed {
			t.Fatal("the prior dialog was answered allowed")
		}
	default:
		t.Fatal("the prior dialog was never answered")
	}
}

// TestBackground_EnvelopeAfterAttachRendersOrdinary: a reader armed while
// detached delivers after attach — the event renders through the ordinary
// path and the lineage ends (no envelope re-arm).
func TestBackground_EnvelopeAfterAttachRendersOrdinary(t *testing.T) {
	m := newTestModel(t)
	ch := make(chan agentMsg, 8)
	m.beginTurn()
	m.running = false
	m.bg = &backgroundRun{started: time.Now(), ch: ch}

	// Attach a live run: the input is busy again, the buffer replays.
	m.chatModel.AppendNotice("detach marker")
	m.handleKey(tea.KeyPressMsg(ctrlB()))
	if m.bg != nil || !m.running {
		t.Fatal("attach of a live run did not return it to the foreground")
	}

	// The pre-attach reader now delivers: rendered, unmarked re-arm.
	env := slotEnvelope{ch: ch, ev: agentTextMsg{text: "after attach"}}
	updated, cmd := m.Update(env)
	m = updated.(*model)
	if !strings.Contains(transcriptText(m), "after attach") {
		t.Fatal("a post-attach envelope event did not render")
	}
	if cmd == nil {
		t.Fatal("the ordinary path returned no re-arm for the live run")
	}
}

// TestBackground_RoutingFollowsTheEnvelopeChannel is the routing proof: the
// channel an envelope names decides everything. The bg channel's events
// buffer and re-arm on the bg channel; the active channel's render and
// re-arm on the active channel; a subagent event — which never rides a run
// channel — keeps the monitor's reader. Then the bg channel's close
// finalizes exactly once, and the monitor survives the run.
func TestBackground_RoutingFollowsTheEnvelopeChannel(t *testing.T) {
	m := newTestModel(t)
	activeCh := make(chan agentMsg, 8)
	bgCh := make(chan agentMsg, 8)
	subCh := make(chan AgentSubEvent, 8)
	m.cfg.AgentEventCh = subCh
	m.agentCh = activeCh
	m.beginTurn()
	m.running = false
	m.bg = &backgroundRun{started: time.Now(), ch: bgCh}

	// One event per channel, delivered detached: the returned commands are
	// the parked readers.
	_, cmdBg := m.Update(slotEnvelope{ch: bgCh, ev: agentTextMsg{text: "bg delta"}})
	_, cmdActive := m.Update(slotEnvelope{ch: activeCh, ev: agentTextMsg{text: "active delta"}})
	_, cmdSub := m.Update(agentSubEventMsg{agentID: "a", kind: "text", content: "sub delta"})
	if cmdBg == nil || cmdActive == nil || cmdSub == nil {
		t.Fatal("the detached routing returned no re-arm")
	}
	if m.bg.seen != 1 {
		t.Fatalf("seen = %d, want 1 — the bg delta must buffer, the active one render", m.bg.seen)
	}
	if !strings.Contains(transcriptText(m), "active delta") {
		t.Fatal("the active delta did not render")
	}
	if strings.Contains(transcriptText(m), "bg delta") {
		t.Fatal("the bg delta rendered instead of buffering")
	}

	// Feed each channel, then ask each parked command to read: every reader
	// must surface ITS channel's event — swapped channels here would mean a
	// parked second reader and a dead monitor.
	bgCh <- agentTextMsg{text: "bg next"}
	activeCh <- agentTextMsg{text: "active next"}
	subCh <- AgentSubEvent{AgentID: "a", Kind: "text", Content: "sub next"}
	bgEnv := readWithTimeout(t, cmdBg).(slotEnvelope)
	activeEnv := readWithTimeout(t, cmdActive).(slotEnvelope)
	subMsg := readWithTimeout(t, cmdSub)
	if bgEnv.ch != bgCh {
		t.Fatal("the bg reader is parked on the wrong channel")
	}
	if _, ok := bgEnv.ev.(agentTextMsg); !ok {
		t.Fatalf("bg reader delivered %T, want agentTextMsg", bgEnv.ev)
	}
	if activeEnv.ch != activeCh {
		t.Fatal("the active reader is parked on the wrong channel")
	}
	if _, ok := activeEnv.ev.(agentTextMsg); !ok {
		t.Fatalf("active reader delivered %T, want agentTextMsg", activeEnv.ev)
	}
	if _, ok := subMsg.(agentSubEventMsg); !ok {
		t.Fatalf("subagent reader delivered %T, want agentSubEventMsg", subMsg)
	}

	// Exactly one finalization at close: the bg channel closes, its reader
	// delivers the empty envelope; nothing else contributes.
	close(bgCh)
	env := readWithTimeout(t, cmdBg).(slotEnvelope)
	if env.ev != nil {
		t.Fatalf("close delivered %v, want a nil event", env.ev)
	}
	_, cmd := m.Update(env)
	if cmd != nil {
		t.Fatalf("finalization returned a command: re-arming a closed channel")
	}
	if n := strings.Count(noticesText(m), "Background session"); n != 1 {
		t.Fatalf("finalization notices = %d, want exactly 1", n)
	}
	// The monitor's reader survives the run: it still reads AgentEventCh.
	subCh <- AgentSubEvent{AgentID: "a", Kind: "text", Content: "after close"}
	if _, ok := cmdSub().(agentSubEventMsg); !ok {
		t.Fatal("the subagent reader died with the run")
	}
}

// TestBackground_TripleToggleFreesSlot: detach → attach → detach wraps the
// same channel in a new backgroundRun, while the reader parked by the first
// detach still delivers the close. Classified by pointer that close would
// land in the wrong run; the channel comparison makes it the real done of
// the run the slot holds now — and the slot's OWN inbox restarts, not the
// active session's queue.
func TestBackground_TripleToggleFreesSlot(t *testing.T) {
	release := make(chan struct{})
	t.Cleanup(func() { close(release) })
	m := newBgTestModel(t, &heldLLM{name: "triple", release: release})
	m.beginTurn()
	m.running = true
	m.agentCh = make(chan agentMsg, 8)

	// 1. Detach, and let one event arm an envelope reader for slot #1.
	newM, _ := m.handleKey(tea.KeyPressMsg(ctrlB()))
	m = newM.(*model)
	first := m.bg
	_, cmd1 := m.Update(slotEnvelope{ch: first.ch, ev: agentTextMsg{text: "delta"}})
	if cmd1 == nil {
		t.Fatal("no re-arm after the detached event")
	}

	// 2. Attach: the run is foreground again, the envelope reader stays
	// parked on the same channel.
	newM, _ = m.handleKey(tea.KeyPressMsg(ctrlB()))
	m = newM.(*model)
	if m.bg != nil || !m.running {
		t.Fatal("the intermediate attach did not return the run to the foreground")
	}

	// 3. Detach again: a fresh slot over the SAME channel. The queued
	// prompt rides along — it belongs to the slot's conversation now.
	m.pendingPrompts = []queuedPrompt{{text: "queued for the slot"}}
	newM, _ = m.handleKey(tea.KeyPressMsg(ctrlB()))
	m = newM.(*model)
	second := m.bg
	if second == nil || second == first {
		t.Fatal("expected a fresh slot over the same channel")
	}
	if second.ch != first.ch {
		t.Fatal("the second detach did not capture the run's channel")
	}
	if len(second.inbox) != 1 {
		t.Fatalf("the queued prompt did not move into the slot's inbox: %d", len(second.inbox))
	}

	// The run finishes in silence: the close arrives through the FIRST
	// detach's reader, in an envelope naming the same channel. The model's
	// agentCh is nil here — detach took the channel, so close the slot's.
	close(second.ch)
	env, ok := cmd1().(slotEnvelope)
	if !ok || env.ev != nil {
		t.Fatalf("the close delivered %T, want an empty slotEnvelope", cmd1())
	}
	m.Update(env)

	if m.bg == nil || m.bg.done {
		t.Fatal("the close did not finalize the current slot — it would hold the run forever")
	}
	if len(m.bg.inbox) != 0 {
		t.Fatalf("the slot inbox holds %d prompts after the restart", len(m.bg.inbox))
	}
	if m.running {
		t.Fatal("the slot's inbox restart leaked into the active slot's state")
	}
	if n := strings.Count(noticesText(m), "Background session"); n != 1 {
		t.Fatalf("finalization notices = %d, want exactly 1", n)
	}
	m.bg.cancel()
}

// TestBackground_UnknownTagQuestionIsDenied covers the question twin of the
// untagged-denial rule: a question with a dead session's tag is canceled,
// never routed.
func TestBackground_UnknownTagQuestionIsDenied(t *testing.T) {
	m := newBgTestModel(t, &heldLLM{name: "q-deny", release: make(chan struct{})})
	m.bg = &backgroundRun{started: time.Now(), sessionID: "sess-bg"}
	questionCh := make(chan SessionQuestion, 4)
	m.cfg.QuestionCh = questionCh
	reply := make(chan tools.QuestionAnswer, 1)

	m.Update(questionRequestMsg{sq: SessionQuestion{
		Req:       tools.QuestionRequest{Question: "stray?", Reply: reply},
		SessionID: "sess-gone",
	}})
	select {
	case res := <-reply:
		if res.Selected != "canceled" {
			t.Fatalf("the unknown-tagged question was answered %q, want canceled", res.Selected)
		}
	default:
		t.Fatal("the unknown-tagged question was never answered")
	}
	if m.question != nil {
		t.Fatal("the unknown-tagged question opened a dialog")
	}
	if m.bg.question != nil {
		t.Fatal("the unknown-tagged question was stashed for the wrong slot")
	}
}
