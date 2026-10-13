package tui

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	llmmodel "google.golang.org/adk/v2/model"

	tea "charm.land/bubbletea/v2"

	"github.com/spa-skyson/pi-rate/internal/agent"
	"github.com/spa-skyson/pi-rate/internal/permission"
	pisession "github.com/spa-skyson/pi-rate/internal/session"
	"github.com/spa-skyson/pi-rate/internal/subagent"
	"github.com/spa-skyson/pi-rate/internal/tools"
)

// The Ctrl+B fork-on-detach state machine over the session registry (issue
// #74, MR2): running → detach → slot; every detach adds one slot, so N
// background sessions come from N sequential run+detach cycles. Attach swaps
// a slot's session with the active one, parking the displaced session back
// into the registry — switching never loses a session. The per-slot
// invariant under test: a slot's run never writes to ANOTHER slot's state or
// to the active slot, so a prompt typed to the fresh session starts
// immediately (the complaint the issue closes) and a background finish
// mid-active-turn leaves the active turn alone.
// The channel-ownership rule is unchanged: each run channel keeps exactly
// one parked reader through detach and attach; slotEnvelope names the
// channel, and the channel names the slot — routing, never a second
// goroutine.

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
// active session while the slot runs. Safe for several loops on one
// instance: the mutex only guards the call counter, the gate is shared.
type heldLLM struct {
	name    string
	mu      sync.Mutex
	calls   int
	release chan struct{}
}

func (l *heldLLM) Name() string { return l.name }

func (l *heldLLM) GenerateContent(_ context.Context, _ *llmmodel.LLMRequest, _ bool) iter.Seq2[*llmmodel.LLMResponse, error] {
	l.mu.Lock()
	l.calls++
	l.mu.Unlock()
	return func(yield func(*llmmodel.LLMResponse, error) bool) {
		<-l.release
		yield(textResp("done"), nil)
	}
}

// bgFactory is the stub session factory detach and /new fork through: a
// fresh real agent (same llm) and a fresh session id per call, the shape
// newSessionAgent in the CLI produces. Only Update calls it —
// single-threaded.
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
// session factory set — the minimum detachRun and /new need to fork.
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
// svc — the events.jsonl assertions need a FileService every session (the
// starting one and every forked one) is registered in.
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

// slotNotices joins the notices of a slot's own transcript — the slot's
// close-out lines live there, not in the active one.
func slotNotices(bg *backgroundRun) string {
	var b strings.Builder
	for _, msg := range bg.messages {
		if msg.isNotice {
			b.WriteString(msg.content + "\n")
		}
	}
	return b.String()
}

// slotByID finds a registry slot by session id in tests — the same lookup
// production routing does, kept explicit so a test names what it drives.
func slotByID(m *model, sid string) *backgroundRun {
	if s := m.slotBySession(sid); s != nil {
		return s
	}
	return nil
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

// TestBackground_RunConfigIsSnapshotted pins the race fix: the loop
// goroutine reads its config from the agentRunConfig snapshot, never from
// live m.cfg — /model may rewrite m.cfg (ProviderName, MidTurnAttempts) in
// Update while a slot's loop runs. Run with -race: the test rewrites both
// fields while a real loop is mid-flight; a loop reading live m.cfg turns
// this into a data race report.
func TestBackground_RunConfigIsSnapshotted(t *testing.T) {
	release := make(chan struct{})
	done := make(chan struct{})
	m := newBgTestModel(t, &heldLLM{name: "snap", release: release})
	ch := make(chan agentMsg, 8)
	other := 9
	// The snapshot is taken on the caller's side — passing m.agentRun()
	// inside the go statement would re-introduce the race it exists to
	// prevent (the argument evaluates in the new goroutine).
	run := m.agentRun()
	go func() {
		defer close(done)
		m.runAgentLoop(context.Background(), "snap", ch, run)
	}()
	// The loop reads its snapshot before the model call it is now blocked
	// in; rewrite the live fields the pre-fix code read mid-loop.
	m.cfg.MidTurnAttempts = &other
	m.cfg.ProviderName = "swapped-provider"
	close(release)
	<-done
	// The assertion is -race staying silent; the drained channel needs no
	// checks (buffered sends survive the test).
}

// TestBackground_DetachForksFreshSession walks the core cycle over a real
// agent: detach mid-stream moves the session whole into a slot and opens a
// fresh one; attach hands the slot back with the run still alive and parks
// the displaced session in the registry; the rest of the turn then renders
// in the foreground exactly once.
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
	m.todoState = &tools.TodoState{}
	newM, tick := m.handleKey(tea.KeyPressMsg(ctrlB()))
	m = newM.(*model)
	if m.running {
		t.Fatal("detach left m.running set")
	}
	if m.todoState != nil {
		t.Fatal("the fresh session shows the departed session's todo plan")
	}
	if len(m.slots) != 1 {
		t.Fatalf("registry holds %d slots after one detach, want 1", len(m.slots))
	}
	if tick == nil {
		t.Fatal("detach did not arm the status tick")
	}
	slot := m.slots[0]
	if slot.sessionID != origSID || slot.agent != origAgent {
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
	if len(slot.messages) == 0 {
		t.Fatal("the slot lost the original transcript")
	}
	if !strings.Contains(noticesText(m), "Session forked") {
		t.Fatal("the fork notice is missing")
	}

	// Ctrl+B again with one slot: attach returns the LIVE run — running,
	// its channel parked as the active one — and the displaced fresh
	// session stays in the registry.
	newM, _ = m.handleKey(tea.KeyPressMsg(ctrlB()))
	m = newM.(*model)
	if !m.running {
		t.Fatal("attach did not return the live run to the foreground")
	}
	if m.cfg.SessionID != origSID {
		t.Fatal("attach did not restore the slot's session id")
	}
	if len(m.slots) != 1 {
		t.Fatalf("registry holds %d slots after attach, want the parked displaced one", len(m.slots))
	}
	if m.slots[0].sessionID == origSID || !m.slots[0].done || m.slots[0].ch != nil {
		t.Fatal("the displaced fresh session was not parked idle in the registry")
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
	slot := m.slots[0]
	if slot == nil || slot.done {
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
	if slot.done != true {
		t.Fatal("the detached run did not finalize on close")
	}
	if strings.Contains(transcriptText(m), "part two") {
		t.Fatal("a detached delta rendered before attach")
	}
	// The finish notice is the slot's: invisible until attach, because the
	// active session must not grow text for a run it knows nothing about.
	if strings.Contains(noticesText(m), "Background session") {
		t.Fatal("the slot finish notice leaked into the active transcript")
	}
	if !strings.Contains(slotNotices(slot), "Background session") {
		t.Fatal("the slot finish notice did not land in the slot's transcript")
	}

	// Attach after the fact: faithful replay, input idle again. The
	// displaced fresh session parks back into the registry.
	newM, _ = m.handleKey(tea.KeyPressMsg(ctrlB()))
	m = newM.(*model)
	if m.running {
		t.Fatal("attaching a finished run marked the model as running")
	}
	if len(m.slots) != 1 {
		t.Fatalf("registry holds %d slots after attach, want the parked displaced one", len(m.slots))
	}
	body := transcriptText(m)
	if !strings.Contains(body, "part one") || !strings.Contains(body, "part two") {
		t.Fatalf("replay incomplete: %q", body)
	}
	if strings.Count(body, "part two") != 1 {
		t.Fatalf("replay duplicated the buffered delta: %q", body)
	}
	// The finish notice came back with the slot's transcript.
	if !strings.Contains(noticesText(m), "Background session") {
		t.Fatal("the slot finish notice did not come back with the attach")
	}
	if !strings.Contains(noticesText(m), "Attached") {
		t.Fatal("the attach notice is missing")
	}
}

// TestBackground_ThreeSlotsLiveParallel is the MR2 core: three sequential
// run+detach cycles leave three slots running in parallel, each persisting
// to ITS OWN events.jsonl, none bleeding into another. Attach walks across
// all three.
func TestBackground_ThreeSlotsLiveParallel(t *testing.T) {
	dir := t.TempDir()
	svc, err := pisession.NewFileService(dir)
	if err != nil {
		t.Fatal(err)
	}
	release := make(chan struct{})
	var closeOnce sync.Once
	closeRelease := func() { closeOnce.Do(func() { close(release) }) }
	t.Cleanup(closeRelease) // the happy path closes early; the cleanup covers failures
	llm := &heldLLM{name: "parallel", release: release}

	a1 := newPersistedTestAgent(t, llm, svc)
	sid1, _, err := a1.CreateSession(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	m := newTestModel(t)
	m.cfg.Agent = a1
	m.cfg.SessionID = sid1
	m.cfg.LLM = llm
	m.cfg.SessionService = svc
	m.cfg.NewSessionAgent = func(ctx context.Context, _ llmmodel.LLM) (*agent.Agent, string, string, error) {
		ag := newPersistedTestAgent(t, llm, svc)
		freshSID, title, err := ag.CreateSessionWithID(ctx, pisession.GenerateSessionID())
		return ag, freshSID, title, err
	}

	// Three run+detach cycles: each active session starts a turn on the
	// blocking LLM, then Ctrl+B detaches it into its own slot.
	m.beginTurn()
	_ = m.startAgentLoop("task one")
	sids := []string{sid1}
	for cycle := 0; cycle < 3; cycle++ {
		newM, _ := m.handleKey(tea.KeyPressMsg(ctrlB()))
		m = newM.(*model)
		if m.running {
			t.Fatalf("cycle %d: detach left the model running", cycle)
		}
		if len(m.slots) != cycle+1 {
			t.Fatalf("cycle %d: registry holds %d slots, want %d", cycle, len(m.slots), cycle+1)
		}
		sids = append(sids, m.cfg.SessionID)
		if cycle < 2 {
			// Start the next cycle's run in the fresh active session.
			m.beginTurn()
			_ = m.startAgentLoop(fmt.Sprintf("task %d", cycle+2))
			if !m.running {
				t.Fatalf("cycle %d: the next run did not start", cycle)
			}
		}
	}
	if len(m.slots) != 3 {
		t.Fatalf("registry holds %d slots, want 3", len(m.slots))
	}
	for i, s := range m.slots {
		if s.done {
			t.Fatalf("slot %d (%s) is not live", i, s.sessionID)
		}
	}

	// Release everything: each slot's loop completes and persists its turn
	// into its own events.jsonl. The parked envelope readers the loops were
	// armed with were dropped by the test on purpose — drain each slot's
	// channel the way Update would, so the closes finalize their slots.
	closeRelease()
	for _, s := range m.slots {
		for ev := range s.ch {
			m.Update(slotEnvelope{ch: s.ch, ev: ev})
		}
		m.Update(slotEnvelope{ch: s.ch, ev: nil}) // the close: the real done
	}
	for i, s := range m.slots {
		if !s.done {
			t.Fatalf("slot %d did not finalize after its channel closed", i)
		}
	}
	for _, s := range m.slots {
		eventsPath := filepath.Join(dir, s.sessionID, "events.jsonl")
		waitForCond(t, 5*time.Second, "events.jsonl of "+s.sessionID+" to appear", func() bool {
			st, err := os.Stat(eventsPath)
			return err == nil && st.Size() > 0
		})
	}
	// The active session started no turn: its file stays empty.
	freshPath := filepath.Join(dir, m.cfg.SessionID, "events.jsonl")
	if st, err := os.Stat(freshPath); err == nil && st.Size() > 0 {
		t.Fatalf("the idle active session's events.jsonl grew: %d bytes", st.Size())
	}

	// Attach walks across all three, parking the displaced each time.
	attached := map[string]bool{}
	for i := 0; i < 3; i++ {
		target := m.slots[0].sessionID
		if attached[target] {
			target = m.slots[1].sessionID
		}
		model2, _ := m.attachBackground(slotByID(m, target))
		m = model2.(*model)
		if m.cfg.SessionID != target {
			t.Fatalf("attach %d: active session = %s, want %s", i, m.cfg.SessionID, target)
		}
		if m.running {
			t.Fatalf("attach %d: a finished slot came back running", i)
		}
		attached[target] = true
	}
	if len(m.slots) != 3 {
		t.Fatalf("registry holds %d slots after three attaches, want 3 (each attach parks the displaced)", len(m.slots))
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
	bgSID := m.slots[0].sessionID

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
		freshSID, title, ferr := ag.CreateSessionWithID(ctx, pisession.GenerateSessionID())
		return ag, freshSID, title, ferr
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
// Ctrl+B itself detaches, so the refusal is reached through attachBackground
// directly here; the popup path carries the same guard and is pinned with
// the popup.
func TestBackground_AttachRefusesWhileActiveRuns(t *testing.T) {
	m := newTestModel(t)
	m.running = true
	activeSID := m.cfg.SessionID
	m.addSlot(&backgroundRun{started: time.Now(), sessionID: "sess-bg"})

	mdl, _ := m.attachBackground(m.slots[0])
	m = mdl.(*model)
	if m.cfg.SessionID != activeSID {
		t.Fatal("the refused attach changed the active session")
	}
	if len(m.slots) != 1 || m.slots[0].sessionID != "sess-bg" {
		t.Fatal("the refused attach displaced the slot")
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
	if len(m.slots) != 0 {
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
	slot := m.slots[0]

	approvalCh := make(chan SessionApproval, 4)
	m.cfg.ApprovalCh = approvalCh
	slot.sessionID = "sess-bg"
	req := permission.ApprovalRequest{Tool: "bash", Command: "rm -rf /", Reply: make(chan permission.ApprovalResult, 1)}

	m.Update(approvalRequestMsg{sa: SessionApproval{Req: req, SessionID: "sess-bg"}})
	if m.approval != nil {
		t.Fatal("the approval dialog opened while the run was detached")
	}
	if slot.approval == nil {
		t.Fatal("the request was not stashed for attach")
	}
	if !strings.Contains(noticesText(m), "needs your input") {
		t.Fatal("the needs-input notice is missing")
	}

	// Attach: the dialog opens from the stash, the slot parks empty.
	newM, _ = m.handleKey(tea.KeyPressMsg(ctrlB()))
	m = newM.(*model)
	if m.approval == nil || m.approval.Tool != "bash" {
		t.Fatalf("attach did not open the stashed dialog: %+v", m.approval)
	}
	if m.slots[0].approval != nil {
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
	slot := m.slots[0]

	questionCh := make(chan SessionQuestion, 4)
	m.cfg.QuestionCh = questionCh
	slot.sessionID = "sess-bg"
	req := tools.QuestionRequest{
		Question: "Proceed?",
		Options:  []tools.QuestionOption{{Label: "yes"}, {Label: "no"}},
		Reply:    make(chan tools.QuestionAnswer, 1),
	}

	m.Update(questionRequestMsg{sq: SessionQuestion{Req: req, SessionID: "sess-bg"}})
	if m.question != nil {
		t.Fatal("the question dialog opened while the run was detached")
	}
	if slot.question == nil {
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
	bgSlot := m.slots[0]

	// 1. Background-tagged: stashes behind a notice, dialog stays closed.
	m.Update(approvalRequestMsg{sa: SessionApproval{
		Req:       permission.ApprovalRequest{Tool: "bg-tool", Reply: bgReply},
		SessionID: bgSlot.sessionID,
	}})
	if m.approval != nil {
		t.Fatal("a background-tagged request opened the active dialog")
	}
	if bgSlot.approval == nil || bgSlot.approval.Tool != "bg-tool" {
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

	// 4. Unknown tag (a dead session's late arrival): denied too.
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
	m.Update(slotEnvelope{ch: bgSlot.ch, ev: nil}) // the slot's real close
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
	slot := m.slots[0]

	// The background run dies mid-active-turn.
	m.Update(slotEnvelope{ch: slot.ch, ev: nil})

	if !m.running {
		t.Fatal("the background finish clobbered the active turn's running flag")
	}
	if m.cfg.SessionID != activeSID {
		t.Fatal("the background finish swapped the active session")
	}
	if m.agentCh != activeCh {
		t.Fatal("the background finish displaced the active run's channel")
	}
	if !slot.done {
		t.Fatal("the background run was not finalized")
	}
	// The close-out notice belongs to the slot's transcript; the active
	// one stays clean.
	if !strings.Contains(slotNotices(slot), "Background session") {
		t.Fatal("the finish notice did not land in the slot's transcript")
	}
	if strings.Contains(noticesText(m), "Background session") {
		t.Fatal("the finish notice leaked into the active transcript")
	}
	if m.approval != nil || m.question != nil {
		t.Fatal("the background finish touched the active slot's dialogs")
	}

	// Cleanup: drain the active turn.
	close(release)
	drainRun(t, m, activeCmd)
}

// TestBackground_EscEscCancelsBackgroundRun: Esc-Esc cancels the newest
// live slot's run through its own cancel func — the run finalizes
// synthetically, a stashed approval is denied, and the slot restarts ITS OWN
// inbox. The active slot stays idle: the queue is the slot's, not the global
// one.
func TestBackground_EscEscCancelsBackgroundRun(t *testing.T) {
	release := make(chan struct{})
	t.Cleanup(func() { close(release) })
	m := newBgTestModel(t, &heldLLM{name: "bg-cancel", release: release})
	m.beginTurn()
	_ = m.startAgentLoop("long task")
	newM, _ := m.handleKey(tea.KeyPressMsg(ctrlB()))
	m = newM.(*model)

	// A stashed approval and a queued prompt wait on the slot's run.
	slot := m.slots[0]
	slotAgent, slotSID := slot.agent, slot.sessionID
	oldCh := slot.ch
	canceled := false
	slot.cancel = func() { canceled = true }
	slot.approval = &permission.ApprovalRequest{Tool: "bash", Reply: make(chan permission.ApprovalResult, 1)}
	slot.inbox = []queuedPrompt{{text: "waiting in the slot"}}
	esc := tea.KeyPressMsg(tea.Key{Code: tea.KeyEsc}).Key()

	newM, _, _ = m.handleInterruptKey(esc)
	m = newM.(*model)
	if m.escCancelCount != 1 {
		t.Fatalf("first Esc: escCancelCount = %d, want 1", m.escCancelCount)
	}
	if slot.done {
		t.Fatal("the first Esc canceled the run outright")
	}

	newM, _, _ = m.handleInterruptKey(esc)
	m = newM.(*model)
	if !canceled {
		t.Fatal("Esc-Esc did not call the slot's cancel func")
	}
	// The synthetic finalization happened — the stash died denied, the old
	// channel was replaced — and the slot's own inbox restarted, which
	// flips done back to false: the slot is busy with ITS conversation
	// again, that is the restart's signature, not a missed finalize.
	if slot.done {
		t.Fatal("Esc-Esc finalized the slot but its inbox did not restart")
	}
	if slot.ch == oldCh {
		t.Fatal("the slot's inbox restart did not give the slot a fresh channel")
	}
	if slot.agent != slotAgent || slot.sessionID != slotSID {
		t.Fatal("the inbox restart left the slot's session")
	}
	if len(slot.inbox) != 0 {
		t.Fatalf("the slot still holds %d inbox prompts after the restart", len(slot.inbox))
	}
	if m.running {
		t.Fatal("the slot's restart leaked into the active slot's state")
	}
	if slot.cancel == nil {
		t.Fatal("the restarted slot run has no cancel func")
	}
	slot.cancel() // stop the restarted run; release unblocks its model call
}

// TestBackground_EscEscTargetsNewestLiveSlot: with several slots, Esc-Esc
// cancels the most recently detached live one — stack discipline — and the
// older slot keeps running.
func TestBackground_EscEscTargetsNewestLiveSlot(t *testing.T) {
	release := make(chan struct{})
	t.Cleanup(func() { close(release) })
	m := newBgTestModel(t, &heldLLM{name: "newest", release: release})
	m.beginTurn()
	_ = m.startAgentLoop("task one")
	newM, _ := m.handleKey(tea.KeyPressMsg(ctrlB()))
	m = newM.(*model)
	first := m.slots[0]
	m.beginTurn()
	_ = m.startAgentLoop("task two")
	newM, _ = m.handleKey(tea.KeyPressMsg(ctrlB()))
	m = newM.(*model)
	second := m.slots[1]
	if first.done || second.done {
		t.Fatal("both slots must be live before the cancel")
	}

	esc := tea.KeyPressMsg(tea.Key{Code: tea.KeyEsc}).Key()
	newM, _, _ = m.handleInterruptKey(esc)
	m = newM.(*model)
	newM, _, _ = m.handleInterruptKey(esc)
	m = newM.(*model)
	if !second.done {
		t.Fatal("Esc-Esc did not cancel the newest live slot")
	}
	if first.done {
		t.Fatal("Esc-Esc canceled the older slot instead of the newest")
	}
	// The registry keeps both: the canceled one stays for review.
	if len(m.slots) != 2 {
		t.Fatalf("registry holds %d slots, want 2", len(m.slots))
	}
}

// TestBackground_QuitCancelsAllSlots: the first Ctrl+C cancels the active
// turn AND every live slot at once, warning before the second press quits.
func TestBackground_QuitCancelsAllSlots(t *testing.T) {
	release := make(chan struct{})
	t.Cleanup(func() { close(release) })
	m := newBgTestModel(t, &heldLLM{name: "quit", release: release})
	m.beginTurn()
	_ = m.startAgentLoop("task one")
	newM, _ := m.handleKey(tea.KeyPressMsg(ctrlB()))
	m = newM.(*model)
	m.beginTurn()
	_ = m.startAgentLoop("task two")
	newM, _ = m.handleKey(tea.KeyPressMsg(ctrlB()))
	m = newM.(*model)
	m.beginTurn()
	_ = m.startAgentLoop("active turn")
	if !m.running || len(m.liveSlots()) != 2 {
		t.Fatalf("setup: running=%v live=%d, want true/2", m.running, len(m.liveSlots()))
	}

	ctrlC := tea.KeyPressMsg(tea.Key{Code: 'c', Mod: tea.ModCtrl}).Key()
	newM, _, _ = m.handleInterruptKey(ctrlC)
	m = newM.(*model)
	if m.running {
		t.Fatal("Ctrl+C did not cancel the active turn")
	}
	for i, s := range m.slots {
		if !s.done {
			t.Fatalf("slot %d survived the quit-cancel", i)
		}
	}
	if m.quitting {
		t.Fatal("the first Ctrl+C quit instead of warning")
	}
}

// TestBackground_LateEventsAfterCancelDie: after a synthetic cancel the dead
// loop may still deliver through its parked envelope reader. Those events
// must be swallowed — no second finalization, no rendering — while the
// subagent monitor keeps its own reader.
func TestBackground_LateEventsAfterCancelDie(t *testing.T) {
	m := newTestModel(t)
	bgCh := make(chan agentMsg, 8)
	subCh := make(chan AgentSubEvent, 8)
	m.cfg.AgentEventCh = subCh
	m.beginTurn()
	m.running = false
	m.addSlot(&backgroundRun{started: time.Now(), ch: bgCh})
	slot := m.slots[0]

	_, cancelCmd, _ := m.handleInterruptKey(tea.KeyPressMsg(tea.Key{Code: tea.KeyEsc}).Key())
	_, _, _ = m.handleInterruptKey(tea.KeyPressMsg(tea.Key{Code: tea.KeyEsc}).Key())
	_ = cancelCmd
	if !slot.done {
		t.Fatal("the cancel did not finalize the run")
	}
	noticesAfterCancel := slotNotices(slot)

	// Late events from the dying loop: swallowed, not rendered, not
	// finalized again.
	m.Update(slotEnvelope{ch: bgCh, ev: agentTextMsg{text: "late"}})
	if strings.Contains(transcriptText(m), "late") {
		t.Fatal("a late event from a canceled run rendered")
	}
	m.Update(slotEnvelope{ch: bgCh, ev: nil})
	if n := strings.Count(slotNotices(slot), "Background session"); n != strings.Count(noticesAfterCancel, "Background session") {
		t.Fatalf("late delivery re-finalized the run: slot notices grew from %d to %d",
			strings.Count(noticesAfterCancel, "Background session"), strings.Count(slotNotices(slot), "Background session"))
	}
	// Subagent events ride the session channel: never swallowed.
	_, subCmd := m.Update(agentSubEventMsg{agentID: "a", kind: "text", content: "s"})
	if subCmd == nil {
		t.Fatal("a subagent event after cancel killed the monitor's reader")
	}
}

// TestBackground_CycleWorksWithBackgroundSlots pins the flipped guard: a
// slot freezes ITS OWN agent, so Shift+Tab rebuilds the ACTIVE agent freely
// while background slots live. Under the v1 global guard this refused.
func TestBackground_CycleWorksWithBackgroundSlots(t *testing.T) {
	var asked []string
	m := newSwitchTestModel(t, &asked)
	m.addSlot(&backgroundRun{started: time.Now(), sessionID: "sess-bg"})

	shiftTab := (tea.KeyPressMsg{Code: tea.KeyTab, Mod: tea.ModShift}).Key()
	newM, _, _ := m.handleToggleKey(shiftTab)
	m = newM.(*model)
	if m.activeAgent != "build" {
		t.Fatalf("agent did not cycle under live background slots: %q", m.activeAgent)
	}
	if len(asked) != 1 {
		t.Fatalf("the switcher ran %d times under live slots, want 1", len(asked))
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
	m.addSlot(&backgroundRun{started: time.Now(), sessionID: "sess-bg"})

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
// ACTIVE session's inbox; background slots never block it from starting.
func TestBackground_QueuedPromptStartsWithSlotAlive(t *testing.T) {
	release := make(chan struct{})
	t.Cleanup(func() { close(release) })
	m := newBgTestModel(t, &heldLLM{name: "queue", release: release})
	// Detached state: the input is free (running=false), the slot runs.
	m.addSlot(&backgroundRun{started: time.Now(), sessionID: "sess-bg"})
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
	m.addSlot(&backgroundRun{started: time.Now(), ch: ch})
	slot := m.slots[0]

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

	if !slot.done {
		t.Fatal("the detached run did not finalize")
	}
	if slot.seen != total {
		t.Fatalf("seen = %d, want %d", slot.seen, total)
	}
	if slot.dropped != total-backgroundBufferCap {
		t.Fatalf("dropped = %d, want %d", slot.dropped, total-backgroundBufferCap)
	}
	for _, ev := range slot.buf {
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
	if len(m.slots) != 0 || m.running {
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
	if len(m.slots) != 0 {
		t.Fatal("a /run cycle was backgrounded")
	}
	if !strings.Contains(m.flash, "/run cycles cannot be backgrounded") {
		t.Fatalf("flash = %q, want the /run refusal", m.flash)
	}

	// A finished cycle is no longer a cycle: detach works.
	m.run.phase = "done"
	newM, _ = m.handleKey(tea.KeyPressMsg(ctrlB()))
	m = newM.(*model)
	if len(m.slots) != 1 {
		t.Fatal("detach refused for a terminal /run state")
	}
	m.slots[0].cancel()
}

// TestBackground_StatusFieldRenders: the bar aggregates the registry — the
// slot count with buffered events, plus the ⚑ badge for slots waiting for
// input; empty registry, no field.
func TestBackground_StatusFieldRenders(t *testing.T) {
	p := paletteOrDark(Palette{})
	if got := statusBackgroundField(StatusRenderInput{}, p); got != nil {
		t.Fatalf("idle status field = %v, want nil", got)
	}
	three := statusBackgroundField(StatusRenderInput{Background: &BackgroundSummary{
		Slots: 3, Live: 2, Events: 34,
	}}, p)
	if len(three) != 1 || !strings.Contains(three[0], "bg 3") || !strings.Contains(three[0], "34 ev") {
		t.Fatalf("three-slot field = %v, want the slot count and event count", three)
	}
	waiting := statusBackgroundField(StatusRenderInput{Background: &BackgroundSummary{
		Slots: 3, Live: 2, Awaiting: 1, Events: 34,
	}}, p)
	if len(waiting) != 2 || !strings.Contains(waiting[1], "⚑ 1") {
		t.Fatalf("waiting field = %v, want the ⚑ badge beside the count", waiting)
	}
}

// TestBackground_StatusSummaryTracksSlots pins the summary's arithmetic over
// the registry: one slot finishing (and one waiting) must move Live and
// Awaiting while Slots holds — the badge the bar renders comes from here.
func TestBackground_StatusSummaryTracksSlots(t *testing.T) {
	m := newTestModel(t)
	m.addSlot(&backgroundRun{started: time.Now(), sessionID: "s1"})
	m.addSlot(&backgroundRun{started: time.Now(), sessionID: "s2"})
	m.addSlot(&backgroundRun{started: time.Now(), sessionID: "s3", done: true})

	sum := m.backgroundSummary()
	if sum == nil || sum.Slots != 3 || sum.Live != 2 {
		t.Fatalf("summary = %+v, want 3 slots / 2 live", sum)
	}
	if sum.Awaiting != 0 {
		t.Fatalf("summary = %+v, want no awaiting slot yet", sum)
	}

	// A stashed request turns the ⚑ on; the finish clears it (the stash
	// dies with the run — the deny needs a buffered Reply to land in).
	m.slots[2].approval = &permission.ApprovalRequest{Tool: "bash", Reply: make(chan permission.ApprovalResult, 1)}
	if sum := m.backgroundSummary(); sum.Awaiting != 1 {
		t.Fatalf("summary = %+v, want 1 awaiting", sum)
	}
	slot2 := m.slots[1]
	m.finishSlotRun(slot2, agentDoneMsg{})
	if sum := m.backgroundSummary(); sum.Live != 1 || sum.Awaiting != 1 || sum.Slots != 3 {
		t.Fatalf("summary after one finish = %+v, want 3 slots / 1 live / 1 awaiting", sum)
	}
	m.finishSlotRun(m.slots[2], agentDoneMsg{})
	if sum := m.backgroundSummary(); sum.Awaiting != 0 {
		t.Fatalf("summary after the waiting slot finished = %+v, want 0 awaiting", sum)
	}
}

// TestBackground_AttachFailClosesAPriorDialog: a dialog left open from
// before the detach is answered denied when the stash opens its own — a
// loop blocks on one request, the old one was abandoned by construction.
func TestBackground_AttachFailClosesAPriorDialog(t *testing.T) {
	m := newTestModel(t)
	oldReply := make(chan permission.ApprovalResult, 1)
	m.approval = &permission.ApprovalRequest{Tool: "old", Reply: oldReply}
	m.addSlot(&backgroundRun{
		started:   time.Now(),
		sessionID: "sess-bg",
		approval:  &permission.ApprovalRequest{Tool: "stashed", Reply: make(chan permission.ApprovalResult, 1)},
	})
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
	m.addSlot(&backgroundRun{started: time.Now(), ch: ch})

	// Attach a live run: the input is busy again, the buffer replays. The
	// displaced idle session parks back into the registry.
	m.chatModel.AppendNotice("detach marker")
	m.handleKey(tea.KeyPressMsg(ctrlB()))
	if !m.running {
		t.Fatal("attach of a live run did not return it to the foreground")
	}
	if len(m.slots) != 1 {
		t.Fatalf("registry holds %d slots after attach, want the parked displaced one", len(m.slots))
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
// channel an envelope names decides everything — now across a registry. The
// slot's events buffer and re-arm on its channel; the active channel's
// render and re-arm on the active channel; a subagent event — which never
// rides a run channel — keeps the monitor's reader. Then the slot's close
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
	m.addSlot(&backgroundRun{started: time.Now(), ch: bgCh})
	slot := m.slots[0]

	// One event per channel, delivered detached: the returned commands are
	// the parked readers.
	_, cmdBg := m.Update(slotEnvelope{ch: bgCh, ev: agentTextMsg{text: "bg delta"}})
	_, cmdActive := m.Update(slotEnvelope{ch: activeCh, ev: agentTextMsg{text: "active delta"}})
	_, cmdSub := m.Update(agentSubEventMsg{agentID: "a", kind: "text", content: "sub delta"})
	if cmdBg == nil || cmdActive == nil || cmdSub == nil {
		t.Fatal("the detached routing returned no re-arm")
	}
	if slot.seen != 1 {
		t.Fatalf("seen = %d, want 1 — the bg delta must buffer, the active one render", slot.seen)
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
	if n := strings.Count(slotNotices(slot), "Background session"); n != 1 {
		t.Fatalf("finalization slot notices = %d, want exactly 1", n)
	}
	// The monitor's reader survives the run: it still reads AgentEventCh.
	subCh <- AgentSubEvent{AgentID: "a", Kind: "text", Content: "after close"}
	if _, ok := cmdSub().(agentSubEventMsg); !ok {
		t.Fatal("the subagent reader died with the run")
	}
}

// TestBackground_TripleToggleFreesSlot: detach → attach → detach wraps the
// same channel in a new slot, while the reader parked by the first detach
// still delivers the close. Classified by pointer that close would land in
// the wrong slot; the channel comparison makes it the real done of the run
// the registry holds now — and the slot's OWN inbox restarts, not the
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
	first := m.slots[0]
	runCh := first.ch // the attach parks the slot channelless — capture now
	_, cmd1 := m.Update(slotEnvelope{ch: first.ch, ev: agentTextMsg{text: "delta"}})
	if cmd1 == nil {
		t.Fatal("no re-arm after the detached event")
	}

	// 2. Attach: the run is foreground again; the displaced fresh session
	// parks back — and the same slot struct now wraps the run's channel
	// for the NEXT detach (the slot is reused in place by the swap).
	newM, _ = m.handleKey(tea.KeyPressMsg(ctrlB()))
	m = newM.(*model)
	if !m.running {
		t.Fatal("the intermediate attach did not return the run to the foreground")
	}
	if len(m.slots) != 1 {
		t.Fatalf("registry holds %d slots after attach, want the parked one", len(m.slots))
	}

	// 3. Detach again: the run moves back into the registry — a new slot
	// over the SAME channel. The queued prompt rides along — it belongs to
	// the slot's conversation now.
	m.pendingPrompts = []queuedPrompt{{text: "queued for the slot"}}
	newM, _ = m.handleKey(tea.KeyPressMsg(ctrlB()))
	m = newM.(*model)
	if len(m.slots) != 2 {
		t.Fatalf("registry holds %d slots after the second detach, want 2", len(m.slots))
	}
	second := m.slots[1]
	if second.ch != runCh {
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

	if second.done {
		t.Fatal("the close did not finalize the current slot — it would hold the run forever")
	}
	if len(second.inbox) != 0 {
		t.Fatalf("the slot inbox holds %d prompts after the restart", len(second.inbox))
	}
	if m.running {
		t.Fatal("the slot's inbox restart leaked into the active slot's state")
	}
	if n := strings.Count(slotNotices(second), "Background session"); n != 1 {
		t.Fatalf("finalization slot notices = %d, want exactly 1", n)
	}
	second.cancel()
}

// TestBackground_AttachRestoresOpenBlocks pins the accumulator restore: the
// slot's transcript may end in an open assistant or thinking block, and the
// buffered deltas replayed right after the swap must EXTEND it — the
// accumulators the live handlers write through have to come back with the
// transcript, or the first replayed delta overwrites the pre-detach part.
// The replay window is before the attach notice, so the pin drives the
// replay through bg.buf, the real path.
func TestBackground_AttachRestoresOpenBlocks(t *testing.T) {
	t.Run("thinking", func(t *testing.T) {
		m := newTestModel(t)
		m.addSlot(&backgroundRun{
			started: time.Now(), done: true, sessionID: "sess-bg", title: "slot title",
			messages: []message{
				{role: "user", content: "q"},
				{role: "thinking", content: "reasoning so far "},
			},
			buf: []agentMsg{agentThinkingMsg{text: "more"}},
		})
		m.todoState = &tools.TodoState{}
		m.chatModel.AppendNotice("fork marker")

		newM, _ := m.handleKey(tea.KeyPressMsg(ctrlB()))
		m = newM.(*model)
		if m.todoState != nil {
			t.Fatal("the displaced plan survived the attach")
		}
		if m.sessionTitle != "slot title" {
			t.Fatalf("sessionTitle = %q, want the slot's title back", m.sessionTitle)
		}
		if m.chatModel.Thinking != "reasoning so far more" {
			t.Fatalf("Thinking = %q, want the accumulator extended by the replay", m.chatModel.Thinking)
		}
		last := m.chatModel.Messages[len(m.chatModel.Messages)-2] // [-1] is the attach notice
		if last.role != "thinking" || last.content != "reasoning so far more" {
			t.Fatalf("thinking block = %q %q, want %q — the pre-detach part was lost", last.role, last.content, "reasoning so far more")
		}
	})

	t.Run("text", func(t *testing.T) {
		m := newTestModel(t)
		m.addSlot(&backgroundRun{
			started: time.Now(), done: true, sessionID: "sess-bg", title: "slot title",
			messages: []message{
				{role: "user", content: "q"},
				{role: "assistant", content: "part one "},
			},
			buf: []agentMsg{agentTextMsg{text: "part two"}},
		})
		m.chatModel.AppendNotice("fork marker")

		newM, _ := m.handleKey(tea.KeyPressMsg(ctrlB()))
		m = newM.(*model)
		last := m.chatModel.Messages[len(m.chatModel.Messages)-2] // [-1] is the attach notice
		want := "part one part two"
		if last.role != "assistant" || last.content != want {
			t.Fatalf("assistant block = %q %q, want %q — the pre-detach part was lost", last.role, last.content, want)
		}
	})
}

// TestBackground_RetryStateFollowsTheSession: Ctrl+R replays the prompt of
// the session that actually failed. The retry trio (lastPrompt, mentions,
// failed flag) moves into the slot on detach and back on attach — never
// across sessions.
func TestBackground_RetryStateFollowsTheSession(t *testing.T) {
	release := make(chan struct{})
	t.Cleanup(func() { close(release) })
	m := newBgTestModel(t, &heldLLM{name: "retry", release: release})

	// The slot session has a failed prompt behind it.
	m.running = true
	m.agentCh = make(chan agentMsg, 8)
	m.lastPrompt = "slot failed prompt"
	m.lastMentions = []string{"notes.md"}
	m.lastPromptFailed = true

	newM, _ := m.handleKey(tea.KeyPressMsg(ctrlB()))
	m = newM.(*model)
	if m.lastPrompt != "" || m.lastPromptFailed {
		t.Fatal("the fresh session inherited the slot's retry state")
	}
	slot := m.slots[0]
	if slot.lastPrompt != "slot failed prompt" || !slot.lastPromptFailed {
		t.Fatal("the slot lost its own retry state on detach")
	}

	// The fresh session runs a prompt and fails it.
	_, _, _ = m.handleInputSubmit(InputSubmitMsg{Text: "fresh question"})
	m.Update(slotEnvelope{ch: m.agentCh, ev: agentDoneMsg{err: errors.New("boom")}})
	if !m.lastPromptFailed || m.lastPrompt != "fresh question" {
		t.Fatalf("fresh retry state = %q failed=%v, want the failed fresh prompt", m.lastPrompt, m.lastPromptFailed)
	}

	// Attach: the slot's retry state comes back, and Ctrl+R replays the
	// SLOT's prompt — not the fresh session's. The slot's run finished
	// while detached, so the attach is a plain swap.
	slot.done = true
	newM, _ = m.handleKey(tea.KeyPressMsg(ctrlB()))
	m = newM.(*model)
	if m.lastPrompt != "slot failed prompt" || !m.lastPromptFailed {
		t.Fatalf("attach restored %q failed=%v, want the slot's failed prompt", m.lastPrompt, m.lastPromptFailed)
	}
	model2, _ := m.handleRetry()
	m = model2.(*model)
	found := false
	for _, msg := range m.chatModel.Messages {
		// submitPrompt wraps mentions into the wire form, so match the
		// prompt's head, not the whole body.
		if msg.role == "user" && strings.HasPrefix(msg.content, "slot failed prompt") {
			found = true
		}
	}
	if !found {
		t.Fatalf("Ctrl+R did not replay the slot's failed prompt: transcript %q", transcriptText(m))
	}
	if !m.running {
		t.Fatal("retry sanity: the replayed prompt should start a turn")
	}
	m.agentCancel() // the retried turn runs on the blocking LLM

	// Detach again: the slot session leaves with its own retry state, and
	// the new fresh session starts with NONE — Ctrl+R there must not
	// replay another session's prompt.
	newM, _ = m.handleKey(tea.KeyPressMsg(ctrlB()))
	m = newM.(*model)
	if m.lastPrompt != "" || m.lastPromptFailed {
		t.Fatalf("second detach left retry state %q failed=%v — a fresh session must not inherit one", m.lastPrompt, m.lastPromptFailed)
	}
	retrySlot := m.slots[len(m.slots)-1]
	if retrySlot.lastPrompt != "slot failed prompt" {
		t.Fatal("the slot lost its retry state on the second detach")
	}
	if model3, _ := m.handleRetry(); model3.(*model).running {
		t.Fatal("Ctrl+R replayed into a session with no failed prompt")
	}
	retrySlot.cancel()
}

// TestBackground_AttachRefusesUnderPlan: symmetric with detach — plan
// finalization writes global model state, so the swap is refused while a
// plan worktree is pending.
func TestBackground_AttachRefusesUnderPlan(t *testing.T) {
	m := newTestModel(t)
	m.mode = "plan"
	m.planWorktree = &subagent.WorktreeManager{}
	m.addSlot(&backgroundRun{started: time.Now(), sessionID: "sess-bg"})

	newM, _ := m.handleKey(tea.KeyPressMsg(ctrlB()))
	m = newM.(*model)
	if len(m.slots) != 1 {
		t.Fatal("attach displaced the slot under a pending plan")
	}
	if !strings.Contains(m.flash, "Cannot attach while a plan session is active") {
		t.Fatalf("flash = %q, want the plan refusal", m.flash)
	}
}

// TestBackground_LateRequestForFinishedSlotIsDenied: a request tagged with a
// finished slot's session dies denied/canceled — the tag is known, the
// addressee is gone. It must not open a dialog, stash, or read as an
// unknown-session factory bug.
func TestBackground_LateRequestForFinishedSlotIsDenied(t *testing.T) {
	m := newBgTestModel(t, &heldLLM{name: "late", release: make(chan struct{})})
	m.addSlot(&backgroundRun{started: time.Now(), done: true, sessionID: "sess-bg"})
	m.cfg.ApprovalCh = make(chan SessionApproval, 4)
	m.cfg.QuestionCh = make(chan SessionQuestion, 4)

	approvalReply := make(chan permission.ApprovalResult, 1)
	m.Update(approvalRequestMsg{sa: SessionApproval{
		Req:       permission.ApprovalRequest{Tool: "late-tool", Reply: approvalReply},
		SessionID: "sess-bg",
	}})
	select {
	case res := <-approvalReply:
		if res.Allowed {
			t.Fatal("the finished slot's approval was allowed")
		}
	default:
		t.Fatal("the finished slot's approval was never answered")
	}
	if m.approval != nil {
		t.Fatal("the finished slot's approval opened a dialog")
	}
	if m.slots[0].approval != nil {
		t.Fatal("the finished slot's approval was stashed")
	}

	questionReply := make(chan tools.QuestionAnswer, 1)
	m.Update(questionRequestMsg{sq: SessionQuestion{
		Req:       tools.QuestionRequest{Question: "late?", Reply: questionReply},
		SessionID: "sess-bg",
	}})
	select {
	case res := <-questionReply:
		if res.Selected != "canceled" {
			t.Fatalf("the finished slot's question was answered %q, want canceled", res.Selected)
		}
	default:
		t.Fatal("the finished slot's question was never answered")
	}
	if m.question != nil || m.slots[0].question != nil {
		t.Fatal("the finished slot's question opened a dialog or stashed")
	}
}

// TestBackground_UnknownTagQuestionIsDenied covers the question twin of the
// untagged-denial rule: a question with no tag (a factory bug) or a dead
// session's tag is canceled, never routed.
func TestBackground_UnknownTagQuestionIsDenied(t *testing.T) {
	m := newBgTestModel(t, &heldLLM{name: "q-deny", release: make(chan struct{})})
	m.addSlot(&backgroundRun{started: time.Now(), sessionID: "sess-bg"})
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
	if m.slots[0].question != nil {
		t.Fatal("the unknown-tagged question was stashed for the wrong slot")
	}

	// Untagged: the factory-bug case. Denied the same way, never guessed.
	untagged := make(chan tools.QuestionAnswer, 1)
	m.Update(questionRequestMsg{sq: SessionQuestion{
		Req: tools.QuestionRequest{Question: "untagged?", Reply: untagged},
	}})
	select {
	case res := <-untagged:
		if res.Selected != "canceled" {
			t.Fatalf("the untagged question was answered %q, want canceled", res.Selected)
		}
	default:
		t.Fatal("the untagged question was never answered")
	}
	if m.question != nil || m.slots[0].question != nil {
		t.Fatal("the untagged question opened a dialog or stashed")
	}
}
