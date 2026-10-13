package tui

import (
	"context"
	"fmt"
	"iter"
	"strings"
	"testing"
	"time"

	llmmodel "google.golang.org/adk/v2/model"

	tea "charm.land/bubbletea/v2"

	"github.com/spa-skyson/pi-rate/internal/permission"
	"github.com/spa-skyson/pi-rate/internal/tools"
)

// The Ctrl+B state machine: running → detach → attach → done, plus the guards
// the one-live-loop invariant hangs on. The channel-ownership rule under test:
// the parked waitForAgent command stays the only reader through detach and
// attach; routing, never a second goroutine.

// twoTextLLM streams two text deltas per call, so a real runAgentLoop emits
// two agentTextMsg events per turn — enough to detach between them.
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

// newBgTestModel builds a model wired to a real agent over llm, ready to have
// startAgentLoop called on it.
func newBgTestModel(t *testing.T, llm llmmodel.LLM) *model {
	t.Helper()
	a, sid := newRunTestAgent(t, llm)
	m := newTestModel(t)
	m.cfg.Agent = a
	m.cfg.SessionID = sid
	return m
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

// noticesText joins the transcript's notices — the detach/finish/attention
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

// TestBackground_DetachBuffersAttachReplays walks the core cycle over a real
// agent: detach mid-turn, events buffer unrendered, attach replays them and
// returns the run to the foreground.
func TestBackground_DetachBuffersAttachReplays(t *testing.T) {
	m := newBgTestModel(t, &twoTextLLM{name: "bg"})
	m.beginTurn()
	cmd := m.startAgentLoop("long task")
	if cmd == nil {
		t.Fatal("startAgentLoop returned no command")
	}

	// Deliver the first text event: it renders, the run is foreground.
	msg := cmd()
	if _, ok := msg.(agentTextMsg); !ok {
		t.Fatalf("first event = %T, want agentTextMsg", msg)
	}
	_, cmd = m.Update(msg)
	if !strings.Contains(transcriptText(m), "part one") {
		t.Fatal("first delta did not render before detach")
	}

	// Ctrl+B: detached. Input frees, the run keeps working.
	newM, tick := m.handleKey(tea.KeyPressMsg(ctrlB()))
	m = newM.(*model)
	if m.running {
		t.Fatal("detach left m.running set")
	}
	if !m.bgAlive() {
		t.Fatal("detach did not arm the background run")
	}
	if tick == nil {
		t.Fatal("detach did not arm the status tick")
	}

	// Second delta arrives detached: buffered, not rendered. The re-arm
	// command from the last Update is the surviving parked reader; its
	// command is dropped here because attach consumes no further event.
	msg = cmd()
	_, _ = m.Update(msg)
	if strings.Contains(transcriptText(m), "part two") {
		t.Fatal("a detached delta rendered instead of buffering")
	}
	if m.bg.seen == 0 {
		t.Fatal("detached events were not counted")
	}

	// Ctrl+B again: attach replays the buffer.
	newM, _ = m.handleKey(tea.KeyPressMsg(ctrlB()))
	m = newM.(*model)
	if m.bg != nil {
		t.Fatal("attach left the background run armed")
	}
	if !m.running {
		t.Fatal("attach did not return the run to the foreground")
	}
	if !strings.Contains(transcriptText(m), "part two") {
		t.Fatal("attach did not replay the buffered delta")
	}
	if !strings.Contains(noticesText(m), "Run sent to background") {
		t.Fatal("detach notice missing")
	}
}

// TestBackground_DoneWhileDetachedStartsQueue pins the resolved design point:
// a background run that finishes on its own frees the agent even though
// nothing was attached, and a queued prompt starts on it.
func TestBackground_DoneWhileDetachedStartsQueue(t *testing.T) {
	m := newBgTestModel(t, &fnLLM{name: "bg-done", gen: func(int) (*llmmodel.LLMResponse, error) {
		return textResp("late answer"), nil
	}})
	m.beginTurn()
	_ = m.startAgentLoop("long task")

	newM, _ := m.handleKey(tea.KeyPressMsg(ctrlB()))
	m = newM.(*model)

	// Release turn #1 before done starts turn #2: two live loops on one
	// agent is exactly what the invariant forbids, even in a test.
	if m.agentCancel != nil {
		m.agentCancel()
	}

	// A prompt typed while detached goes to the queue and stays there.
	model3, _, _ := m.handleInputSubmit(InputSubmitMsg{Text: "queued while detached"})
	m = model3.(*model)
	if len(m.pendingPrompts) != 1 {
		t.Fatalf("queued prompt count = %d, want 1", len(m.pendingPrompts))
	}
	if m.flash != "Queued — background run active" {
		t.Fatalf("flash = %q, want the background queue notice", m.flash)
	}

	// The run finishes detached: the queue starts without an attach.
	m.Update(agentDoneMsg{})
	if m.bg == nil || !m.bg.done {
		t.Fatal("done did not finalize the background run")
	}
	if m.running != true {
		t.Fatal("the queued prompt did not start after the background run finished")
	}
	if len(m.pendingPrompts) != 0 {
		t.Fatalf("queue still holds %d prompts after start", len(m.pendingPrompts))
	}
	if !strings.Contains(noticesText(m), "Background run finished") {
		t.Fatal("finish notice missing")
	}
	// The new turn is live on a fresh channel; release it.
	if m.agentCancel != nil {
		m.agentCancel()
	}
}

// TestBackground_ApprovalDetachedStashOpensOnAttach: a permission request
// arriving from a detached run must not open the dialog — it stashes, notices,
// and attach opens it.
func TestBackground_ApprovalDetachedStashOpensOnAttach(t *testing.T) {
	m := newBgTestModel(t, &fnLLM{name: "bg-appr", gen: func(int) (*llmmodel.LLMResponse, error) {
		return textResp("working"), nil
	}})
	m.beginTurn()
	_ = m.startAgentLoop("long task")
	newM, _ := m.handleKey(tea.KeyPressMsg(ctrlB()))
	m = newM.(*model)

	approvalCh := make(chan permission.ApprovalRequest, 4)
	m.cfg.ApprovalCh = approvalCh
	req := permission.ApprovalRequest{Tool: "bash", Command: "rm -rf /", Reply: make(chan permission.ApprovalResult, 1)}

	m.Update(approvalRequestMsg{req: req})
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

// TestBackground_CycleRefusesWhileBgAlive: Shift+Tab may not cycle agents
// under a live background run — the rebuild would mutate the *Agent the run
// is executing on. Once the run is done, cycling works again.
func TestBackground_CycleRefusesWhileBgAlive(t *testing.T) {
	var asked []string
	m := newSwitchTestModel(t, &asked)
	m.bg = &backgroundRun{started: time.Now()}

	shiftTab := (tea.KeyPressMsg{Code: tea.KeyTab, Mod: tea.ModShift}).Key()
	newM, _, _ := m.handleToggleKey(shiftTab)
	m = newM.(*model)
	if m.activeAgent != "" {
		t.Fatalf("agent cycled to %q under a live background run", m.activeAgent)
	}
	if len(asked) != 0 {
		t.Fatal("the switcher ran under a live background run")
	}
	if !strings.Contains(m.flash, "Background run active") {
		t.Fatalf("flash = %q, want the background refusal", m.flash)
	}

	// A finished run holds nothing: the cycle works.
	m.bg.done = true
	newM, _, _ = m.handleToggleKey(shiftTab)
	m = newM.(*model)
	if m.activeAgent != "build" {
		t.Fatalf("agent did not cycle after the run finished: %q", m.activeAgent)
	}
}

// TestBackground_QueuedPromptWaitsWhileBgAlive: startNextPrompt must not
// start a turn on an agent a background run holds — the queue drains only
// after the run finishes (or is attached and finishes).
func TestBackground_QueuedPromptWaitsWhileBgAlive(t *testing.T) {
	m := newTestModel(t)
	// Detached state: the input is free (running=false), the run holds the
	// agent (bg alive).
	m.bg = &backgroundRun{started: time.Now()}
	m.pendingPrompts = []queuedPrompt{{text: "waiting"}}

	refused, cmd := m.startNextPrompt()
	m = refused.(*model)
	if cmd != nil {
		t.Fatal("startNextPrompt started a turn under a live background run")
	}
	if len(m.pendingPrompts) != 1 || m.running {
		t.Fatal("the queued prompt started instead of waiting")
	}

	// Done: the same call now starts it.
	m.bg.done = true
	updated, cmd := m.startNextPrompt()
	m = updated.(*model)
	if cmd == nil {
		t.Fatal("startNextPrompt did not start after the run finished")
	}
	if m.running != true || len(m.pendingPrompts) != 0 {
		t.Fatal("the queued prompt did not start")
	}
	if m.agentCancel != nil {
		m.agentCancel()
	}
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
// channel while the test pumps it the way the TUI does — command → message →
// Update → re-arm — and every event lands exactly once. Run with -race.
func TestBackground_PumpRace(t *testing.T) {
	const total = 1200 // past the buffer cap: eviction runs under load too

	m := newTestModel(t)
	m.beginTurn()
	m.agentCh = make(chan agentMsg, 64)
	m.running = false // detached by hand; the loop side is a plain producer
	m.bg = &backgroundRun{started: time.Now()}

	ch := m.agentCh
	go func() {
		for i := 0; i < total; i++ {
			ch <- agentTextMsg{text: fmt.Sprint(i)}
		}
		close(ch)
	}()

	seen := map[string]int{}
	parked := waitForAgent(ch)
	for {
		if parked == nil {
			t.Fatal("no reader parked before the channel closed")
		}
		msg := parked()
		if text, ok := msg.(agentTextMsg); ok {
			_, parked = m.Update(msg)
			seen[text.text]++
			continue
		}
		// agentDoneMsg (synthesized by the close): finalize through Update.
		m.Update(msg)
		break
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

// TestBackground_IntegrationDetachMidTurn drives a real runAgentLoop through
// detach and a detached finish, then attaches and checks the transcript
// caught up — the smoke test for the whole path end to end.
func TestBackground_IntegrationDetachMidTurn(t *testing.T) {
	m := newBgTestModel(t, &twoTextLLM{name: "smoke"})
	m.beginTurn()
	cmd := m.startAgentLoop("smoke run")
	if cmd == nil {
		t.Fatal("startAgentLoop returned no command")
	}

	// Pump until the first delta renders, then detach between deltas.
	first := cmd()
	if _, ok := first.(agentTextMsg); !ok {
		t.Fatalf("first event = %T, want agentTextMsg", first)
	}
	_, cmd = m.Update(first)
	newM, _ := m.handleKey(tea.KeyPressMsg(ctrlB()))
	m = newM.(*model)
	if !m.bgAlive() {
		t.Fatal("detach failed mid-turn")
	}

	// Pump the rest detached: events buffer, the close finalizes the run.
	for {
		msg := cmd()
		m.Update(msg)
		if _, ok := msg.(agentDoneMsg); ok {
			break
		}
	}
	if m.bg == nil || !m.bg.done {
		t.Fatal("the detached run did not finalize on close")
	}
	if m.bgAlive() {
		t.Fatal("a finalized run still holds the agent")
	}
	if strings.Contains(transcriptText(m), "part two") {
		t.Fatal("a detached delta rendered before attach")
	}

	// Attach after the fact: faithful replay, input idle again.
	newM, _ = m.handleKey(tea.KeyPressMsg(ctrlB()))
	m = newM.(*model)
	if m.bg != nil {
		t.Fatal("attach after finish left the background run armed")
	}
	if m.running {
		t.Fatal("attaching a finished run marked the model as running")
	}
	body := transcriptText(m)
	if !strings.Contains(body, "part one") || !strings.Contains(body, "part two") {
		t.Fatalf("replay incomplete: %q", body)
	}
	if !strings.Contains(noticesText(m), "Background run finished") {
		t.Fatal("finish notice missing after attach")
	}
}

// TestBackground_QuestionDetachedStashOpensOnAttach mirrors the approval
// stash for the question tool.
func TestBackground_QuestionDetachedStashOpensOnAttach(t *testing.T) {
	m := newBgTestModel(t, &fnLLM{name: "bg-q", gen: func(int) (*llmmodel.LLMResponse, error) {
		return textResp("working"), nil
	}})
	m.beginTurn()
	_ = m.startAgentLoop("long task")
	newM, _ := m.handleKey(tea.KeyPressMsg(ctrlB()))
	m = newM.(*model)

	questionCh := make(chan tools.QuestionRequest, 4)
	m.cfg.QuestionCh = questionCh
	req := tools.QuestionRequest{
		Question: "Proceed?",
		Options:  []tools.QuestionOption{{Label: "yes"}, {Label: "no"}},
		Reply:    make(chan tools.QuestionAnswer, 1),
	}

	m.Update(questionRequestMsg{req: req})
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
