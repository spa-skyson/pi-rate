package tui

import (
	"context"
	"fmt"
	"iter"
	stdlog "log"
	"strings"
	"sync"
	"testing"
	"time"

	llmmodel "google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/genai"

	"github.com/spa-skyson/pi-rate/internal/agent"
	pisession "github.com/spa-skyson/pi-rate/internal/session"
)

// spinLLM reproduces the issue #60 shape: every round emits a bash call with
// identical name and args but a fresh provider-minted id, until round texts —
// then it answers with text and the turn completes. The stuck detector calls
// the run dead on the 10th identical call, which strands that call mid-turn
// exactly like the live incident did.
type spinLLM struct {
	name string
	mu   sync.Mutex
	n    int
	// texts is the round on which the model stops calling the tool and
	// answers with text instead; a negative value never does.
	texts int
}

func (l *spinLLM) Name() string { return l.name }

func (l *spinLLM) GenerateContent(_ context.Context, _ *llmmodel.LLMRequest, _ bool) iter.Seq2[*llmmodel.LLMResponse, error] {
	l.mu.Lock()
	call := l.n
	l.n++
	l.mu.Unlock()
	return func(yield func(*llmmodel.LLMResponse, error) bool) {
		if l.texts >= 0 && call >= l.texts {
			yield(textResp("done talking"), nil)
			return
		}
		yield(&llmmodel.LLMResponse{
			Content: &genai.Content{
				Role: genai.RoleModel,
				Parts: []*genai.Part{{
					FunctionCall: &genai.FunctionCall{
						ID:   fmt.Sprintf("call_%02d", call),
						Name: "bash",
						Args: map[string]any{"command": "echo hi"},
					},
				}},
			},
		}, nil)
	}
}

// newSynthTestAgent builds a real agent over a real on-disk session service —
// the production pairing — plus the created session id.
func newSynthTestAgent(t *testing.T, llm llmmodel.LLM) (*agent.Agent, *pisession.FileService, string) {
	t.Helper()
	svc, err := pisession.NewFileService(t.TempDir())
	if err != nil {
		t.Fatalf("NewFileService: %v", err)
	}
	a, err := agent.New(agent.Config{Model: llm, SessionService: svc, Instruction: "test"})
	if err != nil {
		t.Fatalf("agent.New: %v", err)
	}
	sid, _, err := a.CreateSession(context.Background())
	if err != nil {
		t.Fatalf("CreateSession: %v", err)
	}
	return a, svc, sid
}

// driveSynthRunLoop runs m.runAgentLoop with the session service wired in —
// driveRunLoop's Config leaves it nil, which would no-op the sweep under test.
func driveSynthRunLoop(t *testing.T, a *agent.Agent, svc *pisession.FileService, sid, prompt string) runResult {
	t.Helper()
	m := &model{cfg: Config{Agent: a, SessionID: sid, SessionService: svc}, ctx: context.Background()}
	m.agentCh = make(chan agentMsg, 1024)

	var res runResult
	done := make(chan struct{})
	go func() {
		defer close(done)
		for msg := range m.agentCh {
			switch v := msg.(type) {
			case agentDoneMsg:
				res.doneErr = v.err
				res.doneCount++
			}
		}
	}()

	go m.runAgentLoop(context.Background(), prompt, m.agentCh, m.agentRun())

	select {
	case <-done:
	case <-time.After(30 * time.Second):
		t.Fatal("runAgentLoop did not terminate within 30s")
	}
	return res
}

// loadSessionEvents reads every persisted event back through the service.
func loadSessionEvents(t *testing.T, svc *pisession.FileService, sid string) []*session.Event {
	t.Helper()
	resp, err := svc.Get(context.Background(), &session.GetRequest{
		AppName:   agent.AppName,
		UserID:    agent.DefaultUserID,
		SessionID: sid,
	})
	if err != nil || resp == nil || resp.Session == nil {
		t.Fatalf("reading session %s: %v", sid, err)
	}
	var events []*session.Event
	for ev := range resp.Session.Events().All() {
		events = append(events, ev)
	}
	return events
}

// orphanedCallIDs reports the ids of persisted function calls that no function
// response answers — the same test ADK's content builder applies before it
// drops calls with a warning. A non-empty result here is exactly the bug.
func orphanedCallIDs(events []*session.Event) []string {
	answered := make(map[string]struct{})
	var ids []string
	for i := len(events) - 1; i >= 0; i-- {
		ev := events[i]
		if ev == nil || ev.Content == nil {
			continue
		}
		for _, part := range ev.Content.Parts {
			if part == nil {
				continue
			}
			if fr := part.FunctionResponse; fr != nil && fr.ID != "" {
				answered[fr.ID] = struct{}{}
			}
			if fc := part.FunctionCall; fc != nil && fc.ID != "" {
				if _, ok := answered[fc.ID]; !ok {
					ids = append(ids, fc.ID)
				}
			}
		}
	}
	return ids
}

// syntheticResponses returns the synthesized tool responses: function
// responses whose error names a stopped turn. Real tool results never carry
// that marker, so any hit is one of ours.
func syntheticResponses(events []*session.Event) []*genai.FunctionResponse {
	var out []*genai.FunctionResponse
	for _, ev := range events {
		if ev == nil || ev.Content == nil {
			continue
		}
		for _, part := range ev.Content.Parts {
			if part == nil || part.FunctionResponse == nil {
				continue
			}
			if msg, ok := part.FunctionResponse.Response["error"].(string); ok &&
				strings.Contains(msg, "turn was stopped") {
				out = append(out, part.FunctionResponse)
			}
		}
	}
	return out
}

// captureADKLog redirects the stdlib logger (where ADK prints its content
// builder warnings) into a buffer for the duration of a test.
func captureADKLog(t *testing.T) *strings.Builder {
	t.Helper()
	var buf strings.Builder
	old := stdlog.Writer()
	stdlog.SetOutput(&buf)
	t.Cleanup(func() { stdlog.SetOutput(old) })
	return &buf
}

// TestRunAgentLoop_SynthesizesResponseForAutoStoppedCall is the issue #60
// incident: the loop auto-stop fires after a call was persisted, the recovery
// prompt continues the turn, and the stranded call must come out of the stop
// with a synthesized error response — same id, same tool name — so ADK never
// has an unanswered call to drop. Every call in the session must be paired
// once the turn is done, and the ADK content builder must have nothing to
// warn about on the recovery turn's context build.
func TestRunAgentLoop_SynthesizesResponseForAutoStoppedCall(t *testing.T) {
	// Ten identical calls trip the detector on round 10; round 11 answers
	// with text, so the recovery completes the turn cleanly.
	llm := &spinLLM{name: "spin-then-text", texts: maxRepeatToolCalls}
	a, svc, sid := newSynthTestAgent(t, llm)
	adkLog := captureADKLog(t)

	res := driveSynthRunLoop(t, a, svc, sid, "loop then finish")
	if res.doneErr != nil {
		t.Fatalf("expected clean completion after recovery, got %v", res.doneErr)
	}
	if warnings := adkLog.String(); strings.Contains(warnings, "dropping function calls") {
		t.Fatalf("ADK dropped calls while building context: %s", warnings)
	}

	events := loadSessionEvents(t, svc, sid)
	if orphans := orphanedCallIDs(events); len(orphans) > 0 {
		t.Fatalf("unanswered calls after the turn: %v", orphans)
	}
	synth := syntheticResponses(events)
	if len(synth) != 1 {
		t.Fatalf("synthetic responses = %d, want 1", len(synth))
	}
	// The stranded call is the 10th (rounds 0..9): its id must be answered
	// by the synthetic response, name included — ADK matches on the id.
	wantID := fmt.Sprintf("call_%02d", maxRepeatToolCalls-1)
	if synth[0].ID != wantID {
		t.Errorf("synthetic response id = %q, want %q", synth[0].ID, wantID)
	}
	if synth[0].Name != "bash" {
		t.Errorf("synthetic response name = %q, want bash", synth[0].Name)
	}
	if msg, _ := synth[0].Response["error"].(string); !strings.Contains(msg, "loop detected") {
		t.Errorf("synthetic response error = %q, want it to name the loop stop", msg)
	}
}

// TestRunAgentLoop_SynthesizesResponseForFailedTurn covers the terminal
// fail path — the one a user cancel (Esc → ctx.Err) also lands in through
// the shared fail closure. Recovery is driven to exhaustion, so the final
// attempt strands its call and the turn dies; the sweep at fail must pair
// it anyway.
func TestRunAgentLoop_SynthesizesResponseForFailedTurn(t *testing.T) {
	llm := &spinLLM{name: "spin-forever", texts: -1}
	a, svc, sid := newSynthTestAgent(t, llm)

	res := driveSynthRunLoop(t, a, svc, sid, "never recover")
	if res.doneErr == nil || !strings.Contains(res.doneErr.Error(), "gave up") {
		t.Fatalf("expected recovery-exhaustion error, got %v", res.doneErr)
	}

	events := loadSessionEvents(t, svc, sid)
	if orphans := orphanedCallIDs(events); len(orphans) > 0 {
		t.Fatalf("unanswered calls after the failed turn: %v", orphans)
	}
	if len(syntheticResponses(events)) == 0 {
		t.Fatal("no synthetic response after the failed turn, want the stranded call answered")
	}
}

// TestRunAgentLoop_NoSyntheticOnCleanTurn pins the other half: a turn that
// completes normally — its tool executed and answered — must gain no
// synthetic response. The sweep runs on the clean exit too and must be a
// no-op there.
func TestRunAgentLoop_NoSyntheticOnCleanTurn(t *testing.T) {
	a, svc, sid := newSynthTestAgent(t, &fnLLM{name: "one-call", gen: func(call int) (*llmmodel.LLMResponse, error) {
		if call == 0 {
			return &llmmodel.LLMResponse{
				Content: &genai.Content{
					Role: genai.RoleModel,
					Parts: []*genai.Part{{
						FunctionCall: &genai.FunctionCall{ID: "call_clean", Name: "bash", Args: map[string]any{"command": "echo hi"}},
					}},
				},
			}, nil
		}
		return textResp("finished"), nil
	}})

	res := driveSynthRunLoop(t, a, svc, sid, "one call then stop")
	if res.doneErr != nil {
		t.Fatalf("expected clean completion, got %v", res.doneErr)
	}

	events := loadSessionEvents(t, svc, sid)
	if orphans := orphanedCallIDs(events); len(orphans) > 0 {
		t.Fatalf("unanswered calls after a clean turn: %v", orphans)
	}
	if synth := syntheticResponses(events); len(synth) != 0 {
		t.Fatalf("clean turn gained %d synthetic response(s), want 0", len(synth))
	}
}
