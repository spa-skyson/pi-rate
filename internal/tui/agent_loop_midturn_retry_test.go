package tui

import (
	"context"
	"errors"
	"iter"
	"strings"
	"sync"
	"testing"
	"time"

	llmmodel "google.golang.org/adk/v2/model"

	"github.com/spa-skyson/pi-rate/internal/config"
)

// seqLLM scripts each GenerateContent call as an ordered sequence of
// LLMResponses, yielded with a nil Go error — the shape a provider produces
// when a stream sends content and then dies with a STREAM_ERROR response
// mid-run (see internal/provider/xai.go). Deterministic and sleep-free: the
// only wait in these tests is midturnRetryPause, stubbed to near zero.
//
// errs, when set, makes the numbered call end with a Go-level error after its
// scripted responses — a stream that died under the provider (fnLLM's failure
// mode). A call whose gen yields nothing and that carries an error is a
// request that died before its stream started. These are the failures
// WithRetryContext classifies on its own, so the error text carries a
// server-supplied "retry in 1ms" hint that keeps the inner backoff
// sleep-free too.
type seqLLM struct {
	name string
	mu   sync.Mutex
	n    int
	gen  func(call int) []*llmmodel.LLMResponse
	errs map[int]error
	// prompts records each call's request text, in order, so a test can
	// pin which prompt a replay re-sent.
	prompts []string
}

func (l *seqLLM) Name() string { return l.name }

func (l *seqLLM) GenerateContent(_ context.Context, req *llmmodel.LLMRequest, _ bool) iter.Seq2[*llmmodel.LLMResponse, error] {
	l.mu.Lock()
	call := l.n
	l.n++
	l.prompts = append(l.prompts, requestText(req))
	l.mu.Unlock()
	return func(yield func(*llmmodel.LLMResponse, error) bool) {
		for _, resp := range l.gen(call) {
			if !yield(resp, nil) {
				return
			}
		}
		if err, ok := l.errs[call]; ok {
			yield(nil, err)
		}
	}
}

// requestText joins every text part of a request, in order.
func requestText(req *llmmodel.LLMRequest) string {
	if req == nil {
		return ""
	}
	var b strings.Builder
	for _, c := range req.Contents {
		for _, p := range c.Parts {
			b.WriteString(p.Text)
			b.WriteString("\n")
		}
	}
	return b.String()
}

// calls reports how many times the model was asked, the observable behind
// every "did it retry?" assertion here.
func (l *seqLLM) calls() int {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.n
}

// streamErrResp reproduces the provider shape for a mid-stream failure: a
// content-less response carrying STREAM_ERROR, yielded with a nil Go error.
func streamErrResp(msg string) *llmmodel.LLMResponse {
	return &llmmodel.LLMResponse{ErrorCode: "STREAM_ERROR", ErrorMessage: msg}
}

const stallReason = "llm stream idle timeout after 90s"

// stubMidturnRetryPause swaps the inter-attempt pause for the duration of a
// test and returns its restore func.
func stubMidturnRetryPause(d time.Duration) func() {
	old := midturnRetryPause
	midturnRetryPause = d
	return func() { midturnRetryPause = old }
}

// stallWarnings filters the warning stream down to the mid-turn retry
// announcements, leaving out unrelated warnings (truncation, loop recovery).
func stallWarnings(warnings []string) []string {
	var out []string
	for _, w := range warnings {
		if strings.Contains(w, "stream stalled") {
			out = append(out, w)
		}
	}
	return out
}

// TestRunAgentLoop_MidturnTransientRetriesOnce is the issue #41 scenario: a
// partial answer is on screen, the stream dies with a transient failure, and
// the turn replays instead of dying. The replayed attempt must arrive as a
// fresh message — the partial block stays closed behind the warning, and no
// text is emitted twice.
func TestRunAgentLoop_MidturnTransientRetriesOnce(t *testing.T) {
	restore := stubMidturnRetryPause(time.Millisecond)
	defer restore()

	llm := &seqLLM{name: "stall-once", gen: func(call int) []*llmmodel.LLMResponse {
		if call == 0 {
			// Partial text reaches the user, then the stream stalls.
			return []*llmmodel.LLMResponse{textResp("partial answer "), streamErrResp(stallReason)}
		}
		return []*llmmodel.LLMResponse{textResp("complete answer 42")}
	}}
	a, sid := newRunTestAgent(t, llm)

	res, warnings := driveRunLoopWithWarnings(t, a, sid, "explain something")

	if res.doneErr != nil {
		t.Fatalf("retried turn ended with an error: %v", res.doneErr)
	}
	if got := llm.calls(); got != 2 {
		t.Fatalf("model was called %d times, want 2 (one stall, one replay)", got)
	}
	got := stallWarnings(warnings)
	if len(got) != 1 {
		t.Fatalf("got %d stall warnings, want 1:\n%s", len(got), strings.Join(warnings, "\n"))
	}
	if !strings.Contains(got[0], "retrying 2/3") || !strings.Contains(got[0], stallReason) {
		t.Fatalf("warning = %q, want \"retrying 2/3\" carrying the reason", got[0])
	}
	// Exactly two text emissions, each once: the partial block from the
	// failed attempt, then the replay as a new message. A duplicated
	// partial (replay re-streaming it) or an aggregate re-send fails here.
	want := []string{"partial answer ", "complete answer 42"}
	if strings.Join(res.texts, "|") != strings.Join(want, "|") {
		t.Fatalf("texts = %q, want %q (no duplicated partial, no aggregate re-send)", res.texts, want)
	}
}

// TestRunAgentLoop_MidturnRetriesExhaustAndOfferRetry pins the budget: three
// straight stalls run the turn exactly three times, warn between attempts,
// and end through the ordinary error path whose handler appends the
// "Retry with Ctrl+R or /retry" hint.
func TestRunAgentLoop_MidturnRetriesExhaustAndOfferRetry(t *testing.T) {
	restore := stubMidturnRetryPause(time.Millisecond)
	defer restore()

	llm := &seqLLM{name: "always-stall", gen: func(int) []*llmmodel.LLMResponse {
		return []*llmmodel.LLMResponse{textResp("partial"), streamErrResp(stallReason)}
	}}
	a, sid := newRunTestAgent(t, llm)

	res, warnings := driveRunLoopWithWarnings(t, a, sid, "explain something")

	if res.doneErr == nil || !strings.Contains(res.doneErr.Error(), stallReason) {
		t.Fatalf("final error = %v, want it to carry the stall reason", res.doneErr)
	}
	if got := llm.calls(); got != 3 {
		t.Fatalf("model was called %d times, want exactly 3 (the whole budget)", got)
	}
	if got := stallWarnings(warnings); len(got) != 2 {
		t.Fatalf("got %d stall warnings, want 2 (between the three attempts):\n%s",
			len(got), strings.Join(warnings, "\n"))
	}

	// Exhaustion lands on the existing failed-turn UI: the error block plus
	// the /retry hint.
	m := &model{running: true, lastPrompt: "explain something"}
	m.chatModel = NewChatModel(nil)
	m.handleAgentDone(agentDoneMsg{err: res.doneErr})
	var sawError, sawHint bool
	for _, msg := range m.chatModel.Messages {
		if msg.isError && strings.Contains(msg.content, stallReason) {
			sawError = true
		}
		if msg.isMeta && strings.Contains(msg.content, "Retry with Ctrl+R or /retry") {
			sawHint = true
		}
	}
	if !sawError || !sawHint {
		t.Fatalf("transcript after exhaustion: error=%v hint=%v", sawError, sawHint)
	}
}

// TestRunAgentLoop_NoRetryAfterToolTraffic pins the replay guard: once an
// attempt has shown a FunctionCall card, a transient failure must end the
// turn — replaying would execute the tool a second time. The failure is a
// Go-level error from the iterator (errs), landing on the request that
// follows the executed tool — the shape WithRetryContext wraps when
// hadEvents is true. The user must see the wrapper's verdict — "transient
// error after partial response (not replayed automatically: this turn
// already ran tool calls — use /retry to resend)" — not a silent death and
// not a mid-turn replay (issue #45).
func TestRunAgentLoop_NoRetryAfterToolTraffic(t *testing.T) {
	restore := stubMidturnRetryPause(time.Millisecond)
	defer restore()

	llm := &seqLLM{name: "stall-after-call", gen: func(call int) []*llmmodel.LLMResponse {
		if call == 0 {
			// The model calls a tool and the stream dies right behind
			// the call card: the call goes out, and the request that
			// carries the conversation forward dies with the Go-level
			// transient in errs.
			return []*llmmodel.LLMResponse{
				callResp("bash", map[string]any{"command": "echo hi"}),
			}
		}
		return nil
	}, errs: map[int]error{1: errors.New(stallReason)}}
	a, sid := newRunTestAgent(t, llm)

	res, warnings := driveRunLoopWithWarnings(t, a, sid, "run a command")

	if res.doneErr == nil || !strings.Contains(res.doneErr.Error(), stallReason) {
		t.Fatalf("error = %v, want the stall surfaced without a replay", res.doneErr)
	}
	// The WithRetry wrapper reached the user: the turn died after tool
	// traffic, and the error names that verdict and points at /retry.
	for _, want := range []string{"not replayed automatically", "use /retry to resend"} {
		if !strings.Contains(res.doneErr.Error(), want) {
			t.Fatalf("error = %v, want it to carry %q", res.doneErr, want)
		}
	}
	if got := stallWarnings(warnings); len(got) != 0 {
		t.Fatalf("replayed after tool traffic: %q", got)
	}
	if len(res.toolCalls) == 0 {
		t.Fatal("tool call never surfaced; the scenario did not exercise the guard")
	}
	if len(res.toolCalls) != 1 {
		t.Fatalf("got %d tool call cards, want 1 (the replay would show a second)", len(res.toolCalls))
	}
}

// TestRunAgentLoop_NoRetryOnTerminalError pins the classification: a 402 will
// not clear however often the request is re-sent, so the turn fails on the
// first attempt without spending the budget.
func TestRunAgentLoop_NoRetryOnTerminalError(t *testing.T) {
	restore := stubMidturnRetryPause(time.Millisecond)
	defer restore()

	const reason = "402 Payment Required: plan limit reached"
	llm := &seqLLM{name: "payment", gen: func(int) []*llmmodel.LLMResponse {
		return []*llmmodel.LLMResponse{textResp("partial answer"), streamErrResp(reason)}
	}}
	a, sid := newRunTestAgent(t, llm)

	res, warnings := driveRunLoopWithWarnings(t, a, sid, "explain something")

	if res.doneErr == nil || !strings.Contains(res.doneErr.Error(), "402") {
		t.Fatalf("error = %v, want the 402 surfaced", res.doneErr)
	}
	if got := llm.calls(); got != 1 {
		t.Fatalf("model was called %d times, want 1 (terminal errors are not retried)", got)
	}
	if got := stallWarnings(warnings); len(got) != 0 {
		t.Fatalf("retried a terminal error: %q", got)
	}
}

// TestRunAgentLoop_CancelDuringRetryPause pins that Esc during the
// between-attempts pause breaks the wait: the turn ends with context.Canceled
// well before the full pause elapses, and no further attempt starts.
func TestRunAgentLoop_CancelDuringRetryPause(t *testing.T) {
	restore := stubMidturnRetryPause(2 * time.Second)
	defer restore()

	llm := &seqLLM{name: "stall", gen: func(int) []*llmmodel.LLMResponse {
		return []*llmmodel.LLMResponse{streamErrResp(stallReason)}
	}}
	a, sid := newRunTestAgent(t, llm)

	m := &model{cfg: Config{Agent: a, SessionID: sid}, ctx: context.Background()}
	m.agentCh = make(chan agentMsg, 64)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	done := make(chan struct{})
	var doneErr error
	go func() {
		defer close(done)
		for msg := range m.agentCh {
			if _, ok := msg.(agentWarningMsg); ok {
				cancel() // the user's second Esc, mid-pause
				continue
			}
			if d, ok := msg.(agentDoneMsg); ok {
				doneErr = d.err
			}
		}
	}()

	start := time.Now()
	go m.runAgentLoop(ctx, "hello", m.agentCh, m.agentRun())
	<-done
	elapsed := time.Since(start)

	if !errors.Is(doneErr, context.Canceled) {
		t.Fatalf("cancel during the pause must fail the turn with context.Canceled, got %v", doneErr)
	}
	// The broken behavior sleeps the full pause before noticing; the fixed
	// one ends within a fraction of it. The pause itself is 2s.
	if elapsed >= time.Second {
		t.Fatalf("the pause ignored cancelation: turn ended after %s", elapsed)
	}
	if got := llm.calls(); got != 1 {
		t.Fatalf("model was called %d times after cancel, want 1 (no attempt past the pause)", got)
	}
}

// TestRunAgentLoop_MidturnWarningCarriesCause pins what the stall warning
// quotes. Here the failure is a Go-level transient that lands after partial
// text, so WithRetry hands it up wrapped as "transient error after partial
// response (not replayed automatically: this turn already ran tool calls —
// use /retry to resend): …" — the parenthetical is its verdict on its own
// replay loop. The mid-turn warning says the turn IS being retried, so it
// must carry the cause underneath, not a verdict that contradicts it.
func TestRunAgentLoop_MidturnWarningCarriesCause(t *testing.T) {
	restore := stubMidturnRetryPause(time.Millisecond)
	defer restore()

	const reason = "502 bad gateway"
	llm := &seqLLM{name: "died-after-text", gen: func(call int) []*llmmodel.LLMResponse {
		if call == 0 {
			return []*llmmodel.LLMResponse{textResp("partial answer ")}
		}
		return []*llmmodel.LLMResponse{textResp("complete answer 42")}
	}, errs: map[int]error{0: errors.New(reason)}}
	a, sid := newRunTestAgent(t, llm)

	res, warnings := driveRunLoopWithWarnings(t, a, sid, "explain something")

	if res.doneErr != nil {
		t.Fatalf("retried turn ended with an error: %v", res.doneErr)
	}
	if got := llm.calls(); got != 2 {
		t.Fatalf("model was called %d times, want 2 (one stall, one replay)", got)
	}
	got := stallWarnings(warnings)
	if len(got) != 1 || !strings.Contains(got[0], "retrying 2/3") || !strings.Contains(got[0], reason) {
		t.Fatalf("warning = %q, want \"retrying 2/3\" carrying the cause %q", got, reason)
	}
	if strings.Contains(got[0], "not retrying") {
		t.Fatalf("warning quotes the inner loop's \"(not retrying)\" wrapper: %q", got[0])
	}
	// Clean completion: no done message, and never two.
	if res.doneCount != 0 {
		t.Fatalf("agentDoneMsg emitted %d times on a clean turn, want 0", res.doneCount)
	}
}

// TestRunAgentLoop_MidturnTransientWithoutPartialText covers the stall that
// lands before anything reached the screen: the run's first event is already
// the STREAM_ERROR, with no partial text behind it. The turn replays, and the
// replay arrives as a fresh message — there is no fragment of the failed
// attempt to keep or duplicate.
func TestRunAgentLoop_MidturnTransientWithoutPartialText(t *testing.T) {
	restore := stubMidturnRetryPause(time.Millisecond)
	defer restore()

	llm := &seqLLM{name: "stall-before-content", gen: func(call int) []*llmmodel.LLMResponse {
		if call == 0 {
			return []*llmmodel.LLMResponse{streamErrResp(stallReason)}
		}
		return []*llmmodel.LLMResponse{textResp("complete answer 42")}
	}}
	a, sid := newRunTestAgent(t, llm)

	res, warnings := driveRunLoopWithWarnings(t, a, sid, "explain something")

	if res.doneErr != nil {
		t.Fatalf("retried turn ended with an error: %v", res.doneErr)
	}
	if got := llm.calls(); got != 2 {
		t.Fatalf("model was called %d times, want 2 (one stall, one replay)", got)
	}
	got := stallWarnings(warnings)
	if len(got) != 1 || !strings.Contains(got[0], "retrying 2/3") || !strings.Contains(got[0], stallReason) {
		t.Fatalf("warnings = %q, want one \"retrying 2/3\" carrying the reason", got)
	}
	if strings.Join(res.texts, "|") != "complete answer 42" {
		t.Fatalf("texts = %q, want the replay's answer only", res.texts)
	}
	// A clean completion reports through the channel close alone (the live
	// TUI's waitForSlotAgent wraps the close as a nil-ev slotEnvelope, which
	// handleSlotEnvelope finalizes from), so no done message
	// may arrive here — and certainly not two.
	if res.doneCount != 0 {
		t.Fatalf("agentDoneMsg emitted %d times on a clean turn, want 0", res.doneCount)
	}
}

// TestRunAgentLoop_MidturnReplayAfterStuckRecovery covers both recovery paths
// landing in one turn: the detector stops a degenerate stretch, the recovery
// prompt goes out, and that recovery attempt then stalls mid-reply. The stall
// must replay the recovery prompt — not the original — and the turn must
// still finish clean.
func TestRunAgentLoop_MidturnReplayAfterStuckRecovery(t *testing.T) {
	restore := stubMidturnRetryPause(time.Millisecond)
	defer restore()

	phrase := "Let me reconsider this from the beginning one more time before proceeding further. "
	llm := &seqLLM{name: "stuck-then-stall", gen: func(call int) []*llmmodel.LLMResponse {
		switch call {
		case 0:
			// Enough copies to trip the output-repeat guard on attempt 1.
			return []*llmmodel.LLMResponse{textResp(strings.Repeat(phrase, maxOutputRepeats+4))}
		case 1:
			// The recovery attempt stalls mid-reply.
			return []*llmmodel.LLMResponse{textResp("Taking a different approach. "), streamErrResp(stallReason)}
		default:
			return []*llmmodel.LLMResponse{textResp("The answer is 42.")}
		}
	}}
	a, sid := newRunTestAgent(t, llm)

	res, warnings := driveRunLoopWithWarnings(t, a, sid, "explain something")

	if res.doneErr != nil {
		t.Fatalf("turn ended with an error: %v", res.doneErr)
	}
	if got := llm.calls(); got != 3 {
		t.Fatalf("model was called %d times, want 3 (stuck attempt, stalled recovery, replay)", got)
	}
	if len(llm.prompts) != 3 {
		t.Fatalf("got %d recorded prompts, want 3", len(llm.prompts))
	}
	// The stuck recovery replaced the prompt, and the mid-turn replay
	// re-sent that recovery prompt rather than the original one.
	if strings.Contains(llm.prompts[0], "stopped automatically") {
		t.Fatalf("attempt 1 already carried the recovery prompt: %q", truncateForTest(llm.prompts[0]))
	}
	for _, p := range llm.prompts[1:] {
		if !strings.Contains(p, "stopped automatically") {
			t.Fatalf("attempt after recovery did not re-send the recovery prompt: %q", truncateForTest(p))
		}
	}
	joined := strings.Join(warnings, "\n")
	if strings.Count(joined, "Loop detected") != 1 || strings.Count(joined, "stream stalled") != 1 {
		t.Fatalf("warnings = %q, want one loop notice and one stall notice", warnings)
	}
	if !strings.Contains(strings.Join(res.texts, ""), "42") {
		t.Fatalf("recovered answer never reached the user: %q", truncateForTest(strings.Join(res.texts, "")))
	}
	// Clean completion: no done message (synthesized from the channel close
	// in the live TUI), and never two.
	if res.doneCount != 0 {
		t.Fatalf("agentDoneMsg emitted %d times on a clean turn, want 0", res.doneCount)
	}
}

// intPtr returns a pointer to n — the shape an explicit config value takes.
func intPtr(n int) *int { return &n }

// TestRunAgentLoop_MidturnBudgetFromConfig pins the wiring (issue #43): the
// replay budget is read from the session config, not a compiled constant. A
// turn that stalls on every attempt runs exactly MidTurnAttempts times —
// 2 means one replay, 0 and 1 mean the first transient failure ends the turn
// (the pre-#41 behavior) — and an unset field runs at the default.
func TestRunAgentLoop_MidturnBudgetFromConfig(t *testing.T) {
	restore := stubMidturnRetryPause(time.Millisecond)
	defer restore()

	tests := []struct {
		name      string
		attempts  *int
		wantCalls int
		wantStall int // "stream stalled" warnings, one per replay
	}{
		{name: "config 2 → one replay", attempts: intPtr(2), wantCalls: 2, wantStall: 1},
		{name: "config 1 → replays off", attempts: intPtr(1), wantCalls: 1, wantStall: 0},
		{name: "config 0 → replays off", attempts: intPtr(0), wantCalls: 1, wantStall: 0},
		{name: "unset → default 3", attempts: nil, wantCalls: 3, wantStall: 2},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			llm := &seqLLM{name: "always-stall", gen: func(int) []*llmmodel.LLMResponse {
				return []*llmmodel.LLMResponse{textResp("partial"), streamErrResp(stallReason)}
			}}
			a, sid := newRunTestAgent(t, llm)

			cfg := Config{Agent: a, SessionID: sid, MidTurnAttempts: tt.attempts}
			res, warnings := driveRunLoopConfig(t, cfg, "explain something")

			if got := llm.calls(); got != tt.wantCalls {
				t.Fatalf("model was called %d times, want %d (attempts = %v)",
					got, tt.wantCalls, tt.attempts)
			}
			got := stallWarnings(warnings)
			if len(got) != tt.wantStall {
				t.Fatalf("got %d stall warnings, want %d:\n%s",
					len(got), tt.wantStall, strings.Join(warnings, "\n"))
			}
			// A budget with nothing to replay with ends through the
			// ordinary error path, whose handler offers /retry.
			if tt.wantStall == 0 && (res.doneErr == nil || !strings.Contains(res.doneErr.Error(), stallReason)) {
				t.Fatalf("error = %v, want the stall surfaced without a replay", res.doneErr)
			}
		})
	}
}

// TestRunAgentLoop_MidturnBudgetCoversInnerRetries pins the shared budget:
// a Go-level transient (a request dying under the provider) is retried by
// WithRetryContext inside a single run, and each of those retries is one more
// request against the same turn the mid-turn replay draws from. The combined
// count stays at the default budget: two inner retries plus one stalled run
// spend the whole budget, and the turn ends without a mid-turn replay.
// Counting only the mid-turn layer here costs nine requests — the inner
// retries re-run inside every one of the three mid-turn runs.
func TestRunAgentLoop_MidturnBudgetCoversInnerRetries(t *testing.T) {
	restore := stubMidturnRetryPause(time.Millisecond)
	defer restore()

	// The server-supplied hint keeps IsTransient true and Delay at 1ms, so
	// the inner backoff never sleeps the test.
	gatewayErr := errors.New("502 bad gateway (retry in 1ms)")

	llm := &seqLLM{
		name: "gateway-down",
		errs: map[int]error{
			0: gatewayErr, 1: gatewayErr, // every attempt: two dead requests,
			3: gatewayErr, 4: gatewayErr, // then partial text and a STREAM_ERROR —
			6: gatewayErr, 7: gatewayErr, // the shape the inner loop cannot replay
		},
		gen: func(call int) []*llmmodel.LLMResponse {
			if call%3 == 2 {
				return []*llmmodel.LLMResponse{textResp("partial answer "), streamErrResp(stallReason)}
			}
			// The dead requests produce nothing at all, which is what
			// makes WithRetryContext replay them on its own.
			return nil
		},
	}
	a, sid := newRunTestAgent(t, llm)

	res, warnings := driveRunLoopWithWarnings(t, a, sid, "explain something")

	// The combined budget is spent, so the turn ends through the ordinary
	// error path: the mid-turn layer has nothing left to replay with.
	if res.doneErr == nil || !strings.Contains(res.doneErr.Error(), stallReason) {
		t.Fatalf("final error = %v, want it to carry the stall reason", res.doneErr)
	}
	if got := llm.calls(); got != config.DefaultMidTurnAttempts {
		t.Fatalf("model was called %d times, want %d (2 inner retries + 1 stalled run = the whole budget)",
			got, config.DefaultMidTurnAttempts)
	}
	// No mid-turn replay happened, so nothing announced one; the only
	// warnings are the inner retries the notifier surfaced.
	if got := stallWarnings(warnings); len(got) != 0 {
		t.Fatalf("replayed past the combined budget: %q", got)
	}
	// The failed run's partial text stays as its own block; nothing
	// re-streamed it.
	if strings.Join(res.texts, "|") != "partial answer " {
		t.Fatalf("texts = %q, want [\"partial answer \"]", res.texts)
	}
	// A failed turn reports exactly once, not once per layer.
	if res.doneCount != 1 {
		t.Fatalf("agentDoneMsg emitted %d times, want 1", res.doneCount)
	}
}
