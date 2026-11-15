package agent

import (
	"context"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
)

// actionsCtx is a minimal adkagent.Context whose only live methods are
// Actions and InvocationID: exactly what the step-limit callback touches.
// Everything else panics via StrictContextMock, so an unexpected call fails
// loudly.
type actionsCtx struct {
	agent.StrictContextMock
	actions *session.EventActions
	invID   string
}

func newActionsCtx(invocationID string) *actionsCtx {
	return &actionsCtx{
		StrictContextMock: agent.NewStrictContextMock(context.Background()),
		actions:           &session.EventActions{},
		invID:             invocationID,
	}
}

func (c *actionsCtx) Actions() *session.EventActions { return c.actions }

func (c *actionsCtx) InvocationID() string { return c.invID }

// assertFinalizingResult checks the map the callback returns on the budget
// just exhausted: the model-facing finalizing signal with the limit named and
// the order to stop calling tools.
func assertFinalizingResult(t *testing.T, got map[string]any, limit int) {
	t.Helper()
	if got == nil {
		t.Fatal("finalizing call returned nil result, want the STEPS signal map")
	}
	if got[stepsLimitStatusKey] != stepsLimitStatusExhausted {
		t.Fatalf("result[%q] = %v, want %q", stepsLimitStatusKey, got[stepsLimitStatusKey], stepsLimitStatusExhausted)
	}
	instr, _ := got[stepsLimitInstructionKey].(string)
	if !strings.Contains(instr, "STEPS BUDGET EXHAUSTED (limit "+strconv.Itoa(limit)+")") {
		t.Fatalf("instruction = %q, want it to name the exhausted limit %d", instr, limit)
	}
	if !strings.Contains(instr, "Do NOT call more tools") {
		t.Fatalf("instruction = %q, want the do-not-call-more-tools order", instr)
	}
}

// TestStepLimitCallback pins the subagent iteration budget: the first maxSteps
// calls pass through untouched, the next one — the first over budget — gets
// the finalizing signal instead of an error (one free round-trip to write a
// final report), any further tool call fails with the limit named AND marks
// the event final so ADK ends the turn instead of looping the error back to
// the model, and zero disables the guard. The callback must never mutate the
// result it is handed under budget — it is a control-flow guard, not a
// transform; the finalizing call replaces it outright.
func TestStepLimitCallback(t *testing.T) {
	t.Run("zero disables the limit", func(t *testing.T) {
		cb := NewStepLimitCallback(0)
		ctx := newActionsCtx("e-run")
		for i := range 10 {
			if _, err := cb(ctx, nil, nil, nil, nil); err != nil {
				t.Fatalf("call %d: unexpected error %v", i+1, err)
			}
			if ctx.actions.SkipSummarization {
				t.Fatalf("call %d: SkipSummarization set under a disabled limit", i+1)
			}
		}
	})

	t.Run("within limit passes, exhaustion finalizes, violation hard-stops", func(t *testing.T) {
		cb := NewStepLimitCallback(3)
		ctx := newActionsCtx("e-run")
		for i := range 3 {
			out, err := cb(ctx, nil, nil, nil, nil)
			if err != nil {
				t.Fatalf("call %d: unexpected error %v", i+1, err)
			}
			if out != nil {
				t.Fatalf("call %d: unexpected replacement result %v", i+1, out)
			}
		}
		if ctx.actions.SkipSummarization {
			t.Fatal("SkipSummarization set while within the budget")
		}

		// 4th call — the budget just ran out: the finalizing signal, not an
		// error. The model needs its free round-trip to write the report.
		original := map[string]any{"stdout": "real result"}
		out, err := cb(ctx, nil, nil, original, nil)
		if err != nil {
			t.Fatalf("4th call: unexpected error %v", err)
		}
		assertFinalizingResult(t, out, 3)
		if ctx.actions.SkipSummarization {
			t.Fatal("finalizing call set SkipSummarization — ADK would end the turn and the model would never write its report")
		}
		if len(original) != 1 || original["stdout"] != "real result" {
			t.Fatalf("finalizing call mutated the original result: %v", original)
		}

		// 5th call — the model ignored the signal: hard stop.
		_, err = cb(ctx, nil, nil, nil, nil)
		if err == nil {
			t.Fatal("5th call: expected steps limit error after the ignored signal, got nil")
		}
		if !strings.Contains(err.Error(), "steps limit 3 reached") {
			t.Fatalf("error = %v, want containing %q", err, "steps limit 3 reached")
		}
		if !ctx.actions.SkipSummarization {
			t.Fatal("violating call did not set SkipSummarization; the turn would not terminate")
		}
	})

	t.Run("budget and finalizing flag reset at each new run", func(t *testing.T) {
		// Two sequential runs of limit tools each must both fit: the second
		// run restarts from zero instead of inheriting the first run's count
		// (the session-lifetime counter cut a TUI turn off mid-work). The
		// finalizing flag must reset too: after run 2 exhausts its budget,
		// run 3 gets a fresh finalizing move — not an instant hard-stop from
		// run 2's leftover state.
		const limit = 3
		cb := NewStepLimitCallback(limit)

		first := newActionsCtx("e-run-1")
		for i := range limit {
			if _, err := cb(first, nil, nil, nil, nil); err != nil {
				t.Fatalf("run 1, call %d: unexpected error %v", i+1, err)
			}
		}
		if first.actions.SkipSummarization {
			t.Fatal("run 1: SkipSummarization set while within the budget")
		}

		second := newActionsCtx("e-run-2")
		for i := range limit {
			if _, err := cb(second, nil, nil, nil, nil); err != nil {
				t.Fatalf("run 2, call %d: budget did not reset — %v", i+1, err)
			}
		}
		if second.actions.SkipSummarization {
			t.Fatal("run 2: SkipSummarization set while within the budget")
		}

		out, err := cb(second, nil, nil, nil, nil)
		if err != nil {
			t.Fatalf("run 2, 4th call: expected the finalizing signal, got error %v", err)
		}
		assertFinalizingResult(t, out, limit)
		if !first.actions.SkipSummarization && second.actions.SkipSummarization {
			t.Fatal("run 2 finalizing call leaked SkipSummarization")
		}
		if second.actions.SkipSummarization {
			t.Fatal("run 2 finalizing call set SkipSummarization; the model would lose its report round-trip")
		}
		if first.actions.SkipSummarization {
			t.Fatal("run 1 events must stay untouched by run 2's over-budget call")
		}

		// Violation in run 2: hard stop.
		if _, err := cb(second, nil, nil, nil, nil); err == nil {
			t.Fatal("run 2, 5th call: expected steps limit error after the ignored signal, got nil")
		}

		// Run 3: both the budget and the finalizing flag are fresh.
		third := newActionsCtx("e-run-3")
		for i := range limit {
			if _, err := cb(third, nil, nil, nil, nil); err != nil {
				t.Fatalf("run 3, call %d: state leaked from run 2 — %v", i+1, err)
			}
		}
		out, err = cb(third, nil, nil, nil, nil)
		if err != nil {
			t.Fatalf("run 3, 4th call: finalizing flag did not reset — got hard error %v", err)
		}
		assertFinalizingResult(t, out, limit)
	})

	t.Run("nil ctx keeps counting against the current run", func(t *testing.T) {
		// Nil ctx (test doubles) skips invocation-id detection: calls keep
		// counting against the current run's budget, so the over-budget
		// call still finalizes and the violating call still trips the limit
		// instead of silently passing.
		const limit = 3
		cb := NewStepLimitCallback(limit)
		for i := range limit {
			if _, err := cb(nil, nil, nil, nil, nil); err != nil {
				t.Fatalf("call %d: unexpected error %v", i+1, err)
			}
		}
		out, err := cb(nil, nil, nil, nil, nil)
		if err != nil {
			t.Fatalf("4th call: expected the finalizing signal, got error %v", err)
		}
		assertFinalizingResult(t, out, limit)
		_, err = cb(nil, nil, nil, nil, nil)
		if err == nil {
			t.Fatal("5th call: expected steps limit error after the ignored signal, got nil")
		}
		if !strings.Contains(err.Error(), "steps limit 3 reached") {
			t.Fatalf("error = %v, want containing %q", err, "steps limit 3 reached")
		}
	})

	t.Run("parallel calls of one run share one budget", func(t *testing.T) {
		// A parallel tool batch of one turn shares one invocation id, so a
		// batch of N tools must eat N steps of that turn's single budget —
		// not N per tool and not free. Deterministic: limit+4 concurrent
		// calls against one budget leave exactly one finalizing call (the
		// model's free move) and three hard-stops. Each goroutine carries
		// its own context (its own event actions) — only the invocation id
		// and the callback's state are shared, which is what a real
		// parallel batch shares too.
		const limit = 8
		cb := NewStepLimitCallback(limit)
		var (
			finalized atomic.Int64
			over      atomic.Int64
			wg        sync.WaitGroup
		)
		for range limit + 4 {
			wg.Add(1)
			go func() {
				defer wg.Done()
				ctx := newActionsCtx("e-run")
				out, err := cb(ctx, nil, nil, nil, nil)
				switch {
				case err != nil:
					over.Add(1)
				case out != nil:
					finalized.Add(1)
				}
			}()
		}
		wg.Wait()
		if got := finalized.Load(); got != 1 {
			t.Fatalf("finalizing calls = %d, want exactly 1 (one free finalizing move per budget)", got)
		}
		if got := over.Load(); got != 3 {
			t.Fatalf("over-budget calls = %d, want exactly 3 (limit+4 concurrent calls, one shared budget)", got)
		}
	})

	t.Run("only the violating call marks the event final", func(t *testing.T) {
		// The termination mechanism for the hard stop: SkipSummarization on
		// the event actions makes the function-response event ADK's final
		// response, which is what ends the turn. The finalizing call must
		// NOT set it — the model needs the round-trip — only the call made
		// after the ignored signal does. Without it ADK would feed the
		// error back to the model and the loop would spin forever.
		cb := NewStepLimitCallback(2)
		ctx := newActionsCtx("e-run")
		for range 2 {
			_, _ = cb(ctx, nil, nil, nil, nil)
		}
		if ctx.actions.SkipSummarization {
			t.Fatal("SkipSummarization set while still under the limit")
		}
		_, _ = cb(ctx, nil, nil, nil, nil) // finalizing move
		if ctx.actions.SkipSummarization {
			t.Fatal("finalizing call set SkipSummarization; the model would never get its report round-trip")
		}
		_, _ = cb(ctx, nil, nil, nil, nil) // violation
		if !ctx.actions.SkipSummarization {
			t.Fatal("violating call did not set SkipSummarization; the turn would not terminate")
		}
	})

	t.Run("under-budget result is never mutated or replaced", func(t *testing.T) {
		cb := NewStepLimitCallback(1)
		result := map[string]any{"stdout": "keep me", "truncated": true}
		got, err := cb(nil, nil, nil, result, nil)
		if err != nil {
			t.Fatalf("unexpected error %v", err)
		}
		if got != nil {
			t.Fatalf("callback returned a replacement result %v, want nil", got)
		}
		if len(result) != 2 || result["stdout"] != "keep me" || result["truncated"] != true {
			t.Fatalf("result was mutated: %v", result)
		}
	})
}
