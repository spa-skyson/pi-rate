package agent

import (
	"fmt"
	"sync"

	adkagent "google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool"
)

// The finalizing result handed to the model when the budget runs out. The
// instruction is the model-facing contract: one free round-trip to write a
// coherent final report instead of dying mid-flight, where the last
// intermediate text would stand in for the deliverable.
const (
	stepsLimitStatusKey       = "steps_limit"
	stepsLimitStatusExhausted = "exhausted"
	stepsLimitInstructionKey  = "instruction"
)

func stepsLimitInstruction(limit int) string {
	return fmt.Sprintf("STEPS BUDGET EXHAUSTED (limit %d). This was your last tool call. "+
		"Produce your final report now: what is done, what remains, where things stand. "+
		"Do NOT call more tools.", limit)
}

// NewStepLimitCallback returns an AfterToolCallback that caps a run at
// maxSteps tool calls — max tool-call iterations per run, as the --steps flag
// documents. The count is per run, not per session: the callback tracks the
// ADK invocation id and restarts from zero whenever it changes, so every user
// turn (one Run/RunStreaming call) gets a fresh budget even when the callback
// instance is built once and kept alive for a whole interactive session.
// A resumed run (human-in-the-loop) reuses the paused run's invocation id and
// continues its budget, so pausing for a confirmation cannot bypass the limit.
//
// Exhausting the budget does not abort the run. The (maxSteps+1)-th call still
// runs — the callback is after-tool — and its callback replaces the result
// with the finalizing signal: the model is told its budget is gone and it must
// produce its final report now, without calling more tools. The signal is a
// plain function response — no error, no SkipSummarization — so ADK grants the
// model exactly one more round-trip, and a text answer ends the turn normally
// with the report intact in the events. A hard error here instead would cut
// the run off with the model's last intermediate text standing in for a
// report; production lost subagent runs to exactly that.
//
// The guarantee is one free move, not infinity: if the model answers the
// signal with another tool call, the callback fails with "steps limit %d
// reached" and sets SkipSummarization on the event actions. SkipSummarization
// is the ADK-sanctioned way to stop the loop from inside a tool call
// (RequestConfirmation sets the same flag): the function-response event
// carrying it reports IsFinalResponse() true, so ADK's Flow.Run ends the turn
// after yielding that event and no further model round-trip happens. Without
// the flag the error would only travel back to the model as {"error": ...}
// function-response text — ADK turns every tool-callback error into a result,
// never into a run error — so the loop kept spinning, every later call failing
// with the same message while tokens burned.
//
// Zero or negative maxSteps disables the limit. Under budget the callback
// never touches the result: it is a control-flow guard, not a transform.
func NewStepLimitCallback(maxSteps int) AfterToolCallback {
	// calls, finalizing and runID change together, so a plain mutex guards
	// them rather than independent atomics: a reset paired with the wrong
	// increment would silently widen or narrow the budget.
	var mu sync.Mutex
	var runID string
	var calls int
	var finalizing bool
	return func(ctx adkagent.Context, _ tool.Tool, _, _ map[string]any, _ error) (map[string]any, error) {
		if maxSteps <= 0 {
			return nil, nil
		}
		mu.Lock()
		// A new invocation id is a new user turn: the budget and the
		// finalizing flag restart. Nil ctx (test doubles) skips the
		// detection and keeps counting against the current run, exactly as
		// before the reset existed.
		if ctx != nil {
			if id := ctx.InvocationID(); id != runID {
				runID = id
				calls = 0
				finalizing = false
			}
		}
		if finalizing {
			// The model already got its free finalizing move and called a
			// tool anyway: hard-stop. Mark the function-response event as
			// the turn's final response, so ADK's loop stops here instead
			// of feeding the error back to a model that will keep calling
			// tools. Nil-safe: not every context carries event actions
			// (test doubles, non-tool callback wrappers).
			mu.Unlock()
			if ctx != nil {
				if actions := ctx.Actions(); actions != nil {
					actions.SkipSummarization = true
				}
			}
			return nil, fmt.Errorf("steps limit %d reached", maxSteps)
		}
		calls++
		over := calls > maxSteps
		if over {
			finalizing = true
		}
		mu.Unlock()
		if !over {
			return nil, nil
		}
		// Budget just exhausted: this call already ran (after-tool), so
		// replace its result with the finalizing signal. No error and no
		// SkipSummarization — the model gets one round-trip to write its
		// final report, and a text answer ends the turn with the report in
		// the events.
		return map[string]any{
			stepsLimitStatusKey:      stepsLimitStatusExhausted,
			stepsLimitInstructionKey: stepsLimitInstruction(maxSteps),
		}, nil
	}
}
