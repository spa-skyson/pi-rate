package tools

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool"
)

// The question tool: the model asks the user a multiple-choice question and
// blocks until the interactive session answers it. The bridge is the same
// shape as the permission gate's approval round-trip: the tool hands a
// QuestionRequest to a caller-supplied notifier and parks on req.Reply, the
// TUI dialog answers it once. With no notifier (headless, one-shot, subagent
// runs) the tool never blocks — it returns canceled with a note telling the
// model to decide itself and proceed, so advertising it can never hang a run
// that has no one to answer.

// maxQuestionOptions caps the option list: past ~8 alternatives the dialog
// stops being a quick choice and the model should narrow them first.
const maxQuestionOptions = 8

// questionHeadlessNote rides a canceled result when there is no interactive
// session to answer: it tells the model the question was not refused — there
// was simply no one to ask.
const questionHeadlessNote = "non-interactive session: no one to answer — decide yourself from the context you have and proceed"

// questionTimeoutDefault caps the wait for an answer when
// PI_QUESTION_TIMEOUT_MS is unset: a dialog no one answers must not park the
// turn forever (production hang #32).
const questionTimeoutDefault = 10 * time.Minute

// questionReplyTimeout reads PI_QUESTION_TIMEOUT_MS (milliseconds). Unset,
// unparsable or non-positive values fall back to the 10-minute default.
func questionReplyTimeout() time.Duration {
	if ms, err := strconv.Atoi(os.Getenv("PI_QUESTION_TIMEOUT_MS")); err == nil && ms > 0 {
		return time.Duration(ms) * time.Millisecond
	}
	return questionTimeoutDefault
}

// QuestionOption is one selectable answer.
type QuestionOption struct {
	Label       string `json:"label"`
	Description string `json:"description,omitempty"`
}

// QuestionInput is the question tool's argument struct.
type QuestionInput struct {
	Question string           `json:"question"`
	Options  []QuestionOption `json:"options"`
	// AllowFreeText, when true (the default), lets the user type their own
	// answer instead of picking a listed option. The pointer distinguishes
	// an omitted field from an explicit false.
	AllowFreeText *bool `json:"allow_free_text,omitempty"`
}

// freeText reports whether custom answers are accepted; omitted means yes.
func (in QuestionInput) freeText() bool {
	return in.AllowFreeText == nil || *in.AllowFreeText
}

// QuestionOutput is the question tool's result.
type QuestionOutput struct {
	// Selected is "option" (a listed option was picked), "custom" (free
	// text) or "canceled" (Esc, turn cancel, or no one to answer).
	Selected string `json:"selected"`
	// Label is the picked option's label, or the typed text for custom.
	Label string `json:"label,omitempty"`
	// Index is the picked option's position. It carries omitempty, so the
	// first option's 0 is absent from the encoded result — the label is the
	// authoritative answer either way.
	Index int `json:"index,omitempty"`
	// Note explains a canceled result (headless sessions). Empty otherwise.
	Note string `json:"note,omitempty"`
}

// QuestionRequest is one interactive question round-trip, modeled on
// permission.ApprovalRequest: Reply is buffered to one and created by the
// sender, so the answerer never blocks even when the wait was abandoned
// (turn canceled while the dialog was up).
type QuestionRequest struct {
	Question      string
	Options       []QuestionOption
	AllowFreeText bool
	// Reply carries the user's answer. Buffered to 1.
	Reply chan QuestionAnswer
}

// QuestionAnswer is the answer to one QuestionRequest. Selected mirrors
// QuestionOutput.Selected; Label carries the option label or the typed text;
// Index is the option position, 0 for custom and canceled.
type QuestionAnswer struct {
	Selected string
	Label    string
	Index    int
}

// validateQuestionInput rejects calls the dialog could not render or the
// user could not answer. The wording doubles as the hint the model reads.
func validateQuestionInput(in QuestionInput) error {
	if strings.TrimSpace(in.Question) == "" {
		return fmt.Errorf("question must not be empty — state what you need decided")
	}
	switch n := len(in.Options); {
	case n == 0:
		return fmt.Errorf("options must list 1..%d entries — if the choice is open-ended, list the alternatives you can see", maxQuestionOptions)
	case n > maxQuestionOptions:
		return fmt.Errorf("options lists %d entries, max is %d — collapse the alternatives first", n, maxQuestionOptions)
	}
	for i, o := range in.Options {
		if strings.TrimSpace(o.Label) == "" {
			return fmt.Errorf("option %d: label must not be empty", i+1)
		}
	}
	return nil
}

// newQuestionTool creates the question tool. notifier, when non-nil, receives
// each request and the handler blocks until req.Reply is answered or the
// turn's context is canceled. nil keeps the headless behavior: an immediate
// canceled result with a note, never a block.
func newQuestionTool(notifier func(QuestionRequest)) (tool.Tool, error) {
	return newTool("question",
		`Ask the user a multiple-choice question and wait for the answer. Only for decisions that are genuinely the user's: a fork in approach, a priority call, a missing requirement. Give 1-8 concrete, mutually exclusive options; allow_free_text (default true) allows a typed answer. Never for what you can verify yourself (read, grep, bash), never to ask permission — approval is the permission system's job. Non-interactive sessions and timeouts return selected="canceled" + note: decide from context and proceed.`,
		func(ac agent.Context, in QuestionInput) (QuestionOutput, error) {
			if err := validateQuestionInput(in); err != nil {
				return QuestionOutput{}, err
			}
			if notifier == nil {
				return QuestionOutput{Selected: "canceled", Note: questionHeadlessNote}, nil
			}
			req := QuestionRequest{
				Question:      in.Question,
				Options:       in.Options,
				AllowFreeText: in.freeText(),
				Reply:         make(chan QuestionAnswer, 1),
			}
			notifier(req)
			// agent.Context carries the turn's context (bash.go does the
			// same cast); a nil context is a direct-call path — never cancel.
			ctx := context.Background()
			if ac != nil {
				ctx = ac
			}
			timeout := time.NewTimer(questionReplyTimeout())
			defer timeout.Stop()
			select {
			case ans := <-req.Reply:
				return QuestionOutput{Selected: ans.Selected, Label: ans.Label, Index: ans.Index}, nil
			case <-ctx.Done():
				// The turn died with the dialog up (Ctrl+C, session exit).
				// Reply is buffered, so a user answer landing now is
				// silently dropped.
				return QuestionOutput{Selected: "canceled"}, nil
			case <-timeout.C:
				// No one answered in time (dialog never shown, user walked
				// away). The turn must not hang: report canceled with the
				// reason so the model can decide and proceed.
				return QuestionOutput{
					Selected: "canceled",
					Note:     fmt.Sprintf("timed out waiting for an answer after %s — decide from the context you have and proceed", questionReplyTimeout()),
				}, nil
			}
		})
}
