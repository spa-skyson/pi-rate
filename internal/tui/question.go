package tui

import (
	"fmt"
	"strings"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"

	"github.com/spa-skyson/pi-rate/internal/tools"
)

// The question dialog. The question tool blocks its turn until the user
// answers here — the same bridge shape as the approval dialog (approval.go):
// the CLI sends a tools.QuestionRequest on Config.QuestionCh, the request
// parks in this dialog, and the user's answer travels back over req.Reply.
// The dialog sits in the handleKey overlay chain (kind question) and renders
// in the chat panel's last slot, next to the approval line. In list mode it
// owns navigation, digits, t, Enter and Esc while every other key falls
// through, so typing continues into the prompt; in free-text mode it owns
// every key, because they all edit the answer.

// questionRequestMsg carries one question from the tool, with the session
// tag the tool's per-session notifier stamped on it (types.go).
type questionRequestMsg struct {
	sq SessionQuestion
}

// waitForQuestion parks a reader on the question channel and delivers the
// next request as a message. Re-armed after each delivery, like
// waitForApproval — exactly one reader is ever parked, so a channel close
// cannot multiply wakeups.
func waitForQuestion(ch <-chan SessionQuestion) tea.Cmd {
	if ch == nil {
		return nil
	}
	return func() tea.Msg {
		sq, ok := <-ch
		if !ok {
			return nil
		}
		return questionRequestMsg{sq: sq}
	}
}

// questionDialog is the open dialog's state: the pending request, the
// highlighted option, and the free-text editor.
type questionDialog struct {
	req  tools.QuestionRequest
	sel  int    // highlighted option
	free bool   // free-text input mode
	text string // free-text buffer
}

// questionMaxLines caps the wrapped question, questionMaxDescLines the
// wrapped description: the dialog is a quick choice, not an essay — longer
// text is cut with an ellipsis.
const (
	questionMaxLines     = 6
	questionMaxDescLines = 2
)

// handleQuestionRequest shows the dialog for a new question. The agent loop
// is blocked in the question tool until the user answers.
func (m *model) handleQuestionRequest(msg questionRequestMsg) (tea.Model, tea.Cmd) {
	if m.question != nil {
		// Questions are sequential inside the agent loop, so a second
		// request while one is up means the first was abandoned (the tool
		// is no longer waiting on it — its context died with the turn).
		// Answer it canceled rather than queue it behind a dialog the user
		// has already stopped reading, and show the new one.
		m.question.req.Reply <- tools.QuestionAnswer{Selected: "canceled"}
	}
	m.question = &questionDialog{req: msg.sq.Req}
	m.openOverlay(overlayQuestion)
	// Attention first: the loop is blocked until the dialog is answered, so
	// a user who switched away needs the bell/notification to come back.
	return m, tea.Batch(m.attentionCmd("☠ Pi-rate", "Question: "+truncateLabel(msg.sq.Req.Question, 60)),
		waitForQuestion(m.cfg.QuestionCh))
}

// handleQuestionKey resolves the question dialog. Unowned keys fall through
// to the normal chain (approval's contract): typing edits the prompt as
// usual and is simply not submitted, because Enter belongs to the dialog
// while it is up.
func (m *model) handleQuestionKey(key tea.Key) (tea.Model, tea.Cmd, bool) {
	if m.question == nil {
		return nil, nil, false
	}
	if m.question.free {
		return m.handleQuestionFreeKey(key)
	}

	q := m.question
	opts := q.req.Options
	switch {
	case key.Mod == 0 && (key.Code == tea.KeyUp || key.Code == 'k'):
		if q.sel > 0 {
			q.sel--
		}
	case key.Mod == 0 && (key.Code == tea.KeyDown || key.Code == 'j'):
		if q.sel < len(opts)-1 {
			q.sel++
		}
	case key.Mod == 0 && key.Code >= '1' && key.Code <= '9':
		// Digits highlight, Enter confirms — one stray press cannot commit
		// an option the user never looked at.
		if i := int(key.Code - '1'); i < len(opts) {
			q.sel = i
		}
	case key.Mod == 0 && key.Code == 't' && q.req.AllowFreeText:
		q.free = true
	case key.Mod == 0 && key.Code == tea.KeyEnter:
		if q.sel < len(opts) {
			model, cmd := m.answerQuestion(tools.QuestionAnswer{Selected: "option", Label: opts[q.sel].Label, Index: q.sel})
			return model, cmd, true
		}
	case key.Code == tea.KeyEsc:
		model, cmd := m.answerQuestion(tools.QuestionAnswer{Selected: "canceled"})
		return model, cmd, true
	default:
		// Ctrl+C deliberately falls through to handleInterruptKey: canceling
		// the whole turn is still available mid-dialog, and the canceled
		// turn answers the pending request through the tool's ctx wait.
		return nil, nil, false
	}
	return m, nil, true
}

// handleQuestionFreeKey drives the free-text editor: printable characters
// append, Backspace deletes, Enter submits a non-empty answer, Esc returns
// to the option list. It owns every key — they are all editor input.
func (m *model) handleQuestionFreeKey(key tea.Key) (tea.Model, tea.Cmd, bool) {
	q := m.question
	switch key.Code {
	case tea.KeyEnter:
		if text := strings.TrimSpace(q.text); text != "" {
			model, cmd := m.answerQuestion(tools.QuestionAnswer{Selected: "custom", Label: text})
			return model, cmd, true
		}
	case tea.KeyEsc:
		q.free = false
		q.text = ""
	case tea.KeyBackspace:
		if r := []rune(q.text); len(r) > 0 {
			q.text = string(r[:len(r)-1])
		}
	default:
		// Same guard as the steer mini-input: one printable rune, no chord.
		if singleRuneKey(key.Text) && key.Mod == 0 {
			q.text += key.Text
		}
	}
	return m, nil, true
}

// answerQuestion sends the user's answer back to the blocked question tool,
// drops the dialog and records the exchange in the transcript.
func (m *model) answerQuestion(ans tools.QuestionAnswer) (tea.Model, tea.Cmd) {
	q := m.question
	m.popOverlay(overlayQuestion)
	if q != nil {
		// Reply is buffered to one and this is its only send: never blocks.
		q.req.Reply <- ans
		m.chatModel.AppendNotice("Q: " + truncateLabel(q.req.Question, 60))
		m.chatModel.AppendNotice("A: " + questionAnswerLabel(ans))
	}
	return m, nil
}

// questionAnswerLabel renders the answer side of the transcript record: the
// option label or typed text, or the bare outcome when there is no text.
func questionAnswerLabel(ans tools.QuestionAnswer) string {
	if ans.Label != "" {
		return truncateLabel(ans.Label, 60)
	}
	return ans.Selected
}

// questionFooter is the dialog's key-hint line: only the keys that actually
// work, and "t" only when free text is allowed (convention #20).
func questionFooter(allowFree bool) string {
	if allowFree {
		return "↑/↓ · 1-9 select · t own text · Enter confirm · Esc cancel"
	}
	return "↑/↓ · 1-9 select · Enter confirm · Esc cancel"
}

// renderQuestionDialog renders the pending question in the chat panel's last
// slot — the same slot the approval line and the branch popup use. Every
// line is clipped to the panel width, so the block never spills into the
// rail.
func (m *model) renderQuestionDialog(width int) string {
	q := m.question
	if q == nil {
		return ""
	}

	dim := lipgloss.NewStyle().Foreground(m.palette.Dim)
	bold := lipgloss.NewStyle().Bold(true)
	textW := max(10, width-2)
	var b strings.Builder

	// Question head: a dim "?" glyph, then the text wrapped into the panel,
	// capped at questionMaxLines.
	for i, line := range truncateLines(wordWrap(q.req.Question, textW), questionMaxLines, textW) {
		b.WriteString("  ")
		if i == 0 {
			b.WriteString(dim.Render("? "))
		} else {
			b.WriteString("  ")
		}
		b.WriteString(clipRunes(line, textW))
		b.WriteString("\n")
	}

	// Options: "N. label", the highlighted one bold with an arrow cursor;
	// a description renders dim underneath, indented under the label.
	for i, o := range q.req.Options {
		row := fmt.Sprintf("%d. %s", i+1, o.Label)
		b.WriteString("  ")
		if i == q.sel {
			b.WriteString(bold.Render("❯ " + clipRunes(row, max(1, width-5))))
		} else {
			b.WriteString("  " + clipRunes(row, max(1, width-5)))
		}
		b.WriteString("\n")
		if o.Description != "" {
			descW := max(10, width-7)
			for _, line := range truncateLines(wordWrap(o.Description, descW), questionMaxDescLines, descW) {
				b.WriteString(dim.Render("       " + clipRunes(line, descW)))
				b.WriteString("\n")
			}
		}
	}

	// Free text: the invitation row in list mode, the editor row in free
	// mode. The cursor is a plain "▏" — append-only with backspace, no real
	// caret (same as the steer mini-input).
	if q.req.AllowFreeText {
		b.WriteString("  ")
		if q.free {
			b.WriteString(clipRunes("Your answer: "+q.text+"▏  (Enter send · Esc back)", max(1, width-2)))
		} else {
			b.WriteString(dim.Render("[t] Write your own…"))
		}
		b.WriteString("\n")
	}

	b.WriteString("  " + dim.Render(clipRunes(questionFooter(q.req.AllowFreeText), max(1, width-2))))
	return b.String()
}

// truncateLines caps lines at max, replacing the last one's tail with an
// ellipsis so a cut is never mistaken for the whole text.
func truncateLines(lines []string, maxLines, width int) []string {
	if len(lines) <= maxLines {
		return lines
	}
	out := append([]string(nil), lines[:maxLines]...)
	out[maxLines-1] = truncateLabel(out[maxLines-1], max(1, width-1)) + "…"
	return out
}
