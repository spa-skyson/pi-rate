package tui

import (
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"

	"github.com/spa-skyson/pi-rate/internal/permission"
	"github.com/spa-skyson/pi-rate/internal/tools"
)

// newQuestionModel builds a minimal model with the question bridge channel,
// and returns it with a ready-to-send request. Reply is buffered, so the
// test can read the answer after the key handler returns.
func newQuestionModel(t *testing.T) (*model, *tools.QuestionRequest) {
	t.Helper()
	m := &model{width: 140, height: 30}
	m.agentCh = make(chan agentMsg, 8)
	m.cfg.QuestionCh = make(chan SessionQuestion, 1)
	// Requests route by tag: the dialog only opens for the active session.
	m.cfg.SessionID = "sess-q"
	req := tools.QuestionRequest{
		Question: "Which approach should we take?",
		Options: []tools.QuestionOption{
			{Label: "Rewrite it", Description: "clean but slow"},
			{Label: "Patch it"},
		},
		AllowFreeText: true,
		Reply:         make(chan tools.QuestionAnswer, 1),
	}
	return m, &req
}

func TestQuestionRequest_ShowsDialogAndRenders(t *testing.T) {
	m, req := newQuestionModel(t)

	model, cmd := m.handleQuestionRequest(questionRequestMsg{sq: SessionQuestion{Req: *req, SessionID: "sess-q"}})
	if model != m {
		t.Fatal("handler must return the same model")
	}
	if cmd == nil {
		t.Fatal("handler must re-arm the question listener")
	}
	if m.question == nil || m.question.req.Question != req.Question {
		t.Fatalf("question state not set: %+v", m.question)
	}
	if !m.hasOverlay(overlayQuestion) {
		t.Error("dialog must be on the overlay stack")
	}

	view := m.renderQuestionDialog(m.chatWidth())
	for _, want := range []string{
		"?", "Which approach should we take?",
		"1. Rewrite it", "clean but slow", "2. Patch it",
		"[t] Write your own…",
		"↑/↓", "1-9 select", "t own text", "Enter confirm", "Esc cancel",
	} {
		if !strings.Contains(view, want) {
			t.Errorf("question view missing %q; got:\n%s", want, view)
		}
	}
	// The highlighted option carries the cursor.
	if !strings.Contains(view, "❯ 1. Rewrite it") {
		t.Errorf("first option must be highlighted; got:\n%s", view)
	}
}

func TestQuestionRequest_NoFreeTextRowWhenOff(t *testing.T) {
	m, req := newQuestionModel(t)
	req.AllowFreeText = false
	m.handleQuestionRequest(questionRequestMsg{sq: SessionQuestion{Req: *req, SessionID: "sess-q"}})

	view := m.renderQuestionDialog(m.chatWidth())
	if strings.Contains(view, "[t] Write your own…") {
		t.Errorf("free-text invitation shown while allow_free_text=false:\n%s", view)
	}
	if strings.Contains(view, "t own text") {
		t.Errorf("footer advertises t while free text is off:\n%s", view)
	}
}

// TestUpdateRoutesQuestionRequest is the #32 red-guard: the parked reader
// delivers questionRequestMsg through model.Update — the exact path a
// production message takes. Before the routing fix the msg fell through every
// update* handler unhandled: the dialog never opened, the reader never
// re-armed, and the question tool parked on its Reply for as long as the
// session lived.
func TestUpdateRoutesQuestionRequest(t *testing.T) {
	m, req := newQuestionModel(t)

	model, cmd := m.Update(questionRequestMsg{sq: SessionQuestion{Req: *req, SessionID: "sess-q"}})
	if model != m {
		t.Fatal("Update must return the same model")
	}
	if m.question == nil || m.question.req.Question != req.Question {
		t.Fatalf("Update did not open the question dialog: %+v", m.question)
	}
	if !m.hasOverlay(overlayQuestion) {
		t.Error("dialog must be on the overlay stack after Update")
	}
	if cmd == nil {
		t.Fatal("Update must re-arm the question reader")
	}
}

func TestQuestionKey_NavigationAndDigits(t *testing.T) {
	m, req := newQuestionModel(t)
	m.handleQuestionRequest(questionRequestMsg{sq: SessionQuestion{Req: *req, SessionID: "sess-q"}})

	// j/down move down and clamp; k/up move back and clamp.
	if _, _, handled := m.handleQuestionKey(tea.Key{Code: 'j'}); !handled || m.question.sel != 1 {
		t.Fatalf("j must move to option 1 (handled=%v sel=%d)", handled, m.question.sel)
	}
	if _, _, handled := m.handleQuestionKey(tea.Key{Code: 'j'}); !handled || m.question.sel != 1 {
		t.Fatalf("j must clamp at the last option (sel=%d)", m.question.sel)
	}
	if _, _, handled := m.handleQuestionKey(tea.Key{Code: tea.KeyUp}); !handled || m.question.sel != 0 {
		t.Fatalf("up must move back to option 0 (sel=%d)", m.question.sel)
	}
	if _, _, handled := m.handleQuestionKey(tea.Key{Code: 'k'}); !handled || m.question.sel != 0 {
		t.Fatalf("k must clamp at the first option (sel=%d)", m.question.sel)
	}

	// Digits highlight but do not confirm.
	if _, _, handled := m.handleQuestionKey(tea.Key{Code: '2'}); !handled || m.question.sel != 1 {
		t.Fatalf("digit 2 must highlight option 2 (sel=%d)", m.question.sel)
	}
	select {
	case <-req.Reply:
		t.Fatal("a digit must not answer the request")
	default:
	}

	// Digits beyond the list are swallowed but change nothing.
	if _, _, handled := m.handleQuestionKey(tea.Key{Code: '9'}); !handled || m.question.sel != 1 {
		t.Fatalf("digit 9 is out of range (sel=%d)", m.question.sel)
	}
}

func TestQuestionKey_EnterConfirmsSelectedOption(t *testing.T) {
	m, req := newQuestionModel(t)
	m.handleQuestionRequest(questionRequestMsg{sq: SessionQuestion{Req: *req, SessionID: "sess-q"}})
	m.handleQuestionKey(tea.Key{Code: '2'})

	_, _, handled := m.handleQuestionKey(tea.Key{Code: tea.KeyEnter})
	if !handled {
		t.Fatal("dialog must own Enter")
	}
	if m.question != nil {
		t.Error("dialog state must clear after answering")
	}
	if m.hasOverlay(overlayQuestion) {
		t.Error("overlay must be popped after answering")
	}
	select {
	case got := <-req.Reply:
		if got.Selected != "option" || got.Label != "Patch it" || got.Index != 1 {
			t.Errorf("reply = %+v, want option/Patch it/1", got)
		}
	default:
		t.Fatal("no answer was sent to Reply")
	}

	// The exchange is recorded as Q/A fact lines.
	msgs := m.chatModel.Messages
	if n := len(msgs); n < 2 ||
		msgs[n-2].content != "Q: Which approach should we take?" ||
		msgs[n-1].content != "A: Patch it" {
		t.Errorf("fact lines missing: %q / %q", msgs[n-2].content, msgs[n-1].content)
	}
}

func TestQuestionKey_EscCancels(t *testing.T) {
	m, req := newQuestionModel(t)
	m.handleQuestionRequest(questionRequestMsg{sq: SessionQuestion{Req: *req, SessionID: "sess-q"}})

	if _, _, handled := m.handleQuestionKey(tea.Key{Code: tea.KeyEsc}); !handled {
		t.Fatal("dialog must own Esc")
	}
	if m.question != nil {
		t.Error("dialog state must clear after canceling")
	}
	select {
	case got := <-req.Reply:
		if got.Selected != "canceled" {
			t.Errorf("reply = %+v, want canceled", got)
		}
	default:
		t.Fatal("no answer was sent to Reply")
	}
	last := m.chatModel.Messages[len(m.chatModel.Messages)-1]
	if last.content != "A: canceled" {
		t.Errorf("cancel fact line = %q, want %q", last.content, "A: canceled")
	}
}

func TestQuestionKey_OtherKeysFallThrough(t *testing.T) {
	m, req := newQuestionModel(t)
	m.handleQuestionRequest(questionRequestMsg{sq: SessionQuestion{Req: *req, SessionID: "sess-q"}})

	// A key the dialog does not own reaches the prompt input untouched —
	// approval's contract. Ctrl+C among them: canceling the turn must stay
	// available mid-dialog.
	for _, key := range []tea.Key{{Code: 'x'}, {Code: 'c', Mod: tea.ModCtrl}} {
		if _, _, handled := m.handleQuestionKey(key); handled {
			t.Fatalf("key %v must fall through", key)
		}
	}
	if m.question == nil {
		t.Fatal("unrelated key must not dismiss the dialog")
	}
	select {
	case <-req.Reply:
		t.Fatal("unrelated key must not answer the request")
	default:
	}
}

func TestQuestionKey_FreeTextFlow(t *testing.T) {
	m, req := newQuestionModel(t)
	m.handleQuestionRequest(questionRequestMsg{sq: SessionQuestion{Req: *req, SessionID: "sess-q"}})

	// t opens the editor.
	if _, _, handled := m.handleQuestionKey(tea.Key{Code: 't'}); !handled || !m.question.free {
		t.Fatal("t must open the free-text editor")
	}

	// In free mode every key edits: typing appends, Backspace deletes, and
	// Esc returns to the list without answering.
	for _, r := range "make it so" {
		m.handleQuestionKey(tea.Key{Code: r, Text: string(r)})
	}
	m.handleQuestionKey(tea.Key{Code: tea.KeyBackspace}) // "make it s"
	if got := m.question.text; got != "make it s" {
		t.Fatalf("text buffer = %q, want %q", got, "make it s")
	}
	m.handleQuestionKey(tea.Key{Code: tea.KeyEsc})
	if m.question.free || m.question.text != "" {
		t.Fatalf("Esc must return to the list (free=%v text=%q)", m.question.free, m.question.text)
	}
	select {
	case <-req.Reply:
		t.Fatal("leaving free mode must not answer the request")
	default:
	}

	// Re-enter, type, confirm with Enter: the typed text is the answer.
	m.handleQuestionKey(tea.Key{Code: 't'})
	for _, r := range "ship the patch" {
		m.handleQuestionKey(tea.Key{Code: r, Text: string(r)})
	}
	_, _, handled := m.handleQuestionKey(tea.Key{Code: tea.KeyEnter})
	if !handled {
		t.Fatal("Enter must submit the free-text answer")
	}
	select {
	case got := <-req.Reply:
		if got.Selected != "custom" || got.Label != "ship the patch" {
			t.Errorf("reply = %+v, want custom/ship the patch", got)
		}
	default:
		t.Fatal("no answer was sent to Reply")
	}
	if m.question != nil {
		t.Error("dialog state must clear after answering")
	}
}

func TestQuestionKey_FreeTextEmptyEnterIgnored(t *testing.T) {
	m, req := newQuestionModel(t)
	m.handleQuestionRequest(questionRequestMsg{sq: SessionQuestion{Req: *req, SessionID: "sess-q"}})
	m.handleQuestionKey(tea.Key{Code: 't'})

	if _, _, handled := m.handleQuestionKey(tea.Key{Code: tea.KeyEnter}); !handled {
		t.Fatal("free mode owns Enter even when the text is empty")
	}
	if m.question == nil {
		t.Fatal("an empty answer must not close the dialog")
	}
	select {
	case <-req.Reply:
		t.Fatal("an empty answer must not be sent")
	default:
	}
}

func TestQuestionRequest_SecondReplacesFirstWithCanceled(t *testing.T) {
	m, first := newQuestionModel(t)
	if _, _ = m.handleQuestionRequest(questionRequestMsg{sq: SessionQuestion{Req: *first, SessionID: "sess-q"}}); m.question == nil {
		t.Fatal("first request not active")
	}

	second := tools.QuestionRequest{
		Question: "And now?",
		Options:  []tools.QuestionOption{{Label: "yes"}},
		Reply:    make(chan tools.QuestionAnswer, 1),
	}
	m.handleQuestionRequest(questionRequestMsg{sq: SessionQuestion{Req: second, SessionID: "sess-q"}})

	select {
	case got := <-first.Reply:
		if got.Selected != "canceled" {
			t.Errorf("stale request answered %+v, want canceled", got)
		}
	default:
		t.Fatal("stale request was never answered")
	}
	if m.question == nil || m.question.req.Question != "And now?" {
		t.Fatalf("second request not active: %+v", m.question)
	}
}

func TestAgentDone_DismissesPendingQuestion(t *testing.T) {
	m, req := newQuestionModel(t)
	m.handleQuestionRequest(questionRequestMsg{sq: SessionQuestion{Req: *req, SessionID: "sess-q"}})

	m.handleAgentDone(agentDoneMsg{})

	if m.question != nil {
		t.Error("turn end must dismiss a pending question")
	}
	select {
	case <-req.Reply:
		t.Fatal("TUI must not answer a request it did not present")
	default:
	}
	last := m.chatModel.Messages[len(m.chatModel.Messages)-1]
	if !strings.Contains(last.content, "question canceled") {
		t.Errorf("expected a dismissal notice, got %q", last.content)
	}
	want := "question canceled: turn ended before an answer — «Which approach should we take?»"
	if last.content != want {
		t.Errorf("notice = %q, want %q", last.content, want)
	}
}

func TestAgentDone_QuestionNoticeTruncatesLongText(t *testing.T) {
	m, req := newQuestionModel(t)
	req.Question = strings.Repeat("в", 80)
	m.handleQuestionRequest(questionRequestMsg{sq: SessionQuestion{Req: *req, SessionID: "sess-q"}})

	m.handleAgentDone(agentDoneMsg{})

	last := m.chatModel.Messages[len(m.chatModel.Messages)-1]
	// 60 display cells for the question: 59 runes plus the ellipsis, all
	// narrow glyphs; the full 80-rune question must not survive.
	want := "question canceled: turn ended before an answer — «" + strings.Repeat("в", 59) + "…»"
	if last.content != want {
		t.Errorf("notice = %q, want %q", last.content, want)
	}
}

func TestQuestionListener_NilChannel(t *testing.T) {
	if waitForQuestion(nil) != nil {
		t.Fatal("nil channel must produce no command")
	}
}

// TestQuestionAndApproval_OverlaysCoexist pins the stack behavior when both
// bridge dialogs hold state — by construction questions and approvals are
// sequential in the loop, but the stack must still route keys to whichever
// sits on top, and the survivor keeps working after the top one closes.
func TestQuestionAndApproval_OverlaysCoexist(t *testing.T) {
	m, req := newQuestionModel(t)
	m.handleQuestionRequest(questionRequestMsg{sq: SessionQuestion{Req: *req, SessionID: "sess-q"}})

	m.cfg.ApprovalCh = make(chan SessionApproval, 1)
	approval := permission.ApprovalRequest{Tool: "bash", Reply: make(chan permission.ApprovalResult, 1)}
	m.approval = &approval
	m.openOverlay(overlayApproval)

	m.syncOverlays()
	if !m.hasOverlay(overlayQuestion) || !m.hasOverlay(overlayApproval) {
		t.Fatalf("both overlays must be on the stack: %+v", m.overlays)
	}

	// The top entry sees the key first. Re-opened last, approval is on top;
	// its keys win while it stays open.
	if _, _, handled := m.handleApprovalKey(tea.Key{Code: 'y'}); !handled {
		t.Fatal("top overlay must handle its own key")
	}
	if m.approval != nil {
		t.Fatal("approval must be answered and cleared")
	}

	// With approval gone, the question dialog is reachable again.
	if _, _, handled := m.handleQuestionKey(tea.Key{Code: tea.KeyEsc}); !handled {
		t.Fatal("question dialog must handle Esc once it is on top")
	}
	select {
	case got := <-req.Reply:
		if got.Selected != "canceled" {
			t.Errorf("reply = %+v, want canceled", got)
		}
	default:
		t.Fatal("question was never answered")
	}
}

// TestRenderQuestionDialog_NarrowPanel pins the width contract: the block
// never spills past the panel budget, even with long option labels.
func TestRenderQuestionDialog_NarrowPanel(t *testing.T) {
	m, req := newQuestionModel(t)
	req.Question = "This question is deliberately long so that wrapping kicks in and the truncation path is exercised properly."
	req.Options[0].Label = "An extremely long option label that will not fit a narrow panel"
	req.Options[0].Description = "A description long enough to wrap into several dim lines under the label."
	m.handleQuestionRequest(questionRequestMsg{sq: SessionQuestion{Req: *req, SessionID: "sess-q"}})

	const budget = 44
	view := m.renderQuestionDialog(budget)
	for _, line := range strings.Split(view, "\n") {
		if got := lipgloss.Width(line); got > budget {
			t.Errorf("line is %d cells wide, budget %d: %q", got, budget, line)
		}
	}
	if !strings.Contains(view, "…") {
		t.Errorf("a capped question or description must end in an ellipsis; got:\n%s", view)
	}
}

// TestQuestionKey_FreeTextAcceptsNonASCII pins issue #42: the free-text guard
// counts runes, not bytes — a Cyrillic keystroke (2 bytes in UTF-8) must land
// in the buffer, a chord must not, and Backspace must drop a whole rune, not
// one of its bytes.
func TestQuestionKey_FreeTextAcceptsNonASCII(t *testing.T) {
	m, req := newQuestionModel(t)
	m.handleQuestionRequest(questionRequestMsg{sq: SessionQuestion{Req: *req, SessionID: "sess-q"}})
	m.handleQuestionKey(tea.Key{Code: 't'}) // open the free-text editor

	// A Cyrillic rune is 2 bytes; the old byte guard dropped it silently.
	if _, _, handled := m.handleQuestionKey(tea.Key{Code: 'я', Text: "я"}); !handled {
		t.Fatal("free editor must own printable input")
	}
	if got := m.question.text; got != "я" {
		t.Fatalf("text buffer = %q, want %q", got, "я")
	}

	// Latin still appends as before.
	m.handleQuestionKey(tea.Key{Code: 'a', Text: "a"})
	if got := m.question.text; got != "яa" {
		t.Fatalf("text buffer = %q, want %q", got, "яa")
	}

	// A chord never lands, even when Text looks insertable.
	m.handleQuestionKey(tea.Key{Code: 'r', Text: "r", Mod: tea.ModCtrl})
	if got := m.question.text; got != "яa" {
		t.Fatalf("chord inserted: text buffer = %q, want %q", got, "яa")
	}

	// Backspace drops the whole latin rune...
	m.handleQuestionKey(tea.Key{Code: tea.KeyBackspace})
	if got := m.question.text; got != "я" {
		t.Fatalf("after backspace = %q, want %q", got, "я")
	}
	// ...and then the whole Cyrillic rune — a byte-wise delete would leave a
	// lone UTF-8 continuation byte in the buffer.
	m.handleQuestionKey(tea.Key{Code: tea.KeyBackspace})
	if got := m.question.text; got != "" {
		t.Fatalf("after backspace = %q, want empty", got)
	}
	select {
	case <-req.Reply:
		t.Fatal("editing must not answer the request")
	default:
	}
}
