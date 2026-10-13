package tui

import (
	"fmt"
	"strings"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"

	"github.com/spa-skyson/pi-rate/internal/permission"
	"github.com/spa-skyson/pi-rate/internal/tools"
)

// The session-registry popup (design §5): every background slot as one row —
// short id, state (⚡ elapsed while its loop works, ✓ once finished, idle for
// a parked session), buffered event count, and a ⚑ badge when the slot holds
// a stashed input request. Enter attaches the selected slot (refused while
// the active session runs — the v1 guard), d dismisses it (a live slot is
// canceled first: dropping a writing loop on the floor is not an option),
// Esc closes. The registry list is the popup's rows — insertion order,
// nothing reorders.

// slotsPopupState is the open registry popup: a selection over m.slots. The
// rows are recomputed from the live registry at render time, so a run that
// finishes while the popup is open repaints truthfully on the next frame.
// Navigation lives on *model — the row count is the registry's, not the
// popup's.
type slotsPopupState struct {
	selected  int // index into m.slots
	height    int // visible rows
	scrollOff int // scroll offset when more slots than height
}

// newSlotsPopup opens the registry popup. Empty registry: a flash instead —
// an empty popup explains less than the flash that says there is nothing.
func (m *model) newSlotsPopup() {
	if len(m.slots) == 0 {
		m.setFlash("No background sessions")
		return
	}
	height := len(m.slots)
	if height > 8 {
		height = 8
	}
	m.slotsPopup = &slotsPopupState{selected: 0, height: height}
	m.openOverlay(overlaySessions)
}

// moveUp selects the previous slot, scrolling the window when the selection
// would leave the top of it. It stops at the first slot.
func (m *model) moveSlotsSelectionUp() {
	p := m.slotsPopup
	if p == nil || p.selected <= 0 {
		return
	}
	p.selected--
	if p.selected < p.scrollOff {
		p.scrollOff--
	}
}

// moveDown selects the next slot, scrolling the window when the selection
// would leave the bottom of it. It stops at the last slot.
func (m *model) moveSlotsSelectionDown() {
	p := m.slotsPopup
	if p == nil || p.selected >= len(m.slots)-1 {
		return
	}
	p.selected++
	if p.selected >= p.scrollOff+p.height {
		p.scrollOff++
	}
}

// selectedSlot resolves the popup's selection against the live registry.
// nil when the popup is closed or the selection slid out (slots dismissed
// under the popup).
func (m *model) selectedSlot() *backgroundRun {
	if m.slotsPopup == nil || m.slotsPopup.selected >= len(m.slots) {
		return nil
	}
	return m.slots[m.slotsPopup.selected]
}

// clampSelection pulls the selection back into the registry after a slot
// left it, and pulls the scroll window with it: a dismiss under a scrolled
// popup would otherwise leave scrollOff past the last row — a slice-bounds
// panic in the next render — or park the highlight on a row the selection
// no longer names. The popup never shows an empty selection while rows
// remain.
func (m *model) clampSelection() {
	p := m.slotsPopup
	if p == nil {
		return
	}
	if len(m.slots) == 0 {
		m.popOverlay(overlaySessions)
		return
	}
	if p.selected >= len(m.slots) {
		p.selected = len(m.slots) - 1
	}
	if maxOff := len(m.slots) - p.height; p.scrollOff > maxOff {
		if maxOff < 0 {
			maxOff = 0
		}
		p.scrollOff = maxOff
	}
}

// handleSlotsPopupKey navigates the registry popup. Anything other than the
// navigation keys closes it, the branch popup's contract.
func (m *model) handleSlotsPopupKey(key tea.Key) (tea.Model, tea.Cmd, bool) {
	if m.slotsPopup == nil {
		return nil, nil, false
	}
	switch key.Code {
	case tea.KeyEnter:
		model, cmd := m.attachSelectedSlot()
		m.clampSelection()
		return model, cmd, true
	case tea.KeyUp:
		m.moveSlotsSelectionUp()
	case tea.KeyDown:
		m.moveSlotsSelectionDown()
	default:
		if hotkeyIDFor(key) == "key.slots-dismiss" {
			// The dismissal of a live slot returns the parked reader of
			// whatever run it left behind — dropping it would strand the
			// loop without its one reader.
			cmd := m.dismissSelectedSlot()
			return m, cmd, true
		}
		m.popOverlay(overlaySessions)
	}
	return m, nil, true
}

// attachSelectedSlot attaches the highlighted slot. The busy-active guard
// lives in attachBackground; a refusal keeps the popup open so the user can
// pick a different slot without reopening.
func (m *model) attachSelectedSlot() (tea.Model, tea.Cmd) {
	s := m.selectedSlot()
	if s == nil {
		m.popOverlay(overlaySessions)
		return m, nil
	}
	model, cmd := m.attachBackground(s)
	return model, cmd
}

// dismissSelectedSlot removes the highlighted slot, draining it first. A
// live one is canceled — with its inbox emptied BEFORE the cancel, or
// finishSlotRun would resurrect the next queued prompt and turn every d
// into "kill this run, start the next one". The canceled run leaves no
// restart behind (the inbox is gone with it), so the returned command is
// the close-out batch — the attention signal when one is configured, nil
// when the dismissal is silent. A finished or parked slot leaves the
// registry at once, its transcript staying on disk in events.jsonl.
func (m *model) dismissSelectedSlot() tea.Cmd {
	s := m.selectedSlot()
	if s == nil {
		return nil
	}
	if !s.done {
		// A writing loop is canceled, not dropped: its stash dies denied,
		// the row flips to ✓, and the next d removes it. The inbox goes
		// first — dismissal drains the session, it does not continue it.
		s.inbox = nil
		return m.cancelBackgroundSlot(s)
	}
	// Fail-closed hygiene: a finished slot's stash should already be nil
	// (finishSlotRun denies it), but a slot leaves the registry with no
	// pending request, whatever drifted.
	if s.approval != nil {
		s.approval.Reply <- permission.ApprovalResult{}
		s.approval = nil
	}
	if s.question != nil {
		s.question.Reply <- tools.QuestionAnswer{Selected: "canceled"}
		s.question = nil
	}
	m.removeSlot(s)
	m.clampSelection()
	m.chatModel.AppendNotice(fmt.Sprintf(
		"Background session %s dismissed — its transcript stays on disk: /resume %s.",
		shortSessionID(s.sessionID), s.sessionID))
	return nil
}

// slotRow renders one registry row the popup and tests share: the state
// glyph, the short id, the event count, the attention badge.
func slotRow(s *backgroundRun) string {
	st := backgroundStatus(s)
	id := shortSessionID(s.sessionID)
	var state string
	switch {
	case st.Await:
		state = "⚑ needs input"
	case !st.Done:
		state = fmt.Sprintf("⚡ %s", formatTurnDuration(st.Elapsed))
	case st.Parked:
		state = "idle"
	default:
		state = "✓"
	}
	row := fmt.Sprintf("%s  %s · %d ev", id, state, st.Events)
	if st.Dropped > 0 {
		row += fmt.Sprintf(", %d dropped", st.Dropped)
	}
	return row
}

// renderSessionsPopup renders the registry popup — the branch popup's frame
// (thick border, centered, near the bottom) over slot rows.
func (m *model) renderSessionsPopup() string {
	if m.slotsPopup == nil {
		return ""
	}

	popup := m.slotsPopup
	bg := m.palette.Surface0
	border := m.palette.Surface
	selected := m.palette.Primary
	liveFg := m.palette.Green
	waitFg := m.palette.Peach
	dimFg := m.palette.Faint

	popupWidth := m.width - 10
	style := lipgloss.NewStyle().
		Background(bg).
		Foreground(m.palette.Subtext).
		Border(lipgloss.ThickBorder(), true, true, true, true).
		BorderForeground(border).
		Width(popupWidth)

	var b strings.Builder
	b.WriteString("\n")

	header := lipgloss.NewStyle().
		Background(bg).
		Foreground(m.palette.Subtext).
		Bold(true).
		Width(popupWidth).
		Align(lipgloss.Center).
		Render("Background Sessions (Enter attach · d dismiss · Esc close)")
	b.WriteString(header)
	b.WriteString("\n")

	rows := m.slots
	height := popup.height
	scrollOff := popup.scrollOff
	// Clamp the window before anything else: when every row fits
	// (len(rows) <= height) the only correct window is the top — the half
	// of states the in-if clamp never reached — and a past-cap offset has
	// no business anywhere.
	if maxOff := len(rows) - height; maxOff <= 0 || scrollOff > maxOff {
		scrollOff = max(0, maxOff)
	}
	if len(rows) > height {
		// Defense in depth over clampSelection: the render must survive
		// whatever the state says, never panic on a stale window.
		if scrollOff > len(rows)-height {
			scrollOff = len(rows) - height
		}
		rows = rows[scrollOff : scrollOff+height]
	}

	for i, s := range rows {
		actualIndex := i + scrollOff
		isSelected := actualIndex == popup.selected
		st := backgroundStatus(s)

		marker := "    "
		if isSelected {
			marker = "  > "
		}
		line := marker + slotRow(s)

		var lineStyle lipgloss.Style
		switch {
		case isSelected:
			lineStyle = lipgloss.NewStyle().Background(selected).Foreground(m.palette.White)
		case st.Await:
			lineStyle = lipgloss.NewStyle().Background(bg).Foreground(waitFg)
		case !st.Done:
			lineStyle = lipgloss.NewStyle().Background(bg).Foreground(liveFg)
		default:
			lineStyle = lipgloss.NewStyle().Background(bg).Foreground(dimFg)
		}

		b.WriteString(lineStyle.Width(popupWidth).Render(line))
		b.WriteString("\n")
	}

	if len(m.slots) > popup.height {
		scrollStyle := lipgloss.NewStyle().Background(bg).Foreground(dimFg)
		b.WriteString(scrollStyle.Render("  ↑↓ scroll"))
	}

	return style.Render(b.String())
}
