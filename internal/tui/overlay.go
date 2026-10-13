package tui

import tea "charm.land/bubbletea/v2"

// OverlayKind names one built-in overlay. The kind is the stack's addressing
// unit: openOverlay/popOverlay move entries by kind, so a call site never
// handles stack positions directly.
type OverlayKind int

const (
	overlayApproval OverlayKind = iota
	overlayQuestion
	overlayCommit
	overlayUpdate
	overlayLogin
	overlaySkillCreate
	overlayBranchPopup
	overlaySessions
	overlaySubagentViewer
	overlaySearchPopup
	overlaySteerInput
	overlayDiffViewer

	// overlayCustom marks a synthetic entry pushed by hand (tests today,
	// dynamic overlays later) that has no backing model field.
	overlayCustom
)

// overlayKinds lists the built-in overlays in canonical priority order — the
// fixed handleKey chain this stack replaced. syncOverlays uses it to repair
// drift; nothing else may dispatch from it.
var overlayKinds = []OverlayKind{
	overlayApproval,
	overlayQuestion,
	overlayCommit,
	overlayUpdate,
	overlayLogin,
	overlaySkillCreate,
	overlayBranchPopup,
	overlaySessions,
	overlaySubagentViewer,
	overlaySearchPopup,
	overlaySteerInput,
	overlayDiffViewer,
}

// overlayEntry is one open layer of the modal stack.
type overlayEntry struct {
	kind OverlayKind

	// keyHandler gets every key while the entry is on the stack. It is the
	// same contract as the old fixed-chain handlers: handled=true wins the
	// key, handled=false lets it pass.
	keyHandler keyHandler

	// alive reports whether the overlay is still open. Built-in entries
	// check their backing field, so drift between field and stack cannot
	// outlive one dispatch; nil means always alive (synthetic entries).
	alive func(*model) bool

	// onClose clears the overlay's state. popOverlay runs it, so call sites
	// close by kind and never nil the field by hand. nil means nothing.
	onClose func(*model)

	// swallowsInput consumes a key the handler declined instead of letting
	// it fall to the layers below. Every built-in overlay leaves this false:
	// their handlers already swallow what they own by returning handled=true
	// and deliberately pass the rest through (approval's stray keys reach the
	// prompt, the viewer lets Ctrl+C through to the interrupt layer). Set it
	// for an overlay that must shade everything below it unconditionally.
	swallowsInput bool
}

// pushOverlay puts an entry on top of the stack — the extension point that
// keeps new overlays out of handleKey: the top entry sees every key first.
func (m *model) pushOverlay(e overlayEntry) {
	m.overlays = append(m.overlays, e)
}

// replaceOverlay pushes entry after dropping any earlier entry of the same
// kind, without running its onClose — re-opening over yourself is not
// closing. This is how a fresh search popup replaces the one it succeeds.
func (m *model) replaceOverlay(e overlayEntry) {
	m.removeOverlay(e.kind)
	m.pushOverlay(e)
}

// removeOverlay drops the topmost entry of kind without running onClose.
func (m *model) removeOverlay(kind OverlayKind) bool {
	for i := len(m.overlays) - 1; i >= 0; i-- {
		if m.overlays[i].kind != kind {
			continue
		}
		m.overlays = append(m.overlays[:i], m.overlays[i+1:]...)
		return true
	}
	return false
}

// hasOverlay reports whether kind is on the stack.
func (m *model) hasOverlay(kind OverlayKind) bool {
	for _, e := range m.overlays {
		if e.kind == kind {
			return true
		}
	}
	return false
}

// popOverlay closes the topmost overlay of kind: drops its stack entry and
// runs its onClose, which clears the backing field. If the entry is already
// gone — the field was cleared behind the stack — onClose runs anyway, so a
// pop by kind always closes the overlay's state. Reports whether an entry
// was removed.
func (m *model) popOverlay(kind OverlayKind) bool {
	for i := len(m.overlays) - 1; i >= 0; i-- {
		if m.overlays[i].kind != kind {
			continue
		}
		e := m.overlays[i]
		m.overlays = append(m.overlays[:i], m.overlays[i+1:]...)
		if e.onClose != nil {
			e.onClose(m)
		}
		return true
	}
	if close := m.overlayEntryFor(kind).onClose; close != nil {
		close(m)
	}
	return false
}

// clearOverlays pops every entry, innermost first, running each onClose.
func (m *model) clearOverlays() {
	for i := len(m.overlays) - 1; i >= 0; i-- {
		if close := m.overlays[i].onClose; close != nil {
			close(m)
		}
	}
	m.overlays = m.overlays[:0]
}

// openOverlay registers a built-in overlay as open, on top of the stack. Call
// it right where the overlay's field is assigned; re-opening an open kind
// replaces its entry, which also lifts it back to the top — the commit flow
// re-enters modal when its message arrives, and must outrank what opened
// under it meanwhile.
func (m *model) openOverlay(kind OverlayKind) {
	m.replaceOverlay(m.overlayEntryFor(kind))
}

// dispatchOverlays offers the key to the stack, top entry first. An entry
// that handles the key wins; one that declines passes the key below, and a
// swallowsInput entry shades the layers below it instead. A handler may pop
// only its own entry, and only when returning handled, so the slice is stable
// while a key walks down.
func (m *model) dispatchOverlays(key tea.Key) (tea.Model, tea.Cmd, bool) {
	for i := len(m.overlays) - 1; i >= 0; i-- {
		e := m.overlays[i]
		model, cmd, handled := e.keyHandler(key)
		if handled {
			return model, cmd, true
		}
		if e.swallowsInput {
			return m, nil, true
		}
	}
	return m, nil, false
}

// syncOverlays reconciles the stack with the overlay fields before a dispatch:
// entries whose overlay closed behind the stack are dropped, and open
// overlays missing an entry are filled in on top, walking the canonical list
// back to front so the highest-priority kind lands topmost. All real opens
// and closes go through openOverlay/popOverlay; this is the guard that keeps
// a missed helper call — or a test poking a field directly — from changing
// what the next keystroke does.
func (m *model) syncOverlays() {
	kept := m.overlays[:0]
	for _, e := range m.overlays {
		if e.alive == nil || e.alive(m) {
			kept = append(kept, e)
		}
	}
	m.overlays = kept

	for i := len(overlayKinds) - 1; i >= 0; i-- {
		e := m.overlayEntryFor(overlayKinds[i])
		if !e.alive(m) || m.hasOverlay(e.kind) {
			continue
		}
		m.pushOverlay(e)
	}
}

// overlayEntryFor builds the stack entry for a built-in overlay, bound to m.
// The handlers are the same ones the fixed chain called; only the dispatch
// moved.
func (m *model) overlayEntryFor(kind OverlayKind) overlayEntry {
	switch kind {
	case overlayApproval:
		return overlayEntry{
			kind:       kind,
			keyHandler: m.handleApprovalKey,
			alive:      func(*model) bool { return m.approval != nil },
			onClose:    func(*model) { m.approval = nil },
		}
	case overlayQuestion:
		return overlayEntry{
			kind:       kind,
			keyHandler: m.handleQuestionKey,
			alive:      func(*model) bool { return m.question != nil },
			onClose:    func(*model) { m.question = nil },
		}
	case overlayCommit:
		return overlayEntry{
			kind:       kind,
			keyHandler: m.handleCommitKey,
			// Any phase counts: the handler itself stays passive while the
			// message generates and turns modal on "confirming".
			alive:   func(*model) bool { return m.commit != nil },
			onClose: func(*model) { m.commit = nil },
		}
	case overlayUpdate:
		return overlayEntry{
			kind:       kind,
			keyHandler: m.handleUpdateKey,
			// Same as commit: the check runs passive, the confirm turns modal.
			alive:   func(*model) bool { return m.update != nil },
			onClose: func(*model) { m.update = nil },
		}
	case overlayLogin:
		return overlayEntry{
			kind:       kind,
			keyHandler: m.handleLoginKey,
			alive:      func(*model) bool { return m.login != nil },
			onClose:    func(*model) { m.login = nil },
		}
	case overlaySkillCreate:
		return overlayEntry{
			kind:       kind,
			keyHandler: m.handleSkillCreateKey,
			alive:      func(*model) bool { return m.pendingSkillCreate != nil },
			onClose:    func(*model) { m.pendingSkillCreate = nil },
		}
	case overlayBranchPopup:
		return overlayEntry{
			kind:       kind,
			keyHandler: m.handleBranchPopupKey,
			alive:      func(*model) bool { return m.branchPopup != nil },
			onClose:    func(*model) { m.branchPopup = nil },
		}
	case overlaySessions:
		return overlayEntry{
			kind:       kind,
			keyHandler: m.handleSlotsPopupKey,
			alive:      func(*model) bool { return m.slotsPopup != nil },
			onClose:    func(*model) { m.slotsPopup = nil },
		}
	case overlaySubagentViewer:
		return overlayEntry{
			kind:       kind,
			keyHandler: m.handleSubagentViewerKey,
			alive:      func(*model) bool { return m.subagentViewer != nil },
			onClose:    func(*model) { m.subagentViewer = nil },
		}
	case overlaySearchPopup:
		return overlayEntry{
			kind: kind,
			keyHandler: func(key tea.Key) (tea.Model, tea.Cmd, bool) {
				cmd, handled := m.handleSearchPopupKey(key)
				return m, cmd, handled
			},
			alive:   func(*model) bool { return m.searchPopup != nil },
			onClose: func(*model) { m.searchPopup = nil },
		}
	case overlaySteerInput:
		return overlayEntry{
			kind:       kind,
			keyHandler: m.handleSteerOverlayKey,
			alive:      func(*model) bool { return m.steerInput != nil },
			onClose:    func(*model) { m.steerInput = nil },
		}
	case overlayDiffViewer:
		return overlayEntry{
			kind:       kind,
			keyHandler: m.handleDiffViewerKey,
			alive:      func(*model) bool { return m.diffViewer != nil },
			onClose:    func(*model) { m.diffViewer = nil },
		}
	default:
		return overlayEntry{kind: kind}
	}
}
