package tui

import (
	"fmt"
	"strings"
	"testing"
	"time"

	tea "charm.land/bubbletea/v2"

	"github.com/spa-skyson/pi-rate/internal/config"
	"github.com/spa-skyson/pi-rate/internal/permission"
)

// The session-registry popup (MR2, design §5): rows, navigation, attach and
// dismiss, plus the /new and /slots command surfaces and the status-bar
// badges built from the registry summary.

// threeRegistrySlots builds a registry with a slot in each state: live
// (running for a while, events buffered), finished-awaiting (a stashed
// request), finished (plain ✓), parked (a /new'd session, nothing to
// review). Rows render in that order.
func threeRegistrySlots(m *model) (*backgroundRun, *backgroundRun, *backgroundRun, *backgroundRun) {
	live := &backgroundRun{
		started:   time.Now().Add(-2*time.Minute - 13*time.Second),
		ch:        make(chan agentMsg, 8),
		sessionID: "sess-live",
	}
	live.record(agentTextMsg{text: "x"})
	live.record(agentTextMsg{text: "y"})
	awaiting := &backgroundRun{
		started:   time.Now().Add(-time.Hour),
		finished:  time.Now().Add(-time.Hour + 3*time.Minute),
		done:      true,
		sessionID: "sess-done",
		approval:  &permission.ApprovalRequest{Tool: "bash", Reply: make(chan permission.ApprovalResult, 1)},
	}
	awaiting.record(agentTextMsg{text: "z"})
	finished := &backgroundRun{
		started:   time.Now().Add(-2 * time.Hour),
		finished:  time.Now().Add(-2*time.Hour + 5*time.Minute),
		done:      true,
		sessionID: "sess-fin",
	}
	finished.record(agentTextMsg{text: "w"})
	parked := &backgroundRun{
		started:   time.Now(),
		finished:  time.Now(),
		done:      true,
		parked:    true,
		sessionID: "sess-park",
	}
	m.addSlot(live)
	m.addSlot(awaiting)
	m.addSlot(finished)
	m.addSlot(parked)
	return live, awaiting, finished, parked
}

// TestSlotsPopup_RendersRegistry: the popup lists every slot with its short
// id, state glyph, event count and the ⚑ badge — and marks the selection.
func TestSlotsPopup_RendersRegistry(t *testing.T) {
	m := newTestModel(t)
	_, _, _, _ = threeRegistrySlots(m)

	m.newSlotsPopup()
	if m.slotsPopup == nil {
		t.Fatal("the popup did not open")
	}
	view := m.renderSessionsPopup()
	for _, want := range []string{
		"Background Sessions",
		"sess-live", "sess-done", "sess-fin", "sess-park",
		"⚡", "✓", "idle", "⚑ needs input",
		"2m", "2 ev",
	} {
		if !strings.Contains(view, want) {
			t.Fatalf("popup view misses %q:\n%s", want, view)
		}
	}
	if !strings.Contains(view, ">") {
		t.Fatal("the popup does not mark a selection")
	}

	// The empty registry explains itself with a flash, no popup.
	m2 := newTestModel(t)
	m2.newSlotsPopup()
	if m2.slotsPopup != nil {
		t.Fatal("an empty registry opened a popup")
	}
	if m2.flash != "No background sessions" {
		t.Fatalf("flash = %q, want the empty-registry notice", m2.flash)
	}
}

// TestSlotsPopup_NavigationAndAttach: arrows move the selection, Enter
// attaches the highlighted slot and closes the popup, a busy active session
// refuses without closing it.
func TestSlotsPopup_NavigationAndAttach(t *testing.T) {
	m := newBgTestModel(t, &heldLLM{name: "popup", release: make(chan struct{})})
	m.chatModel.AppendNotice("active content")
	_, _, _ = m.handleInputSubmit(InputSubmitMsg{Text: "make the active session non-empty"})
	m.cancelAgent() // the submit started a turn on the blocking LLM; stop it
	// Park it: /new moves the session into the registry without a run.
	mdl2, _ := m.handleNewCommand()
	m = mdl2.(*model)
	if len(m.slots) != 1 {
		t.Fatalf("registry holds %d slots after /new, want 1", len(m.slots))
	}
	parkedSID := m.slots[0].sessionID

	m.newSlotsPopup()
	down := tea.KeyPressMsg(tea.Key{Code: tea.KeyDown}).Key()
	if _, _, handled := m.handleSlotsPopupKey(down); !handled {
		t.Fatal("the popup did not claim the arrow key")
	}
	if m.slotsPopup.selected != 0 {
		t.Fatalf("selection = %d, want 0 (one slot, nowhere to move)", m.slotsPopup.selected)
	}

	// Enter attaches the selected slot: the parked session comes back, the
	// popup closes.
	mdl3, _, _ := m.handleSlotsPopupKey(tea.KeyPressMsg(tea.Key{Code: tea.KeyEnter}).Key())
	m = mdl3.(*model)
	if m.cfg.SessionID != parkedSID {
		t.Fatalf("Enter attached %s, want %s", m.cfg.SessionID, parkedSID)
	}
	if m.slotsPopup != nil {
		t.Fatal("a successful attach left the popup open")
	}
	if !hasConversationContent(m.chatModel.Messages) {
		t.Fatal("the attached session's transcript did not come back")
	}
}

// TestSlotsPopup_AttachRefusedKeepsPopup: with a running active session,
// Enter refuses and the popup stays for a different pick.
func TestSlotsPopup_AttachRefusedKeepsPopup(t *testing.T) {
	m := newTestModel(t)
	m.running = true
	m.addSlot(&backgroundRun{started: time.Now(), sessionID: "sess-bg"})

	m.newSlotsPopup()
	mdl, _, _ := m.handleSlotsPopupKey(tea.KeyPressMsg(tea.Key{Code: tea.KeyEnter}).Key())
	m = mdl.(*model)
	if m.slotsPopup == nil {
		t.Fatal("the refused attach closed the popup")
	}
	if !strings.Contains(m.flash, "active session is running") {
		t.Fatalf("flash = %q, want the busy-active refusal", m.flash)
	}
}

// TestSlotsPopup_Dismiss: d cancels a live slot first (the row flips to a
// finished state), a second d removes it; a finished slot leaves at once,
// named in a notice.
func TestSlotsPopup_Dismiss(t *testing.T) {
	m := newTestModel(t)
	live, awaiting, finished, parked := threeRegistrySlots(m)
	_ = parked // dismissed last, driven by selection
	m.newSlotsPopup()

	// d on the live slot (row 0): canceled, kept for review.
	m.dismissSelectedSlot()
	if !live.done {
		t.Fatal("dismissing a live slot did not cancel it")
	}
	if len(m.slots) != 4 {
		t.Fatalf("registry holds %d slots after canceling the live one, want 4", len(m.slots))
	}
	select {
	case <-awaiting.approval.Reply:
		t.Fatal("dismiss touched an unrelated slot's stash")
	default:
	}

	// Move to the finished-awaiting slot (row 1) and dismiss it: leaves the
	// registry, its stash denied on the way out. Capture the reply channel
	// first — the dismissal nils the stash.
	awaitReply := awaiting.approval.Reply
	m.moveSlotsSelectionDown()
	m.dismissSelectedSlot()
	if len(m.slots) != 3 {
		t.Fatalf("registry holds %d slots after the dismissal, want 3", len(m.slots))
	}
	for _, s := range m.slots {
		if s == awaiting {
			t.Fatal("the dismissed slot stayed in the registry")
		}
	}
	select {
	case res := <-awaitReply:
		if res.Allowed {
			t.Fatal("the dismissed slot's stash was answered allowed")
		}
	default:
		t.Fatal("the dismissed slot's stash was never denied")
	}

	// The plain finished slot dismisses too (row 1 after the clamp), then
	// the parked one; the selection clamps each time.
	m.dismissSelectedSlot()
	if len(m.slots) != 2 || m.slotBySession(finished.sessionID) == finished {
		t.Fatalf("registry holds %d slots, want 2 without the finished one", len(m.slots))
	}
	m.dismissSelectedSlot()
	if len(m.slots) != 1 {
		t.Fatalf("registry holds %d slots, want only the canceled live one", len(m.slots))
	}
	if m.slotsPopup.selected != 0 {
		t.Fatalf("selection = %d after the clamp, want 0", m.slotsPopup.selected)
	}
	if !strings.Contains(noticesText(m), "dismissed") {
		t.Fatal("the dismissal notice is missing")
	}
}

// TestSlotsPopup_EscCloses: any unclaimed key closes the popup and clears
// its state through the overlay's onClose.
func TestSlotsPopup_EscCloses(t *testing.T) {
	m := newTestModel(t)
	m.addSlot(&backgroundRun{started: time.Now(), sessionID: "sess-bg"})
	m.newSlotsPopup()

	esc := tea.KeyPressMsg(tea.Key{Code: tea.KeyEsc}).Key()
	mdl, _, handled := m.handleSlotsPopupKey(esc)
	m = mdl.(*model)
	if !handled {
		t.Fatal("the popup did not claim Esc")
	}
	if m.slotsPopup != nil {
		t.Fatal("Esc left the popup state open")
	}
}

// TestSlotsCommand_OpensPopup: /slots dispatches through the command
// registry into the popup, the same table every other command uses.
func TestSlotsCommand_OpensPopup(t *testing.T) {
	m := newTestModel(t)
	m.addSlot(&backgroundRun{started: time.Now(), sessionID: "sess-bg"})

	mdl, _ := m.handleSlashCommand("/slots")
	m = mdl.(*model)
	if m.slotsPopup == nil {
		t.Fatal("/slots did not open the registry popup")
	}
}

// TestNewCommandStateMatrix walks /new's three states: busy refuses, an
// empty session says so, a session with content parks into the registry and
// a fresh empty one takes the screen.
func TestNewCommandStateMatrix(t *testing.T) {
	t.Run("busy refuses", func(t *testing.T) {
		m := newBgTestModel(t, &heldLLM{name: "new-busy", release: make(chan struct{})})
		m.running = true
		mdl, _ := m.handleNewCommand()
		m = mdl.(*model)
		if len(m.slots) != 0 {
			t.Fatal("/new parked a session while the turn was running")
		}
		if !strings.Contains(m.flash, "running") {
			t.Fatalf("flash = %q, want the busy refusal", m.flash)
		}
	})

	t.Run("empty says so", func(t *testing.T) {
		m := newBgTestModel(t, &heldLLM{name: "new-empty", release: make(chan struct{})})
		mdl, _ := m.handleNewCommand()
		m = mdl.(*model)
		if len(m.slots) != 0 {
			t.Fatal("/new parked an empty session")
		}
		if m.flash != "This session is already empty" {
			t.Fatalf("flash = %q, want the already-empty notice", m.flash)
		}
	})

	t.Run("idle parks", func(t *testing.T) {
		m := newBgTestModel(t, &heldLLM{name: "new-idle", release: make(chan struct{})})
		m.chatModel.AppendNotice("fork marker") // chrome does not count as content
		_, _, _ = m.handleInputSubmit(InputSubmitMsg{Text: "conversation to keep"})
		m.cancelAgent() // the submit started a turn on the blocking LLM; stop it
		prevSID := m.cfg.SessionID

		mdl, _ := m.handleNewCommand()
		m = mdl.(*model)
		if len(m.slots) != 1 {
			t.Fatalf("registry holds %d slots after /new, want 1", len(m.slots))
		}
		parked := m.slots[0]
		if parked.sessionID != prevSID {
			t.Fatal("/new did not park the previous session")
		}
		if !parked.done || parked.ch != nil {
			t.Fatal("a parked session must be idle: no loop, done from birth")
		}
		if m.running {
			t.Fatal("/new left the model running")
		}
		if hasConversationContent(m.chatModel.Messages) {
			t.Fatal("the fresh session inherited content")
		}
		if !strings.Contains(noticesText(m), "moved to the background registry") {
			t.Fatal("the /new notice is missing")
		}

		// The parked session attaches back whole.
		m.newSlotsPopup()
		mdl2, _, _ := m.handleSlotsPopupKey(tea.KeyPressMsg(tea.Key{Code: tea.KeyEnter}).Key())
		m = mdl2.(*model)
		if m.cfg.SessionID != prevSID {
			t.Fatalf("attach restored %s, want the parked %s", m.cfg.SessionID, prevSID)
		}
		cameBack := false
		for _, msg := range m.chatModel.Messages {
			if msg.role == "user" && msg.content == "conversation to keep" {
				cameBack = true
			}
		}
		if !cameBack {
			t.Fatal("the parked transcript did not come back")
		}
	})
}

// TestBackground_ToggleOpensPopupWhenMany: Ctrl+B with an idle active
// session attaches the single slot, opens the popup over several, and
// detaches while running no matter how many slots exist.
func TestBackground_ToggleOpensPopupWhenMany(t *testing.T) {
	m := newBgTestModel(t, &heldLLM{name: "toggle-many", release: make(chan struct{})})
	m.addSlot(&backgroundRun{started: time.Now(), sessionID: "s1"})
	m.addSlot(&backgroundRun{started: time.Now(), sessionID: "s2"})

	// Idle over several slots: Ctrl+B opens the popup.
	newM, _ := m.handleKey(tea.KeyPressMsg(ctrlB()))
	m = newM.(*model)
	if m.slotsPopup == nil {
		t.Fatal("Ctrl+B over several slots did not open the registry popup")
	}
	m.popOverlay(overlaySessions)

	// Running: Ctrl+B detaches regardless of the registry size.
	m.running = true
	m.agentCh = make(chan agentMsg, 8)
	newM, _ = m.handleKey(tea.KeyPressMsg(ctrlB()))
	m = newM.(*model)
	if len(m.slots) != 3 {
		t.Fatalf("registry holds %d slots after a running detach, want 3", len(m.slots))
	}
	if m.running {
		t.Fatal("the detach left the model running")
	}

	// Idle with exactly one slot: Ctrl+B attaches it back.
	m2 := newTestModel(t)
	m2.addSlot(&backgroundRun{started: time.Now(), sessionID: "only"})
	newM, _ = m2.handleKey(tea.KeyPressMsg(ctrlB()))
	m2 = newM.(*model)
	if m2.cfg.SessionID != "only" {
		t.Fatalf("Ctrl+B with one slot attached %s, want only", m2.cfg.SessionID)
	}
	if m2.slotsPopup != nil {
		t.Fatal("a single-slot Ctrl+B opened the popup")
	}
}

// TestBackground_StatusBadgesUpdateOnEvents: the summary the bar renders
// moves with the registry — an event bumps the event count, a stash turns
// the ⚑ on, a finish moves Live while Slots holds.
func TestBackground_StatusBadgesUpdateOnEvents(t *testing.T) {
	m := newTestModel(t)
	live := &backgroundRun{started: time.Now(), sessionID: "s1", ch: make(chan agentMsg, 8)}
	fin := &backgroundRun{started: time.Now(), done: true, sessionID: "s2"}
	m.addSlot(live)
	m.addSlot(fin)

	sum := m.backgroundSummary()
	if sum.Slots != 2 || sum.Live != 1 || sum.Events != 0 || sum.Awaiting != 0 {
		t.Fatalf("initial summary = %+v, want 2 slots / 1 live / 0 ev / 0 waiting", sum)
	}

	// A buffered event moves the event count the bar shows.
	m.Update(slotEnvelope{ch: live.ch, ev: agentTextMsg{text: "delta"}})
	if sum := m.backgroundSummary(); sum.Events != 1 {
		t.Fatalf("summary after one event = %+v, want 1 ev", sum)
	}

	// A stashed request turns the ⚑ on.
	live.approval = &permission.ApprovalRequest{Tool: "bash", Reply: make(chan permission.ApprovalResult, 1)}
	if sum := m.backgroundSummary(); sum.Awaiting != 1 {
		t.Fatalf("summary after the stash = %+v, want 1 waiting", sum)
	}

	// The slot's run finishes: Live drops, Slots holds, the stash is gone.
	m.Update(slotEnvelope{ch: live.ch, ev: nil})
	if sum := m.backgroundSummary(); sum.Live != 0 || sum.Slots != 2 || sum.Awaiting != 0 {
		t.Fatalf("summary after the finish = %+v, want 2 slots / 0 live / 0 waiting", sum)
	}
}

// TestSlotsPopup_DismissClampsScrollWindow: with more slots than the popup
// shows, dismissing the bottom row must not push the scroll window past the
// registry (a slice-bounds panic in the next render) and must not leave the
// highlight on a row the selection no longer names — Enter still attaches
// exactly the highlighted row.
func TestSlotsPopup_DismissClampsScrollWindow(t *testing.T) {
	m := newTestModel(t)
	for i := 0; i < 10; i++ {
		m.addSlot(&backgroundRun{started: time.Now(), done: true, sessionID: fmt.Sprintf("sess-%02d", i)})
	}
	m.newSlotsPopup()
	if m.slotsPopup.height != 8 {
		t.Fatalf("popup height = %d, want the 8-row cap", m.slotsPopup.height)
	}

	// Scroll to the bottom: selection 9, window at offset 2.
	for i := 0; i < 9; i++ {
		m.moveSlotsSelectionDown()
	}
	if m.slotsPopup.selected != 9 || m.slotsPopup.scrollOff != 2 {
		t.Fatalf("setup: selected=%d scrollOff=%d, want 9/2", m.slotsPopup.selected, m.slotsPopup.scrollOff)
	}
	bottom := m.selectedSlot().sessionID
	if bottom != "sess-09" {
		t.Fatalf("bottom selection = %s, want sess-09", bottom)
	}

	// Dismiss the bottom slot: the registry shrinks to 9, the window and
	// the selection clamp, the render neither panics nor highlights a
	// stranger.
	m.dismissSelectedSlot()
	if len(m.slots) != 9 {
		t.Fatalf("registry holds %d slots, want 9", len(m.slots))
	}
	if m.slotsPopup.selected != 8 || m.slotsPopup.scrollOff != 1 {
		t.Fatalf("after dismiss: selected=%d scrollOff=%d, want 8/1", m.slotsPopup.selected, m.slotsPopup.scrollOff)
	}
	highlighted := m.slots[m.slotsPopup.selected]
	wantID := highlighted.sessionID // captured: attach swaps the slot's sessionID with the displaced one
	view := m.renderSessionsPopup()
	if !strings.Contains(view, "> "+shortSessionID(wantID)) {
		t.Fatalf("the highlight does not sit on the selected row:\n%s", view)
	}

	// Enter attaches exactly the highlighted row.
	mdl, _ := m.attachSelectedSlot()
	m = mdl.(*model)
	if m.cfg.SessionID != wantID {
		t.Fatalf("Enter attached %s, want the highlighted %s", m.cfg.SessionID, wantID)
	}

	// Dismissing down to an empty registry pulls scrollOff to zero and
	// closes the popup. The attach above closed it too (attachBackground
	// pops the popup on success), so every iteration reopens it first.
	for len(m.slots) > 0 {
		if m.slotsPopup == nil {
			m.newSlotsPopup()
		}
		m.dismissSelectedSlot()
	}
	if m.slotsPopup != nil {
		t.Fatal("an emptied registry left the popup open")
	}
}

// TestSlotsPopup_DismissDrainsInbox: dismissing a live slot empties its
// inbox BEFORE the cancel — the dismissal ends the session, it does not
// resurrect the next queued prompt — and the returned command is the parked
// reader of the slot's channel, never dropped.
func TestSlotsPopup_DismissDrainsInbox(t *testing.T) {
	release := make(chan struct{})
	t.Cleanup(func() { close(release) })
	m := newBgTestModel(t, &heldLLM{name: "dismiss", release: release})
	// Attention on: the cancel path's close-out carries a real command, so
	// the test can pin that the d key PROPAGATES it instead of dropping it.
	m.cfg.Attention = &config.AttentionConfig{}
	m.beginTurn()
	reader := m.startAgentLoop("long task")
	newM, _ := m.handleKey(tea.KeyPressMsg(ctrlB()))
	m = newM.(*model)
	slot := m.slots[0]
	slot.inbox = []queuedPrompt{{text: "next"}, {text: "after next"}}

	m.newSlotsPopup()
	// d on the live slot: canceled, inbox drained — no resurrection. The
	// key handler returns the dismissal's command (the close-out batch);
	// dropping it here is how a parked reader gets lost.
	d := tea.KeyPressMsg(tea.Key{Code: 'd'}).Key()
	mdl, cmd, handled := m.handleSlotsPopupKey(d)
	m = mdl.(*model)
	if !handled {
		t.Fatal("the popup did not claim the d key")
	}
	if !slot.done {
		t.Fatal("the dismissal did not finalize the live slot")
	}
	if len(slot.inbox) != 0 {
		t.Fatalf("the dismissal left %d inbox prompts — the session was continued, not drained", len(slot.inbox))
	}
	if cmd == nil {
		t.Fatal("the d key dropped the dismissal's command")
	}
	_ = reader

	// The second d removes the finished slot.
	m.dismissSelectedSlot()
	if len(m.slots) != 0 {
		t.Fatalf("registry holds %d slots after the second d, want 0", len(m.slots))
	}
	// The drained slot never restarted, so there is no cancel func to call
	// (finishSlotRun nils it; startSlotInbox found an empty inbox) — and no
	// loop left writing behind the test.
}
