package tui

import (
	"os"
	"regexp"
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"
)

// -----------------------------------------------------------------------------
// Registry integrity
// -----------------------------------------------------------------------------

func TestKeyRegistry_Integrity(t *testing.T) {
	t.Parallel()
	seenIDs := make(map[string]bool, len(keyRegistry))
	seenKeys := make(map[string]string, len(keyRegistry))
	for _, b := range keyRegistry {
		if b.ID == "" {
			t.Error("registry row with empty ID")
		} else if seenIDs[b.ID] {
			t.Errorf("duplicate registry ID %q", b.ID)
		}
		seenIDs[b.ID] = true

		if b.Key == "" {
			t.Errorf("%s: empty key", b.ID)
		}
		if dup, ok := seenKeys[b.Key]; ok {
			t.Errorf("%s: key %q already bound by %s", b.ID, b.Key, dup)
		}
		seenKeys[b.Key] = b.ID

		if b.Description == "" {
			t.Errorf("%s: empty description", b.ID)
		}
		if b.Category == "" {
			t.Errorf("%s: empty category", b.ID)
		}
		if b.Kind != kindSlash && b.Kind != kindHotkey {
			t.Errorf("%s: unknown kind %q", b.ID, b.Kind)
		}
		if b.Kind == kindSlash && !strings.HasPrefix(b.Key, "/") {
			t.Errorf("%s: slash row key %q is not a command", b.ID, b.Key)
		}
	}
}

// currentKeyOf is the seam phase 2 (config overrides) plugs into. Until then
// it must be a pass-through of the default.
func TestCurrentKeyOf(t *testing.T) {
	t.Parallel()
	if got := currentKeyOf(keyBinding{Key: "ctrl+t"}); got != "ctrl+t" {
		t.Errorf("currentKeyOf without override = %q, want the default", got)
	}
	if got := currentKeyOf(keyBinding{Key: "ctrl+t", Override: "ctrl+g"}); got != "ctrl+g" {
		t.Errorf("currentKeyOf with override = %q, want the override", got)
	}
}

// The hotkey rows are the snapshot of every chord the dispatchers answered
// before the registry existed. A removed row is a removed keystroke.
func TestKeyRegistry_CoversHotkeys(t *testing.T) {
	t.Parallel()
	want := map[string]string{ // key → ID
		"ctrl+r":    "key.retry",
		"ctrl+o":    "key.compact-tools",
		"ctrl+b":    "key.background",
		"ctrl+g":    "key.branch",
		"shift+tab": "key.agent-cycle",
		"ctrl+h":    "key.history",
		"ctrl+t":    "key.monitor",
		"esc":       "key.cancel",
		"ctrl+c":    "key.quit",
		"ctrl+z":    "key.suspend",
		"up":        "key.history-window",
		"down":      "key.monitor-down",
		"pgup":      "key.chat-up",
		"pgdn":      "key.chat-down",
		"s":         "key.subagents-steer",
		"d":         "key.slots-dismiss",
	}
	byID := make(map[string]keyBinding, len(keyRegistry))
	for _, b := range keyRegistry {
		if b.Kind == kindHotkey {
			byID[b.ID] = b
		}
	}
	if len(byID) != len(want) {
		t.Fatalf("registry has %d hotkey rows, want %d", len(byID), len(want))
	}
	for key, id := range want {
		b, ok := byID[id]
		if !ok {
			t.Errorf("hotkey %q missing from the registry", id)
			continue
		}
		if b.Key != key {
			t.Errorf("%s bound to %q, want %q", id, b.Key, key)
		}
		if b.Description == "" || b.Category == "" {
			t.Errorf("%s: help dialog needs a description and category", id)
		}
	}
}

// Every slash row has a handler and every handler belongs to a slash row —
// the lazy table and the registry cannot drift apart.
func TestKeyRegistry_HandlersMatchSlashRows(t *testing.T) {
	t.Parallel()
	slashHandlerFor("cmd.help") // force the lazy build
	rows := 0
	for _, b := range keyRegistry {
		if b.Kind != kindSlash {
			continue
		}
		rows++
		if slashHandlers[b.ID] == nil {
			t.Errorf("%s (%s): no handler in the lazy table", b.ID, b.Key)
		}
	}
	if len(slashHandlers) != rows {
		t.Errorf("lazy handler table has %d entries for %d slash rows", len(slashHandlers), rows)
	}
}

// -----------------------------------------------------------------------------
// Dispatch through the registry
// -----------------------------------------------------------------------------

func TestHotkeyFor_ResolvesRegistryKeys(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name string
		key  tea.Key
		want string
	}{
		{"ctrl+r retries", tea.Key{Code: 'r', Mod: tea.ModCtrl}, "key.retry"},
		{"ctrl+o compacts", tea.Key{Code: 'o', Mod: tea.ModCtrl}, "key.compact-tools"},
		{"ctrl+b backgrounds", tea.Key{Code: 'b', Mod: tea.ModCtrl}, "key.background"},
		{"ctrl+g branch", tea.Key{Code: 'g', Mod: tea.ModCtrl}, "key.branch"},
		{"shift+tab cycles", tea.Key{Code: tea.KeyTab, Mod: tea.ModShift}, "key.agent-cycle"},
		{"ctrl+h history", tea.Key{Code: 'h', Mod: tea.ModCtrl}, "key.history"},
		{"ctrl+t monitor", tea.Key{Code: 't', Mod: tea.ModCtrl}, "key.monitor"},
		{"esc cancels", tea.Key{Code: tea.KeyEsc}, "key.cancel"},
		{"ctrl+c quits", tea.Key{Code: 'c', Mod: tea.ModCtrl}, "key.quit"},
		{"ctrl+z suspends", tea.Key{Code: 'z', Mod: tea.ModCtrl}, "key.suspend"},
		{"up history window", tea.Key{Code: tea.KeyUp}, "key.history-window"},
		{"down monitor", tea.Key{Code: tea.KeyDown}, "key.monitor-down"},
		{"pgup scroll", tea.Key{Code: tea.KeyPgUp}, "key.chat-up"},
		{"pgdn scroll", tea.Key{Code: tea.KeyPgDown}, "key.chat-down"},
		{"s steers", tea.Key{Code: 's', Text: "s"}, "key.subagents-steer"},
		{"unregistered chord", tea.Key{Code: 'x', Mod: tea.ModCtrl}, ""},
		{"bare tab", tea.Key{Code: tea.KeyTab}, ""},
		{"uppercase s", tea.Key{Code: 'S', Text: "S"}, ""},
		// Exact-modifier semantics, as the dispatchers always had: a chord
		// with an extra modifier does not read as its base binding.
		{"ctrl+shift+r is not retry", tea.Key{Code: 'r', Mod: tea.ModCtrl | tea.ModShift}, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := hotkeyIDFor(tt.key); got != tt.want {
				t.Errorf("hotkeyIDFor(%+v) = %q, want %q", tt.key, got, tt.want)
			}
		})
	}
}

// -----------------------------------------------------------------------------
// Source guards — the registry is the one table; these keep it that way.
// -----------------------------------------------------------------------------

// nonTestSources returns the package's non-test .go files, name → contents.
// Line endings are normalised to LF so that funcBody and other consumers do not
// need to handle CRLF (which git's autocrlf produces on Windows checkouts).
func nonTestSources(t *testing.T) map[string]string {
	t.Helper()
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatalf("read package dir: %v", err)
	}
	sources := make(map[string]string)
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		data, err := os.ReadFile(name)
		if err != nil {
			t.Fatalf("read %s: %v", name, err)
		}
		sources[name] = strings.ReplaceAll(string(data), "\r\n", "\n")
	}
	return sources
}

// funcBody extracts a method body by cutting at the first closing brace in
// column 0 — the bodies under test have no top-level braces inside.
// Line endings are normalised to LF so that CRLF checkouts (Windows with
// core.autocrlf=true) do not prevent the "\n}\n" search from matching.
func funcBody(t *testing.T, src, name string) string {
	t.Helper()
	src = strings.ReplaceAll(src, "\r\n", "\n")
	start := strings.Index(src, "func (m *model) "+name)
	if start < 0 {
		t.Fatalf("%s not found", name)
	}
	rel := strings.Index(src[start:], "\n}\n")
	if rel < 0 {
		t.Fatalf("%s: no closing brace", name)
	}
	return src[start : start+rel]
}

func TestKeyRegistry_SourceGuards(t *testing.T) {
	sources := nonTestSources(t)

	// extraSlashCommands must stay dead: the registry owns /todos now, and
	// the side table would quietly fork the command list again.
	for name, src := range sources {
		if strings.Contains(src, "extraSlashCommands") {
			t.Errorf("%s references extraSlashCommands — absorbed by keyRegistry, do not revive", name)
		}
		// The /todos special case must not come back in the dispatcher:
		// only the registry row itself may spell the command.
		if name != "keybind.go" && strings.Contains(src, `"/todos"`) {
			t.Errorf("%s hard-codes \"/todos\" — register it in keyRegistry instead", name)
		}
	}

	// The four global dispatchers must resolve keys through the registry,
	// not compare chords inline. An inline `key.Code == 'x'` here is a
	// binding that bypasses keyRegistry and never reaches the help dialog.
	chord := regexp.MustCompile(`key\.Code == '`)
	for _, fn := range []string{"handleInterruptKey", "handleToggleKey", "handleHistoryKey", "handleScrollKey"} {
		body := funcBody(t, sources["tui.go"], fn)
		if !strings.Contains(body, "hotkeyIDFor(key)") {
			t.Errorf("%s does not resolve keys through hotkeyIDFor", fn)
		}
		for _, hit := range chord.FindAllString(body, -1) {
			t.Errorf("%s compares chords inline (%s...) — add the binding to keyRegistry and switch on its ID", fn, hit)
		}
	}
}

func TestFuncBody_CRLF(t *testing.T) {
	// Regression: on Windows with core.autocrlf=true the source files arrive
	// with \r\n line endings. funcBody searches for "\n}\n" and must not fail
	// when the closing brace is preceded by \r.
	src := "package tui\r\n\r\nfunc (m *model) handleInterruptKey() tea.Cmd {\r\n\treturn func() tea.Msg {\r\n\t\treturn nil\r\n\t}\r\n}\r\n"
	body := funcBody(t, src, "handleInterruptKey")
	if body == "" {
		t.Fatal("funcBody returned empty on CRLF input")
	}
	if !strings.Contains(body, "return nil") {
		t.Errorf("funcBody = %q, want it to contain the method body", body)
	}
}

// -----------------------------------------------------------------------------
// Help dialog (searchModeHelp)
// -----------------------------------------------------------------------------

func TestHelpPopup_OpensFromSlashCommand(t *testing.T) {
	m := cplxModel(t)
	m.handleSlashCommand("/help")
	if m.searchPopup == nil {
		t.Fatal("/help did not open a popup")
	}
	if m.searchPopup.mode != searchModeHelp {
		t.Fatalf("/help opened mode %q, want %q", m.searchPopup.mode, searchModeHelp)
	}
	if len(m.searchPopup.entries) != len(keyRegistry) {
		t.Errorf("help popup shows %d rows, registry has %d", len(m.searchPopup.entries), len(keyRegistry))
	}

	// Every registry row is present: slash commands by command, hotkeys by
	// key spelling.
	seen := make(map[string]bool, len(m.searchPopup.entries))
	for _, it := range m.searchPopup.entries {
		seen[it.Text] = true
		if it.Description == "" || !strings.Contains(it.Description, " — ") {
			t.Errorf("help row %q lost its category/description: %q", it.Text, it.Description)
		}
	}
	for _, b := range keyRegistry {
		if !seen[b.Key] {
			t.Errorf("help popup is missing registry row %q (%s)", b.Key, b.ID)
		}
	}
}

func TestHelpPopup_FuzzyFilter(t *testing.T) {
	t.Run("slash rows match by name", func(t *testing.T) {
		m := cplxModel(t)
		m.handleSlashCommand("/help")
		sp := m.searchPopup
		sp.search = "todos"
		sp.filterSearch()
		if len(sp.filtered) == 0 {
			t.Fatal("query \"todos\" matched nothing")
		}
		if first := sp.filtered[0].Text; first != "/todos" {
			t.Errorf("query \"todos\": first row = %q, want /todos", first)
		}
	})

	t.Run("hotkey rows match by description", func(t *testing.T) {
		m := cplxModel(t)
		m.handleSlashCommand("/help")
		sp := m.searchPopup
		sp.search = "monitor"
		sp.filterSearch()
		found := false
		for _, it := range sp.filtered {
			if it.Text == "ctrl+t" {
				found = true
			}
		}
		if !found {
			t.Error("query \"monitor\" did not surface the ctrl+t monitor binding")
		}
	})
}

func TestHelpPopup_EnterRunsSlashRow(t *testing.T) {
	m := cplxModel(t)
	m.handleSlashCommand("/help")
	sp := m.searchPopup
	for i, it := range sp.filtered {
		if it.Text == "/session" {
			sp.selected = i
		}
	}
	m.acceptSearchPopupSelection()

	if m.searchPopup != nil {
		t.Error("running a slash row must close the help popup")
	}
	if len(m.chatModel.Messages) != 1 || !strings.Contains(m.chatModel.Messages[0].content, "Session:") {
		t.Errorf("Enter on /session did not run the command; messages = %+v", m.chatModel.Messages)
	}
}

// Hotkey rows are view-only: Enter consumes the key, changes nothing, and
// keeps the dialog open.
func TestHelpPopup_EnterOnHotkeyRowDoesNothing(t *testing.T) {
	m := cplxModel(t)
	m.handleSlashCommand("/help")
	sp := m.searchPopup
	for i, it := range sp.filtered {
		if it.Text == "ctrl+t" {
			sp.selected = i
		}
	}
	if cmd := m.acceptSearchPopupSelection(); cmd != nil {
		t.Error("hotkey row must not return a command")
	}
	if m.searchPopup == nil {
		t.Error("Enter on a hotkey row must not close the help popup")
	}
	if len(m.chatModel.Messages) != 0 {
		t.Errorf("Enter on a hotkey row ran something: %+v", m.chatModel.Messages)
	}
}

func TestHelpPopup_FooterAndHeader(t *testing.T) {
	if footer := searchPopupFooter(searchModeHelp); !strings.Contains(footer, "Enter run") {
		t.Errorf("help footer = %q, want it to advertise Enter run", footer)
	}
	m := cplxModel(t)
	m.handleSlashCommand("/help")
	st := m.searchPopupStyles(searchModeHelp, 40)
	if st.header != "Help" {
		t.Errorf("help header = %q, want Help", st.header)
	}
}
