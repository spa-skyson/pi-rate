package tui

import (
	"sync"

	tea "charm.land/bubbletea/v2"
)

// The central registry of everything the TUI can be told to do: one row per
// command or key binding, carrying the stable ID, the default key, the
// description shown in autocomplete and the help dialog, and the category
// that groups them there. Slash dispatch, the hotkey dispatchers, autocomplete
// and the /help dialog all read this one table — the description switches and
// hand-written help text they used to duplicate are gone.

// bindingKind splits the registry into the two dispatch families.
type bindingKind string

const (
	// kindSlash is a "/command" typed into the prompt.
	kindSlash bindingKind = "slash"
	// kindHotkey is a global chord handled by the key dispatchers.
	kindHotkey bindingKind = "hotkey"
)

// Help-dialog categories. The slash rows reuse the groups the old hand-written
// /help text had; every hotkey lands in catKeys.
const (
	catGeneral = "General"
	catSession = "Session"
	catGit     = "Git & Planning"
	catDisplay = "Display"
	catSystem  = "System"
	catSkills  = "Skills"
	catKeys    = "Keys"
)

// keyBinding is one row of the registry.
type keyBinding struct {
	// ID is the stable identity dispatchers switch on. Slash rows are
	// "cmd.<name>", hotkeys "key.<action>".
	ID string
	// Key is the default binding. Slash rows carry the command itself
	// ("/help"); hotkeys carry the chord in the spelling the help dialog
	// shows ("ctrl+t", "shift+tab", "pgup").
	Key string
	// Description is the autocomplete and help text.
	Description string
	// Category groups rows in the help dialog.
	Category string
	// Kind selects the dispatch family.
	Kind bindingKind
	// Hidden keeps a slash command out of slashCommands — autocomplete and
	// the command popup — while it still dispatches and still has a
	// description (the /skill-* subcommands).
	Hidden bool
	// Override is the user-configured replacement for Key. Phase 2 of the
	// registry (key bindings from config) writes it; it is always empty
	// today and currentKeyOf falls back to Key. Dispatchers must resolve
	// keys through hotkeyFor — never by comparing chords inline — so an
	// override starts working without touching them again.
	Override string
}

// keyRegistry is the single source of truth. Slash rows keep the historical
// flat order: autocomplete returns the first prefix match, so "/r" completing
// to /run and "/pr" to /pr-autofix is behavior, not presentation (pinned by
// TestCplxSlashCommands_DerivedOrder).
//
// The table is deliberately pure data — no handler method values. The /todos
// handler opens the search popup, which reads the derived command list, which
// reads this table: capturing that method value in this var's initializer is
// an initialization cycle. Handlers attach lazily (bindSlashHandlers) on
// first dispatch instead, which is what lets every command — /todos and the
// argument-less /model, /agent, /subagents popups included — live in the one
// table without special cases in the dispatcher.
var keyRegistry = []keyBinding{
	// --- slash commands ---
	{ID: "cmd.help", Key: "/help", Description: "Show help", Category: catGeneral, Kind: kindSlash},
	{ID: "cmd.clear", Key: "/clear", Description: "Clear conversation", Category: catGeneral, Kind: kindSlash},
	{ID: "cmd.copy", Key: "/copy", Description: "Copy conversation to clipboard", Category: catGeneral, Kind: kindSlash},
	{ID: "cmd.model", Key: "/model", Description: "Show or switch model", Category: catSession, Kind: kindSlash},
	{ID: "cmd.agent", Key: "/agent", Description: "Show or switch the session agent", Category: catSession, Kind: kindSlash},
	{ID: "cmd.session", Key: "/session", Description: "Show session info", Category: catSession, Kind: kindSlash},
	{ID: "cmd.context", Key: "/context", Description: "Show context usage", Category: catSession, Kind: kindSlash},
	{ID: "cmd.branch", Key: "/branch", Description: "Manage branches", Category: catGit, Kind: kindSlash},
	{ID: "cmd.compact", Key: "/compact", Description: "Compact context", Category: catSession, Kind: kindSlash},
	{ID: "cmd.subagents", Key: "/subagents", Description: "Monitor subagents", Category: catSession, Kind: kindSlash},
	{ID: "cmd.history", Key: "/history", Description: "Command history", Category: catSession, Kind: kindSlash},
	{ID: "cmd.login", Key: "/login", Description: "Configure API keys (codex, openai, anthropic, gemini)", Category: catSession, Kind: kindSlash},
	{ID: "cmd.commit", Key: "/commit", Description: "Create commit from staged changes", Category: catGit, Kind: kindSlash},
	{ID: "cmd.diff", Key: "/diff", Description: "Fullscreen diff viewer (working tree / last commit)", Category: catGit, Kind: kindSlash},
	{ID: "cmd.plan", Key: "/plan", Description: "Start PDD planning session", Category: catGit, Kind: kindSlash},
	{ID: "cmd.run", Key: "/run", Description: "Execute a spec with task agent (verifies subagent exit status before merging)", Category: catGit, Kind: kindSlash},
	// After /run and /pr-autofix: autocomplete returns the first prefix
	// match, so "/r" must keep completing to /run and "/pr" to /pr-autofix.
	{ID: "cmd.pr-autofix", Key: "/pr-autofix", Description: "Watch a GitHub PR's checks and fix them until it is green", Category: catGit, Kind: kindSlash},
	{ID: "cmd.retry", Key: "/retry", Description: "Re-send the prompt of a turn that failed", Category: catSession, Kind: kindSlash},
	{ID: "cmd.skills", Key: "/skills", Description: "List skills (create, load)", Category: catSkills, Kind: kindSlash},
	{ID: "cmd.skill-list", Key: "/skill-list", Description: "List all loaded skills", Category: catSkills, Kind: kindSlash, Hidden: true},
	{ID: "cmd.skill-load", Key: "/skill-load", Description: "Reload skills from disk", Category: catSkills, Kind: kindSlash, Hidden: true},
	{ID: "cmd.skill-create", Key: "/skill-create", Description: "Create a new skill", Category: catSkills, Kind: kindSlash, Hidden: true},
	{ID: "cmd.theme", Key: "/theme", Description: "Switch theme or list themes", Category: catDisplay, Kind: kindSlash},
	{ID: "cmd.ping", Key: "/ping", Description: "Test LLM connectivity", Category: catSystem, Kind: kindSlash},
	{ID: "cmd.model-price-refresh", Key: "/model-price-refresh", Description: "Refresh model prices from models.dev", Category: catSystem, Kind: kindSlash},
	{ID: "cmd.rtk", Key: "/rtk", Description: "Output compaction stats", Category: catSystem, Kind: kindSlash},
	{ID: "cmd.update", Key: "/update", Description: "Check for updates, upgrade pirate (or: /update check)", Category: catSystem, Kind: kindSlash},
	{ID: "cmd.mcp", Key: "/mcp", Description: "List MCP servers and tool status", Category: catSystem, Kind: kindSlash},
	{ID: "cmd.exit", Key: "/exit", Description: "Exit", Category: catSystem, Kind: kindSlash},
	{ID: "cmd.quit", Key: "/quit", Description: "Exit", Category: catSystem, Kind: kindSlash},
	// Absorbed from the extra-commands side table: /todos opens the plan
	// popup, so its handler could not live in an eagerly-initialized table.
	{ID: "cmd.todos", Key: "/todos", Description: "Show todo/plan list", Category: catGit, Kind: kindSlash},

	// --- hotkeys (the chord dispatchers resolve through hotkeyFor) ---
	{ID: "key.retry", Key: "ctrl+r", Description: "Re-send the prompt of a turn that failed", Category: catKeys, Kind: kindHotkey},
	{ID: "key.compact-tools", Key: "ctrl+o", Description: "Toggle compact tool output", Category: catKeys, Kind: kindHotkey},
	{ID: "key.background", Key: "ctrl+b", Description: "Send the running task to background / attach it back", Category: catKeys, Kind: kindHotkey},
	{ID: "key.branch", Key: "ctrl+g", Description: "Toggle the branch popup", Category: catKeys, Kind: kindHotkey},
	{ID: "key.agent-cycle", Key: "shift+tab", Description: "Cycle the session agent", Category: catKeys, Kind: kindHotkey},
	{ID: "key.history", Key: "ctrl+h", Description: "Open history search", Category: catKeys, Kind: kindHotkey},
	{ID: "key.monitor", Key: "ctrl+t", Description: "Toggle the subagent monitor", Category: catKeys, Kind: kindHotkey},
	{ID: "key.cancel", Key: "esc", Description: "Dismiss an overlay, or cancel the running turn (press twice)", Category: catKeys, Kind: kindHotkey},
	{ID: "key.quit", Key: "ctrl+c", Description: "Cancel the running turn; press twice to quit", Category: catKeys, Kind: kindHotkey},
	{ID: "key.suspend", Key: "ctrl+z", Description: "Suspend the process", Category: catKeys, Kind: kindHotkey},
	{ID: "key.history-window", Key: "up", Description: "Open the prompt history window, or scroll the chat up", Category: catKeys, Kind: kindHotkey},
	{ID: "key.monitor-down", Key: "down", Description: "Scroll the chat down; at the bottom opens the subagent monitor", Category: catKeys, Kind: kindHotkey},
	{ID: "key.chat-up", Key: "pgup", Description: "Scroll the chat up by a page", Category: catKeys, Kind: kindHotkey},
	{ID: "key.chat-down", Key: "pgdn", Description: "Scroll the chat down by a page", Category: catKeys, Kind: kindHotkey},
	// Popup-local: only answers inside the subagent monitor.
	{ID: "key.subagents-steer", Key: "s", Description: "Steer the selected running subagent (in the monitor)", Category: catKeys, Kind: kindHotkey},
}

// registryByCommand indexes the slash rows by command for dispatch and
// description lookups. Pure data over pure data, so the initializer is safe.
var registryByCommand = func() map[string]keyBinding {
	byCmd := make(map[string]keyBinding, len(keyRegistry))
	for _, b := range keyRegistry {
		if b.Kind == kindSlash {
			byCmd[b.Key] = b
		}
	}
	return byCmd
}()

// slashCommands is the visible slash command list for autocomplete: every
// non-hidden slash row, in registry order. The skill subcommands
// (/skill-list, /skill-load, /skill-create) stay out of it to keep the list
// concise; they remain dispatchable and described.
var slashCommands = func() []string {
	names := make([]string, 0, len(keyRegistry))
	for _, b := range keyRegistry {
		if b.Kind == kindSlash && !b.Hidden {
			names = append(names, b.Key)
		}
	}
	return names
}()

// slashCommandDesc returns the description for a slash command, or "" for
// anything that is not a registered command.
func slashCommandDesc(cmd string) string {
	return registryByCommand[cmd].Description
}

// slashHandler is what typing a slash command does.
type slashHandler = func(*model, []string) (tea.Model, tea.Cmd)

// slashCmdArgs adapts a handler that mutates the model and returns nothing.
func slashCmdArgs(f func(*model, []string)) slashHandler {
	return func(m *model, args []string) (tea.Model, tea.Cmd) {
		f(m, args)
		return m, nil
	}
}

// slashCmdBare adapts a handler that takes no args and returns a command.
func slashCmdBare(f func(*model) (tea.Model, tea.Cmd)) slashHandler {
	return func(m *model, _ []string) (tea.Model, tea.Cmd) {
		return f(m)
	}
}

// slashCmdVoid adapts a handler that takes no args and returns nothing.
func slashCmdVoid(f func(*model)) slashHandler {
	return func(m *model, _ []string) (tea.Model, tea.Cmd) {
		f(m)
		return m, nil
	}
}

var (
	slashHandlersOnce sync.Once
	slashHandlers     map[string]slashHandler
)

// slashHandlerFor returns the handler registered for a command ID, building
// the table on first use. Lazy because the handler method values reach
// newSearchPopup → allSearchCandidates → slashCommands → keyRegistry: a
// package-level initializer capturing them would be an initialization cycle —
// the very reason /todos used to be special-cased in handleSlashCommand. At
// call time there is no cycle, so every command dispatches through the one
// table.
func slashHandlerFor(id string) slashHandler {
	slashHandlersOnce.Do(bindSlashHandlers)
	return slashHandlers[id]
}

// bindSlashHandlers attaches handlers to the registry's slash rows. The three
// pickers wrap their argument commands: with no argument they open their
// popup instead of printing a listing into the chat (previously special-cased
// in handleSlashCommand for the same cycle).
func bindSlashHandlers() {
	slashHandlers = map[string]slashHandler{
		// /help is the help dialog itself: the registry rendered as a popup.
		"cmd.help": func(m *model, _ []string) (tea.Model, tea.Cmd) {
			m.newSearchPopup(searchModeHelp)
			return m, nil
		},
		"cmd.clear": slashCmdVoid((*model).clearConversation),
		"cmd.copy":  slashCmdBare((*model).handleCopyCommand),
		// /model, /agent and /subagents with no argument open their picker
		// popups instead of printing a listing into the chat — previously
		// special-cased in handleSlashCommand for the same init cycle.
		"cmd.model": func(m *model, args []string) (tea.Model, tea.Cmd) {
			if len(args) == 0 {
				return m, m.openModelsPopup()
			}
			return m.handleModelCommand(args)
		},
		"cmd.agent": func(m *model, args []string) (tea.Model, tea.Cmd) {
			if len(args) == 0 {
				m.newSearchPopup(searchModeAgents)
				return m, nil
			}
			return m.handleAgentCommand(args)
		},
		"cmd.subagents": func(m *model, args []string) (tea.Model, tea.Cmd) {
			if len(args) == 0 {
				m.newSearchPopup(searchModeSubagents)
				return m, nil
			}
			m.handleAgentsCommand()
			return m, nil
		},
		"cmd.session":             slashCmdVoid((*model).showSessionMessage),
		"cmd.context":             slashCmdVoid((*model).showContextMessage),
		"cmd.branch":              slashCmdArgs((*model).handleBranchCommand),
		"cmd.compact":             slashCmdVoid((*model).handleCompactCommand),
		"cmd.history":             slashCmdArgs((*model).handleHistoryCommand),
		"cmd.login":               (*model).handleLoginCommand,
		"cmd.commit":              slashCmdBare((*model).handleCommitCommand),
		"cmd.diff":                slashCmdBare((*model).openDiffViewer),
		"cmd.plan":                (*model).handlePlanCommand,
		"cmd.run":                 (*model).handleRunCommand,
		"cmd.pr-autofix":          (*model).handlePRAutofixCommand,
		"cmd.retry":               slashCmdBare((*model).handleRetry),
		"cmd.skills":              (*model).handleSkillsCommand,
		"cmd.skill-list":          slashCmdBare((*model).handleSkillListCommand),
		"cmd.skill-load":          slashCmdBare((*model).handleSkillLoadCommand),
		"cmd.skill-create":        (*model).handleSkillCreateCommand,
		"cmd.theme":               (*model).handleThemeCommand,
		"cmd.ping":                (*model).handlePingCommand,
		"cmd.model-price-refresh": (*model).handleModelPriceRefreshCommand,
		"cmd.rtk":                 slashCmdArgs((*model).handleRTKCommand),
		"cmd.update":              (*model).handleUpdateCommand,
		"cmd.mcp":                 slashCmdVoid((*model).handleMCPCommand),
		"cmd.exit":                slashCmdBare((*model).handleQuitCommand),
		"cmd.quit":                slashCmdBare((*model).handleQuitCommand),
		"cmd.todos": func(m *model, _ []string) (tea.Model, tea.Cmd) {
			// Takes no arguments: trailing whitespace or stray text is
			// ignored, the popup opens.
			m.newSearchPopup(searchModeTodos)
			return m, nil
		},
	}
}

// keyLabel spells a pressed key the way the registry does. Only exact,
// single-modifier chords resolve — matching what the dispatchers matched
// before the registry, so ctrl+shift+r stays unmatched rather than reading
// as ctrl+r.
func keyLabel(key tea.Key) string {
	switch {
	case key.Mod == tea.ModShift && key.Code == tea.KeyTab:
		return "shift+tab"
	case key.Mod == tea.ModCtrl && key.Code >= 'a' && key.Code <= 'z':
		return "ctrl+" + string(key.Code)
	case key.Mod == 0 && key.Code >= 'a' && key.Code <= 'z':
		return string(key.Code)
	case key.Mod == 0 && len(key.Text) == 1 && key.Text[0] >= 'a' && key.Text[0] <= 'z':
		// Tests (and some terminals) carry the letter in Text only.
		return key.Text
	}
	switch key.Code {
	case tea.KeyEsc:
		return "esc"
	case tea.KeyUp:
		return "up"
	case tea.KeyDown:
		return "down"
	case tea.KeyPgUp:
		return "pgup"
	case tea.KeyPgDown:
		return "pgdn"
	}
	return ""
}

// currentKeyOf returns the binding's effective key: the phase-2 override when
// one is set, the default otherwise.
func currentKeyOf(b keyBinding) string {
	if b.Override != "" {
		return b.Override
	}
	return b.Key
}

// hotkeyFor resolves the pressed key to its hotkey binding. The scan is over
// a few dozen rows per keypress and honors the override, so phase 2 needs no
// dispatcher changes.
func hotkeyFor(key tea.Key) (keyBinding, bool) {
	pressed := keyLabel(key)
	if pressed == "" {
		return keyBinding{}, false
	}
	for _, b := range keyRegistry {
		if b.Kind == kindHotkey && currentKeyOf(b) == pressed {
			return b, true
		}
	}
	return keyBinding{}, false
}

// hotkeyIDFor resolves the pressed key to its registry ID, or "" when no
// hotkey binding claims it.
func hotkeyIDFor(key tea.Key) string {
	if b, ok := hotkeyFor(key); ok {
		return b.ID
	}
	return ""
}

// helpSearchItems renders the registry as help-popup rows: every binding,
// slash commands then hotkeys, in registry order. The category leads the
// description, so the dialog groups visually and fuzzy search matches
// category words for free.
func helpSearchItems() []SearchItem {
	items := make([]SearchItem, 0, len(keyRegistry))
	for _, b := range keyRegistry {
		items = append(items, SearchItem{
			Text:        b.Key,
			Description: b.Category + " — " + b.Description,
		})
	}
	return items
}
