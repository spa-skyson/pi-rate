// Package tui implements the interactive terminal UI using Bubble Tea v2.
package tui

import (
	"context"
	"fmt"
	"image/color"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"charm.land/lipgloss/v2"

	tea "charm.land/bubbletea/v2"

	"github.com/spa-skyson/pi-rate/internal/auth"
	"github.com/spa-skyson/pi-rate/internal/config"
	"github.com/spa-skyson/pi-rate/internal/extension"
	"github.com/spa-skyson/pi-rate/internal/palace"
	"github.com/spa-skyson/pi-rate/internal/permission"
	"github.com/spa-skyson/pi-rate/internal/sop"
	"github.com/spa-skyson/pi-rate/internal/subagent"
	"github.com/spa-skyson/pi-rate/internal/tools"
)

// model is the Bubble Tea model for the interactive TUI.
type model struct {
	cfg    Config
	ctx    context.Context
	cancel context.CancelFunc

	// UI state.
	width  int
	height int

	// Input sub-model.
	inputModel InputModel

	// Chat sub-model (messages, scroll, rendering).
	chatModel ChatModel

	// Status bar sub-model.
	statusModel StatusModel

	// Theme manager.
	themeManager *ThemeManager

	// palette is the resolved color palette for the active theme, recomputed
	// each frame in View so a /theme switch takes effect immediately.
	palette Palette

	// bgDetected records that the terminal background question is settled —
	// either the reply to RequestBackgroundColor arrived, or the user picked a
	// theme explicitly. Detection never overrides an explicit choice.
	bgDetected bool

	// Active /plan worktree. Planning artifacts remain isolated until the
	// planner completes, then the branch is merged and the backup ref retained.
	planWorktreeAgentID string
	planWorktreePath    string
	sopGraphs           map[string]*sop.Compiled // compiled SOPs for the sidebar diagram, per name
	planPhases          []PlanPhase              // cached phase checklist; recomputed when planPhasesStale
	planPhasesStale     bool                     // an agent event may have changed the spec artifacts
	planBackupBranch    string
	planTaskName        string
	planWorktree        *subagent.WorktreeManager

	// planFixCycle counts automatic validation-repair attempts for the current
	// plan. validatePlanArtifacts fails the same way an unmet Done Criterion
	// does, so it gets the same treatment: feed the findings back to the planner
	// automatically rather than waiting for a human to notice and type "continue".
	// Bounded so a plan that cannot satisfy the contract stops and reports
	// instead of looping forever.
	planFixCycle int

	// Agent state.
	running        bool
	mode           string             // "chat" or "plan" — shown in status bar
	agentCh        chan agentMsg      // channel for receiving agent events
	agentCancel    context.CancelFunc // cancels the active agent response without quitting the TUI
	pendingPrompts []queuedPrompt     // prompts submitted while a response is active

	// bg is the detached run (Ctrl+B): non-nil from detach until attach.
	// While it is alive it holds the one live main loop — new prompts queue,
	// agent and model switches refuse — and its events buffer instead of
	// rendering. See background.go.
	bg *backgroundRun

	// turnStarted is when the current turn began (set by beginTurn next to
	// running). The done-attention signal in handleAgentDone measures the
	// turn against attentionDoneAfter from it.
	turnStarted time.Time

	// focused mirrors the terminal window focus (tea.FocusMsg / tea.BlurMsg,
	// reported because the View sets ReportFocus while attention is on).
	// Starts false — "assume away": a terminal that does not answer focus
	// reporting never sends either message, and for such a terminal the bell
	// must stay usable, not go silently dead. A focus-reporting terminal
	// sends FocusMsg almost immediately, so the unfocused window is brief.
	focused bool

	// modelFallbacks is the fallback chain of the current session, latched on
	// the first terminal provider error and drained one entry per switch:
	// frontmatter fallback-models of the active agent, else the role's
	// fallbackModels (its own, else "default"). Nil when no terminal error has
	// happened or nothing is configured. In-memory, session-only — the same
	// scope as agentModelOverrides.
	modelFallbacks []string

	// modelFallbackUsed counts how many entries of modelFallbacks have been
	// consumed; index of the next entry is modelFallbackUsed. Reset on a
	// successful turn, a manual /model, and a rebuild on the same chain.
	modelFallbackUsed int

	// activeAgent names the primary agent the main session runs on
	// ("" = the built-in default). Set by /agent, Shift+Tab, and startup
	// defaultAgent; shown in the sidebar next to the model.
	activeAgent string

	// agentModelOverrides holds the session model per primary agent
	// (agent name → model), recorded when /model is typed while that agent
	// is active. In-memory only — never persisted — and re-applied by
	// applyAgent on every switch back to the agent. Nil until the first
	// override.
	agentModelOverrides map[string]string

	// bgRunning counts how many background subagents are currently running.
	// Incremented on spawn+bg, decremented on done+bg; rendered in the sidebar
	// next to the agent name.
	bgRunning int

	// The prompt of the most recent turn, kept so a turn that failed can be
	// re-sent without retyping it. lastPromptFailed marks it as worth offering:
	// a successful turn clears the flag, so /retry never replays a prompt whose
	// answer is already on screen.
	lastPrompt       string
	lastMentions     []string
	lastPromptFailed bool

	// Agent face renderer with mood expressions.
	face *FaceRenderer

	// hookQueue serializes lifecycle-hook execution off the Update goroutine.
	// Created lazily by enqueueHook (see agent_loop.go), which is only ever
	// reached from Update, so the lazy init needs no lock.
	hookQueue chan func()

	// Sea scene animation state for the top bar while a turn runs.
	sea seaState

	// Mouse text selection. lastFrame is the frame the selection's coordinates
	// refer to — pi selects text itself (see selection.go), so it needs the
	// pixels it drew in order to know what is under the cursor. frameRows is its
	// height, which is what converts a mouse row into a frame row.
	sel       selection
	lastFrame string
	frameRows int
	// sessionTitle is the title derived from the user's most recent prompt. It
	// is surfaced to the terminal via View().WindowTitle so Bubble Tea's
	// renderer emits the escape sequence in-band with the frame it draws.
	sessionTitle string
	// titleSpin is the phase of the animated title prefix ("π Pi-rate ⠋",
	// "π Pi-rate ⠙", …) shown while a turn runs. It is advanced in Update from the
	// sea tick, next to ToolDisplay.BlinkOn, so View stays a pure
	// function of model state.
	titleSpin int
	// topSectionRows is the number of rows in the top section (messages + sidebar)
	// before the full-width status bar. The rail and sidebar only cover these rows.
	topSectionRowsVal int
	// The selectable rows: the message viewport, half-open. Everything else in
	// the panel is chrome — the sea scene, the rules, the status bar, the
	// prompt — and selecting it yields nothing anyone wants on their clipboard.
	msgTop, msgBottom int

	// Transient status-bar notice. flashSeq invalidates the timer of a flash
	// that has already been superseded, so an old expiry cannot clear a newer
	// message.
	flash    string
	flashSeq int

	// Deferred initialization state.
	loading      bool
	loadingItems map[string]bool // item name -> done?
	loadingTotal int             // planned init item count when known
	initCh       <-chan InitEvent
	loadingDots  int // animation dots (0-3): ., .., ..., ....

	// Git diff stats (refreshed after tool completions).
	diffAdded   int
	diffRemoved int

	// Commit flow state.
	commit *commitState

	// Update flow state (/update command): the confirm prompt or the running
	// installer, or nil.
	update *updateState

	// Tool-approval dialog state: the pending permission request, or nil.
	// Set by the ApprovalCh listener, answered and cleared by the key handler;
	// see approval.go.
	approval *permission.ApprovalRequest

	// Question-dialog state: the pending question request plus its cursor
	// and free-text editor, or nil. Set by the QuestionCh listener, answered
	// and cleared by the key handler; see question.go.
	question *questionDialog

	// Login flow state.
	login *loginState

	// Skill-create pending overwrite confirmation.
	pendingSkillCreate *pendingSkillCreate

	// Run flow state (/run command).
	run *runState

	// prAutofix is the in-flight /pr-autofix run, or nil. Unlike run, it holds
	// no copy of the workflow: the SOP is the workflow and the engine walks it.
	prAutofix *prAutofixState

	// Branch popup state (shown on status bar click).
	branchPopup *branchPopupState

	// Unified search popup for slash commands and history.
	searchPopup *searchPopupState

	// Recent slash commands for the popup's Suggested section: a
	// most-recent-first list capped at suggestedCommandsMax plus a usage
	// count. In-memory by design — the popup reflects this session's own
	// usage; persisting it across restarts is a future need, not this one.
	cmdRecent []string
	cmdCounts map[string]int

	// subagentViewer is the fullscreen stream viewer opened from the
	// subagent monitor (Enter on an agent). It renders the selected card's
	// full agentEvents; nil when closed. See subagent_monitor.go.
	subagentViewer *subagentViewerState

	// pasteViewer is the fullscreen read-only view of one collapsed paste's
	// full text, opened from the prompt with Alt+V; nil when closed.
	// See paste.go.
	pasteViewer *pasteViewerState

	// diffViewer is the fullscreen diff viewer (/diff) over the working
	// tree or the last commit; nil when closed. See diffview.go.
	diffViewer *diffViewerState

	// steerInput is the monitor's mini-input for sending a follow-up message
	// to a running subagent (`s` on a running row); nil when closed. See
	// subagent_monitor.go.
	steerInput *subagentSteerState

	// overlays is the modal stack: the top entry sees every key first, a
	// handled key never reaches the layers below, and a declined one falls
	// through unless the entry swallows input. The fields above hold each
	// overlay's STATE; the stack holds the DISPATCH. openOverlay/popOverlay
	// keep the two in step at every open and close site, and syncOverlays
	// repairs drift before each dispatch. See overlay.go.
	overlays []overlayEntry

	// Legacy selection index for slash commands (used in tests).
	slashCommandSelected int

	// Quit.
	quitting bool
	initErr  error // fatal init error → propagated from Run()

	// Ctrl+C handling: show warning on first press, quit on second.
	ctrlCCount int

	// Esc handling: warn on the first press, cancel a running turn on the
	// second within 2 seconds. Reset by the resetEscCancelCountMsg timer and
	// at turn boundaries.
	escCancelCount int

	// escResetSeq invalidates the timer of a previous warn window, the same
	// trick flashSeq plays for flashes: a timer fired for an old window must
	// not clear the counter of a freshly armed one.
	escResetSeq int

	// Memory palace status for sidebar (nil if no palace DB).
	memoryStatus *palace.PalaceStatus

	// resizeAt records when the last WindowSizeMsg arrived. Key/paste input
	// is suppressed briefly after resize to let terminal response sequences
	// (OSC color replies, DECRPM, CPR) drain without leaking into the input.
	resizeAt time.Time

	// todoState is the latest todo/plan state from todo_write, or nil when
	// no plan has been set or todo tracking is disabled.
	todoState *tools.TodoState
}

// branchPopupState manages the git branch list popup.
type branchPopupState struct {
	branches  []string // list of git branches
	selected  int      // currently selected index
	active    string   // the currently active branch
	height    int      // popup height (number of visible items)
	scrollOff int      // scroll offset when more branches than height
}

// moveUp selects the previous branch, scrolling the window when the selection
// would leave the top of it. It stops at the first branch.
func (p *branchPopupState) moveUp() {
	if p.selected <= 0 {
		return
	}
	p.selected--
	if p.selected < p.scrollOff {
		p.scrollOff--
	}
}

// moveDown selects the next branch, scrolling the window when the selection
// would leave the bottom of it. It stops at the last branch.
func (p *branchPopupState) moveDown() {
	if p.selected >= len(p.branches)-1 {
		return
	}
	p.selected++
	if p.selected >= p.scrollOff+p.height {
		p.scrollOff++
	}
}

// newBranchPopup creates a new branch popup with the list of git branches.
func (m *model) newBranchPopup() {
	branches := listGitBranches(m.cfg.WorkDir)
	if len(branches) == 0 {
		return
	}

	active := m.statusModel.GitBranch
	selected := 0
	for i, b := range branches {
		if b == active {
			selected = i
			break
		}
	}

	popupHeight := len(branches)
	if popupHeight > 8 {
		popupHeight = 8
	}

	m.branchPopup = &branchPopupState{
		branches:  branches,
		selected:  selected,
		active:    active,
		height:    popupHeight,
		scrollOff: 0,
	}
	m.openOverlay(overlayBranchPopup)
}

// listGitBranches returns a list of all local git branches, with the active one first.
func listGitBranches(workDir string) []string {
	cmd := exec.Command("git", "branch")
	if workDir != "" {
		cmd.Dir = workDir
	}
	out, err := cmd.Output()
	if err != nil {
		return nil
	}

	var branches []string
	active := ""
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		line = strings.TrimSpace(line)
		if len(line) == 0 {
			continue
		}
		// Active branch starts with '*'
		if strings.HasPrefix(line, "* ") {
			active = strings.TrimPrefix(line, "* ")
		} else {
			branches = append(branches, strings.TrimSpace(line))
		}
	}

	// Put active branch first
	if active != "" {
		result := []string{active}
		result = append(result, branches...)
		return result
	}
	return branches
}

// searchPopupState is a unified search window for slash commands and history.
// The mode determines what items are shown and how they are selected.
type searchPopupState struct {
	mode       searchMode   // "commands" or "history"
	entries    []SearchItem // all items (commands or history entries)
	filtered   []SearchItem // filtered by search query
	suggested  []SearchItem // Suggested candidates (recent commands, the active model/agent/…); matched against entries while the filter is empty
	suggestedN int          // how many of the leading filtered rows are the pinned Suggested section (0 when the filter is non-empty)
	selected   int          // currently selected index in filtered list
	search     string       // current search query
	height     int          // popup height (number of visible items)
	scrollOff  int          // scroll offset when more entries than height
}

// searchMode determines what the popup displays.
type searchMode string

const (
	searchModeCommands searchMode = "commands"
	searchModeHistory  searchMode = "history"
	searchModeModels   searchMode = "models"
	searchModeAgents   searchMode = "agents"
	// searchModeSubagents is the /subagents monitor. It lists the session's
	// subagents (orchestrator statuses stitched to transcript cards) and
	// opens the fullscreen stream viewer on Enter. Deliberately not
	// searchModeAgents — that popup selects the session's primary agent.
	searchModeSubagents searchMode = "subagents"
	// searchModeTodos shows the current todo/plan list. View-only: Enter
	// does nothing; Esc closes it.
	searchModeTodos searchMode = "todos"
	// searchModeFiles is the @file-mention popup. No command opens it: the
	// trigger is the prompt text itself — an @ where a mention can start —
	// and the filter is the text typed after it. The keys it owns (arrows,
	// Enter, Esc) live in handleFilesPopupKey; every other key falls through
	// to the prompt so typing keeps editing the mention.
	searchModeFiles searchMode = "files"
	// searchModeHelp is the /help dialog: every keyRegistry row — slash
	// commands then hotkeys — with the category leading each description.
	// Enter on a slash row runs the command through the normal dispatch;
	// hotkey rows are view-only.
	searchModeHelp searchMode = "help"
)

// SearchItem represents an item in the search popup (command or history entry).
type SearchItem struct {
	Text        string // the command or history text
	Description string // for commands: the description
	// ID, when set, is the action target behind the row rather than text to
	// insert: the subagent monitor carries the agentID so Enter can open that
	// agent's stream viewer. Empty for every other mode.
	ID string
}

// searchPopupChrome is the number of rows the search popup reserves for chrome
// — two border rows (top, bottom), a header row, a search-prompt row, and a
// footer key-hint row.  height = items.  The item list fills the remaining
// space, so the popup's total height is item count + searchPopupChrome.  The
// render loop and the Up/Down scroll math both treat height as the visible
// item count, so it has to reflect what actually fits, not the total number
// of items — otherwise scrolling never fires and the selected row renders
// off-screen.
const searchPopupChrome = 5

// searchPopupMaxItems caps the popup so a huge command list does not eat the
// whole message area.  25 mirrors the previous cap and is enough to scan at a
// glance.
const searchPopupMaxItems = 25

// searchPopupListHeight returns how many item rows the popup should expose.
// availableRows is the number of rows the message viewport can spare for the
// popup; the popup uses that many minus its own chrome.  The result is
// clamped to [1, max] when there are items so the popup is always usable,
// even on terminals too small for the chrome plus a margin.  An empty list
// returns 0 — the "no matches" path renders the chrome only.
func searchPopupListHeight(itemCount, availableRows int) int {
	if itemCount <= 0 {
		return 0
	}
	if availableRows <= searchPopupChrome {
		// Not enough room for the chrome plus any item; show one anyway
		// and let the overlay clip the bottom.  Picking something is more
		// useful than seeing an empty list.
		return 1
	}
	height := itemCount
	if max := availableRows - searchPopupChrome; height > max {
		height = max
	}
	if height > searchPopupMaxItems {
		height = searchPopupMaxItems
	}
	return height
}

// newSearchPopup creates a unified search popup with the given mode.
func (m *model) newSearchPopup(mode searchMode) {
	var items []SearchItem
	var search string // files mode opens pre-filtered by the typed @prefix

	switch mode {
	case searchModeCommands:
		// Get all slash command candidates.
		allCandidates := m.allSearchCandidates()
		// Filter by current input text if it starts with /.
		inputText := m.inputModel.Text
		filter := ""
		showAll := inputText == "/" // Show all commands when just "/" is typed
		if strings.HasPrefix(inputText, "/") && !showAll {
			filter = strings.ToLower(inputText)
		}
		for _, c := range allCandidates {
			if filter == "" || strings.HasPrefix(strings.ToLower(c.Text), filter) {
				items = append(items, SearchItem{Text: c.Text, Description: c.Description})
			}
		}

	case searchModeHistory:
		entries := m.inputModel.History
		if len(entries) == 0 {
			return
		}
		// Show oldest first (last item in entries at top).
		items = make([]SearchItem, len(entries))
		for i, e := range entries {
			items[len(entries)-1-i] = SearchItem{Text: e.Text}
		}

	case searchModeModels:
		// Copy before marking: cfg.ModelCandidates is shared with the cli
		// that built it, and an in-place append here would compound into
		// "— active — active" on the next open.
		items = make([]SearchItem, len(m.cfg.ModelCandidates))
		copy(items, m.cfg.ModelCandidates)
		markActiveRole(items, m.cfg.ActiveRole)
		// Nothing declared and no background source: the popup could never
		// show anything, so do not open it.
		if len(items) == 0 && m.cfg.ModelCandidatesRefresh == nil {
			return
		}

	case searchModeAgents:
		items = m.agentSearchItems()

	case searchModeSubagents:
		items = m.subagentSearchItems()
		// The monitor always opens, even with nothing to show: the user
		// asked for it explicitly, and the empty state says why it is empty
		// better than a silent refusal would.

	case searchModeTodos:
		if m.todoState == nil || len(m.todoState.Items) == 0 {
			items = []SearchItem{{Text: "No plan set. Use todo_write to create one."}}
		} else {
			items = make([]SearchItem, len(m.todoState.Items))
			for i, it := range m.todoState.Items {
				items[i] = SearchItem{Text: todoPrefix(it.Status) + " " + it.Content}
			}
		}

	case searchModeFiles:
		workDir := m.inputModel.WorkDir
		if workDir == "" {
			workDir = m.cfg.WorkDir
		}
		_, search = m.mentionAtCursor()
		// matchingFiles already narrowed to prefix and fuzzy hits; the
		// popup's filterSearch then ranks them. Paths are the rows; the
		// per-file "file" description adds nothing here.
		for _, c := range matchingFiles(search, workDir, mentionPopupFileLimit) {
			items = append(items, SearchItem{Text: c.Text})
		}

	case searchModeHelp:
		items = helpSearchItems()

	}

	availableRows := m.messageViewportHeight()
	// The overlay leaves a 1-row margin between the popup and the bottom of
	// the viewport when the popup fits.  Reserve that row too so the chrome
	// does not push the last item into that margin.
	if availableRows > 0 {
		availableRows--
	}
	popupHeight := searchPopupListHeight(len(items), availableRows)

	// A fresh popup never inherits a steer input from a previous monitor
	// session — an orphaned mini-input would own keys with nothing on screen.
	m.popOverlay(overlaySteerInput)

	m.searchPopup = &searchPopupState{
		mode:      mode,
		entries:   items,
		suggested: m.suggestedItems(mode),
		filtered:  items,
		selected:  0,
		search:    search,
		height:    popupHeight,
		scrollOff: 0,
	}
	m.openOverlay(overlaySearchPopup)
	// Pin the Suggested rows for the empty query the popup opens with.
	m.searchPopup.filterSearch()
}

// suggestedCommandsMax caps the Suggested section of the commands popup.
const suggestedCommandsMax = 5

// recordCommandUse notes a picked command for the popup's Suggested section:
// a most-recent-first list capped at suggestedCommandsMax plus a usage
// count. In-memory only — see the cmdRecent field comment.
func (m *model) recordCommandUse(cmd string) {
	if cmd == "" {
		return
	}
	if m.cmdCounts == nil {
		m.cmdCounts = make(map[string]int)
	}
	m.cmdCounts[cmd]++
	for i, c := range m.cmdRecent {
		if c == cmd {
			m.cmdRecent = append(m.cmdRecent[:i], m.cmdRecent[i+1:]...)
			break
		}
	}
	m.cmdRecent = append([]string{cmd}, m.cmdRecent...)
	if len(m.cmdRecent) > suggestedCommandsMax {
		m.cmdRecent = m.cmdRecent[:suggestedCommandsMax]
	}
}

// suggestedItems returns the rows a popup pins above the list while the
// filter is empty: recent commands, or the active model/agent/subagent/todo.
// Rows are matched against the popup's entries at filter time, so stale
// suggestions drop out silently (see pinnedSuggested).
func (m *model) suggestedItems(mode searchMode) []SearchItem {
	switch mode {
	case searchModeCommands:
		items := make([]SearchItem, 0, len(m.cmdRecent))
		for _, cmd := range m.cmdRecent {
			items = append(items, SearchItem{Text: cmd})
		}
		return items
	case searchModeModels:
		role := m.cfg.ActiveRole
		if role == "" {
			role = "default"
		}
		return []SearchItem{{Text: role}}
	case searchModeAgents:
		role := m.activeAgent
		if role == "" {
			role = "default"
		}
		return []SearchItem{{Text: role}}
	case searchModeSubagents:
		// The running subagent is the "current" one; a monitor with nothing
		// running gets no Suggested row at all.
		for _, r := range m.subagentRows() {
			if r.status != "running" {
				continue
			}
			title := r.title
			if title == "" {
				title = r.typ
			}
			return []SearchItem{{ID: r.agentID, Text: title}}
		}
		return nil
	case searchModeTodos:
		// The in-progress todo is the current one.
		if m.todoState == nil {
			return nil
		}
		for _, it := range m.todoState.Items {
			if it.Status == "in_progress" {
				return []SearchItem{{Text: todoPrefix(it.Status) + " " + it.Content}}
			}
		}
		return nil
	}
	return nil
}

// todoPrefix is the checkbox glyph a todo status renders as in the popup.
func todoPrefix(status string) string {
	switch status {
	case "in_progress":
		return "[~]"
	case "completed":
		return "[x]"
	default:
		return "[ ]"
	}
}

// refreshSearchPopupHeight recomputes the popup's visible item count after
// the terminal has been resized.  The popup keeps its selection; height, and
// scrollOff are reconciled so the selection stays visible and the rendered
// window never overshoots len(filtered) — otherwise renderSearchPopup would
// index past the end and panic on a tall→short→tall resize cycle.
func (m *model) refreshSearchPopupHeight() {
	if m.searchPopup == nil {
		return
	}
	availableRows := m.messageViewportHeight()
	if availableRows > 0 {
		availableRows--
	}
	m.searchPopup.height = searchPopupListHeight(len(m.searchPopup.filtered), availableRows)
	n := len(m.searchPopup.filtered)
	if n == 0 {
		m.searchPopup.selected = 0
		m.searchPopup.scrollOff = 0
		return
	}
	if m.searchPopup.selected >= n {
		m.searchPopup.selected = n - 1
	}
	// Bring scrollOff in sync with the new height. A previous resize could
	// have left it pointing past the new bottom of the window (short→tall
	// means more items visible, so a stale scrollOff overshoots), or past the
	// end of the list entirely (tall→short can shrink the visible window
	// below the old scrollOff). The previous value is not safe to keep.
	if m.searchPopup.selected < m.searchPopup.scrollOff {
		m.searchPopup.scrollOff = m.searchPopup.selected
	} else if m.searchPopup.selected >= m.searchPopup.scrollOff+m.searchPopup.height {
		m.searchPopup.scrollOff = m.searchPopup.selected - m.searchPopup.height + 1
	}
	// Defense in depth: never overshoot the end of filtered, so the
	// i-based loop in renderSearchPopup cannot read past the slice. The
	// in-view clamps above keep scrollOff within selected±height; this is
	// the final tie-breaker for the case where a previous scrollOff is
	// still larger than the new visible window (e.g. n shrunk or height
	// grew). max() guards against a negative maxScroll, which can only
	// happen if some external code wrote a height larger than n between
	// when this function read it and when it ran this clamp.
	if maxScroll := n - m.searchPopup.height; m.searchPopup.scrollOff > maxScroll {
		m.searchPopup.scrollOff = max(0, maxScroll)
	}
}

// allSearchCandidates returns all slash command candidates for the search popup.
func (m *model) allSearchCandidates() []CompletionCandidate {
	skills := m.inputModel.Skills
	if len(skills) == 0 {
		skills = m.cfg.Skills
	}
	return allSlashCommandCandidates(skills)
}

// markActiveRole appends the active marker to the role candidate the session
// currently runs on — the same way the /agent popup marks the active agent.
// An empty ActiveRole means the default role.
func markActiveRole(items []SearchItem, activeRole string) {
	if activeRole == "" {
		activeRole = "default"
	}
	for i, it := range items {
		if it.Text == activeRole {
			items[i].Description += " — active"
			return
		}
	}
}

// filterSearch rebuilds filtered from entries. An empty query pins the
// Suggested rows (recent commands, the active model/agent/subagent/todo)
// above the rest in entry order; a non-empty one fuzzy-scores every row —
// text counts double, description adds — and sorts by score, keeping entry
// order for ties. A query nothing matches falls back to the full list, as
// before.
func (sp *searchPopupState) filterSearch() {
	sp.selected, sp.scrollOff = 0, 0
	sp.suggestedN = 0
	if sp.search == "" {
		sp.filtered = sp.entries
		head, used := sp.pinnedSuggested()
		if len(head) > 0 {
			sp.suggestedN = len(head)
			filtered := make([]SearchItem, 0, len(sp.entries))
			filtered = append(filtered, head...)
			for i, e := range sp.entries {
				if !used[i] {
					filtered = append(filtered, e)
				}
			}
			sp.filtered = filtered
		}
		return
	}
	type scored struct {
		item  SearchItem
		score int
	}
	var matches []scored
	for _, e := range sp.entries {
		if s := fuzzyScore(sp.search, e.Text, e.Description); s >= 0 {
			matches = append(matches, scored{item: e, score: s})
		}
	}
	if len(matches) == 0 {
		sp.filtered = sp.entries // nothing matched: show everything, as before
		return
	}
	sort.SliceStable(matches, func(i, j int) bool { return matches[i].score > matches[j].score })
	sp.filtered = make([]SearchItem, len(matches))
	for i, m := range matches {
		sp.filtered[i] = m.item
	}
}

// pinnedSuggested returns the suggested rows that still exist in entries —
// in suggested order, with each row taken from entries so descriptions and
// IDs travel with it — plus which entry indexes were consumed. Stale
// suggestions (an unloaded skill, a finished subagent) drop out silently.
func (sp *searchPopupState) pinnedSuggested() (head []SearchItem, used []bool) {
	if len(sp.suggested) == 0 {
		return nil, nil
	}
	used = make([]bool, len(sp.entries))
	for _, s := range sp.suggested {
		for i, e := range sp.entries {
			if !used[i] && e.Text == s.Text && e.ID == s.ID {
				head = append(head, e)
				used[i] = true
				break
			}
		}
	}
	return head, used
}

// fuzzyScore scores query against an item's text and description as a
// case-insensitive subsequence match. Text hits count double, description
// hits add on top; -1 means neither matched. The score orders the popup
// list: prefix matches outrank word-start matches, which outrank plain
// subsequence hits ("/he" → /help before /theme).
func fuzzyScore(query, text, desc string) int {
	score := -1
	if s := fuzzyMatchScore(query, text); s > 0 {
		score = 2 * s
	}
	if s := fuzzyMatchScore(query, desc); s > 0 {
		if score < 0 {
			score = s
		} else {
			score += s
		}
	}
	return score
}

// fuzzyMatchScore scores query against s as a case-insensitive subsequence
// match: ≥1 on a hit, -1 on a miss. A full prefix match outranks a match at
// a word boundary, which outranks a plain hit.
func fuzzyMatchScore(query, s string) int {
	if query == "" {
		return 0
	}
	lq, ls := strings.ToLower(query), strings.ToLower(s)
	t := []rune(ls)
	first, idx := -1, 0
	for _, r := range lq {
		found := -1
		for k := idx; k < len(t); k++ {
			if t[k] == r {
				found = k
				break
			}
		}
		if found < 0 {
			return -1
		}
		if first < 0 {
			first = found
		}
		idx = found + 1
	}
	switch {
	case strings.HasPrefix(ls, lq):
		return 7
	case wordStartAt(t, first):
		return 4
	default:
		return 1
	}
}

// wordStartAt reports whether rune i in t sits right after a separator or at
// the string start, so a match there begins a word ("claude" inside
// "anthropic/claude-opus").
func wordStartAt(t []rune, i int) bool {
	if i <= 0 {
		return true
	}
	switch t[i-1] {
	case ' ', '-', '_', '.', '/', ':', '(', '[':
		return true
	}
	return false
}

// syncPalette resolves the active theme's palette and fans it out to every
// renderer that holds one. It runs once per frame from View, and again from
// applyResize and the theme-switch paths so the palette is never stale when the
// markdown renderer is rebuilt. A nil manager (tests) falls back to the dark
// palette, keeping existing output unchanged.
func (m *model) syncPalette() {
	m.palette = darkPalette
	if m.themeManager != nil {
		m.palette = paletteFor(m.themeManager.Current())
	}
	m.chatModel.Palette = m.palette
	// The framed reply headers name the session model; read fresh so a
	// /model switch shows up without extra plumbing.
	m.chatModel.ModelLabel = m.cfg.ModelName
	m.chatModel.ToolDisplay.Palette = m.palette
	m.inputModel.Palette = m.palette
	m.sea.palette = m.palette
}

// applyTheme fans out a theme change: it repaints the lipgloss chrome via
// syncPalette, then rebuilds the pieces that cache a palette internally — the
// glamour markdown renderer and the text input's prompt and cursor styles.
// Without this a /theme switch would leave the transcript in the old theme.
func (m *model) applyTheme() {
	m.syncPalette()
	m.chatModel.RefreshTheme()
	m.inputModel.RefreshTheme()
}

// newModel assembles the root model. It is separate from Run so the startup
// wiring is testable without standing up a Bubble Tea program — notably the
// ordering below, where building the renderer before the theme was known is
// what pinned the whole transcript to the dark stylesheet.
func newModel(ctx context.Context, cancel context.CancelFunc, cfg Config) model {
	// Initialize the theme manager before the markdown renderer: the glamour
	// stylesheet is chosen from the palette, so the renderer cannot be built
	// until the theme is known.
	tm := NewThemeManager()
	// Custom themes load before the configured name is applied, so the config
	// can name a theme that only exists in ~/.pirate/themes or a project.
	tm.LoadCustomThemes(cfg.WorkDir)
	if cfg.ThemeName != "" && cfg.ThemeName != "default" {
		_ = tm.SetTheme(cfg.ThemeName) // ignore error, falls back to tokyo-night
	}
	palette := paletteFor(tm.Current())

	renderer, _ := newMarkdownRenderer(100, palette)

	// Load persistent command history from ~/.pirate/history.jsonl.
	history := loadHistory()
	if history == nil {
		history = make([]HistoryEntry, 0)
	}

	m := model{
		cfg:          cfg,
		ctx:          ctx,
		cancel:       cancel,
		inputModel:   NewInputModel(history, cfg.Skills, cfg.SkillDirs, cfg.WorkDir),
		chatModel:    NewChatModel(renderer),
		statusModel:  StatusModel{},
		themeManager: tm,
		palette:      palette,
		face:         NewFaceRenderer(),
	}
	m.syncPalette()
	// The renderer above was built from this palette, so record its key rather
	// than letting the first RefreshTheme rebuild it for nothing.
	m.chatModel.rendererPaletteKey = paletteKey(palette)

	if cfg.DeferredInit != nil {
		m.loading = true
		m.loadingItems = make(map[string]bool)
		m.initCh = cfg.DeferredInit
	} else {
		m.statusModel.GitBranch = detectBranch(cfg.WorkDir)
	}
	return m
}

// Run starts the interactive TUI.
func Run(ctx context.Context, cfg Config) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	m := newModel(ctx, cancel, cfg)

	// Clear inherited terminal state before the first frame is drawn. pi renders
	// on the normal screen, so whatever the previous command left set is still
	// in effect until something resets it.
	prepareTerminal()

	opts := append(terminalProgramOptions(os.Stdout, os.Environ()), tea.WithContext(ctx))
	p := tea.NewProgram(&m, opts...)
	// Everything after p.Run() may write to the terminal: the program owns
	// stdout/stderr until it returns, and any print while it is live corrupts
	// the drawn frame (AGENTS.md, TUI output safety). Callers keep the same
	// contract for their post-Run epilogues — see internal/cli/interactive.go.
	_, err := p.Run()
	drainTerminalResponses()
	if m.initErr != nil {
		// Log the fatal init error so crash loops are diagnosable.
		if m.cfg.Logger != nil {
			m.cfg.Logger.Errorf("fatal init error: %v", m.initErr)
		}
		return m.initErr
	}
	return err
}

func (m *model) Init() tea.Cmd {
	// Ask the terminal for its background color so an unconfigured theme can
	// match it. The reply arrives as tea.BackgroundColorMsg; terminals that do
	// not answer simply leave the configured default in place. This is a
	// program-level command, deliberately not something the render path does —
	// see TestUpdateRendererDoesNotQueryTerminalBackground.
	requestBg := tea.RequestBackgroundColor

	if m.initCh != nil {
		// Deferred init: start listening for init events.
		// Heavy initialization runs in a background goroutine (started by cli).
		cmds := []tea.Cmd{
			requestBg,
			waitForInitEvent(m.initCh),
			tea.Tick(300*time.Millisecond, func(t time.Time) tea.Msg { return loadingTickMsg{} }),
		}
		// Drain notices from the first frame. Deferred init is exactly when
		// they are raised — skipped MCP servers, blocked skills, an OAuth
		// re-login — and it finishes long after they start arriving, so a
		// reader armed only from the InitResult would leave the whole startup
		// run buffered.
		if m.cfg.SystemNoticeCh != nil {
			cmds = append(cmds, waitForSystemNotice(m.cfg.SystemNoticeCh))
		}
		if m.cfg.ApprovalCh != nil {
			cmds = append(cmds, waitForApproval(m.cfg.ApprovalCh))
		}
		if m.cfg.QuestionCh != nil {
			cmds = append(cmds, waitForQuestion(m.cfg.QuestionCh))
		}
		if m.cfg.TodoCh != nil {
			cmds = append(cmds, waitForTodoUpdate(m.cfg.TodoCh))
		}
		return tea.Batch(cmds...)
	}

	// Synchronous init (non-deferred path, used by tests and non-interactive modes).
	m.refreshDiffStats()
	var cmds []tea.Cmd
	if m.cfg.AgentEventCh != nil {
		cmds = append(cmds, waitForSubEvent(m.cfg.AgentEventCh))
	}
	if m.cfg.SystemNoticeCh != nil {
		cmds = append(cmds, waitForSystemNotice(m.cfg.SystemNoticeCh))
	}
	if m.cfg.ApprovalCh != nil {
		cmds = append(cmds, waitForApproval(m.cfg.ApprovalCh))
	}
	if m.cfg.QuestionCh != nil {
		cmds = append(cmds, waitForQuestion(m.cfg.QuestionCh))
	}
	if m.cfg.TodoCh != nil {
		cmds = append(cmds, waitForTodoUpdate(m.cfg.TodoCh))
	}
	cmds = append(cmds, memoryTickCmd(m.cwd(), m.cfg.Palace), requestBg)
	return tea.Batch(cmds...)
}

// defaultLightTheme is the theme used when the terminal reports a light
// background and the user has not configured one.
const defaultLightTheme = "catppuccin-latte"

// handleBackgroundColor applies the terminal's reported background color.
//
// It only ever supplies a *default*: a theme named in config, or picked with
// /theme, is a deliberate choice and is never overridden. Nothing is persisted
// — the detected theme is a property of the terminal pi happens to be running
// in, not of the user's configuration.
func (m *model) handleBackgroundColor(msg tea.BackgroundColorMsg) {
	if m.bgDetected {
		return
	}
	m.bgDetected = true

	if m.themeManager == nil {
		return
	}
	if m.cfg.ThemeName != "" && m.cfg.ThemeName != "default" {
		return
	}
	if msg.IsDark() {
		return // the built-in default is already a dark theme
	}
	if err := m.themeManager.SetTheme(defaultLightTheme); err != nil {
		return
	}
	m.applyTheme()
}

// msgHandler consumes a message, reporting whether it handled it. When handled
// is false the model and command returned are ignored.
type msgHandler func(tea.Msg) (_ tea.Model, _ tea.Cmd, handled bool)

// Update routes a message to the group that owns it — terminal input, the
// agent event stream, the /run workflow, or the session side-channels.
func (m *model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	for _, update := range []msgHandler{
		m.updateTerminal,
		m.updateAgentStream,
		m.updateRunWorkflow,
		m.updateSession,
	} {
		if model, cmd, handled := update(msg); handled {
			return model, cmd
		}
	}

	// Deliberately no waitForAgent re-arm here. Every message that arrives on
	// m.agentCh is claimed by updateAgentStream, and each of those handlers
	// re-arms exactly once, so a single reader is always parked on the channel.
	// Re-arming for messages that did *not* come from the channel adds a reader
	// without consuming one: the readers then accumulate for the whole turn, and
	// on close(agentCh) every one of them delivers its own agentDoneMsg, running
	// handleAgentDone — lifecycle hooks, startNextPrompt — once per surplus
	// reader. See TestUnclaimedMessageDoesNotArmAgentListener and
	// TestWindowSizeDoesNotArmAgentListener.
	return m, nil
}

// updateTerminal handles messages that originate at the terminal — resize,
// keys, mouse, paste — plus the UI's own animation ticks.
func (m *model) updateTerminal(msg tea.Msg) (tea.Model, tea.Cmd, bool) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		return m.handleWindowSize(msg)

	case tea.BackgroundColorMsg:
		m.handleBackgroundColor(msg)
		return m, nil, true

	case tea.FocusMsg:
		// The terminal window gained focus (ReportFocus is on while
		// attention is configured): bells stay quiet while the user is
		// looking at the screen.
		m.focused = true
		return m, nil, true

	case tea.BlurMsg:
		m.focused = false
		return m, nil, true

	case tea.PasteMsg:
		return m.handlePaste(msg)

	case tea.KeyPressMsg:
		return m.handleKeyPressMsg(msg)

	case tea.MouseMsg:
		model, cmd := m.handleMouse(msg)
		return model, cmd, true

	case InputSubmitMsg:
		return m.handleInputSubmit(msg)

	case resetCtrlCCountMsg:
		model, cmd := m.handleResetCtrlCCount()
		return model, cmd, true

	case resetEscCancelCountMsg:
		model, cmd := m.handleResetEscCancelCount(msg)
		return model, cmd, true

	case resizeDrainDoneMsg:
		if msg.resizeAt.Equal(m.resizeAt) {
			m.resizeAt = time.Time{}
		}
		return m, nil, true

	case flashExpiredMsg:
		// Ignore a timer whose flash has already been replaced.
		if msg.seq == m.flashSeq {
			m.flash = ""
		}
		return m, nil, true

	case loadingTickMsg:
		return m.handleLoadingTick()

	case seaTickMsg:
		return m.handleSeaTick()

	case cardTickMsg:
		// Repaint, then re-arm while a live card remains. The repaint is
		// implicit — Update is always followed by View — so the handler only
		// decides whether the clock keeps ticking.
		return m, m.armCardTick(), true
	}
	return nil, nil, false
}

// handleKeyPressMsg routes a keystroke, dropping the stray text fragments a
// resize can leave in the input queue.
func (m *model) handleKeyPressMsg(msg tea.KeyPressMsg) (tea.Model, tea.Cmd, bool) {
	if m.resizeDraining() && isResizeTextFragment(msg) {
		return m, resizeDrainDoneCmd(m.resizeAt), true
	}
	model, cmd := m.handleKey(msg)
	return model, cmd, true
}

// handleInputSubmit runs a submitted line: a slash command directly, anything
// else queued as a prompt. Slash commands are ignored while a turn is running
// or a backgrounded run holds the agent.
//
// A prompt submitted mid-turn is queued without canceling the running turn
// (opencode semantics: typing is never lost and the running turn completes
// before the queued prompt starts). Esc or Ctrl+C still cancels explicitly.
// Detached, the same queue takes the prompt — the backgrounded run holds the
// agent, so nothing may start (the one-live-loop invariant) — with a flash
// that says why the prompt is waiting.
func (m *model) handleInputSubmit(msg InputSubmitMsg) (tea.Model, tea.Cmd, bool) {
	if strings.HasPrefix(msg.Text, "/") {
		if m.running || m.bgAlive() {
			// The loop reads the session the command would rewrite
			// (/compact, /branch, /clear): a backgrounded run holds it, the
			// same reason queued prompts wait. The running case stays
			// silent as it always was — there, the running status explains
			// itself; detached, the free input makes silence read as a
			// swallowed keystroke, so name the reason.
			if m.bgAlive() {
				m.chatModel.AppendNotice("Slash commands are unavailable while a run is in the background — Ctrl+B to attach.")
			}
			return m, nil, true
		}
		model, cmd := m.handleSlashCommand(msg.Text)
		return model, cmd, true
	}
	if m.running || m.bgAlive() {
		_, cmd := m.enqueuePrompt(msg.Text, msg.Mentions)
		// Flash so Enter is not silent. enqueuePrompt sets flash
		// to "Prompt queue full" when the queue is maxed; keep that.
		if m.flash != "Prompt queue full" {
			label := "Queued"
			if m.bgAlive() {
				// The input is free while a run is backgrounded, so a plain
				// "Queued" would read as a stall: name the reason it waits.
				label = "Queued — background run active"
			}
			flashCmd := m.setFlash(label)
			if cmd != nil {
				cmd = tea.Batch(cmd, flashCmd)
			} else {
				cmd = flashCmd
			}
		}
		return m, cmd, true
	}
	model, cmd := m.enqueuePrompt(msg.Text, msg.Mentions)
	return model, cmd, true
}

// handleLoadingTick advances the loading dots.
//
// The loading-dots ticker was the engine behind TODO §30/§42: every 300 ms fire
// it mutated m.loadingDots and re-armed unconditionally, producing a 3.3 Hz
// full-history re-render for the life of the process. The dots only animate the
// deferred-init splash (m.loading is set at tui.go:455 and cleared at
// :1963/:1985), so the re-arm belongs to that lifecycle, not to process uptime.
// Re-arming only while m.loading is true stops the chain at init completion and
// removes the idle 3.3 Hz floor.
func (m *model) handleLoadingTick() (tea.Model, tea.Cmd, bool) {
	if !m.loading {
		return m, nil, true
	}
	m.loadingDots = (m.loadingDots + 1) % 4
	return m, tea.Tick(300*time.Millisecond, func(time.Time) tea.Msg { return loadingTickMsg{} }), true
}

// handleSeaTick advances the sea scene and the animations that ride on it.
func (m *model) handleSeaTick() (tea.Model, tea.Cmd, bool) {
	if !m.running {
		return m, nil, true
	}
	m.sea.tick(m.mainWidth())
	// The pending-tool bullet blinks on a ~1s cycle. The phase advances
	// here, in Update, so View stays a pure function of model state —
	// the sea tick (150ms while running) re-renders often enough to
	// animate it.
	now := time.Now()
	m.chatModel.ToolDisplay.BlinkOn = now.UnixMilli()/500%2 == 0
	// Same idea one level out: the tab title's prefix symbol rotates so
	// a backgrounded session still shows it is working.
	m.titleSpin = terminalTitleSpinIndex(now)
	return m, seaTickCmd(), true
}

// handleWindowSize re-lays out the frame. It does not touch the agent listener:
// a tea.WindowSizeMsg never came off m.agentCh, so re-arming here would leave an
// extra reader parked on the channel for every resize taken mid-turn.
func (m *model) handleWindowSize(msg tea.WindowSizeMsg) (tea.Model, tea.Cmd, bool) {
	m.resizeAt = time.Now()
	m.width, m.height = msg.Width, msg.Height
	m.applyResize()

	return m, resizeDrainDoneCmd(m.resizeAt), true
}

// handlePaste inserts pasted text into the prompt, ignoring pastes that arrive
// while the agent runs or that are really resize noise. Outside a resize drain
// the message is left unhandled so later handler groups still get a look at it.
func (m *model) handlePaste(msg tea.PasteMsg) (tea.Model, tea.Cmd, bool) {
	if !m.resizeDraining() && isUserPaste(msg.Content) {
		m.inputModel.InsertText(msg.Content)
		// A pasted @ is a mention like a typed one: sync opens, refilters,
		// or closes the file popup to match the text at the cursor. A paste
		// without a triggerable @ is a no-op here.
		m.syncFilesPopup()
	}
	if m.resizeDraining() {
		return m, resizeDrainDoneCmd(m.resizeAt), true
	}
	return nil, nil, false
}

// handleMouse dispatches to the click, drag, release and wheel handlers.
func (m *model) handleMouse(msg tea.MouseMsg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.MouseClickMsg:
		return m.handleMouseClick(msg)
	case tea.MouseMotionMsg:
		return m.handleMouseMotion(msg)
	case tea.MouseReleaseMsg:
		return m.handleMouseRelease(msg)
	case tea.MouseWheelMsg:
		return m.handleMouseWheel(msg)
	}
	return m, nil
}

// updateAgentStream handles the events streamed by a running agent turn.
func (m *model) updateAgentStream(msg tea.Msg) (tea.Model, tea.Cmd, bool) {
	// An event read from a backgrounded run's channel: the envelope names
	// the run, so a run canceled while detached cannot have its late events
	// mistaken for the next turn's (background.go).
	if env, ok := msg.(bgEnvelope); ok {
		model, cmd := m.handleBgEnvelope(env)
		return model, cmd, true
	}
	// A detached run's events buffer instead of rendering. Routing, not a
	// second reader: the parked command delivered this message and the
	// handler below re-arms by the message's source (background.go).
	if m.bgAlive() {
		if _, ok := msg.(agentMsg); ok {
			model, cmd := m.handleBackgroundAgentMsg(msg)
			return model, cmd, true
		}
	}
	switch msg := msg.(type) {
	case agentThinkingMsg:
		model, cmd := m.handleAgentThinking(msg)
		return model, cmd, true
	case agentTextMsg:
		model, cmd := m.handleAgentText(msg)
		return model, cmd, true
	case agentToolCallMsg:
		model, cmd := m.handleAgentToolCall(msg)
		return model, cmd, true
	case agentToolResultMsg:
		model, cmd := m.handleAgentToolResult(msg)
		return model, cmd, true
	case agentSubEventMsg:
		model, cmd := m.handleAgentSubEvent(msg)
		return model, cmd, true
	case agentWarningMsg:
		model, cmd := m.handleAgentWarning(msg)
		return model, cmd, true
	case agentUsageMsg:
		model, cmd := m.handleAgentUsage(msg)
		return model, cmd, true
	case systemNoticeMsg:
		// AppendNotice, not a bare append: a notice carries role "assistant",
		// so an unflagged one is absorbed by the next streamed text delta — it
		// is overwritten before the repaint it just triggered can paint it.
		// This is the path every LSP hook warning takes, so on an .rs write it
		// was the notice that vanished, never the repaint that failed.
		m.chatModel.AppendNotice(msg.text)
		return m, waitForSystemNotice(m.cfg.SystemNoticeCh), true
	case todoUpdateMsg:
		m.todoState = &msg.state
		return m, waitForTodoUpdate(m.cfg.TodoCh), true
	case agentDoneMsg:
		model, cmd := m.handleAgentDone(msg)
		return model, cmd, true
	}
	return nil, nil, false
}

// updateRunWorkflow handles the /run spec workflow: agent progress, the
// quality gate, and the merge back.
func (m *model) updateRunWorkflow(msg tea.Msg) (tea.Model, tea.Cmd, bool) {
	switch msg := msg.(type) {
	case runAgentEventMsg:
		model, cmd := m.handleRunAgentEvent(msg)
		return model, cmd, true
	case runAgentDoneMsg:
		model, cmd := m.handleRunAgentDone(msg)
		return model, cmd, true
	case runGateResultMsg:
		model, cmd := m.handleRunGateResult(msg)
		return model, cmd, true
	case runMergeResultMsg:
		model, cmd := m.handleRunMergeResult(msg)
		return model, cmd, true
	case prAutofixChan:
		model, cmd := m.handlePRAutofixMsg(msg)
		return model, cmd, true
	}
	return nil, nil, false
}

// updateSession handles the side-channels that outlive a single turn: startup,
// login, commit, memory polling and ping.
func (m *model) updateSession(msg tea.Msg) (tea.Model, tea.Cmd, bool) {
	switch msg := msg.(type) {
	case initEventMsg:
		model, cmd := m.handleInitEvent(msg)
		return model, cmd, true
	case loginSSOResultMsg:
		model, cmd := m.handleLoginSSOResult(msg)
		return model, cmd, true
	case commitGeneratedMsg:
		model, cmd := m.handleCommitGenerated(msg)
		return model, cmd, true
	case commitDoneMsg:
		model, cmd := m.handleCommitDone(msg)
		return model, cmd, true
	case updateCheckMsg:
		model, cmd := m.handleUpdateCheckDone(msg)
		return model, cmd, true
	case updateAppliedMsg:
		model, cmd := m.handleUpdateApplied(msg)
		return model, cmd, true
	case memoryTickMsg:
		model, cmd := m.handleMemoryTick(msg)
		return model, cmd, true
	case pingDoneMsg:
		model, cmd := m.handlePingDone(msg)
		return model, cmd, true
	case modelPriceRefreshDoneMsg:
		model, cmd := m.handleModelPriceRefreshDone(msg)
		return model, cmd, true
	case modelCandidatesMsg:
		model, cmd := m.handleModelCandidates(msg)
		return model, cmd, true
	case approvalRequestMsg:
		// Detached, the dialog does not open: the request waits in the bg
		// stash behind a notice, and attach opens it (background.go).
		if m.bgAlive() {
			model, cmd := m.handleBackgroundApproval(msg)
			return model, cmd, true
		}
		model, cmd := m.handleApprovalRequest(msg)
		return model, cmd, true
	case questionRequestMsg:
		if m.bgAlive() {
			model, cmd := m.handleBackgroundQuestion(msg)
			return model, cmd, true
		}
		model, cmd := m.handleQuestionRequest(msg)
		return model, cmd, true
	case backgroundTickMsg:
		// Repaints the detached run's elapsed-time field; dies with the run.
		if m.bgAlive() {
			return m, armBackgroundTick(), true
		}
		return m, nil, true
	}
	return nil, nil, false
}

// handleMemoryTick stores the latest memory palace status and reschedules the
// next poll.
func (m *model) handleMemoryTick(msg memoryTickMsg) (tea.Model, tea.Cmd) {
	m.memoryStatus = msg.status
	return m, tea.Tick(memoryTickInterval, func(time.Time) tea.Msg {
		return memoryTickCmd(m.cwd(), m.cfg.Palace)()
	})
}

// handlePingDone replaces the "Pinging model..." placeholder with the ping
// result, and feeds the model's reply to the matrix rain.
func (m *model) handlePingDone(msg pingDoneMsg) (tea.Model, tea.Cmd) {
	content := msg.output
	if msg.err != nil {
		content += fmt.Sprintf("\n\n✗ Ping failed: %v", msg.err)
	}

	reply := message{role: "assistant", content: content}
	if n := len(m.chatModel.Messages); n > 0 && m.chatModel.Messages[n-1].role == "thinking" {
		m.chatModel.Messages[n-1] = reply
	} else {
		m.chatModel.Messages = append(m.chatModel.Messages, reply)
	}

	if msg.reply != "" {
		m.sea.feed(msg.reply, m.mainWidth())
	}
	return m, nil
}

// handleMouseClick starts a text selection and clears any previous one.
//
// pi selects text itself: mouse reporting is on so the wheel can scroll the
// chat, and that takes click-drag away from the terminal. See selection.go.
func (m *model) handleMouseClick(msg tea.MouseClickMsg) (tea.Model, tea.Cmd) {
	mouse := msg.Mouse()
	if mouse.Button != tea.MouseLeft {
		return m, nil
	}
	// The gauge's [x] is a control, not text. It sits to the right of the chat
	// panel, so it has to be claimed before the off-panel branch below swallows
	// the click as "deselect".
	if m.hitContextClear(mouse.X, mouse.Y) {
		m.sel = selection{}
		m.clearConversation()
		return m, m.setFlash("Context cleared")
	}
	// Only the chat panel is selectable. A press on the rail or in the sidebar
	// starts nothing — and clears any highlight, the same as a click anywhere
	// else would.
	if mouse.X >= m.chatWidth() {
		m.sel = selection{}
		return m, nil
	}
	x, y := m.clampToChat(mouse.X, mouse.Y)

	// Clicking inside a highlight copies it again, rather than throwing it away.
	// The highlight is the affordance: what is lit up is what a click takes.
	if m.sel.contains(x, y, m.chatWidth()) {
		return m, m.copyAndFlash(selectedText(m.lastFrame, m.sel, m.chatWidth()))
	}

	m.sel = selection{dragging: true, anchorX: x, anchorY: y, cursorX: x, cursorY: y}
	return m, nil
}

// copyAndFlash puts text on the clipboard and says so in the status bar.
func (m *model) copyAndFlash(text string) tea.Cmd {
	cmd := copySelection(text)
	if cmd == nil {
		return nil
	}
	return tea.Batch(cmd, m.setFlash("Copied!"))
}

// flashDuration is how long a transient status notice stays up.
const flashDuration = 3 * time.Second

// flashExpiredMsg clears a flash. It carries the sequence number of the flash it
// belongs to, so a stale timer cannot wipe a message raised after it.
type flashExpiredMsg struct{ seq int }

func (m *model) setFlash(text string) tea.Cmd {
	m.flashSeq++
	m.flash = text
	seq := m.flashSeq
	return tea.Tick(flashDuration, func(time.Time) tea.Msg {
		return flashExpiredMsg{seq: seq}
	})
}

// handleMouseMotion extends an in-progress selection. Motion only arrives while
// a button is held, which is exactly a drag.
func (m *model) handleMouseMotion(msg tea.MouseMotionMsg) (tea.Model, tea.Cmd) {
	if !m.sel.dragging {
		return m, nil
	}
	mouse := msg.Mouse()
	m.sel.cursorX, m.sel.cursorY = m.clampToChat(mouse.X, mouse.Y)
	m.sel.present = true
	return m, nil
}

// handleMouseRelease ends the drag and copies the selection.
//
// Copy on release, with no extra keystroke: that is what the terminal used to do
// for us, and reproducing it is the whole point. The highlight stays up so it is
// obvious what landed on the clipboard.
func (m *model) handleMouseRelease(msg tea.MouseReleaseMsg) (tea.Model, tea.Cmd) {
	if !m.sel.dragging {
		return m, nil
	}
	m.sel.dragging = false

	// Take the release's own position, not the last motion's. Terminals coalesce
	// motion events, so the final mouse position often arrives only with the
	// release: copying the last motion made the selection stop a row short of
	// where the user let go.
	mouse := msg.Mouse()
	m.sel.cursorX, m.sel.cursorY = m.clampToChat(mouse.X, mouse.Y)

	if m.sel.empty() {
		m.sel = selection{} // a plain click, not a drag: just clear
		return m, nil
	}
	return m, m.copyAndFlash(selectedText(m.lastFrame, m.sel, m.chatWidth()))
}

// clampToChat converts a mouse position into a chat-panel cell.
//
// The Y translation matters: the UI runs on the normal screen, not the alternate
// one, so the frame is drawn *below* whatever the terminal already had on it —
// a shell prompt, a "pprof listening" line — and Bubble Tea reports the mouse in
// absolute terminal rows, not frame rows. Indexing the frame with a raw mouse Y
// therefore lands N rows too low, N being the height of that prior output.
//
// The frame is bottom-anchored (its last row is the screen's last row), so the
// offset is whatever is left over: height - frameRows. Deriving it beats
// hardcoding it, because it is right for any amount of prior output.
//
// X needs no such translation — the frame starts at column 0 — but it is clamped
// to the panel so a drag into the sidebar, or off the window entirely, pins to
// the edge instead of sweeping the rail and sidebar into the selection.
func (m *model) clampToChat(x, y int) (int, int) {
	x = min(max(x, 0), max(0, m.chatWidth()-1))

	// Rows are clamped to the message viewport, so a drag that strays into the
	// sea scene above or the status bar and prompt below pins to the nearest
	// message row instead of copying chrome. Clamping rather than refusing keeps
	// a drag that overshoots by a row from doing nothing at all.
	top, bottom := m.msgTop, m.msgBottom
	if bottom <= top { // no viewport yet (first frame)
		return x, 0
	}
	y = min(max(m.frameRow(y), top), bottom-1)
	return x, y
}

// frameRow maps an absolute terminal row to a row of the rendered frame.
func (m *model) frameRow(y int) int {
	return y - m.frameTop()
}

// frameTop is the terminal row the frame's first line occupies.
func (m *model) frameTop() int {
	return max(0, m.height-m.frameRows)
}

// handleMouseWheel processes mouse wheel events for scrolling the chat viewport.
func (m *model) handleMouseWheel(msg tea.MouseWheelMsg) (tea.Model, tea.Cmd) {
	mouse := msg.Mouse()
	switch mouse.Button {
	case tea.MouseWheelUp:
		m.chatModel.ScrollUp(3, m.height)
	case tea.MouseWheelDown:
		m.chatModel.ScrollDown(3)
	}
	return m, nil
}

// keyHandler consumes a key press, reporting whether it handled the key. When
// handled is false the model and command returned are ignored and the key
// falls through to the next handler.
type keyHandler func(tea.Key) (_ tea.Model, _ tea.Cmd, handled bool)

// handleKey routes a key press through the overlay stack — the top entry
// sees every key first — then the interrupt layer, the editing keys, and
// finally the prompt input. Esc against an open overlay is the overlay's
// (the viewer steps back to the monitor, the monitor closes); only an empty
// stack reaches the interrupt layer, where Esc cancels a running turn. A new
// overlay joins by pushing an entry — handleKey does not grow.
func (m *model) handleKey(msg tea.KeyPressMsg) (tea.Model, tea.Cmd) {
	key := msg.Key()

	// The stack, top first. syncOverlays runs first so the stack mirrors the
	// overlay fields no matter what cleared or opened one behind the helpers.
	m.syncOverlays()
	if model, cmd, handled := m.dispatchOverlays(key); handled {
		return model, cmd
	}

	// The interrupt layer sits below the stack: with an overlay open, Esc
	// belongs to that overlay, and only an empty stack cancels the turn.
	if model, cmd, handled := m.handleInterruptKey(key); handled {
		return model, cmd
	}

	// Everything below edits the prompt, which is read-only while busy.
	if m.loading {
		return m, nil
	}

	for _, handle := range []keyHandler{
		m.handleToggleKey,
		m.handleHistoryKey,
		m.handleScrollKey,
	} {
		if model, cmd, handled := handle(key); handled {
			return model, cmd
		}
	}

	return m.handleInputKey(msg)
}

// isCancelKey reports whether key is one of the two ways to back out of a
// modal prompt: Esc or Ctrl+C.
func isCancelKey(key tea.Key) bool {
	return key.Code == tea.KeyEsc || (key.Code == 'c' && key.Mod == tea.ModCtrl)
}

// handleCommitKey resolves the commit confirmation prompt, swallowing every
// other key so a stray press cannot commit.
func (m *model) handleCommitKey(key tea.Key) (tea.Model, tea.Cmd, bool) {
	if m.running || m.commit == nil || m.commit.phase != "confirming" {
		return nil, nil, false
	}
	switch {
	case key.Code == tea.KeyEnter:
		model, cmd := m.handleCommitConfirm()
		return model, cmd, true
	case isCancelKey(key):
		model, cmd := m.handleCommitCancel()
		return model, cmd, true
	default:
		return m, nil, true
	}
}

// handleLoginKey drives the login overlay. In the phases that collect text the
// unhandled keys fall through to the prompt input; in every other phase the
// overlay swallows them.
func (m *model) handleLoginKey(key tea.Key) (tea.Model, tea.Cmd, bool) {
	if m.running || m.login == nil {
		return nil, nil, false
	}
	if isCancelKey(key) {
		model, cmd := m.handleLoginCancel()
		return model, cmd, true
	}

	collecting := m.login.phase == "waiting" || m.login.phase == "manual-code"
	if key.Code == tea.KeyEnter && collecting {
		if m.login.phase == "waiting" {
			return m.submitLoginInput(m.handleLoginSave)
		}
		return m.submitLoginInput(m.handleLoginCodeSubmit)
	}
	return m, nil, !collecting
}

// submitLoginInput hands the trimmed prompt text to submit. Enter on a blank
// prompt does nothing rather than submitting an empty credential.
func (m *model) submitLoginInput(submit func(string) (tea.Model, tea.Cmd)) (tea.Model, tea.Cmd, bool) {
	text := strings.TrimSpace(m.inputModel.Text)
	if text == "" {
		return m, nil, true
	}
	m.inputModel.Clear()
	model, cmd := submit(text)
	return model, cmd, true
}

// handleSkillCreateKey resolves the skill-create overwrite confirmation.
func (m *model) handleSkillCreateKey(key tea.Key) (tea.Model, tea.Cmd, bool) {
	if m.running || m.pendingSkillCreate == nil {
		return nil, nil, false
	}
	switch {
	case key.Code == tea.KeyEnter:
		model, cmd := m.handleSkillCreateConfirm()
		return model, cmd, true
	case isCancelKey(key):
		model, cmd := m.handleSkillCreateCancel()
		return model, cmd, true
	default:
		return m, nil, true
	}
}

// handleBranchPopupKey navigates the branch picker. Anything other than the
// arrows and Enter dismisses it.
func (m *model) handleBranchPopupKey(key tea.Key) (tea.Model, tea.Cmd, bool) {
	if m.branchPopup == nil {
		return nil, nil, false
	}
	switch key.Code {
	case tea.KeyEnter:
		model, cmd := m.handleBranchSelect()
		return model, cmd, true
	case tea.KeyUp:
		m.branchPopup.moveUp()
	case tea.KeyDown:
		m.branchPopup.moveDown()
	default:
		m.popOverlay(overlayBranchPopup)
	}
	return m, nil, true
}

// handleInterruptKey handles the keys that stay live while the agent runs:
// Esc to dismiss or cancel, Ctrl+C to cancel then quit, and Ctrl+Z to suspend.
// The chords resolve through the registry (hotkeyIDFor); F12 is swallowed
// before it — a dead-man key with no action, deliberately not in the help.
func (m *model) handleInterruptKey(key tea.Key) (tea.Model, tea.Cmd, bool) {
	if key.Code == tea.KeyF12 {
		return m, nil, true
	}
	switch hotkeyIDFor(key) {
	case "key.cancel":
		if m.steerInput != nil {
			// The monitor's steer mini-input: Esc backs out of the input
			// first, a second Esc closes the monitor itself. Reached only
			// when called below the stack — through handleKey the steer
			// entry owns the key first — but kept so the layer stays
			// correct on a direct call.
			m.popOverlay(overlaySteerInput)
			return m, nil, true
		}
		if m.searchPopup != nil {
			m.popOverlay(overlaySearchPopup)
			return m, nil, true
		}
		if m.running {
			// First Esc only warns; the second within 2 seconds cancels
			// the turn. The timer resets the counter, and cancelAgent
			// plus beginTurn cover the turn boundaries, so a stale count
			// cannot survive into an unrelated turn.
			m.escCancelCount++
			if m.escCancelCount >= 2 {
				m.escCancelCount = 0
				return m, m.cancelAgent(), true
			}
			m.chatModel.AppendWarning("\nEsc again to cancel...")
			return m, resetEscCancelCount(m), true
		}
		if m.bgAlive() {
			// The backgrounded run deserves the same double-Esc contract:
			// the input is free, so a bare Esc must stay a no-op, but a
			// deliberate Esc-Esc reaches the run the input no longer holds.
			m.escCancelCount++
			if m.escCancelCount >= 2 {
				m.escCancelCount = 0
				return m, m.cancelBackgroundRun(), true
			}
			m.chatModel.AppendWarning("\nEsc again to cancel the background run...")
			return m, resetEscCancelCount(m), true
		}
		return m, nil, true

	case "key.quit":
		if m.running || m.bgAlive() {
			var cmd tea.Cmd
			if m.bgAlive() {
				cmd = m.cancelBackgroundRun()
			} else {
				cmd = m.cancelAgent()
			}
			m.ctrlCCount++
			m.chatModel.AppendWarning("\nCtrl+C again to quit...")
			return m, tea.Batch(cmd, resetCtrlCCount(m)), true
		}
		m.ctrlCCount++
		if m.ctrlCCount >= 2 {
			m.quitting = true
			return m, tea.Quit, true
		}
		// First press: warn, and reset the count after 2 seconds.
		m.chatModel.AppendWarning("\nCtrl+C again to quit...")
		return m, resetCtrlCCount(m), true

	case "key.suspend":
		// Ctrl+Z: suspend the process, classic job-control style. Bubble
		// Tea v2 owns the terminal in raw mode, so the shell would never
		// see the keystroke — the model must forward it. tea.Suspend has
		// the program restore the terminal, SIGTSTP the process group,
		// and on resume (SIGCONT) re-init input and fully repaint the
		// renderer, so no manual termios/stop machinery lives here. Lives
		// in the always-live interrupt chain: suspending while a turn
		// runs (its main use) and under open overlays both work, and the
		// Ctrl+C scheme above is untouched.
		return m, tea.Suspend, true
	}
	return nil, nil, false
}

// handleToggleKey handles the Ctrl-chord toggles and the search popup. The
// chords resolve through the registry (hotkeyIDFor → keyRegistry); the action
// bodies and their gates are unchanged.
func (m *model) handleToggleKey(key tea.Key) (tea.Model, tea.Cmd, bool) {
	switch hotkeyIDFor(key) {
	case "key.retry":
		// Ctrl+R: re-send the prompt of a turn that failed. Checked before
		// the input fallthrough, and only when the prompt is empty, so a
		// user editing a new prompt never has it replaced by the previous
		// one. A search popup owns the key while it is open — retrying
		// under it would run a turn behind the popup.
		if m.inputModel.Text == "" && m.searchPopup == nil {
			model, cmd := m.handleRetry()
			return model, cmd, true
		}

	case "key.compact-tools":
		// Ctrl+O: toggle compact/expanded tool output.
		m.chatModel.ToolDisplay.CompactTools = !m.chatModel.ToolDisplay.CompactTools
		return m, nil, true

	case "key.background":
		// Ctrl+B sends the running turn to the background or attaches a
		// backgrounded one back. Deliberately ungated on the prompt text,
		// unlike the editing chords: the action is non-destructive and its
		// own undo (press again), and while busy is exactly when it is needed.
		model, cmd := m.toggleBackground()
		return model, cmd, true

	case "key.branch":
		// Ctrl+G toggles the branch popup only when the prompt is empty.
		// Formerly Ctrl+B — that chord now backgrounds the running task —
		// and still gated because the standard text input uses Ctrl+G as a
		// goto-line/readline chord, and some terminals emit it for navigation.
		if m.inputModel.Text == "" {
			if m.statusModel.GitBranch != "" {
				if m.branchPopup == nil {
					m.newBranchPopup()
				} else {
					m.popOverlay(overlayBranchPopup)
				}
			}
			return m, nil, true
		}

	case "key.agent-cycle":
		// Shift+Tab cycles the main session's agent: default → primary agents
		// (alphabetical) → default. An open search popup never gets here: its
		// stack entry owns Tab/Shift+Tab for list navigation and wins the key
		// before the globals run.
		model, cmd := m.cycleAgent()
		return model, cmd, true

	case "key.history":
		// Ctrl+H: open history search popup (reverse-i-search style). With no
		// history to search the key falls through to the input.
		//
		// Gated on an empty prompt because Ctrl+H is also the readline "delete
		// backward" chord: with text in the buffer the input keeps it, so the
		// editing key still works where it is actually needed. The popup gate
		// stays too: the popup declines chords it does not own, so without the
		// gate this would open a second popup under the open one.
		if m.searchPopup == nil && len(m.inputModel.History) > 0 && m.inputModel.Text == "" {
			m.newSearchPopup(searchModeHistory)
			return m, nil, true
		}

	case "key.monitor":
		// Ctrl+T toggles the subagent monitor — the popup /subagents opens.
		// Reachable while a response runs, like the other popups: watching a
		// subagent is exactly what you do mid-turn. Gated on an empty prompt
		// because Ctrl+T is a readline editing chord (transpose) with text in
		// the buffer. With the stream viewer open its stack entry swallows the
		// chord before the globals run.
		if m.inputModel.Text == "" {
			m.toggleSubagentsPopup()
			return m, nil, true
		}
	}
	return nil, nil, false
}

// handleHistoryKey maps Up to the prompt-history window and Down to the chat.
//
// Up used to cycle history inline, replacing whatever was typed. The window
// shows the whole history at once and leaves the prompt untouched until an
// entry is chosen, so a stray Up costs one Esc rather than your draft.
//
// The mouse wheel cannot land here: View enables mouse reporting, so a wheel
// tick is a MouseWheelMsg handled by handleMouseWheel, never a KeyUp. Up still
// scrolls the chat before any history exists, so the keyboard can scroll on a
// fresh session.
//
// Down keeps scrolling the chat, with one exception: on an empty prompt, with
// the view already at the bottom — where a scroll-down is a no-op — and
// subagents running, ↓ opens the subagent monitor. That is the one state where
// the key it replaces is dead, so nothing is taken away: scrolled-up reading
// still walks back down first, and a second ↓ then opens the monitor.
//
// A prompt starting with "/" is excluded: those arrows drive the slash-command
// popup instead. An open search popup never gets here either: its stack entry
// owns Up/Down for list navigation and wins the key before the globals run.
//
// With a multi-line prompt the two edges belong to the input: when the cursor
// sits on an inner display row, Up/Down move it between lines (the engine's
// job) and never reach this handler. Only from the very first row does Up
// open the history window, and only from the very last row does Down scroll
// the chat — so single-line prompts behave exactly as they always did.
func (m *model) handleHistoryKey(key tea.Key) (tea.Model, tea.Cmd, bool) {
	if m.shouldShowSlashCommandPopup() {
		return nil, nil, false
	}
	switch hotkeyIDFor(key) {
	case "key.history-window":
		if !m.inputModel.CursorOnFirstVisualRow() {
			return nil, nil, false
		}
		if len(m.inputModel.History) == 0 {
			m.chatModel.ScrollUp(3, m.height)
			return m, nil, true
		}
		m.newSearchPopup(searchModeHistory)
		return m, nil, true
	case "key.monitor-down":
		if !m.inputModel.CursorOnLastVisualRow() {
			return nil, nil, false
		}
		if m.inputModel.Text == "" && m.chatModel.Scroll == 0 && m.runningSubagentCount() > 0 {
			m.newSearchPopup(searchModeSubagents)
			return m, nil, true
		}
		m.chatModel.ScrollDown(3)
		return m, nil, true
	}
	return nil, nil, false
}

// handleScrollKey pages the chat viewport.
func (m *model) handleScrollKey(key tea.Key) (tea.Model, tea.Cmd, bool) {
	switch hotkeyIDFor(key) {
	case "key.chat-up":
		m.chatModel.ScrollUp(5, m.height)
		return m, nil, true
	case "key.chat-down":
		m.chatModel.ScrollDown(5)
		return m, nil, true
	}
	return nil, nil, false
}

// handleInputKey delegates to the prompt input, keeping the slash-command
// popup in sync with the text as it changes.
func (m *model) handleInputKey(msg tea.KeyPressMsg) (tea.Model, tea.Cmd) {
	key := msg.Key()

	// Alt+V: view the collapsed paste nearest the cursor fullscreen. No
	// markers in the prompt → no-op. The viewer opens over the prompt and
	// swallows keys until Esc (see handlePasteViewerKey).
	if key.Code == 'v' && key.Mod == tea.ModAlt {
		m.openPasteViewer()
		return m, nil
	}

	// Show the commands popup when the input starts with "/".
	if m.shouldShowSlashCommandPopup() {
		if m.searchPopup == nil || m.searchPopup.mode != searchModeCommands {
			m.newSearchPopup(searchModeCommands)
		}
		// Immediately handle Tab/Up/Down to navigate the popup.
		switch key.Code {
		case tea.KeyTab, tea.KeyUp, tea.KeyDown:
			if cmd, handled := m.handleSearchPopupKey(key); handled {
				return m, cmd
			}
		}
	}

	prevText := m.inputModel.Text
	cmd := m.inputModel.HandleKey(msg)
	// The @file popup mirrors the mention at the cursor on every key —
	// including ones that only move the cursor — so it can never outlive
	// its trigger.
	m.syncFilesPopup()
	if m.inputModel.Text == prevText {
		return m, cmd
	}

	// The text changed: open or close the popup to match the new prompt.
	if m.searchPopup != nil && m.searchPopup.mode == searchModeCommands && !m.shouldShowSlashCommandPopup() {
		m.popOverlay(overlaySearchPopup)
	}
	if m.searchPopup == nil && m.shouldShowSlashCommandPopup() {
		m.newSearchPopup(searchModeCommands)
	}
	return m, cmd
}

func (m *model) handleSearchPopupKey(key tea.Key) (tea.Cmd, bool) {
	if m.searchPopup == nil {
		return nil, false
	}

	sp := m.searchPopup

	// The steer mini-input (subagent monitor) owns every key while open, and
	// `s` on a running row opens it before the key reaches the filter below.
	if m.steerInput != nil {
		return m.handleSubagentSteerKey(key), true
	}
	if sp.mode == searchModeFiles {
		// The file popup is text-driven: letters never land here.
		return m.handleFilesPopupKey(key)
	}
	if sp.mode == searchModeSubagents && hotkeyIDFor(key) == "key.subagents-steer" {
		m.tryOpenSubagentSteer()
		// Consumed either way: a refusal is a notice, not a filter edit.
		return nil, true
	}

	switch key.Code {
	case tea.KeyUp:
		sp.selectPrev()
		return nil, true
	case tea.KeyDown:
		sp.selectNext()
		return nil, true
	case tea.KeyTab:
		sp.selectByTab(key.Mod == tea.ModShift)
		return nil, true
	case tea.KeyEnter:
		return m.acceptSearchPopupSelection(), true
	case tea.KeyEsc:
		m.popOverlay(overlaySearchPopup)
		return nil, true
	case tea.KeyBackspace:
		if len(sp.search) == 0 {
			// If search is empty, close popup on backspace
			m.popOverlay(overlaySearchPopup)
			return nil, true
		}
		// Drop the last rune, not the last byte — a byte off a multi-byte
		// rune leaves invalid UTF-8 in the filter (same slice as the question
		// and steer inputs).
		r := []rune(sp.search)
		sp.search = string(r[:len(r)-1])
		sp.filterSearch()
		return nil, true
	default:
		// Type to search (any single printable rune, no chord).
		if singleRuneKey(key.Text) && key.Mod == 0 {
			sp.search += key.Text
			sp.filterSearch()
			return nil, true
		}
		return nil, false
	}
}

// selectPrev moves the selection up one, wrapping to the last item from the
// first, and scrolls to keep the selection visible.
func (sp *searchPopupState) selectPrev() {
	if sp.selected > 0 {
		sp.selected--
	} else if len(sp.filtered) > 1 {
		// Wrap to last item on Up from first.
		sp.selected = len(sp.filtered) - 1
	}
	sp.scrollOff = max(0, sp.selected-sp.height+1)
}

// selectNext moves the selection down one, wrapping to the first item from the
// last, and scrolls only when the selection would fall past the visible window.
func (sp *searchPopupState) selectNext() {
	if sp.selected < len(sp.filtered)-1 {
		sp.selected++
	} else if len(sp.filtered) > 1 {
		// Wrap to first item on Down from last.
		sp.selected = 0
	}
	if sp.selected >= sp.scrollOff+sp.height {
		sp.scrollOff = sp.selected - sp.height + 1
	}
}

// selectByTab moves the selection for Tab / Shift+Tab. Shift+Tab wraps from the
// first item to the last; Tab stops at the last item rather than wrapping.
func (sp *searchPopupState) selectByTab(backwards bool) {
	if len(sp.filtered) == 0 {
		return
	}
	if backwards {
		if sp.selected > 0 {
			sp.selected--
		} else {
			// Wrap to last item on Shift+Tab from first.
			sp.selected = len(sp.filtered) - 1
		}
	} else if sp.selected < len(sp.filtered)-1 {
		// Tab advances; stays at last item (no wrap).
		sp.selected++
	}
	sp.scrollOff = max(0, sp.selected-sp.height+1)
}

// acceptSearchPopupSelection puts the selected item into the prompt and closes
// the popup. A command gets a trailing space so its arguments can be typed
// straight on; a history entry is inserted verbatim. In the models and agents
// modes the item IS the action: it is executed through the same handlers as
// `/model <text>` and `/agent <name>` — including their refusals while a
// response is running — and the returned command (a flash-expiry tick, for
// instance) is handed back to the caller.
func (m *model) acceptSearchPopupSelection() tea.Cmd {
	sp := m.searchPopup
	if len(sp.filtered) == 0 || sp.selected >= len(sp.filtered) {
		return nil
	}
	item := sp.filtered[sp.selected]
	switch sp.mode {
	case searchModeCommands:
		m.recordCommandUse(item.Text)
		m.inputModel.SetText(item.Text + " ")
		m.popOverlay(overlaySearchPopup)
	case searchModeHistory:
		m.inputModel.SetText(item.Text)
		m.popOverlay(overlaySearchPopup)
	case searchModeFiles:
		m.insertFilesPopupSelection(item.Text)
		m.popOverlay(overlaySearchPopup)
	case searchModeModels:
		m.popOverlay(overlaySearchPopup)
		_, cmd := m.handleModelCommand([]string{item.Text})
		return cmd
	case searchModeAgents:
		m.popOverlay(overlaySearchPopup)
		_, cmd := m.handleAgentCommand([]string{item.Text})
		return cmd
	case searchModeSubagents:
		// Enter opens the fullscreen stream viewer for the selected agent.
		// The monitor stays open behind it — Esc in the viewer comes back
		// here. A status-only row has no transcript card, so no stream to
		// show: the key is consumed but nothing opens.
		if item.ID != "" && m.agentCardByID(item.ID) != nil {
			m.subagentViewer = &subagentViewerState{agentID: item.ID}
			m.openOverlay(overlaySubagentViewer)
		}
	case searchModeTodos:
		// View-only: Enter does nothing. Esc closes the popup.
	case searchModeHelp:
		// Slash rows run the command through the normal dispatch — the same
		// path typing it would take. Hotkey rows are view-only: the key is
		// consumed, the popup stays open.
		if _, ok := registryByCommand[item.Text]; !ok {
			return nil
		}
		m.popOverlay(overlaySearchPopup)
		m.recordCommandUse(item.Text)
		_, cmd := m.handleSlashCommand(item.Text)
		return cmd
	}
	return nil
}

func (m *model) View() tea.View {
	if m.quitting {
		return tea.NewView("Goodbye!\n")
	}

	m.syncPalette()

	if m.width == 0 {
		return m.startupView()
	}

	// Layout: messages + sidebar on top, status bar + input spanning the full
	// width below. The sidebar sits beside the messages only; the status bar
	// extends to the right edge, giving it more room for tools and clock info.
	mainWidth := m.mainWidth()
	if m.statusModel.Width != m.width || m.chatModel.Width != m.chatWidth() {
		m.applyResize()
		mainWidth = m.mainWidth()
	}
	bodyWidth := m.chatWidth()
	sideW := m.width - mainWidth
	showSidebar := sideW > 0

	// Render components.
	// The editable width shrinks by the box chrome (two border columns) plus
	// the two prompt columns and one column of slack, so the bordered prompt
	// occupies exactly the row width the unbordered one did.
	m.inputModel.SetWidth(max(0, m.width-5))
	// The input grows with its content up to a third of the terminal and
	// scrolls internally past that; the message viewport yields the rows.
	m.inputModel.SetMaxHeight(max(1, m.height/3))
	messagesView, lineKinds := m.chatModel.renderMessages(m.running)
	statusBar := m.statusModel.Render(m.statusRenderInput())
	inputArea := m.inputModel.View(m.loading)
	var inputCursor *tea.Cursor
	if !m.loading {
		inputCursor = m.inputModel.Cursor()
	}

	// Calculate available height for messages.
	availableHeight := m.messageViewportHeight()

	// Truncate messages to fit viewport.
	visibleMessages, startLine, endLine := clipMessagesToViewport(
		messagesView, availableHeight, m.chatModel.Scroll)
	visibleMessages = m.overlaySearchPopup(visibleMessages, bodyWidth)
	// The stream viewer covers the whole message viewport, so it paints after
	// the search popup: the monitor stays open underneath it and the viewer
	// simply covers it up.
	visibleMessages = m.overlaySubagentViewer(visibleMessages, bodyWidth)
	// The paste viewer paints last: opened from the prompt, it sits on top of
	// whatever popup happens to be live beneath it.
	visibleMessages = m.overlayPasteViewer(visibleMessages, bodyWidth)
	// The diff viewer (/diff) is likewise a prompt-opened fullscreen mode;
	// it paints on top so the command works no matter what is live below.
	visibleMessages = m.overlayDiffViewer(visibleMessages, bodyWidth)

	// Note: width constraint is handled by glamour's WithWordWrap(contentWidth) in chatModel.UpdateRenderer.
	// lipgloss.Width() counts raw bytes including invisible ANSI codes, causing wrapping issues.

	// Render the sea scene as a full-width top bar (when active).
	seaBar := m.sea.render()

	// Horizontal rule for separating sections. The panel hr is bodyWidth; the
	// full-width hr spans the entire terminal (used below the sidebar).
	hrStyle := lipgloss.NewStyle().Foreground(m.palette.Surface)
	hr := hrStyle.Render(strings.Repeat("─", bodyWidth))
	fullHr := hrStyle.Render(strings.Repeat("─", m.width))

	var b strings.Builder
	if seaBar != "" {
		b.WriteString(hr)
		b.WriteString("\n")
		b.WriteString(seaBar)
		b.WriteString("\n")
		b.WriteString(hr)
		b.WriteString("\n")
	}
	// One blank row above the messages, matched by one below, so the block is
	// inset evenly between the rules whether or not the matrix bar is showing.
	b.WriteString("\n")

	// The rows the message viewport occupies: where the rail switches from a
	// plain divider to the minimap, and how many cells it needs. A block that
	// ends in a newline is already terminated, so its trailing "" is not a row —
	// counting it as one pushed a minimap bar onto the rule below.
	msgStart := strings.Count(b.String(), "\n")
	msgRows := strings.Count(visibleMessages, "\n") + 1
	if strings.HasSuffix(visibleMessages, "\n") {
		msgRows--
	}

	b.WriteString(visibleMessages)
	// Terminate the message block. When the viewport is exactly full nothing
	// pads it, and the blank row below would otherwise be appended to the last
	// message line.
	if !strings.HasSuffix(visibleMessages, "\n") {
		b.WriteString("\n")
	}
	b.WriteString("\n") // the matching blank row below

	// Render branch popup if open.
	if m.branchPopup != nil {
		popupView := m.renderBranchPopup()
		b.WriteString(popupView)
		b.WriteString("\n")
	}

	// Render the tool-approval dialog while a request is pending. It sits in
	// the same slot as the branch popup: the last block of the chat panel,
	// above the closing rule, where the user's eye already is.
	if m.approval != nil {
		b.WriteString(m.renderApprovalDialog(bodyWidth))
		b.WriteString("\n")
	}

	// Render the question dialog while a request is pending — same slot,
	// below the approval line when both are somehow up.
	if m.question != nil {
		b.WriteString(m.renderQuestionDialog(bodyWidth))
		b.WriteString("\n")
	}

	// The rule closes the panel — no newline after it. A trailing newline made
	// the panel one row taller than its content: an empty row below the rule,
	// which the sidebar filled with its own last row, so its filler dots ran on
	// beside the status bar and the two rules never met.
	b.WriteString(hr)

	// The top section: body | rail | sidebar. The sidebar only covers the
	// messages area, not the status bar below — the status bar extends the
	// full terminal width underneath.
	body := padLinesTo(b.String(), bodyWidth)
	panelRows := strings.Count(body, "\n") + 1
	rail := railColumn(panelRows, msgStart, renderMinimap(lineKinds, startLine, endLine, msgRows, m.palette), m.palette)
	leftPanel := lipgloss.JoinHorizontal(lipgloss.Top, body, rail)

	var topSection string
	if showSidebar {
		sidebar := RenderSidebar(m.sidebarRenderInput(sideW, panelRows))
		topSection = joinPanelSidebar(leftPanel, sidebar, mainWidth, sideW)
	} else {
		topSection = padLinesTo(leftPanel, m.width)
	}

	// Bottom section: full-width status bar + input. These span the entire
	// terminal width — the sidebar ends at the hr above, so the status bar has
	// room for all its segments (tools, clock, context) without competing with
	// the sidebar for columns.
	var bottom strings.Builder
	bottom.WriteString(statusBar)
	bottom.WriteString("\n")
	bottom.WriteString(fullHr)
	bottom.WriteString("\n")
	inputCursorY := strings.Count(topSection, "\n") + 1 + strings.Count(bottom.String(), "\n")
	bottom.WriteString(inputArea)
	bottom.WriteString("\n")
	// The opencode-style line under the prompt: session identity and run
	// status on the left, mode key hints on the right.
	bottom.WriteString(m.bottomStatusLine(m.width))
	bottom.WriteString("\n")
	// The closing rule doubles as the session context gauge — same row, same
	// width, now carrying a reading instead of only closing the frame.
	bottom.WriteString(renderContextRule(m.contextRuleFor(m.width), m.palette))

	// Pad the bottom section to the full terminal width so no row is wider or
	// narrower than m.width. The status bar's Width style handles it for that
	// line, but the input area and rules need explicit padding.
	bottomStr := padLinesTo(bottom.String(), m.width)
	final := topSection + "\n" + bottomStr

	// Remember the frame the mouse is pointing at, then draw the selection over
	// it. The selection is in screen coordinates, so it has to be applied to the
	// composed frame — and the copy on release reads back from this same string,
	// which is why it is kept rather than re-derived.
	m.lastFrame = final
	m.frameRows = strings.Count(final, "\n") + 1
	m.topSectionRowsVal = panelRows
	m.msgTop, m.msgBottom = msgStart, msgStart+msgRows
	final = highlight(final, m.sel, m.chatWidth(), m.width)

	v := tea.NewView(final)
	// The renderer diffs WindowTitle against the last frame and only emits the
	// escape sequence when it changes, then clears it again on teardown. Going
	// through the View is what keeps the sequence ordered against the frame
	// writes — writing it to os.Stdout directly races the renderer and lands
	// mid-frame, which corrupts the drawn output.
	//
	// While a turn runs the "π Pi-rate -" prefix becomes a rotating
	// "π Pi-rate ⠋ / ⠙ / ⠹ / …" so the tab shows progress even when the window
	// is not visible.
	v.WindowTitle = formatTerminalTitleWithPrefix(
		terminalTitlePrefix(m.running, m.titleSpin), m.sessionTitle, m.cfg.WorkDir)
	if inputCursor != nil {
		inputCursor.Y += inputCursorY
		v.Cursor = inputCursor
	}
	// Stay on the normal terminal screen, but report mouse events so the wheel
	// scrolls the chat viewport (handleMouseWheel) instead of the terminal's
	// scrollback, which only ever showed stale frames of a redrawing UI.
	//
	// This is also what keeps the wheel off the history window: with reporting
	// on, a wheel tick arrives as a MouseWheelMsg and can never be delivered as
	// an Up key, so scrolling and Up stay structurally separate rather than
	// racing over the same event.
	//
	// Reporting takes click-drag away from the terminal, so the terminal can no
	// longer select text for us. Rather than make the user hold a bypass modifier,
	// pi does the selection itself — drag to select, release to copy. See
	// selection.go. CellMotion is the narrowest mode that carries wheel events;
	// Bubble Tea has no wheel-only mode.
	v.AltScreen = false
	v.MouseMode = tea.MouseModeCellMotion
	// Focus reporting backs the attention policy: FocusMsg/BlurMsg keep the
	// model's m.focused current, and attentionCmd rings only while blurred.
	// Enabled only when attention is actually configured — a session with the
	// signals off should not flip the terminal's focus-report mode for
	// nothing.
	v.ReportFocus = m.attentionEnabled()
	return v
}

// startupView is the matrix-style startup text shown before the first terminal
// size arrives, listing deferred-init progress once there is any to show.
func (m *model) startupView() tea.View {
	matrixLine := renderStartupMatrixLine(m.loadingDots, m.cfg.AppVersion, m.loadingItems, m.loadingTotal, m.palette)
	if m.loadingItems == nil {
		return tea.NewView(matrixLine + "\n")
	}
	var lines []string
	lines = append(lines, matrixLine)
	for _, item := range sortedKeys(m.loadingItems) {
		done := m.loadingItems[item]
		mark := " "
		if done {
			mark = "✓"
		}
		lines = append(lines, "  "+mark+" "+item)
	}
	return tea.NewView(strings.Join(lines, "\n") + "\n")
}

// clipMessagesToViewport takes the rendered message block down to the rows the
// viewport can show, scrolled back by scroll rows, and pads it out when the
// conversation is shorter than the viewport. It returns the visible block along
// with the line range it covers, which the minimap needs to place its bar.
func clipMessagesToViewport(messagesView string, availableHeight, scroll int) (visible string, startLine, endLine int) {
	msgLines := strings.Split(messagesView, "\n")
	totalLines := len(msgLines)

	startLine = totalLines - availableHeight - scroll
	if startLine < 0 {
		startLine = 0
	}
	endLine = startLine + availableHeight
	if endLine > totalLines {
		endLine = totalLines
	}

	visible = strings.Join(msgLines[startLine:endLine], "\n")

	// A visible window whose last line is the empty string ends in a bare
	// newline — the row is open. The blank-row bookkeeping in View assumes a
	// terminated last row (its "blank row below" newline then closes the
	// open row instead of adding one), which silently dropped a row and left
	// the frame one line shorter than the terminal whenever an exact-fit
	// clip landed on a blank line. Materialize the row with a space, the
	// same cell the pad rows below use.
	if strings.HasSuffix(visible, "\n") {
		visible += " "
	}

	// Pad to fill the viewport. availableHeight is message rows only — the blank
	// rows that inset the block from the rules are budgeted separately.
	// Keep the final padding row materialized. A trailing newline only
	// terminates the preceding row; treating its trailing empty string as a row
	// made the composed frame one line shorter than the terminal.
	//
	// Built in one allocation: `visible += "\n "` in a loop recopies the whole
	// viewport per padding row, which on a tall terminal with a short
	// conversation made this the second-largest allocation site in the process.
	if visibleLineCount := strings.Count(visible, "\n") + 1; visibleLineCount < availableHeight {
		visible += strings.Repeat("\n ", availableHeight-visibleLineCount)
	}
	return visible, startLine, endLine
}

// a2aSidebarEntries flattens the configured A2A agents into sidebar rows, in
// config order. A nil/empty config yields nil, hiding the section.
func a2aSidebarEntries(cfg *config.A2AConfig) []A2AAgentEntry {
	if cfg == nil {
		return nil
	}
	entries := make([]A2AAgentEntry, 0, len(cfg.Agents))
	for _, a := range cfg.Agents {
		entries = append(entries, A2AAgentEntry{Name: a.Name})
	}
	return entries
}

// sidebarRenderInput gathers everything the sidebar draws for this frame.
func (m *model) sidebarRenderInput(sidebarWidth, panelRows int) SidebarRenderInput {
	in := SidebarRenderInput{
		Width: sidebarWidth,
		// Exactly as tall as the panel beside it. Sized to the terminal
		// instead, the sidebar outran the panel — JoinHorizontal padded the
		// panel with blank rows, leaving a gap below the prompt while the
		// sidebar's filler dots carried on past it.
		Height:       panelRows,
		Mascot:       m.mascot(),
		Mode:         m.mode,
		ProviderName: m.providerDisplayName(),
		ModelName:    m.cfg.ModelName,
		// Read from cfg on every render rather than captured once. Nothing
		// changes the level mid-session today — it is fixed at startup from
		// ~/.pirate/config.json — so this is not load-bearing yet. Reading it
		// fresh keeps a future /thinking command from having to remember to
		// republish the sidebar input as well.
		ThinkingLevel: m.cfg.ThinkingLevel,
		AgentName:     m.activeAgent,
		GitBranch:     m.statusModel.GitBranch,
		DiffAdded:     m.diffAdded,
		DiffRemoved:   m.diffRemoved,
		Running:       m.running,
		TokenTracker:  m.cfg.TokenTracker,
		AppVersion:    m.cfg.AppVersion,
		HostName:      cachedHostname(),
		FolderName:    sidebarFolderName(m.cwd()),
		Messages:      m.chatModel.Messages,
		ActiveTool:    m.statusModel.ActiveTool,
		LoadingItems:  m.loadingItems,
		SeaLines:      "",
		StatusLine:    "",
		Orchestrator:  m.cfg.Orchestrator,
		MCPTools:      extension.BuildMCPToolEntries(m.cfg.MCPToolsets),
		A2AAgents:     a2aSidebarEntries(m.cfg.A2A),
		MemoryStatus:  m.memoryStatus,
		Artifacts:     m.artifactList(),
		Palette:       m.palette,
		BgRunning:     m.bgRunning,
		TodoState:     m.todoState,
	}
	if m.run != nil && m.run.phase != "" {
		in.RunChecklist = m.run.checklist
		in.RunPhase = m.run.phase
		in.RunSpec = m.run.specName
		in.RunCycle = m.run.retries + 1
		in.RunMaxCycle = m.run.maxRetries
	}
	if m.mode == "plan" && m.planWorktreePath != "" && m.planTaskName != "" {
		in.PlanPhases = m.currentPlanPhases()
		if g := m.sopGraph("plan"); g != nil {
			in.Graph = &SOPGraph{
				Order:  g.Order,
				Edges:  g.GraphEdges(),
				Status: planStageStatus(in.PlanPhases),
			}
		}
	}
	if in.Graph == nil && m.prAutofix != nil {
		// The engine drives this mode, so its statuses come from the event
		// stream rather than from a projection of some parallel state.
		if g := m.sopGraph("pr-autofix"); g != nil {
			in.Graph = &SOPGraph{
				Order:  g.Order,
				Edges:  g.GraphEdges(),
				Status: m.prAutofix.tracker.statuses(),
			}
		}
	}
	if in.Graph == nil && in.RunPhase != "" {
		if g := m.sopGraph("run"); g != nil {
			in.Graph = &SOPGraph{
				Order:  g.Order,
				Edges:  g.GraphEdges(),
				Status: runStageStatus(g.Order, in.RunPhase),
			}
		}
	}
	return in
}

// currentPlanPhases returns the phase checklist, re-reading the spec artifacts
// only when an agent event may have changed them.
//
// The artifacts on disk stay the source of truth — they survive a resumed plan
// and a restarted TUI, which an event log alone would not — but stat-ing seven
// paths and reading their opening bytes on every keystroke is work the render
// path should not do. Events decide *when* to look, not *what* is true.
func (m *model) currentPlanPhases() []PlanPhase {
	if m.planPhasesStale || m.planPhases == nil {
		specDir := filepath.Join(m.planWorktreePath, "specs", m.planTaskName)
		m.planPhases = detectPlanPhases(specDir)
		m.planPhasesStale = false
	}
	return m.planPhases
}

// invalidatePlanPhases marks the checklist for recomputation. It is called from
// every event that can precede an artifact write — a tool result, a subagent
// event, the end of a turn — because missing one would freeze the checklist,
// which is the bug this whole area just came from.
func (m *model) invalidatePlanPhases() {
	m.planPhasesStale = true
}

// sopGraph returns the compiled SOP named name, compiling it once per session.
//
// Compiling parses and lints the embedded YAML, which is far too much work for
// a function the render path calls on every keystroke — hence the cache. A SOP
// that fails to compile caches nothing and simply hides the diagram; the stage
// list still renders.
func (m *model) sopGraph(name string) *sop.Compiled {
	if m.sopGraphs == nil {
		m.sopGraphs = map[string]*sop.Compiled{}
	}
	if g, ok := m.sopGraphs[name]; ok {
		return g
	}

	var compiled *sop.Compiled
	if def, err := sop.LoadEmbeddedDefinition(name); err == nil {
		if c, err := sop.Compile(def, sop.DescribeFactory{}); err == nil {
			compiled = c
		}
	}
	m.sopGraphs[name] = compiled // nil is cached too: do not retry every frame
	return compiled
}

// drainTerminalResponses discards any pending terminal response sequences
// (e.g. cursor position reports, DECRQM replies) that may arrive after the
// TUI exits. Without this, late responses leak into the shell prompt as garbage
// like "[14;1R[?2026;2$y".
func drainTerminalResponses() {
	f := os.Stdin
	// Switch stdin to non-blocking so we can read without waiting.
	if err := setNonBlock(f); err != nil {
		return
	}
	defer setBlock(f) //nolint:errcheck

	buf := make([]byte, 256)
	deadline := time.Now().Add(50 * time.Millisecond)
	for time.Now().Before(deadline) {
		n, _ := f.Read(buf)
		if n == 0 {
			break
		}
	}
}

const startupProgressBarWidth = 8

// matrixChars is the braille tape of the startup line's wave — the last
// surviving use of the old matrix rain's tape (the sidebar animation is now
// the sea scene in sea.go).
//
// Braille only, and that is a hard requirement rather than a style choice. The
// tape rides on a row whose columns matter, so any glyph that renders at the
// wrong advance width shifts every column after it. Braille (U+2800–U+28FF) is
// East Asian Neutral — one cell in every width table — and is fully covered by
// the monospace fonts pi ships against, so it never falls back to another font
// with a different advance. The earlier tape mixed in ⌒ ⌣ ⌁ ∿ ⊹ ˖ and friends:
// some are East Asian Ambiguous (two cells wherever the terminal is configured
// for CJK), and several are missing from JetBrains Mono entirely, which is
// what drew the "???" clusters over the rules. TestMatrixTapeIsWidthSafe pins
// this.
const matrixChars = "⠀⠀⠁⠂⠄⠠⡀⢀⣀⣠⣤⣶⣷⣿⣿⣿⣷⣶⣤⣠⣀⢀⡀⠠⠄⠂⠁⠃⠇⡇⣇⣧⣷⣿⣷⣧⣇⡇⠇⠃⠐⠈⠈⠐⠠⢀⣠⣴⣶⣷⣾⣿⣿⣾⣷⣶⣴⣠⢀⠠⠐⠈⠀⠁⠉⠙⠹⢹⣹⣽⣿⣽⣹⢹⠹⠙⠉⠁⠀⠁⠂⠄⡀⢀⣀⣠⣤⣶⣿⣶⣤⣠⣀⢀⡀⠄⠂⠁⠀"

// matrixRunes is the precomputed rune slice from matrixChars.
var matrixRunes = []rune(matrixChars)

func renderStartupMatrixLine(phase int, appVersion string, loadingItems map[string]bool, loadingTotal int, p Palette) string {
	versionSuffix := ""
	if appVersion != "" {
		versionSuffix = " " + appVersion
	}
	progress := renderStartupProgress(loadingItems, loadingTotal)
	detail := renderStartupDetail(loadingItems)
	width := 48
	if width < 1 || len(matrixRunes) == 0 {
		return "Loading Pi" + versionSuffix + progress + detail + " .."
	}
	bright := lipgloss.NewStyle().Foreground(p.Teal).Bold(true)
	mid := lipgloss.NewStyle().Foreground(p.Blue)
	dim := lipgloss.NewStyle().Foreground(p.Surface1)
	accent := lipgloss.NewStyle().Foreground(p.Mauve).Bold(true)

	dotCount := 2 + phase%3
	wave := phase % (2 * (width - 1))
	if wave >= width {
		wave = 2*(width-1) - wave
	}

	var b strings.Builder
	b.WriteString(accent.Render("Loading Pi" + versionSuffix + progress + detail + strings.Repeat(".", dotCount)))
	b.WriteString(" ")
	for i := 0; i < width; i++ {
		r := matrixRunes[(i+phase*7)%len(matrixRunes)]
		delta := i - wave
		if delta < 0 {
			delta = -delta
		}
		switch {
		case delta == 0:
			b.WriteString(bright.Render(string(r)))
		case delta <= 2:
			b.WriteString(mid.Render(string(r)))
		default:
			b.WriteString(dim.Render(string(r)))
		}
	}
	return b.String()
}

func renderStartupProgress(loadingItems map[string]bool, loadingTotal int) string {
	if loadingItems == nil {
		return ""
	}

	done := 0
	for _, itemDone := range loadingItems {
		if itemDone {
			done++
		}
	}

	total := loadingTotal
	if total < len(loadingItems) {
		total = len(loadingItems)
	}
	if total < 1 {
		return fmt.Sprintf(" [%s 0%%]", barGlyphs(0, startupProgressBarWidth))
	}

	pct := min(done*100/total, 100)

	// Any progress at all claims a block. Init finishes a long tail of small
	// items, and a bar that reads empty for the first few of them looks stuck.
	filled := done * startupProgressBarWidth / total
	if done > 0 && filled == 0 {
		filled = 1
	}

	return fmt.Sprintf(" [%s %d%% %d/%d]", barGlyphs(filled, startupProgressBarWidth), pct, done, total)
}

func renderStartupDetail(loadingItems map[string]bool) string {
	if loadingItems == nil {
		return ""
	}
	if len(loadingItems) == 0 {
		return " starting init pipeline"
	}

	var pending []string
	for _, item := range sortedKeys(loadingItems) {
		if !loadingItems[item] {
			pending = append(pending, item)
		}
	}
	if len(pending) == 0 {
		return " finalizing init"
	}
	return " working: " + strings.Join(pending, ", ")
}

func (m *model) applyResize() {
	// Resolve the palette first: UpdateRenderer picks the glamour stylesheet
	// from ChatModel.Palette, and on the first resize View has not run yet, so
	// without this the renderer would be built from the zero palette.
	m.syncPalette()

	// The chat viewport is sized to the panel minus the rail, which owns the
	// last column. The status bar spans the full terminal width — it sits below
	// the sidebar, not beside it — so it gets m.width, not chatWidth.
	chatWidth := m.chatWidth()
	m.statusModel.Width = m.width
	if m.chatModel.Width != chatWidth {
		m.chatModel.UpdateRenderer(chatWidth)
	}
	m.clampScroll()
	// Pre-render or reflow the sea scene so width changes are visible
	// immediately. It sits in the top bar, sized to the panel like everything
	// else there.
	if !m.sea.active {
		m.sea.feed("ahoy", chatWidth)
	} else {
		m.sea.tick(chatWidth)
	}
	// Scene height can affect the message viewport, so clamp again after it updates.
	m.clampScroll()
	// The input's row cap follows the new terminal height (a third of it),
	// so a resize re-bounds how much prompt the layout can spend.
	m.inputModel.SetMaxHeight(max(1, m.height/3))
	// The search popup exposes a fixed number of item rows.  Recompute that
	// here so a resize keeps the highlighted row inside the new window —
	// otherwise the Up/Down math still trusts the old budget and the
	// selection can land off-screen.
	m.refreshSearchPopupHeight()
}

func (m *model) clampScroll() {
	maxScroll := m.chatModel.MaxScroll(m.messageViewportHeight())
	if m.chatModel.Scroll > maxScroll {
		m.chatModel.Scroll = maxScroll
	}
	if m.chatModel.Scroll < 0 {
		m.chatModel.Scroll = 0
	}
}

func (m *model) messageViewportHeight() int {
	if m.statusModel.Width != m.width {
		m.statusModel.Width = m.width
	}
	statusBar := m.statusModel.Render(m.statusRenderInput())
	inputArea := m.inputModel.View(m.loading)
	bottomLine := m.bottomStatusLine(m.width)
	statusLines := strings.Count(statusBar, "\n") + 1
	inputLines := strings.Count(inputArea, "\n") + 1
	bottomLines := strings.Count(bottomLine, "\n") + 1
	// The chrome around the messages: the panel's closing rule plus the two rules
	// that frame the input below it, and the status line under the prompt. The
	// two blank rows that inset the messages from those rules are not message
	// rows either. Counting them as such made the panel one row taller than the
	// terminal, so the terminal scrolled the frame and tore the panel away from
	// the sidebar.
	availableHeight := m.height - statusLines - inputLines - bottomLines - 3 - 2
	if m.sea.visible() {
		// The sea scene adds its rows plus two framing rules above the messages.
		// visible() answers the same question as render() != "" without
		// building the scene: render() assembled the whole animation purely
		// for this check, and View renders it again a few lines later.
		availableHeight -= seaLines + 2
	}
	if m.branchPopup != nil {
		availableHeight -= m.branchPopup.height + 6
	}
	if availableHeight < 1 {
		return 1
	}
	return availableHeight
}

// resizeDraining returns true for a short window after a terminal resize,
// during which key and paste input is suppressed to let terminal response
// sequences (OSC color replies, DECRPM, cursor position reports) drain.
func (m *model) resizeDraining() bool {
	return !m.resizeAt.IsZero() && time.Since(m.resizeAt) < 150*time.Millisecond
}

type resizeDrainDoneMsg struct {
	resizeAt time.Time
}

func resizeDrainDoneCmd(resizeAt time.Time) tea.Cmd {
	if resizeAt.IsZero() {
		return nil
	}
	return tea.Tick(150*time.Millisecond, func(time.Time) tea.Msg {
		return resizeDrainDoneMsg{resizeAt: resizeAt}
	})
}

func isResizeTextFragment(msg tea.KeyPressMsg) bool {
	return msg.Key().Text != ""
}

// mainWidth returns the width of the main panel (excluding sidebar).
func (m *model) mainWidth() int {
	if m.width <= 0 {
		return 1
	}
	sw := sidebarWidth(m.width)
	chatW := m.width - sw
	// Show the sidebar only when the chat panel has at least 60 columns — wide
	// enough for readable message text. Below that threshold the sidebar is
	// hidden and the full terminal width belongs to the chat.
	if chatW >= 60 {
		return chatW
	}
	return m.width
}

// chatWidth is the width of the left panel's content — everything in it, not
// just messages: the panel minus the rail on its right edge. Content plus rail
// always add up to mainWidth.
func (m *model) chatWidth() int {
	return max(1, m.mainWidth()-railWidth)
}

// topSectionRows returns the number of rows occupied by the top section — the
// messages area plus the sidebar beside it, ending at the horizontal rule above
// the status bar. The status bar and input below span the full width without a
// rail or sidebar. The value is set during View(); call it after a render.
func (m *model) topSectionRows() int {
	return m.topSectionRowsVal
}

func (m *model) eyes() string {
	if m.face != nil {
		return m.face.Eyes()
	}
	return MoodIdle.Eyes()
}

func (m *model) mascot() string {
	if m.face != nil {
		return m.face.Mascot()
	}
	return MoodIdle.Mascot()
}

// artifactList returns the artifacts attached to the current session,
// formatted for the sidebar. Returns nil when no artifact service is
// wired — the sidebar treats nil/empty identically and renders nothing.
//
// This is a stub: the artifact service is plumbed through agent.Config
// in the image-paste feature; once that lands, this method reads the
// service off the runner and translates List+Load responses into
// []ArtifactEntry.
func (m *model) artifactList() []ArtifactEntry {
	return nil
}

// refreshDiffStats updates the git diff line counts.
func (m *model) refreshDiffStats() {
	cwd := m.cwd()
	if cwd == "" {
		return
	}
	cmd := exec.Command("git", "diff", "--numstat", "HEAD")
	cmd.Dir = cwd
	out, err := cmd.Output()
	if err != nil {
		return
	}
	var added, removed int
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		if line == "" {
			continue
		}
		var a, r int
		if _, err := fmt.Sscanf(line, "%d\t%d\t", &a, &r); err == nil {
			added += a
			removed += r
		}
	}
	added += countUntrackedLines(cwd)
	m.diffAdded = added
	m.diffRemoved = removed
}

// countUntrackedLines counts total lines across untracked files.
func countUntrackedLines(cwd string) int {
	cmd := exec.Command("git", "ls-files", "--others", "--exclude-standard")
	cmd.Dir = cwd
	out, err := cmd.Output()
	if err != nil {
		return 0
	}
	total := 0
	for _, file := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		if file == "" {
			continue
		}
		wc := exec.Command("wc", "-l", file)
		wc.Dir = cwd
		wcOut, err := wc.Output()
		if err != nil {
			continue
		}
		var lines int
		if _, err := fmt.Sscanf(strings.TrimSpace(string(wcOut)), "%d", &lines); err == nil {
			total += lines
		}
	}
	return total
}

// statusRenderInput builds the StatusRenderInput from the current model state.
func (m *model) statusRenderInput() StatusRenderInput {
	var rc *runCycleInfo
	if m.run != nil && m.run.phase != "done" && m.run.phase != "failed" {
		rc = &runCycleInfo{
			SpecName:   m.run.specName,
			Cycle:      m.run.retries + 1,
			MaxRetries: m.run.maxRetries,
		}
	}
	mode := m.mode
	if mode == "" {
		mode = "chat"
	}
	hostName := cachedHostname()
	return StatusRenderInput{
		ProviderName:  m.providerDisplayName(),
		ModelName:     m.cfg.ModelName,
		Running:       m.running,
		Pending:       len(m.pendingPrompts),
		Mode:          mode,
		Eyes:          m.eyes(),
		Messages:      m.chatModel.Messages,
		TokenTracker:  m.cfg.TokenTracker,
		DiffAdded:     m.diffAdded,
		DiffRemoved:   m.diffRemoved,
		RunCycle:      rc,
		FolderName:    sidebarFolderName(m.cwd()),
		HostName:      hostName,
		LoadingItems:  m.loadingItems,
		Flash:         m.flash,
		Palette:       m.palette,
		RunningAgents: m.runningSubagentCount(),
		Background:    m.backgroundStatus(),
	}
}

// providerDisplayName returns the provider label shown in the status bar
// and sidebar. When the configured provider is "openai" but the loaded
// OPENAI_API_KEY is actually a codex ChatGPT OAuth token (a JWT with the
// OpenAI auth claim), "codex" is shown instead so the user knows which
// credential is active.
func (m *model) providerDisplayName() string {
	name := m.cfg.ProviderName
	if strings.EqualFold(name, "openai") && auth.IsCodexOAuthToken(os.Getenv("OPENAI_API_KEY")) {
		return "codex"
	}
	return name
}

// detectBranch returns the current git branch name, or empty string.
func detectBranch(workDir string) string {
	cmd := exec.Command("git", "rev-parse", "--abbrev-ref", "HEAD")
	if workDir != "" {
		cmd.Dir = workDir
	}
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

// handleBranchSelect switches to the selected branch.
func (m *model) handleBranchSelect() (tea.Model, tea.Cmd) {
	if m.branchPopup == nil || len(m.branchPopup.branches) == 0 {
		m.popOverlay(overlayBranchPopup)
		return m, nil
	}

	selectedBranch := m.branchPopup.branches[m.branchPopup.selected]

	// Don't switch if already on this branch
	if selectedBranch == m.branchPopup.active {
		m.popOverlay(overlayBranchPopup)
		return m, nil
	}

	cwd := m.cwd()

	// Run git checkout in the background
	cmd := exec.Command("git", "checkout", selectedBranch)
	if cwd != "" {
		cmd.Dir = cwd
	}

	err := cmd.Run()
	if err != nil {
		m.chatModel.AppendWarning(fmt.Sprintf("Failed to switch branch: %v", err))
	} else {
		m.statusModel.GitBranch = selectedBranch
		m.refreshDiffStats()
	}

	m.popOverlay(overlayBranchPopup)
	return m, nil
}

// resetCtrlCCount is a tea.Cmd that resets the Ctrl+C counter after a delay.
func resetCtrlCCount(m *model) tea.Cmd {
	return func() tea.Msg {
		time.Sleep(2 * time.Second)
		return resetCtrlCCountMsg{}
	}
}

// msgResetCtrlCCount resets the Ctrl+C counter.
type resetCtrlCCountMsg struct{}

// loadingTickMsg advances the loading dots animation.
type loadingTickMsg struct{}

func (m *model) handleResetCtrlCCount() (tea.Model, tea.Cmd) {
	m.ctrlCCount = 0
	return m, nil
}

// resetEscCancelCount is a tea.Cmd that resets the double-Esc counter after a
// delay. It is deliberately a separate msg type from resetCtrlCCountMsg: the
// two counters guard different actions, and a shared type would reset both.
func resetEscCancelCount(m *model) tea.Cmd {
	m.escResetSeq++
	seq := m.escResetSeq
	return func() tea.Msg {
		time.Sleep(2 * time.Second)
		return resetEscCancelCountMsg{seq: seq}
	}
}

// resetEscCancelCountMsg resets the double-Esc cancel counter. It carries the
// sequence number of the warn window it belongs to, so a timer fired for a
// superseded window is ignored.
type resetEscCancelCountMsg struct{ seq int }

func (m *model) handleResetEscCancelCount(msg resetEscCancelCountMsg) (tea.Model, tea.Cmd) {
	// Ignore a timer whose warn window has already been superseded.
	if msg.seq == m.escResetSeq {
		m.escCancelCount = 0
	}
	return m, nil
}

// --- Deferred initialization ---

// initEventMsg wraps an InitEvent received from the deferred init channel.
type initEventMsg struct {
	event InitEvent
	ch    <-chan InitEvent
}

// waitForInitEvent returns a Cmd that reads the next event from the init channel.
func waitForInitEvent(ch <-chan InitEvent) tea.Cmd {
	return func() tea.Msg {
		ev, ok := <-ch
		if !ok {
			return initEventMsg{event: InitEvent{Err: fmt.Errorf("init channel closed unexpectedly")}, ch: ch}
		}
		return initEventMsg{event: ev, ch: ch}
	}
}

// handleInitEvent processes deferred initialization progress.
func (m *model) handleInitEvent(msg initEventMsg) (tea.Model, tea.Cmd) {
	ev := msg.event

	if ev.Err != nil {
		m.loading = false
		m.loadingItems = nil
		m.loadingTotal = 0
		m.loadingDots = 0
		m.initErr = ev.Err
		return m, tea.Quit
	}

	if ev.Total > 0 {
		m.loadingTotal = ev.Total
	}

	// Track item progress.
	if ev.Item != "" {
		if m.loadingItems == nil {
			m.loadingItems = make(map[string]bool)
		}
		m.loadingItems[ev.Item] = ev.Done
	}

	// Final result: apply all initialized subsystems.
	if ev.Result != nil {
		m.loading = false
		m.loadingItems = nil
		m.loadingTotal = 0
		m.loadingDots = 0

		r := ev.Result
		m.cfg.Agent = r.Agent
		m.cfg.SessionID = r.SessionID
		// Seed the terminal window/tab title with the default (git repo / CWD
		// basename) so the OSC 0 sequence on the next frame already carries a
		// sensible label — before the user types the first prompt. The prompt
		// later overwrites it via applySessionTitle, so this only matters for
		// the pre-prompt frames.
		if m.sessionTitle == "" && r.SessionTitle != "" {
			m.sessionTitle = r.SessionTitle
		}
		m.cfg.SessionService = r.SessionService
		m.cfg.Orchestrator = r.Orchestrator
		m.cfg.Logger = r.Logger
		if r.Logger != nil {
			auth.SetDebugLogger(func(msg string) { r.Logger.Info("auth: " + msg) })
		} else {
			auth.SetDebugLogger(nil)
		}
		// Hook failures must never reach stderr while the TUI holds the
		// alternate screen. Installed unconditionally: with no logger the sink
		// swallows the message, which still beats corrupting the UI.
		extension.SetHookLogger(func(msg string) { r.Logger.Error("hook: " + msg) })
		m.cfg.Skills = r.Skills
		m.cfg.SkillDirs = r.SkillDirs
		m.cfg.GenerateCommitMsg = r.GenerateCommitMsg
		m.cfg.AgentEventCh = r.AgentEventCh
		// Remembered before the assignment: a front end that handed the same
		// channel to the initial config already has a reader waiting on it,
		// and arming a second would leave two goroutines blocked on one
		// channel for the rest of the session.
		prevNoticeCh := m.cfg.SystemNoticeCh
		m.cfg.SystemNoticeCh = r.SystemNoticeCh
		m.cfg.ContextBreakdown = r.ContextBreakdown
		m.cfg.TokenTracker = r.TokenTracker
		m.cfg.CompactMetrics = r.CompactMetrics
		m.statusModel.GitBranch = r.GitBranch
		m.diffAdded = r.DiffAdded
		m.diffRemoved = r.DiffRemoved
		m.cfg.MCPToolsets = r.MCPToolsets
		m.cfg.MCPServers = r.MCPServers
		// defaultAgent: deferred init may have restarted the session on the
		// agent's own model and handed over the switchable-agent list.
		if r.LLM != nil {
			m.cfg.LLM = r.LLM
			if r.ModelName != "" {
				m.cfg.ModelName = r.ModelName
				m.cfg.ProviderName = r.ProviderName
			}
		}
		if r.ActiveAgent != "" {
			m.activeAgent = r.ActiveAgent
		}
		m.cfg.PrimaryAgents = r.PrimaryAgents

		// Update input model with loaded skills.
		m.inputModel.Skills = r.Skills
		m.inputModel.SkillDirs = r.SkillDirs

		// Rebuild the transcript last: it reads the session service and token
		// tracker that were just installed above.
		if r.Resumed {
			m.restoreSession()
		}

		var cmds []tea.Cmd
		if r.AgentEventCh != nil {
			cmds = append(cmds, waitForSubEvent(r.AgentEventCh))
		}
		if r.SystemNoticeCh != nil && r.SystemNoticeCh != prevNoticeCh {
			cmds = append(cmds, waitForSystemNotice(r.SystemNoticeCh))
		}
		cmds = append(cmds, memoryTickCmd(m.cwd(), m.cfg.Palace))
		return m, tea.Batch(cmds...)
	}

	// Keep reading init events.
	return m, waitForInitEvent(msg.ch)
}

func (m *model) shouldShowSlashCommandPopup() bool {
	if m.running || m.loading || m.login != nil || m.commit != nil || m.pendingSkillCreate != nil {
		return false
	}
	text := m.inputModel.Text
	return strings.HasPrefix(text, "/") && !strings.ContainsAny(text, " \t\n\r")
}

func (m *model) overlaySearchPopup(messages string, mainWidth int) string {
	popup := m.renderSearchPopup(max(0, mainWidth-4))
	if popup == "" {
		return messages
	}

	lines := strings.Split(messages, "\n")
	viewportHeight := len(lines)

	popupLines := strings.Split(popup, "\n")
	if viewportHeight > 0 && len(popupLines) > viewportHeight {
		popupLines = popupLines[:viewportHeight]
	}
	if len(popupLines) == 0 {
		return strings.Join(lines, "\n")
	}

	popupWidth := maxLineWidth(popupLines)
	left := 0
	if mainWidth > popupWidth {
		left = (mainWidth - popupWidth) / 2
	}
	start := 0
	if viewportHeight > len(popupLines) {
		start = viewportHeight - len(popupLines) - 1
	}
	if start < 0 {
		start = 0
	}

	for i, line := range popupLines {
		idx := start + i
		if idx >= len(lines) {
			break
		}
		lines[idx] = overlayPopupLine(line, left, mainWidth)
	}
	return strings.Join(lines, "\n")
}

func maxLineWidth(lines []string) int {
	width := 0
	for _, line := range lines {
		width = max(width, lipgloss.Width(line))
	}
	return width
}

func overlayPopupLine(line string, left int, totalWidth int) string {
	if totalWidth <= 0 {
		return line
	}
	if left < 0 {
		left = 0
	}
	prefix := strings.Repeat(" ", left)
	rendered := prefix + line
	if width := lipgloss.Width(rendered); width < totalWidth {
		rendered += strings.Repeat(" ", totalWidth-width)
	}
	return rendered
}

func (m *model) renderSearchPopup(width int) string {
	if m.searchPopup == nil {
		return ""
	}
	if width < 24 {
		width = 24
	}

	sp := m.searchPopup
	st := m.searchPopupStyles(sp.mode, width)

	var b strings.Builder

	// Header with count.
	header := st.header
	if len(sp.filtered) > 0 {
		header = fmt.Sprintf("%s (%d)", header, len(sp.filtered))
	}
	b.WriteString(st.headerStyle.Render(header))

	// Search prompt line.
	if sp.search != "" {
		b.WriteString("\n")
		searchLine := fmt.Sprintf("  Search: %s", sp.search)
		b.WriteString(st.searchStyle.Width(width).Render(clipRunes(searchLine, width)))
	} else {
		b.WriteString("\n")
		b.WriteString(st.searchStyle.Width(width).Render("  Search... (type to filter)"))
	}

	if len(sp.filtered) == 0 {
		b.WriteString("\n")
		msg := "  No matching " + string(sp.mode) // "commands", "history", "models", "agents"
		b.WriteString(st.searchStyle.Width(width).Render(msg))
		b.WriteString("\n")
		b.WriteString(st.searchStyle.Width(width).Render(clipRunes(searchPopupFooter(sp.mode), width)))
		return st.popupStyle.Render(b.String())
	}

	writeSearchPopupItems(&b, sp, st, width)

	// The steer mini-input renders below the list; the cursor is a plain "▏"
	// — the input is append-only with backspace, no real caret.
	if m.steerInput != nil {
		line := fmt.Sprintf("  Steer %s: %s▏  (Enter send · Esc cancel)",
			agentTitleFit(m.steerInput.label, 32), m.steerInput.text)
		b.WriteString("\n")
		b.WriteString(st.searchStyle.Width(width).Render(clipRunes(line, width)))
	}

	// Footer: only the keys this mode actually answers. PageUp/PageDown and
	// Home/End are not bound in the popup, so they are not advertised;
	// Backspace is obvious.
	b.WriteString("\n")
	b.WriteString(st.searchStyle.Width(width).Render(clipRunes(searchPopupFooter(sp.mode), width)))

	return st.popupStyle.Render(b.String())
}

// searchPopupFooter is the popup's key-hint line for a mode: only keys that
// actually work there. Todos is view-only, so it promises no Enter; the
// subagent monitor adds its `s` steer key.
func searchPopupFooter(mode searchMode) string {
	switch mode {
	case searchModeTodos:
		return "  ↑/↓ move · Esc close"
	case searchModeSubagents:
		return "  ↑/↓ move · Enter view · s steer · Esc close"
	case searchModeCommands:
		return "  ↑/↓ move · Enter run · Esc close"
	case searchModeHelp:
		return "  ↑/↓ move · Enter run command · Esc close"
	case searchModeHistory:
		return "  ↑/↓ move · Enter insert · Esc close"
	case searchModeFiles:
		return "  ↑/↓ move · Enter insert · Esc close"
	case searchModeModels, searchModeAgents:
		return "  ↑/↓ move · Enter switch · Esc close"
	default:
		return "  ↑/↓ move · Enter select · Esc close"
	}
}

// searchPopupStyleSet is the per-mode palette of the search popup: the same
// chrome with a different accent color and title for commands and for history.
type searchPopupStyleSet struct {
	header            string
	popupStyle        lipgloss.Style
	headerStyle       lipgloss.Style
	searchStyle       lipgloss.Style
	itemStyle         lipgloss.Style
	selectedItemStyle lipgloss.Style
}

// searchPopupStyles builds the style set for a popup mode. Commands are cyan
// and history is orange; an unknown mode gets the unaccented base styles.
func (m *model) searchPopupStyles(mode searchMode, width int) searchPopupStyleSet {
	bg := m.palette.Surface0
	st := searchPopupStyleSet{
		popupStyle:  lipgloss.NewStyle().Background(bg),
		headerStyle: lipgloss.NewStyle().Background(bg).Bold(true),
		searchStyle: lipgloss.NewStyle().Background(bg),
		itemStyle:   lipgloss.NewStyle().Background(bg),
		// Each mode below sets its own selection background; this is just the
		// starting point.
		selectedItemStyle: lipgloss.NewStyle().Background(m.palette.Surface0),
	}

	var accent color.Color
	switch mode {
	case searchModeCommands:
		accent = m.palette.Cyan                                // cyan for commands
		st.itemStyle = st.itemStyle.Foreground(m.palette.Teal) // teal
		st.header = "Commands"
	case searchModeHistory:
		accent = m.palette.Peach                                // orange for history
		st.itemStyle = st.itemStyle.Foreground(m.palette.Peach) // orange
		st.header = "History"
	case searchModeModels:
		accent = m.palette.Mauve // mauve for models
		st.itemStyle = st.itemStyle.Foreground(m.palette.Lavender)
		st.header = "Models"
	case searchModeAgents:
		accent = m.palette.Green                                // green for agents
		st.itemStyle = st.itemStyle.Foreground(m.palette.Green) // green
		st.header = "Agents"
	case searchModeSubagents:
		accent = m.palette.Blue // blue for the subagent monitor
		st.itemStyle = st.itemStyle.Foreground(m.palette.Text)
		st.header = "Subagents"
	case searchModeTodos:
		accent = m.palette.Green // green for todos
		st.itemStyle = st.itemStyle.Foreground(m.palette.Text)
		st.header = "Todos"
	case searchModeFiles:
		accent = m.palette.Sapphire // sapphire for file mentions
		st.itemStyle = st.itemStyle.Foreground(m.palette.Text)
		st.header = "Files"
	case searchModeHelp:
		accent = m.palette.Yellow // yellow for the help dialog
		st.itemStyle = st.itemStyle.Foreground(m.palette.Text)
		st.header = "Help"
	default:
		return st
	}

	st.popupStyle = st.popupStyle.
		Foreground(m.palette.Subtext).
		Border(lipgloss.RoundedBorder(), true, true, true, true).
		BorderForeground(accent).
		Width(width)
	st.headerStyle = st.headerStyle.Foreground(m.palette.Subtext).Width(width)
	st.searchStyle = st.searchStyle.Foreground(m.palette.Dim)
	// Pair the accent selection background with a foreground derived from the
	// accent's own luminance. The theme's Background cannot serve: themes
	// with "none" fills (opencode) resolve it to NoColor, and the terminal's
	// default foreground — light on the terminals those themes target — is
	// unreadable on the violet selection. Luminance keeps every theme's
	// selection readable without per-theme tables.
	st.selectedItemStyle = st.selectedItemStyle.
		Background(accent).
		Foreground(selectionForeground(accent))
	return st
}

// writeSearchPopupItems renders the visible slice of the item list.
//
// Both i and scrollOff+i are bounded by len(filtered) so a stale scrollOff from
// before a resize cannot index past the end and panic —
// refreshSearchPopupHeight is the primary guard, this is the belt to its braces.
//
// While the filter is empty, the first suggestedN rows are the Suggested
// section: a header line sits above its first row (and scrolls away with it),
// and each suggested row carries the ● current marker instead of the plain
// indent.
func writeSearchPopupItems(b *strings.Builder, sp *searchPopupState, st searchPopupStyleSet, width int) {
	for i := 0; i < sp.height; i++ {
		idx := sp.scrollOff + i
		if idx < 0 || idx >= len(sp.filtered) {
			break
		}
		if idx == 0 && sp.suggestedN > 0 {
			b.WriteString("\n")
			b.WriteString(st.searchStyle.Width(width).Render(clipRunes("  Suggested", width)))
		}
		item := sp.filtered[idx]
		prefix := "  "
		currentItemStyle := st.itemStyle
		switch {
		case idx == sp.selected:
			// Always highlight the selected item.
			prefix = "> "
			currentItemStyle = st.selectedItemStyle
		case idx < sp.suggestedN:
			prefix = "● "
		}

		line := prefix + item.Text
		// Add description for commands, models and agents (history has none).
		if item.Description != "" && sp.mode != searchModeHistory {
			desc := clipRunes(item.Description, width*50/100)
			if desc != "" {
				line += "  " + desc
			}
		}

		b.WriteString("\n")
		b.WriteString(currentItemStyle.Width(width).Render(clipRunes(line, width)))
	}
}

func (m *model) slashCommandCandidates(prefix string) []CompletionCandidate {
	skills := m.inputModel.Skills
	if len(skills) == 0 {
		skills = m.cfg.Skills
	}

	if prefix == "/" {
		return allSlashCommandCandidates(skills)
	}

	workDir := m.inputModel.WorkDir
	if workDir == "" {
		workDir = m.cfg.WorkDir
	}
	result := Complete(prefix, skills, workDir)
	if result == nil || len(result.Candidates) == 0 {
		return nil
	}

	candidates := make([]CompletionCandidate, 0, len(result.Candidates))
	for _, candidate := range result.Candidates {
		if candidate.Type == CompletionTypeCommand || candidate.Type == CompletionTypeSkill {
			candidates = append(candidates, candidate)
		}
	}
	return candidates
}

func allSlashCommandCandidates(skills []extension.Skill) []CompletionCandidate {
	seen := make(map[string]bool)
	candidates := make([]CompletionCandidate, 0, len(slashCommands)+len(skills))
	for _, cmd := range slashCommands {
		if seen[cmd] {
			continue
		}
		seen[cmd] = true
		candidates = append(candidates, CompletionCandidate{
			Text:        cmd,
			Description: slashCommandDesc(cmd),
			Type:        CompletionTypeCommand,
		})
	}
	for _, skill := range skills {
		cmd := "/" + skill.Name
		if seen[cmd] {
			continue
		}
		seen[cmd] = true
		candidates = append(candidates, CompletionCandidate{
			Text:        cmd,
			Description: skill.Description,
			Type:        CompletionTypeSkill,
		})
	}
	sort.Slice(candidates, func(i, j int) bool {
		return strings.ToLower(candidates[i].Text) < strings.ToLower(candidates[j].Text)
	})
	return candidates
}

func clipRunes(s string, width int) string {
	if width <= 0 {
		return ""
	}
	runes := []rune(s)
	if len(runes) <= width {
		return s
	}
	if width <= 3 {
		return string(runes[:width])
	}
	return string(runes[:width-3]) + "..."
}

// renderBranchPopup renders the branch list popup.
func (m *model) renderBranchPopup() string {
	if m.branchPopup == nil {
		return ""
	}

	popup := m.branchPopup
	bg := m.palette.Surface0
	border := m.palette.Surface
	selected := m.palette.Primary
	activeFg := m.palette.Green
	dimFg := m.palette.Faint

	style := lipgloss.NewStyle().
		Background(bg).
		Foreground(m.palette.Subtext).
		Border(lipgloss.ThickBorder(), true, true, true, true).
		BorderForeground(border).
		Width(m.width - 10)

	// Calculate popup position (centered horizontally, near the bottom)
	popupWidth := m.width - 10

	var b strings.Builder
	b.WriteString("\n")

	// Header
	header := lipgloss.NewStyle().
		Background(bg).
		Foreground(m.palette.Subtext).
		Bold(true).
		Width(popupWidth).
		Align(lipgloss.Center).
		Render("Git Branches (Enter to switch, Esc to close)")
	b.WriteString(header)
	b.WriteString("\n")

	// Render visible branches
	branches := popup.branches
	height := popup.height
	scrollOff := popup.scrollOff

	if len(branches) > height {
		branches = branches[scrollOff : scrollOff+height]
	}

	for i, branch := range branches {
		actualIndex := i + scrollOff
		isSelected := actualIndex == popup.selected
		isActive := branch == popup.active

		var line string
		if isActive {
			line = fmt.Sprintf("  ◉ %s (current)", branch)
		} else {
			line = fmt.Sprintf("    %s", branch)
		}

		if isSelected {
			line = "> " + line[2:] // Replace leading spaces with ">"
		}

		var lineStyle lipgloss.Style
		switch {
		case isSelected:
			lineStyle = lipgloss.NewStyle().Background(selected).Foreground(m.palette.White)
		case isActive:
			lineStyle = lipgloss.NewStyle().Background(bg).Foreground(activeFg)
		default:
			lineStyle = lipgloss.NewStyle().Background(bg).Foreground(dimFg)
		}

		b.WriteString(lineStyle.Width(popupWidth).Render(line))
		b.WriteString("\n")
	}

	// Show scroll indicator if needed
	if len(popup.branches) > popup.height {
		scrollStyle := lipgloss.NewStyle().Background(bg).Foreground(dimFg)
		b.WriteString(scrollStyle.Render("  ↑↓ scroll"))
	}

	return style.Render(b.String())
}
