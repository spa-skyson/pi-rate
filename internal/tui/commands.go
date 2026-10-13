package tui

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/spa-skyson/pi-rate/internal/agent"
	"github.com/spa-skyson/pi-rate/internal/config"
	"github.com/spa-skyson/pi-rate/internal/extension"
	pisession "github.com/spa-skyson/pi-rate/internal/session"
	"github.com/spa-skyson/pi-rate/internal/subagent"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
)

func (m *model) handleSlashCommand(input string) (tea.Model, tea.Cmd) {
	parts := strings.Fields(input)
	cmd := strings.ToLower(parts[0])

	// Log all slash commands.
	if m.cfg.Logger != nil {
		m.cfg.Logger.UserMessage(input)
	}

	// Sync the terminal window/tab title and the persisted session metadata
	// with the slash command the user just typed. The next View() will emit
	// OSC 0 carrying the new title (formatTerminalTitle scrubs C0 controls,
	// so the envelope stays well-formed). Skipped for the three commands
	// whose semantics aren't "make this the title":
	//   - /clear  resets the title to the app default
	//   - /exit, /quit  tear the program down before any frame can render
	switch cmd {
	case "/clear", "/exit", "/quit":
	default:
		m.setSessionTitle(input)
	}

	// Dispatch through the central registry: the row carries the identity,
	// description and category; the handler attaches lazily (slashHandlerFor)
	// so the popup-opening commands — /todos, /model, /agent, /subagents —
	// live in the same table without an initialization cycle.
	if binding, ok := registryByCommand[cmd]; ok {
		if h := slashHandlerFor(binding.ID); h != nil {
			return h(m, parts[1:])
		}
	}

	// Check if it's a dynamic skill command.
	skillName := strings.TrimPrefix(cmd, "/")
	if skill, ok := extension.FindSkill(m.cfg.Skills, skillName); ok {
		return m.handleSkillCommand(skill, parts[1:])
	}
	m.chatModel.Messages = append(m.chatModel.Messages, message{
		role:    "assistant",
		content: fmt.Sprintf("Unknown command: `%s`. Type `/help` for available commands.", cmd),
	})

	return m, nil
}

// hasConversationContent reports whether the transcript carries anything a
// user would miss — prompts, replies, errors, reasoning — as opposed to the
// chrome notices a fresh fork already shows.
func hasConversationContent(messages []message) bool {
	for _, msg := range messages {
		if msg.isNotice || msg.isMeta || msg.isWarning {
			continue
		}
		if msg.role == "user" || msg.content != "" {
			return true
		}
	}
	return false
}

// handleNewCommand starts a fresh active session without a run. The current
// one is not discarded: an idle session with content parks in the registry —
// the same ownership move detach performs, minus the loop. Busy refuses (a
// running turn holds its session), an already-empty session says so.
func (m *model) handleNewCommand() (tea.Model, tea.Cmd) {
	if m.running {
		return m, m.setFlash("The active session is running — /new when it finishes")
	}
	if m.mode == "plan" && m.planWorktree != nil {
		return m, m.setFlash("Cannot start a new session while a plan is pending")
	}
	if !hasConversationContent(m.chatModel.Messages) {
		return m, m.setFlash("This session is already empty")
	}
	model, cmd, ok := m.parkActiveSession(true)
	if !ok {
		return model, nil
	}
	return model, cmd
}

// showSessionMessage appends the current session ID as an assistant message.
func (m *model) showSessionMessage() {
	m.chatModel.Messages = append(m.chatModel.Messages, message{
		role:    "assistant",
		content: fmt.Sprintf("Session: `%s`", m.cfg.SessionID),
	})
}

// showContextMessage appends the context usage breakdown as an assistant message.
func (m *model) showContextMessage() {
	m.chatModel.Messages = append(m.chatModel.Messages, message{
		role:    "assistant",
		content: m.formatContextUsage(),
	})
}

// handleQuitCommand tears the program down.
func (m *model) handleQuitCommand() (tea.Model, tea.Cmd) {
	m.quitting = true
	return m, tea.Quit
}

func (m *model) handleCopyCommand() (tea.Model, tea.Cmd) {
	transcript := m.chatModel.PlainTranscript()
	if transcript == "" {
		m.chatModel.Messages = append(m.chatModel.Messages, message{
			role:    "assistant",
			content: "Nothing to copy yet.",
		})
		return m, nil
	}

	m.chatModel.Messages = append(m.chatModel.Messages, message{
		role:    "assistant",
		content: "Copied conversation to clipboard.",
	})
	return m, tea.SetClipboard(transcript)
}

// handleBranchCommand handles /branch subcommands: create, switch, list.
func (m *model) handleBranchCommand(args []string) {
	if m.cfg.SessionService == nil {
		m.chatModel.Messages = append(m.chatModel.Messages, message{
			role:    "assistant",
			content: "Branching not available (no session service configured).",
		})
		return
	}

	if len(args) == 0 {
		m.chatModel.Messages = append(m.chatModel.Messages, message{
			role:    "assistant",
			content: "Usage: `/branch <name>` to create, `/branch switch <name>` to switch, `/branch list` to list.",
		})
		return
	}

	subcmd := strings.ToLower(args[0])

	switch subcmd {
	case "list":
		branches, active, err := m.cfg.SessionService.ListBranches(
			m.cfg.SessionID, agent.AppName, agent.DefaultUserID,
		)
		if err != nil {
			m.chatModel.Messages = append(m.chatModel.Messages, message{
				role:    "assistant",
				content: fmt.Sprintf("Error listing branches: %v", err),
			})
			return
		}
		var b strings.Builder
		b.WriteString("**Branches:**\n")
		for _, br := range branches {
			marker := " "
			if br.Name == active {
				marker = "*"
			}
			fmt.Fprintf(&b, "- %s `%s` (head: %d)\n", marker, br.Name, br.Head)
		}
		m.chatModel.Messages = append(m.chatModel.Messages, message{role: "assistant", content: b.String()})

	case "switch":
		if len(args) < 2 {
			m.chatModel.Messages = append(m.chatModel.Messages, message{
				role:    "assistant",
				content: "Usage: `/branch switch <name>`",
			})
			return
		}
		branchName := args[1]
		err := m.cfg.SessionService.SwitchBranch(
			m.cfg.SessionID, agent.AppName, agent.DefaultUserID, branchName,
		)
		if err != nil {
			m.chatModel.Messages = append(m.chatModel.Messages, message{
				role:    "assistant",
				content: fmt.Sprintf("Error switching branch: %v", err),
			})
			return
		}
		m.chatModel.Messages = append(m.chatModel.Messages, message{
			role:    "assistant",
			content: fmt.Sprintf("Switched to branch `%s`.", branchName),
		})

	default:
		// Treat as branch name to create.
		branchName := subcmd
		err := m.cfg.SessionService.CreateBranch(
			m.cfg.SessionID, agent.AppName, agent.DefaultUserID, branchName,
		)
		if err != nil {
			m.chatModel.Messages = append(m.chatModel.Messages, message{
				role:    "assistant",
				content: fmt.Sprintf("Error creating branch: %v", err),
			})
			return
		}
		// Auto-switch to the new branch.
		_ = m.cfg.SessionService.SwitchBranch(
			m.cfg.SessionID, agent.AppName, agent.DefaultUserID, branchName,
		)
		m.chatModel.Messages = append(m.chatModel.Messages, message{
			role:    "assistant",
			content: fmt.Sprintf("Created and switched to branch `%s`.", branchName),
		})
	}
}

// clearConversation resets the transcript and the session context behind it.
//
// This is the body of /clear, split out so the context gauge's [x] button runs
// exactly this path. A second copy of these fifteen assignments would drift the
// first time one of them changed, and the failure mode is silent: a transcript
// that looks cleared while the model still receives every prior turn.
func (m *model) clearConversation() {
	m.chatModel.Messages = m.chatModel.Messages[:0]
	m.chatModel.Scroll = 0
	m.chatModel.Streaming = ""
	m.chatModel.Thinking = ""
	m.chatModel.TraceLog = nil
	m.running = false
	m.run = nil
	m.statusModel.ActiveTool = ""
	m.statusModel.ToolStart = time.Time{}
	m.loadingItems = nil
	m.sea.clear()
	m.sea.feed("ahoy", m.mainWidth())
	// Drop the retry offer with the transcript: /retry after /clear would
	// otherwise re-send a prompt whose turn is no longer on screen.
	m.lastPrompt = ""
	m.lastMentions = nil
	m.lastPromptFailed = false
	// Drop the session's events and zero the context gauge. Without this the
	// transcript looks empty while the model still receives — and still pays
	// for — every prior turn.
	m.clearSessionContext()
	// Reset the terminal window/tab title and the persisted session metadata
	// title. Funnel through setSessionTitle("") so the agent's SetSessionTitle
	// is also called and meta.json stays in sync with the OSC 0 frame that the
	// next View() will emit.
	m.setSessionTitle("")
}

// clearSessionContext empties the session's event history and zeroes the
// context gauge, so /clear resets what the model sees rather than only what
// the user sees.
//
// Order matters: the gauge is only zeroed once the events are actually gone.
// If ClearEvents fails the history survives, and a zeroed gauge would then
// under-report a still-full window — a worse lie than the stale reading it
// replaced. So on error the failure is surfaced in the chat and the tracker is
// left alone.
//
// Both dependencies are optional: a model can be built without a session
// service (nothing to clear) or without a token tracker (no gauge to zero).
func (m *model) clearSessionContext() {
	if m.cfg.SessionService != nil {
		if err := m.cfg.SessionService.ClearEvents(
			m.cfg.SessionID, agent.AppName, agent.DefaultUserID,
		); err != nil {
			m.chatModel.Messages = append(m.chatModel.Messages, message{
				role:    "assistant",
				content: fmt.Sprintf("Context not cleared: %v", err),
			})
			return
		}
	}

	if tt := m.cfg.TokenTracker; tt != nil {
		tt.ResetContextWindow()
	}
}

// handleCompactCommand triggers session compaction.
//
// After a successful compact, the gauge is updated immediately with the
// post-compaction token count so the user sees the window shrink on the same
// keystroke, rather than waiting for the next LLM response. Without this
// push the gauge keeps reading the pre-compaction number — visually
// indistinguishable from no compaction at all.
func (m *model) handleCompactCommand() {
	if m.cfg.SessionService == nil {
		m.chatModel.Messages = append(m.chatModel.Messages, message{
			role:    "assistant",
			content: "Compaction not available (no session service configured).",
		})
		return
	}

	err := m.cfg.SessionService.Compact(
		m.cfg.SessionID, agent.AppName, agent.DefaultUserID,
		pisession.SimpleSummarizer, pisession.CompactConfig{},
	)
	if err != nil {
		m.chatModel.Messages = append(m.chatModel.Messages, message{
			role:    "assistant",
			content: fmt.Sprintf("Compaction error: %v", err),
		})
		return
	}

	// Refresh the gauge. EstimateTokens is the same chars/4 heuristic the
	// session uses internally, so it matches what the next LLM response will
	// report up to that fuzz. The token-tracker's cache prefix is also reset
	// by SetLastPromptTokens, since the new window starts a fresh prefix.
	if tt := m.cfg.TokenTracker; tt != nil {
		if est, estErr := m.cfg.SessionService.EstimateTokens(
			m.cfg.SessionID, agent.AppName, agent.DefaultUserID,
		); estErr == nil {
			tt.SetLastPromptTokens(int64(est))
		}
	}

	m.chatModel.Messages = append(m.chatModel.Messages, message{
		role:    "assistant",
		content: "Session context compacted.",
	})
}

// countAgentsByStatus counts agents in each status category.
func countAgentsByStatus(agents []subagent.AgentStatus) (running, done, failed int) {
	for _, a := range agents {
		switch a.Status {
		case "running":
			running++
		case "completed":
			done++
		case "failed":
			failed++
			// "killed" and "canceled" are not counted as failed
		}
	}
	return
}

// agentStatusIcon returns a display icon for an agent status.
func agentStatusIcon(status string) string {
	switch status {
	case "running":
		return "▶ "
	case "completed":
		return "✓ "
	case "failed":
		return "✗ "
	case "canceled":
		return "◼ "
	case "killed":
		return "⚠ "
	default:
		return "  "
	}
}

// formatAgentsList formats a list of agents for display.
func formatAgentsList(agents []subagent.AgentStatus) string {
	if len(agents) == 0 {
		return "No subagents have been spawned yet."
	}

	running, done, failed := countAgentsByStatus(agents)

	var b strings.Builder
	fmt.Fprintf(&b, "**Subagents** — %d total, %d running, %d done", len(agents), running, done)
	if failed > 0 {
		fmt.Fprintf(&b, ", %d failed", failed)
	}
	b.WriteString("\n\n")

	for _, a := range agents {
		icon := agentStatusIcon(a.Status)

		prompt := a.Prompt
		if len(prompt) > 70 {
			prompt = prompt[:67] + "..."
		}

		dur := a.Duration
		if dur == "" {
			dur = time.Since(a.StartedAt).Truncate(time.Second).String()
		}

		fmt.Fprintf(&b, "%s `%s` **%s** [%s] %s (%s)\n", icon, a.AgentID[:8], a.Type, a.Status, prompt, dur)
	}
	return b.String()
}

// handleAgentsCommand shows the status of running and recent subagents.
func (m *model) handleAgentsCommand() {
	if m.cfg.Orchestrator == nil {
		m.chatModel.Messages = append(m.chatModel.Messages, message{
			role:    "assistant",
			content: "Subagent system not available.",
		})
		return
	}

	m.chatModel.Messages = append(m.chatModel.Messages, message{
		role:    "assistant",
		content: formatAgentsList(m.cfg.Orchestrator.List()),
	})
}

// modelCandidatesMsg carries the background catalog refresh for an open
// models popup. nil items mean "nothing new" (fetch failed, nothing
// configured) and leave the popup's list alone.
type modelCandidatesMsg struct{ items []SearchItem }

// openModelsPopup opens the /model picker and kicks off the background
// catalog refresh. The refresh runs only while the popup is open — its result
// is dropped if the user has already moved on. A nil refresh source simply
// skips the fetch; the popup shows the declared list on its own.
func (m *model) openModelsPopup() tea.Cmd {
	m.newSearchPopup(searchModeModels)
	if m.cfg.ModelCandidatesRefresh == nil {
		return nil
	}
	refresh, ctx := m.cfg.ModelCandidatesRefresh, m.ctx
	return func() tea.Msg {
		return modelCandidatesMsg{items: refresh(ctx)}
	}
}

// handleModelCandidates applies a refreshed candidate list to an open models
// popup: entries are replaced, the current search query re-filtered, and the
// visible window recomputed for the new count.
func (m *model) handleModelCandidates(msg modelCandidatesMsg) (tea.Model, tea.Cmd) {
	if len(msg.items) == 0 || m.searchPopup == nil || m.searchPopup.mode != searchModeModels {
		return m, nil
	}
	sp := m.searchPopup
	markActiveRole(msg.items, m.cfg.ActiveRole)
	sp.entries = msg.items
	sp.suggested = m.suggestedItems(sp.mode)
	sp.filterSearch()
	m.refreshSearchPopupHeight()
	return m, nil
}

// handleModelCommand handles /model <name>: switch to the named model (or
// role if name matches a role). The argument-less popup is routed by
// handleSlashCommand (see the note there); direct calls with no argument are
// a no-op.
func (m *model) handleModelCommand(args []string) (tea.Model, tea.Cmd) {
	if len(args) == 0 {
		return m, nil
	}

	target := strings.TrimSpace(args[0])

	// Resolve: if target is a configured role name, use that role's model.
	var modelName, roleName string
	if rc, ok := m.cfg.Roles[target]; ok {
		modelName = rc.Model
		roleName = target
	} else {
		modelName = target
	}

	if modelName == "" {
		m.chatModel.Messages = append(m.chatModel.Messages, message{
			role:    "assistant",
			content: fmt.Sprintf("Role `%s` has no model configured.", target),
		})
		return m, nil
	}

	if m.cfg.ModelSwitcher == nil {
		m.chatModel.Messages = append(m.chatModel.Messages, message{
			role:    "assistant",
			content: "Model switching is not available in this context.",
		})
		return m, nil
	}

	if m.running {
		m.chatModel.Messages = append(m.chatModel.Messages, message{
			role:    "assistant",
			content: "Cannot switch model while a response is running. Wait for it to finish or cancel it first.",
		})
		return m, nil
	}
	// No background gate here: the slot freezes its own agent while its loop
	// lives, and RebuildWithModel below targets only the active slot's agent.

	newLLM, newName, newProvider, err := m.cfg.ModelSwitcher(m.ctx, modelName)
	if err != nil {
		m.chatModel.Messages = append(m.chatModel.Messages, message{
			role:    "assistant",
			content: fmt.Sprintf("Failed to switch model: %s", err),
		})
		return m, nil
	}

	// Rebuild the agent with the new LLM.
	if m.cfg.Agent != nil {
		if err := m.cfg.Agent.RebuildWithModel(newLLM); err != nil {
			m.chatModel.Messages = append(m.chatModel.Messages, message{
				role:    "assistant",
				content: fmt.Sprintf("Failed to rebuild agent: %s", err),
			})
			return m, nil
		}
	}

	// Update TUI state.
	m.cfg.LLM = newLLM
	m.cfg.ModelName = newName
	m.cfg.ProviderName = newProvider

	// A deliberate user-driven switch resets the fallback session: the chain
	// was chosen against the previous model, and the user just said this one
	// works.
	m.modelFallbacks = nil
	m.modelFallbackUsed = 0

	var sb strings.Builder
	if m.activeAgent != "" {
		// Session-only override for the active primary agent: the model
		// belongs to this agent, not to the default role. Persisting here
		// would rewrite the default role behind the user's back, so the
		// choice lives in memory and re-applies on every switch back to the
		// agent.
		m.setAgentModelOverride(m.activeAgent, newName)
		fmt.Fprintf(&sb, "Model for agent **%s** switched to **%s** (provider: %s).",
			m.activeAgent, newName, newProvider)
		m.chatModel.Messages = append(m.chatModel.Messages, message{
			role:    "assistant",
			content: sb.String(),
		})
		return m, nil
	}

	if roleName != "" {
		m.cfg.ActiveRole = roleName
	} else {
		m.cfg.ActiveRole = "default"
		// Persist the new model as the default role so it survives restart.
		//
		// The provider is deliberately not persisted: newName carries its
		// provider prefix whenever the bare name would not resolve to the same
		// backend (switchedModelName), and a provider written into the role
		// then outranks the prefix on every later switch — so one `/model` to
		// another provider used to pin the role to it and break the next one.
		saveModelToConfig(newName)
	}

	fmt.Fprintf(&sb, "Switched model to **%s** (provider: %s).", newName, newProvider)
	if roleName != "" && roleName != "default" {
		fmt.Fprintf(&sb, " Role: `%s`.", roleName)
	}
	m.chatModel.Messages = append(m.chatModel.Messages, message{
		role:    "assistant",
		content: sb.String(),
	})
	return m, nil
}

// saveModelToConfig persists the model as the default role in
// ~/.pirate/config.json. Silently ignores errors (best-effort persistence).
func saveModelToConfig(modelName string) {
	_ = config.SaveDefaultRole(modelName, "")
}

// formatContextUsage builds a context usage display similar to Claude Code's /context.
func (m *model) formatContextUsage() string {
	var b strings.Builder

	m.writeCtxBreakdownSection(&b)

	est := m.estimateCtxRoleTokens()

	// Header.
	b.WriteString("**Context Usage**\n\n")

	m.writeCtxModelLine(&b, est.total)
	m.writeCtxCategorySection(&b, est)
	m.writeCtxDailySection(&b)
	m.writeCtxWindowSection(&b)
	m.writeCtxSkillsSection(&b)
	m.writeCtxSubagentSection(&b)

	// Compaction stats.
	if cm := m.cfg.CompactMetrics; cm != nil {
		stats := cm.FormatStats()
		if stats != "" {
			b.WriteString("\n*Output compaction*\n")
			b.WriteString(stats)
		}
	}

	return b.String()
}

// writeCtxBreakdownSection writes the "what is filling the window" breakdown.
// It answers the question a user asks before deciding what to trim, so it
// leads the /context output.
func (m *model) writeCtxBreakdownSection(b *strings.Builder) {
	bd := m.cfg.ContextBreakdown
	if bd == nil {
		return
	}
	used := int64(0)
	window := int64(0)
	if tt := m.cfg.TokenTracker; tt != nil {
		used = tt.LastPromptTokens()
		window = tt.ContextWindowSize()
	}
	if used == 0 {
		used = estimateContextTokenCount(m.chatModel.Messages) + bd.FixedTotal()
	}
	if window <= 0 {
		window = autoRangeWindow(used)
	}
	b.WriteString("*Context usage*\n\n")
	b.WriteString(RenderContextBreakdown(
		bd.withConversationFrom(used), window, min(m.chatWidth()-4, 64), m.palette))
	b.WriteString("\n\n")
}

// ctxRoleTokens is the per-role token estimate behind the /context category
// breakdown. total is estimated over the summed characters, not as the sum of
// the per-role estimates, so it can differ from user+assistant+tool by rounding.
type ctxRoleTokens struct {
	user      int64
	assistant int64
	tool      int64
	total     int64
}

// estimateCtxRoleTokens splits the same estimate the status bar shows by role.
func (m *model) estimateCtxRoleTokens() ctxRoleTokens {
	userChars, assistantChars, toolChars := 0, 0, 0
	for _, msg := range m.chatModel.Messages {
		size := messageChars(msg)
		switch msg.role {
		case "user":
			userChars += size
		case "assistant":
			assistantChars += size
		default: // tool
			toolChars += size
		}
	}
	return ctxRoleTokens{
		user:      estimateTokens(userChars),
		assistant: estimateTokens(assistantChars),
		tool:      estimateTokens(toolChars),
		total:     estimateTokens(userChars + assistantChars + toolChars),
	}
}

// writeCtxModelLine writes the 20-block usage bar and the model label beside it.
func (m *model) writeCtxModelLine(b *strings.Builder, totalTokens int64) {
	const barLen = 20
	var usedBlocks int
	var limitTokens int64
	if tt := m.cfg.TokenTracker; tt != nil && tt.Limit() > 0 {
		limitTokens = tt.Limit()
		usedBlocks = barFill(float64(tt.TotalUsed())/float64(limitTokens), barLen)
	} else if totalTokens > 0 {
		// No limit to measure against — show the context against a nominal 100k
		// window instead, and never less than one block, so a short conversation
		// still reads as "something is in here".
		usedBlocks = 1
		if totalTokens > 10000 {
			usedBlocks = max(barFill(float64(totalTokens)/100000, barLen), 1)
		}
	}
	bar := barGlyphs(usedBlocks, barLen)

	modelLabel := m.cfg.ModelName
	if m.cfg.ProviderName != "" {
		modelLabel = m.cfg.ProviderName + " | " + modelLabel
	}
	if limitTokens > 0 {
		tt := m.cfg.TokenTracker
		fmt.Fprintf(b, "`%s`  %s · %s/%s tokens (%.0f%%)\n\n",
			bar, modelLabel,
			formatTokenCount(tt.TotalUsed()), formatTokenCount(limitTokens), tt.PercentUsed())
		return
	}
	fmt.Fprintf(b, "`%s`  %s · ctx ~%s tokens\n\n",
		bar, modelLabel, formatTokenCount(totalTokens))
}

// writeCtxCategorySection writes the estimated per-role usage list.
func (m *model) writeCtxCategorySection(b *strings.Builder, est ctxRoleTokens) {
	b.WriteString("*Estimated usage by category*\n")
	fmt.Fprintf(b, "- **User messages**: ~%s tokens (%d msgs)\n",
		formatTokenCount(est.user), countByRole(m.chatModel.Messages, "user"))
	fmt.Fprintf(b, "- **Assistant messages**: ~%s tokens (%d msgs)\n",
		formatTokenCount(est.assistant), countByRole(m.chatModel.Messages, "assistant"))
	fmt.Fprintf(b, "- **Tool calls**: ~%s tokens (%d calls)\n",
		formatTokenCount(est.tool), countByRole(m.chatModel.Messages, "tool"))
	fmt.Fprintf(b, "- **Total context**: ~%s tokens (%d messages)\n",
		formatTokenCount(est.total), len(m.chatModel.Messages))
}

// writeCtxDailySection writes actual (not estimated) daily token consumption.
func (m *model) writeCtxDailySection(b *strings.Builder) {
	tt := m.cfg.TokenTracker
	if tt == nil {
		return
	}
	total := tt.TotalUsed()
	if total == 0 {
		return
	}
	b.WriteString("\n*Daily token usage*\n")
	fmt.Fprintf(b, "- **Consumed today**: %s tokens\n", formatTokenCount(total))
	if tt.Limit() > 0 {
		fmt.Fprintf(b, "- **Remaining**: %s tokens\n", formatTokenCount(tt.Remaining()))
	}
}

// writeCtxWindowSection writes context window occupancy from the last LLM
// response, followed by the prompt cache section.
func (m *model) writeCtxWindowSection(b *strings.Builder) {
	tt := m.cfg.TokenTracker
	if tt == nil || tt.LastPromptTokens() <= 0 {
		return
	}
	promptTokens := tt.LastPromptTokens()
	ctxWindow := tt.ContextWindowSize()

	b.WriteString("\n*Context window*\n")
	if ctxWindow > 0 {
		pct := tt.ContextPercentUsed()
		freeTokens := ctxWindow - promptTokens
		if freeTokens < 0 {
			freeTokens = 0
		}

		const ctxBarLen = 20
		ctxBar := barGlyphs(barFill(pct/100, ctxBarLen), ctxBarLen)

		fmt.Fprintf(b, "`%s`  %s / %s (%.0f%%)\n",
			ctxBar,
			formatTokenCount(promptTokens), formatTokenCount(ctxWindow), pct)
		fmt.Fprintf(b, "- **Used**: %s tokens\n", formatTokenCount(promptTokens))
		fmt.Fprintf(b, "- **Free**: %s tokens (%.0f%%)\n",
			formatTokenCount(freeTokens), 100-pct)
	} else {
		fmt.Fprintf(b, "- **Last prompt**: %s tokens (window size unknown)\n",
			formatTokenCount(promptTokens))
	}

	m.writeCtxCacheSection(b, promptTokens)
}

// writeCtxCacheSection writes prompt cache statistics. Reported explicitly even
// at zero hits — a silent section reads the same whether caching works or is
// entirely absent.
func (m *model) writeCtxCacheSection(b *strings.Builder, promptTokens int64) {
	tt := m.cfg.TokenTracker
	b.WriteString("\n*Prompt cache*\n")
	cached := tt.LastCachedTokens()
	if cached > 0 {
		fmt.Fprintf(b, "- **Last request**: %s of %s prompt tokens cached (%.0f%%)\n",
			formatTokenCount(cached), formatTokenCount(promptTokens),
			float64(cached)/float64(promptTokens)*100)
	} else {
		b.WriteString("- **Last request**: no cache hit\n")
	}
	if today := tt.CachedTokensToday(); today > 0 {
		fmt.Fprintf(b, "- **Today**: %s tokens read from cache (%.0f%% of input)\n",
			formatTokenCount(today), tt.CacheHitRateToday())
	}
	if prefix := tt.CachePrefixTokens(); prefix > 0 {
		fmt.Fprintf(b, "- **Stable prefix**: %s tokens · **body since**: %s tokens\n",
			formatTokenCount(prefix), formatTokenCount(tt.BodyTokens()))
	}
}

// writeCtxSkillsSection lists loaded skills alphabetically, for predictable
// /context output.
func (m *model) writeCtxSkillsSection(b *strings.Builder) {
	if len(m.cfg.Skills) == 0 {
		return
	}
	b.WriteString("\n*Skills* ")
	fmt.Fprintf(b, "(%d loaded)\n", len(m.cfg.Skills))
	names := make([]string, 0, len(m.cfg.Skills))
	byName := make(map[string]extension.Skill, len(m.cfg.Skills))
	for _, s := range m.cfg.Skills {
		names = append(names, s.Name)
		byName[s.Name] = s
	}
	sort.Strings(names)
	for _, name := range names {
		s := byName[name]
		source := s.Source
		if source == "" {
			source = "user"
		}
		var bodyDesc string
		if size, ok := extension.SkillBodySize(m.cfg.Skills, s.Name); ok {
			bodyDesc = fmt.Sprintf("body: %s", formatTokenCount(int64(size)))
		} else {
			bodyDesc = "body: not loaded"
		}
		desc := s.Description
		if desc == "" {
			desc = "(no description)"
		}
		fmt.Fprintf(b, "- /%s — %s [%s]  %s\n", s.Name, desc, source, bodyDesc)
	}
}

// writeCtxSubagentSection writes the running/done/failed subagent tally.
func (m *model) writeCtxSubagentSection(b *strings.Builder) {
	if m.cfg.Orchestrator == nil {
		return
	}
	agents := m.cfg.Orchestrator.List()
	if len(agents) == 0 {
		return
	}
	running, done, failed := 0, 0, 0
	for _, a := range agents {
		switch a.Status {
		case "running":
			running++
		case "failed":
			failed++
		default:
			done++
		}
	}
	b.WriteString("\n*Subagents*\n")
	fmt.Fprintf(b, "- **Total**: %d (running: %d, done: %d, failed: %d)\n",
		len(agents), running, done, failed)
}

// showCommandList displays available slash commands as an assistant message.
func (m *model) showCommandList() {
	var b strings.Builder
	b.WriteString("**Commands:**\n")
	for _, cmd := range slashCommands {
		desc := slashCommandDesc(cmd)
		b.WriteString("  `" + cmd + "`")
		if desc != "" {
			b.WriteString(" — " + desc)
		}
		b.WriteString("\n")
	}
	if len(m.cfg.Skills) > 0 {
		b.WriteString("\n**Skills:**\n")
		for _, skill := range m.cfg.Skills {
			fmt.Fprintf(&b, "  `/%s`", skill.Name)
			if skill.Description != "" {
				b.WriteString(" — " + skill.Description)
			}
			b.WriteString("\n")
		}
	}
	m.chatModel.Messages = append(m.chatModel.Messages, message{
		role:    "assistant",
		content: b.String(),
	})
}

// handleSkillsCommand handles /skills and its subcommands: create, load, list (default).
func (m *model) handleSkillsCommand(args []string) (tea.Model, tea.Cmd) {
	if len(args) == 0 {
		return m.handleSkillListCommand()
	}
	sub := strings.ToLower(args[0])
	switch sub {
	case "create":
		return m.handleSkillCreateCommand(args[1:])
	case "load", "reload":
		return m.handleSkillLoadCommand()
	case "list":
		return m.handleSkillListCommand()
	default:
		m.chatModel.Messages = append(m.chatModel.Messages, message{
			role:    "assistant",
			content: "Usage: `/skills` — list  |  `/skills create <name>` — create  |  `/skills load` — reload",
		})
		return m, nil
	}
}

// formatThemeList builds the theme list: one row per theme, each carrying its
// own colors, so the list is the preview — no separate command to run and no
// switching themes one at a time to see them.
//
// The rows carry ANSI, so the message must be marked preRendered — glamour
// prints escape bytes as text.
func formatThemeList(themes []Theme, currentName string, p Palette) string {
	p = paletteOrDark(p)
	name := lipgloss.NewStyle().Foreground(p.Text)
	active := lipgloss.NewStyle().Foreground(p.Success).Bold(true)

	nameWidth := 0
	for _, t := range themes {
		if len(t.Name) > nameWidth {
			nameWidth = len(t.Name)
		}
	}

	custom := lipgloss.NewStyle().Foreground(p.Dim)

	var b strings.Builder
	for i, t := range themes {
		if i > 0 {
			b.WriteString("\n")
		}
		icon := "🌙"
		if t.ThemeType == "light" {
			icon = "☀️"
		}
		nameStyle := name
		if t.Name == currentName {
			nameStyle = active
		}
		fmt.Fprintf(&b, "  %s  %s  %s",
			icon,
			nameStyle.Render(fmt.Sprintf("%-*s", nameWidth, t.Name)),
			swatchStrip(t.Colors))
		if t.Custom {
			b.WriteString("  " + custom.Render("(custom)"))
		}
	}
	return b.String()
}

// formatThemeError builds an error message with optional close-match suggestions.
func formatThemeError(name string, matches []string) string {
	msg := fmt.Sprintf("Unknown theme `%s`.", name)
	if len(matches) > 0 {
		msg += " Did you mean: " + strings.Join(matches, ", ") + "?"
	}
	return msg
}

// handleThemeCommand handles /theme: list themes, switch theme, or show current.
func (m *model) handleThemeCommand(args []string) (tea.Model, tea.Cmd) {
	if m.themeManager == nil {
		m.chatModel.Messages = append(m.chatModel.Messages, message{
			role:    "assistant",
			content: "Theme system not available.",
		})
		return m, nil
	}

	if len(args) == 0 {
		// Cheap re-scan so themes written since startup show up without a
		// restart; broken files surface here, where the user is looking for
		// them. Warnings are transient (status flash) — the list still renders.
		var cmd tea.Cmd
		if warnings := m.themeManager.LoadCustomThemes(m.cfg.WorkDir); len(warnings) > 0 {
			cmd = m.setFlash("Custom themes: " + strings.Join(warnings, "; "))
		}
		m.chatModel.Messages = append(m.chatModel.Messages, message{
			role:        "assistant",
			preRendered: true,
			content:     formatThemeList(m.themeManager.List(), m.themeManager.CurrentName(), m.palette),
		})
		return m, cmd
	}

	if strings.ToLower(args[0]) == "palette" {
		m.chatModel.Messages = append(m.chatModel.Messages, message{
			role:        "assistant",
			preRendered: true,
			content:     formatPalettePreview(m.palette),
		})
		return m, nil
	}

	name := strings.ToLower(args[0])
	if err := m.themeManager.SetTheme(name); err != nil {
		m.chatModel.Messages = append(m.chatModel.Messages, message{
			role:    "assistant",
			content: formatThemeError(name, m.themeManager.ClosestMatches(name, 5)),
		})
		return m, nil
	}

	saveThemeToConfig(name)

	// An explicit choice wins over terminal background detection for the rest
	// of the session.
	m.bgDetected = true
	m.applyTheme()

	cur := m.themeManager.Current()
	m.chatModel.Messages = append(m.chatModel.Messages, message{
		role:    "assistant",
		content: fmt.Sprintf("Theme switched to `%s` (%s).", cur.Name, cur.DisplayName),
	})
	return m, nil
}

// handleRTKCommand handles the /rtk command and subcommands.
func (m *model) handleRTKCommand(args []string) {
	sub := "stats"
	if len(args) > 0 {
		sub = strings.ToLower(args[0])
	}
	switch sub {
	case "stats":
		if m.cfg.CompactMetrics == nil {
			m.chatModel.Messages = append(m.chatModel.Messages, message{
				role:    "assistant",
				content: "Output compactor is not active.",
			})
			return
		}
		m.chatModel.Messages = append(m.chatModel.Messages, message{
			role:    "assistant",
			content: m.cfg.CompactMetrics.FormatStats(),
		})
	default:
		m.chatModel.Messages = append(m.chatModel.Messages, message{
			role:    "assistant",
			content: "Usage: `/rtk` or `/rtk stats` — Show output compaction statistics",
		})
	}
}

// handleMCPCommand shows the status of configured MCP servers and their tools.
func (m *model) handleMCPCommand() {
	servers := m.cfg.MCPServers
	toolsets := m.cfg.MCPToolsets

	if len(servers) == 0 {
		m.chatModel.Messages = append(m.chatModel.Messages, message{
			role:    "assistant",
			content: "No MCP servers configured. Add servers under `mcp.servers` in `~/.pirate/config.json`.",
		})
		return
	}

	statuses := extension.ToolsetStatuses(toolsets)
	// Index statuses by name for lookup.
	statusByName := make(map[string]extension.MCPServerStatus, len(statuses))
	for _, s := range statuses {
		statusByName[s.Name] = s
	}

	var b strings.Builder
	fmt.Fprintf(&b, "**MCP Servers** — %d configured\n\n", len(servers))
	b.WriteString("| Server | URL | Status | Tools |\n")
	b.WriteString("|--------|-----|--------|-------|\n")
	for _, srv := range servers {
		st, ok := statusByName[srv.Name]
		statusIcon := "⏳ pending"
		toolCount := "—"
		if ok {
			switch st.Status {
			case "connected":
				statusIcon = "✓ connected"
				toolCount = fmt.Sprintf("%d", st.ToolCount)
			case "failed":
				statusIcon = "✗ failed"
				toolCount = "—"
			default:
				statusIcon = "⏳ pending"
			}
		}
		displayURL := maskServerURL(srv.URL)
		fmt.Fprintf(&b, "| `%s` | %s | %s | %s |\n", srv.Name, displayURL, statusIcon, toolCount)
	}

	// Hint about pending tool counts.
	var connected []extension.MCPServerStatus
	for _, s := range statuses {
		if s.Status == "connected" {
			connected = append(connected, s)
		}
	}
	if len(connected) > 0 {
		b.WriteString("\n> Tool counts are available after the first agent interaction with each server.\n")
	} else if len(statuses) > 0 {
		b.WriteString("\n> MCP tool counts populate after the first agent request — start a task to connect.\n")
	}

	m.chatModel.Messages = append(m.chatModel.Messages, message{
		role:    "assistant",
		content: b.String(),
	})
}

// maskServerURL masks API keys in MCP server URLs for TUI display.
// Replaces query parameter values (e.g., ?key=xxx) with "***".
func maskServerURL(url string) string {
	if url == "" {
		return "_none_"
	}
	if !strings.Contains(url, "?") {
		return url
	}
	masked := url
	sensitive := []string{
		"tavilyApiKey", "apiKey", "key", "token", "secret", "password",
	}
	for _, param := range sensitive {
		prefix := param + "="
		if idx := strings.Index(masked, prefix); idx >= 0 {
			end := idx + len(prefix)
			if amp := strings.Index(masked[end:], "&"); amp >= 0 {
				masked = masked[:idx] + param + "=" + "***" + masked[end+amp:]
			} else {
				masked = masked[:idx] + param + "=***"
			}
		}
	}
	return masked
}
