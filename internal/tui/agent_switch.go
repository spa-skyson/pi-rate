package tui

import (
	"fmt"
	"sort"
	"strings"

	tea "charm.land/bubbletea/v2"

	"github.com/spa-skyson/pi-rate/internal/subagent"
)

// handleAgentCommand handles /agent <name>: applies that agent — or
// "default" — to the main session. The argument-less popup is routed by
// handleSlashCommand (see the note there); direct calls with no argument are
// a no-op.
func (m *model) handleAgentCommand(args []string) (tea.Model, tea.Cmd) {
	if len(args) == 0 {
		return m, nil
	}
	name := strings.TrimSpace(args[0])
	if strings.EqualFold(name, "default") {
		name = ""
	}
	return m.applyAgent(name)
}

// cycleAgent advances the Shift+Tab cycle: default (the built-in Pi-rate
// agent) → primary agents alphabetically → back to default.
func (m *model) cycleAgent() (tea.Model, tea.Cmd) {
	return m.applyAgent(nextAgentName(m.activeAgent, primaryAgentNames(m.cfg.PrimaryAgents)))
}

// applyAgent switches the main session to the named primary agent ("" = the
// built-in default agent), shared by /agent <name> and Shift+Tab. Like
// /model, it refuses while a response is running or a backgrounded run holds
// the agent: the agent object is being rebuilt underneath the loop, and the
// rebuild must never run under a live one — the one-live-loop invariant.
func (m *model) applyAgent(name string) (tea.Model, tea.Cmd) {
	if m.running {
		return m, m.setFlash("Cannot switch agent while a response is running")
	}
	if m.bgAlive() {
		return m, m.setFlash("Background run active — attach (Ctrl+B) or wait for it to finish")
	}
	if m.cfg.Agent == nil || m.cfg.AgentSwitcher == nil {
		m.chatModel.AppendNotice("Agent switching is not available in this context.")
		return m, nil
	}

	sw, err := m.cfg.AgentSwitcher(m.ctx, name, m.agentModelOverride(name))
	if err != nil {
		m.chatModel.AppendNotice(fmt.Sprintf("Agent switch failed: %v", err))
		return m, nil
	}
	if err := m.cfg.Agent.RebuildWithSession(sw.Name, sw.Instruction, sw.LLM, sw.BeforeTool, sw.AfterTool); err != nil {
		m.chatModel.AppendNotice(fmt.Sprintf("Failed to rebuild agent: %v", err))
		return m, nil
	}

	// A nil LLM from the switcher means "no model change" — keep the names.
	if sw.LLM != nil {
		m.cfg.LLM = sw.LLM
		m.cfg.ModelName = sw.ModelName
		m.cfg.ProviderName = sw.Provider
	}
	m.activeAgent = name

	// Success is silent: the sidebar already shows the agent and model, and a
	// notice per Shift+Tab stop just spams the transcript. Only failures
	// above surface as notices.
	return m, nil
}

// agentModelOverride returns the session model recorded for the named agent
// by a /model typed while it was active. "" for the default agent and for
// agents without an override — their own model specification stands.
func (m *model) agentModelOverride(name string) string {
	if name == "" || m.agentModelOverrides == nil {
		return ""
	}
	return m.agentModelOverrides[name]
}

// setAgentModelOverride records a session-only model override for a primary
// agent. In-memory by design: it lives as long as the TUI session, is never
// persisted to config, and re-applies on every switch back to the agent.
func (m *model) setAgentModelOverride(name, modelName string) {
	if name == "" {
		return
	}
	if m.agentModelOverrides == nil {
		m.agentModelOverrides = make(map[string]string)
	}
	m.agentModelOverrides[name] = modelName
}

// agentConfig looks up a primary agent's config by name. ok is false for the
// default agent ("") and names that are not in PrimaryAgents.
func (m *model) agentConfig(name string) (subagent.AgentConfig, bool) {
	if name == "" {
		return subagent.AgentConfig{}, false
	}
	for _, ac := range m.cfg.PrimaryAgents {
		if ac.Name == name {
			return ac, true
		}
	}
	return subagent.AgentConfig{}, false
}

// primaryAgentNames returns the switchable agents' names, sorted for a
// stable cycle and listing.
func primaryAgentNames(agents []subagent.AgentConfig) []string {
	names := make([]string, 0, len(agents))
	for _, ac := range agents {
		names = append(names, ac.Name)
	}
	sort.Strings(names)
	return names
}

// nextAgentName returns the next stop of the Shift+Tab cycle: default ("")
// hands off to the first agent, the last agent wraps back to default. An
// agent no longer on the list restarts the cycle from its beginning.
func nextAgentName(current string, names []string) string {
	if len(names) == 0 {
		return ""
	}
	if current == "" {
		return names[0]
	}
	for i, n := range names {
		if n == current {
			if i+1 < len(names) {
				return names[i+1]
			}
			return ""
		}
	}
	return names[0]
}

// agentSearchItems builds the /agent popup list: the built-in default first,
// then the primary agents alphabetically. Text stays a clean agent name — it
// is executed as `/agent <name>` — so the active marker lives in the
// description beside the agent's own description and model.
func (m *model) agentSearchItems() []SearchItem {
	cfgs := make(map[string]subagent.AgentConfig, len(m.cfg.PrimaryAgents))
	for _, ac := range m.cfg.PrimaryAgents {
		cfgs[ac.Name] = ac
	}

	mark := func(name string) string {
		if name == m.activeAgent || (name == "default" && m.activeAgent == "") {
			return " — active"
		}
		return ""
	}

	items := make([]SearchItem, 0, len(m.cfg.PrimaryAgents)+1)
	items = append(items, SearchItem{Text: "default", Description: "built-in pirate agent" + mark("default")})
	for _, name := range primaryAgentNames(m.cfg.PrimaryAgents) {
		ac := cfgs[name]
		desc := ac.Description
		if desc == "" {
			desc = "(no description)"
		}
		if model := agentDisplayModel(m.cfg, ac); model != "" {
			desc += " · " + model
		}
		items = append(items, SearchItem{Text: name, Description: desc + mark(name)})
	}
	return items
}

// agentDisplayModel names the model an agent runs on: its `model:`, or the
// model behind its `role:`. Empty when neither resolves.
func agentDisplayModel(cfg Config, ac subagent.AgentConfig) string {
	if ac.Model != "" {
		return ac.Model
	}
	if ac.Role != "" {
		if rc, ok := cfg.Roles[ac.Role]; ok {
			return rc.Model
		}
	}
	return ""
}
