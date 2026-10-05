package server

import (
	"strings"

	acp "github.com/coder/acp-go-sdk"

	"github.com/spa-skyson/pi-rate/internal/acp/server/adapter"
	"github.com/spa-skyson/pi-rate/internal/config"
	"github.com/spa-skyson/pi-rate/internal/extension"
	"github.com/spa-skyson/pi-rate/internal/subagent"
)

// DiscoverAvailableCommands resolves slash commands for a specific session cwd.
// User skills are always included, while project skills are discovered from the
// nearest ancestor directories of the session working directory.
func DiscoverAvailableCommands(cwd string) []acp.AvailableCommand {
	cwd = normalizeDiscoveryCWD(cwd)

	// Skill discovery honors skillsDirs / disableLegacySkillDirs (and their
	// env overrides) from config.json; a load failure falls back to the
	// default discovery.
	cfg, err := config.LoadFrom(cwd)
	if err != nil {
		cfg = config.Config{}
	}
	skills, _ := extension.LoadSkills(extension.DefaultSkillDirsIn(cwd, cfg)...)

	var subagents []subagent.AgentConfig
	if discovery, err := subagent.DiscoverAgents(cwd, subagent.ScopeBoth); err == nil && discovery != nil {
		subagents = discovery.All
	}

	return adapter.BuildAvailableCommands(skills, subagents)
}

func normalizeDiscoveryCWD(cwd string) string {
	cwd = strings.TrimSpace(cwd)
	if cwd != "" {
		return cwd
	}
	return "."
}
