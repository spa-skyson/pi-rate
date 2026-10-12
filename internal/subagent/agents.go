package subagent

import (
	"bufio"
	"fmt"
	"io/fs"
	"log/slog"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/spa-skyson/pi-rate/internal/config"
	"github.com/spa-skyson/pi-rate/internal/permission"
)

// minAgentTimeoutMs is the smallest frontmatter `timeout:` treated as a real
// value. Below this the author almost certainly meant seconds; honoring it
// would kill the agent before it could produce anything.
const minAgentTimeoutMs = 1000

// AgentScope defines the scope for agent discovery.
type AgentScope string

const (
	ScopeBoth    AgentScope = "both"    // Bundled + user/project
	ScopeBundled AgentScope = "bundled" // Only embedded agents
	ScopeProject AgentScope = "project" // Only user/project agents
)

// AgentMode classifies where an agent may run, from frontmatter `mode:`.
// The zero value is the bundled default: subagent-only, so existing agent
// files keep their current behavior when the key is absent.
type AgentMode string

const (
	// ModeSubagent is the default: the agent is spawned as a subagent only.
	ModeSubagent AgentMode = "subagent"
	// ModePrimary reserves the agent for the main session (Shift+Tab /
	// /agent); it is not advertised as a subagent.
	ModePrimary AgentMode = "primary"
	// ModeAll allows both: switchable into the main session and spawnable.
	ModeAll AgentMode = "all"
)

// AgentConfig represents a parsed agent definition from markdown.
type AgentConfig struct {
	Name        string // Agent identifier (e.g., "explore", "plan")
	Description string // One-line description from frontmatter
	Role        string // Config role name for model resolution (e.g., "smol", "plan", "slow")
	// Mode is the raw frontmatter `mode:` value ("primary", "subagent" or
	// "all"). Empty means ModeSubagent. Use IsPrimary/PrimaryOnly to
	// classify; the raw value matters only for round-tripping.
	Mode AgentMode
	// Model names the model this agent runs on, set via frontmatter `model:`.
	// It overrides `role:` and may carry a provider prefix — a built-in one
	// ("openai/gpt-5.6") or a declared provider's name from the config.json
	// `providers` section ("corp-codex/gpt-5.6-sol"). Empty means `role:`
	// decides.
	Model string
	// FallbackModels lists the models tried in order when the agent's
	// primary model (frontmatter `model:`, else its role) is unavailable —
	// quota exhausted, auth failure, or repeated 5xx. Set via frontmatter
	// `fallback-models:` as a comma-separated list of the same
	// "provider/model" spellings `model:` takes. The consumer caps the list;
	// parsing keeps every entry.
	FallbackModels []string
	Worktree       bool     // Whether this agent runs in an isolated git worktree
	Timeout        int      // Absolute timeout in milliseconds (0 = use default)
	Instruction    string   // System prompt (markdown body)
	Tools          []string // Allowed tool names (empty = all tools)
	// LSP selects this agent's language-server surface: "off", "min" or
	// "full". Empty means inherit the child process default (min). Set it in
	// frontmatter on agents that navigate code, so the wide LSP surface is
	// bought per-agent instead of by every session.
	LSP string
	// Temperature is the sampling temperature passed to the child as
	// --temperature. Zero means unset: frontmatter temperatures are
	// practically always > 0 (0.2-1.0), so 0 is safe as the "not asked for"
	// sentinel — an agent that genuinely wants 0 sampling temperature is
	// rare enough to lose the tie against keeping the plain float field.
	Temperature float64
	// ReasoningEffort is the raw frontmatter `reasoningEffort:` value
	// (trimmed, lowercased). It is normalized to a thinking level at spawn
	// time by NormalizeReasoningEffort; empty means inherit.
	ReasoningEffort string
	// StreamIdleTimeout is the frontmatter `streamIdleTimeout:` value in
	// milliseconds, handed to the child as PI_STREAM_IDLE_TIMEOUT_MS (issue
	// #37). Nil inherits — the child uses its own config.json value or the
	// 90s default; 0 disables the idle abort for this agent; a positive value
	// is the silence budget. A pointer because "unset" and "explicitly 0"
	// must reach the child differently.
	StreamIdleTimeout *int
	// MCP holds the frontmatter `mcp:` server names, handed to the child as
	// PI_MCP (issue #68): the child is a one-shot json run and lifts no MCP
	// server unless the run names one. Empty means no PI_MCP — the child
	// starts none unless its own --mcp or PI_MCP says otherwise; the special
	// name "all" restores every configured server.
	MCP []string
	// Steps caps the child's tool-call iterations (0 = no limit).
	Steps int
	// Permission holds the agent's frontmatter `permission:` rules. They gate
	// the child's tool calls; the orchestrator serializes them into the
	// child's environment and the child layers them over the global config
	// rules. Zero value = no agent-specific rules.
	Permission permission.Rules
	Source     string // "bundled", "user", or "project"
}

// IsPrimary reports whether the agent may be switched into the main session
// (frontmatter `mode: primary` or `all`). Empty mode — the bundled default —
// is subagent-only.
func (c AgentConfig) IsPrimary() bool {
	return c.Mode == ModePrimary || c.Mode == ModeAll
}

// PrimaryOnly reports whether the agent is reserved for the main session
// (`mode: primary`) and must not be advertised as a spawnable subagent.
// `mode: all` agents are both, so they stay advertised.
func (c AgentConfig) PrimaryOnly() bool {
	return c.Mode == ModePrimary
}

// AgentDiscoveryResult contains all discovered agents.
type AgentDiscoveryResult struct {
	Bundled []AgentConfig
	User    []AgentConfig
	Project []AgentConfig
	All     []AgentConfig // Merged with priority: project > user > bundled
}

// ParseAgentFile parses a single agent markdown file.
// Expected format:
// ---
// name: agent-name
// description: One-line description
// role: smol
// model: corp-codex/gpt-5.6-sol
// fallback-models: openai/gpt-5.6-sol, anthropic/claude-sonnet-4-6
// worktree: false
// tools: read, write, edit
// mcp: serena, agentmemory
// temperature: 0.3
// reasoningEffort: high
// streamIdleTimeout: 90s
// steps: 150
// timeout: 30m
// mode: primary
// ---
// Markdown instruction body...
//
// `model:` overrides `role:` when both are set; it may name a built-in
// provider's model or a declared one ("provider/model", see the `providers`
// section of config.json). `fallback-models:` is a comma-separated list of
// models tried in order when the primary one is unavailable. `temperature:`,
// `reasoningEffort:` and `steps:`
// tune the child's sampling and iteration budget; unusable values warn and
// leave the field at its zero value (= inherit / unlimited). `mcp:` is a
// comma-separated list of MCP server names (or "all") that rides to the
// child as PI_MCP: a child is a one-shot run and lifts no MCP server unless
// the run names one. `timeout:` is
// milliseconds by default and also takes unit suffixes — 30m, 1h, 90s,
// 500ms; a value resolving to under a second warns and falls back to the
// default. `streamIdleTimeout:` takes the same spellings and rides to the
// child as PI_STREAM_IDLE_TIMEOUT_MS; a bare `0` disables the child's
// stream-idle abort, any other unusable value inherits.
func ParseAgentFile(path string) (AgentConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return AgentConfig{}, err
	}
	return parseAgentContent(string(data), path)
}

// ParseAgentFileFromFS parses an agent file from an embedded filesystem.
func ParseAgentFileFromFS(fsys fs.FS, path string) (AgentConfig, error) {
	data, err := fs.ReadFile(fsys, path)
	if err != nil {
		return AgentConfig{}, err
	}
	return parseAgentContent(string(data), path)
}

// parseAgentContent is the shared parsing logic for agent markdown content.
func parseAgentContent(content, path string) (AgentConfig, error) {
	// Derive default name from filename: explore.md → explore
	name := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))

	cfg := AgentConfig{Name: name}

	scanner := bufio.NewScanner(strings.NewReader(content))
	inFrontmatter := false
	frontmatterDone := false
	var body strings.Builder
	var perm permissionParser

	for scanner.Scan() {
		line := scanner.Text()
		trimmed := strings.TrimSpace(line)

		if trimmed == "---" && !frontmatterDone {
			if !inFrontmatter {
				inFrontmatter = true
				continue
			}
			// End of frontmatter.
			inFrontmatter = false
			frontmatterDone = true
			continue
		}

		if inFrontmatter {
			// Inside an open `permission:` block every indented line belongs
			// to it; the block ends at the first non-indented line, which is
			// then processed as a regular frontmatter key.
			if perm.open() && perm.feed(cfg.Name, line) {
				continue
			}
			if key, value, ok := parseAgentFrontmatterLine(line); ok {
				if key == "permission" && value == "" {
					perm.begin()
					continue
				}
				applyAgentFrontmatterKey(&cfg, key, value)
			}
		} else {
			body.WriteString(line)
			body.WriteString("\n")
		}
	}

	if err := scanner.Err(); err != nil {
		return AgentConfig{}, err
	}

	cfg.Instruction = strings.TrimSpace(body.String())
	cfg.Permission = perm.rules
	return cfg, nil
}

// applyAgentFrontmatterKey sets the field of cfg named by a frontmatter key.
// Unknown keys are ignored.
func applyAgentFrontmatterKey(cfg *AgentConfig, key, value string) {
	switch key {
	case "name":
		cfg.Name = value
	case "description":
		cfg.Description = value
	case "role":
		cfg.Role = value
	case "model":
		cfg.Model = strings.TrimSpace(value)
	case "fallback-models":
		cfg.FallbackModels = parseAgentToolList(value)
	case "worktree":
		cfg.Worktree = strings.ToLower(value) == "true"
	case "timeout":
		if ms, ok := parseAgentTimeout(cfg.Name, value); ok {
			cfg.Timeout = ms
		}
	case "lsp":
		cfg.LSP = strings.ToLower(strings.TrimSpace(value))
	case "tools":
		cfg.Tools = append(cfg.Tools, parseAgentToolList(value)...)
	case "mcp":
		cfg.MCP = parseAgentMCPList(cfg.Name, value)
	case "temperature":
		if t, ok := parseAgentTemperature(cfg.Name, value); ok {
			cfg.Temperature = t
		}
	case "reasoningEffort":
		cfg.ReasoningEffort = strings.ToLower(strings.TrimSpace(value))
	case "streamIdleTimeout":
		if ms, ok := parseAgentStreamIdleTimeout(cfg.Name, value); ok {
			cfg.StreamIdleTimeout = &ms
		}
	case "mode":
		switch v := strings.ToLower(strings.TrimSpace(value)); v {
		case "":
			cfg.Mode = ""
		case string(ModePrimary), string(ModeSubagent), string(ModeAll):
			cfg.Mode = AgentMode(v)
		default:
			slog.Warn("subagent: unknown mode ignored; treating as subagent",
				"agent", cfg.Name, "mode", value, "known", "primary, subagent, all")
		}
	case "steps":
		if n, ok := parseAgentSteps(cfg.Name, value); ok {
			cfg.Steps = n
		}
	case "permission":
		// The block form is consumed by parseAgentContent before this switch
		// sees the key; reaching it here means a scalar value like
		// `permission: deny`, which is not a rule set.
		if value != "" {
			slog.Warn("subagent: permission must be a block of rules; scalar value ignored",
				"agent", cfg.Name, "value", value)
		}
	}
}

// permissionParser accumulates rules from an indented `permission:` block in
// agent frontmatter. It understands exactly the two levels the opencode
// permission format uses — tool-name rules and a nested `bash:` block of
// command-pattern rules — not YAML in general:
//
//	permission:
//	  edit: deny
//	  serena*: deny
//	  bash:
//	    "*": ask
//	    "git *": allow
type permissionParser struct {
	rules      permission.Rules
	begun      bool // saw the opening `permission:` line
	inBash     bool // inside the nested bash: block
	bashIndent int  // indent of the `bash:` line itself
}

// open reports whether a `permission:` block is being read.
func (p *permissionParser) open() bool { return p.begun }

// begin marks the parser as inside a `permission:` block.
func (p *permissionParser) begin() {
	p.begun = true
	p.inBash = false
}

// feed consumes one line while the block is open, reporting whether the line
// belonged to it. Blank lines stay inside; a non-indented line closes the
// block and is returned to the regular frontmatter keys.
func (p *permissionParser) feed(agentName, line string) bool {
	trimmed := strings.TrimSpace(line)
	if trimmed == "" {
		return true
	}
	indent := len(line) - len(strings.TrimLeft(line, " \t"))
	if indent == 0 {
		return false
	}
	// A line deeper than the `bash:` key itself is a command pattern.
	if p.inBash && indent > p.bashIndent {
		p.addBash(agentName, trimmed)
		return true
	}
	p.inBash = false

	key, value, ok := parseAgentFrontmatterLine(trimmed)
	if !ok {
		slog.Warn("subagent: unusable permission line ignored", "agent", agentName, "line", trimmed)
		return true
	}
	key = unquoteFrontmatter(key)
	value = unquoteFrontmatter(value)
	if key == "bash" {
		if value == "" {
			p.inBash = true
			p.bashIndent = indent
			return true
		}
		// A scalar `bash: deny` gates the tool wholesale — the command
		// patterns are the block form above.
		p.addTool(agentName, "bash", value)
		return true
	}
	p.addTool(agentName, key, value)
	return true
}

// addTool records one tool-name rule; an unusable directive warns and is
// skipped, matching how the other frontmatter values behave.
func (p *permissionParser) addTool(agentName, pattern, value string) {
	d, err := permission.ParseDirective(value)
	if err != nil {
		slog.Warn("subagent: permission rule ignored", "agent", agentName, "rule", pattern, "error", err)
		return
	}
	if p.rules.Tools == nil {
		p.rules.Tools = make(map[string]permission.Directive)
	}
	p.rules.Tools[pattern] = d
}

// addBash records one command-pattern rule from inside the bash: block.
func (p *permissionParser) addBash(agentName, trimmed string) {
	pattern, value, ok := parseAgentFrontmatterLine(trimmed)
	if !ok {
		slog.Warn("subagent: unusable bash permission line ignored", "agent", agentName, "line", trimmed)
		return
	}
	d, err := permission.ParseDirective(value)
	if err != nil {
		slog.Warn("subagent: permission rule ignored", "agent", agentName, "rule", pattern, "error", err)
		return
	}
	p.rules.Bash = append(p.rules.Bash, permission.BashRule{
		Pattern:   unquoteFrontmatter(pattern),
		Directive: d,
	})
}

// unquoteFrontmatter strips one pair of matching surrounding quotes, so both
// `"git *": allow` and `git *: allow` parse to the same pattern.
func unquoteFrontmatter(s string) string {
	s = strings.TrimSpace(s)
	if len(s) >= 2 && (s[0] == '"' && s[len(s)-1] == '"' || s[0] == '\'' && s[len(s)-1] == '\'') {
		return s[1 : len(s)-1]
	}
	return s
}

// parseAgentTimeout reads a frontmatter `timeout:` value, reporting ok=false
// when it is unusable and the default should stand instead.
//
// The value is a bare count of milliseconds ("600000", the original spelling)
// or a number with one unit suffix — "1h", "45m", "90s", "500ms". The suffix
// is case-insensitive and a space between the number and the unit is
// tolerated ("45 m"); combined forms ("1h30m") are not supported and fall
// back to the default with a warning.
//
// The result is still guarded against sub-second values, explicit "500ms"
// included: the unit is milliseconds, which reads as seconds at a glance — a
// bundled agent shipped `timeout: 30` and was SIGKILLed 30ms in, every time,
// unable to emit a single token. Anything under a second cannot be deliberate,
// so treat it as the unit mistake it is rather than honoring a value that
// guarantees the agent never runs.
func parseAgentTimeout(agentName, value string) (int, bool) {
	ms, err := parseTimeoutMs(value)
	if err != nil {
		slog.Warn("subagent: unusable timeout ignored",
			"agent", agentName, "timeout", value, "using", "default")
		return 0, false
	}
	if ms <= 0 {
		return 0, false
	}
	if ms < minAgentTimeoutMs {
		slog.Warn("subagent: implausibly small timeout ignored; the unit is milliseconds",
			"agent", agentName, "timeout_ms", ms, "using", "default")
		return 0, false
	}
	return ms, true
}

// parseAgentStreamIdleTimeout reads a frontmatter `streamIdleTimeout:` value,
// reporting ok=false when it is unusable and the child should inherit.
//
// Same spellings as `timeout:` (milliseconds or one unit suffix — "90s",
// "2m"), with one difference: a bare "0" is meaningful here, it disables the
// idle abort for this agent, so it is accepted rather than rejected. Values
// under a second are still refused — a stream-silence budget that short
// aborts healthy streams, and it is the same unit mistake `timeout:` guards
// against.
func parseAgentStreamIdleTimeout(agentName, value string) (int, bool) {
	if strings.TrimSpace(value) == "0" {
		return 0, true
	}
	return parseAgentTimeout(agentName, value)
}

// timeoutUnitMs maps a `timeout:` unit suffix to its length in milliseconds.
var timeoutUnitMs = map[string]int{
	"ms": 1,
	"s":  1000,
	"m":  60 * 1000,
	"h":  60 * 60 * 1000,
}

// parseTimeoutMs converts a `timeout:` value to milliseconds: a bare integer
// counts as milliseconds, otherwise an integer followed by exactly one unit
// suffix (ms, s, m, h — case-insensitive, surrounding space tolerated).
// Combined forms ("1h30m") and anything non-numeric are an error.
func parseTimeoutMs(value string) (int, error) {
	v := strings.ToLower(strings.TrimSpace(value))
	i := 0
	if i < len(v) && (v[i] == '-' || v[i] == '+') {
		i++
	}
	for i < len(v) && v[i] >= '0' && v[i] <= '9' {
		i++
	}
	ms, err := strconv.Atoi(v[:i])
	if err != nil {
		return 0, err
	}
	unit := strings.TrimSpace(v[i:])
	if unit == "" {
		return ms, nil
	}
	factor, ok := timeoutUnitMs[unit]
	if !ok {
		return 0, fmt.Errorf("unknown unit %q (want ms, s, m or h, or bare milliseconds)", unit)
	}
	scaled := ms * factor
	if scaled/factor != ms {
		return 0, fmt.Errorf("timeout %q overflows milliseconds", value)
	}
	return scaled, nil
}

// parseAgentTemperature reads a frontmatter `temperature:` value, reporting
// ok=false when it is unusable and the field should stay zero (= unset, child
// inherits). Sampling temperatures are 0..2 by every provider's contract, so a
// negative value is the same unit-class mistake as a non-numeric one.
func parseAgentTemperature(agentName, value string) (float64, bool) {
	t, err := strconv.ParseFloat(value, 64)
	if err != nil || t < 0 {
		slog.Warn("subagent: unusable temperature ignored",
			"agent", agentName, "temperature", value, "using", "default")
		return 0, false
	}
	return t, true
}

// parseAgentSteps reads a frontmatter `steps:` value — the cap on the child's
// tool-call iterations — reporting ok=false when it is unusable and the field
// should stay zero (= no limit).
func parseAgentSteps(agentName, value string) (int, bool) {
	n, err := strconv.Atoi(value)
	if err != nil || n < 0 {
		slog.Warn("subagent: unusable steps ignored",
			"agent", agentName, "steps", value, "using", "unlimited")
		return 0, false
	}
	return n, true
}

// NormalizeReasoningEffort maps a raw `reasoningEffort:` value onto the
// thinking levels the provider layer understands. "minimal" is OpenAI's
// spelling of "none"; everything else passes through lowercased. Unknown
// values warn and come back empty so the child inherits instead of failing.
func NormalizeReasoningEffort(value string) string {
	switch v := strings.ToLower(strings.TrimSpace(value)); v {
	case "":
		return ""
	case "minimal":
		return "none"
	case "none", "low", "medium", "high", "max":
		return v
	default:
		slog.Warn("subagent: unknown reasoningEffort ignored",
			"reasoningEffort", value, "known", "none, minimal, low, medium, high, max")
		return ""
	}
}

// parseAgentMCPList splits a frontmatter `mcp:` value into lowercase server
// names — the spelling the child's PI_MCP resolution matches on; "all" keeps
// its special meaning of "every configured server". A value that leaves no
// usable name warns and returns nil, so the child keeps the one-shot default
// instead of a silently empty request.
func parseAgentMCPList(agentName, value string) []string {
	var names []string
	for _, n := range strings.Split(value, ",") {
		if n = strings.ToLower(strings.TrimSpace(n)); n != "" {
			names = append(names, n)
		}
	}
	if len(names) == 0 {
		slog.Warn("subagent: mcp: names no server; ignoring",
			"agent", agentName, "value", value)
		return nil
	}
	return names
}

// parseAgentToolList splits a comma-separated frontmatter `tools:` value,
// dropping empty entries.
func parseAgentToolList(value string) []string {
	var tools []string
	for _, t := range strings.Split(value, ",") {
		if t = strings.TrimSpace(t); t != "" {
			tools = append(tools, t)
		}
	}
	return tools
}

// parseAgentFrontmatterLine parses "key: value" from a frontmatter line.
func parseAgentFrontmatterLine(line string) (key, value string, ok bool) {
	parts := strings.SplitN(line, ":", 2)
	if len(parts) != 2 {
		return "", "", false
	}
	return strings.TrimSpace(parts[0]), strings.TrimSpace(parts[1]), true
}

// LoadAgentsFromDir loads all agent markdown files from a directory.
// Returns empty slice if directory doesn't exist (not an error).
func LoadAgentsFromDir(dir string) ([]AgentConfig, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("reading agents dir %s: %w", dir, err)
	}

	var agents []AgentConfig
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".md") {
			continue
		}
		path := filepath.Join(dir, entry.Name())
		agent, err := ParseAgentFile(path)
		if err != nil {
			return nil, fmt.Errorf("parsing %s: %w", path, err)
		}
		agents = append(agents, agent)
	}
	return agents, nil
}

// findNearestProjectAgentsDir walks up from cwd looking for .pirate/agents/.
func findNearestProjectAgentsDir(cwd string) (string, error) {
	dir := cwd
	for {
		agentsDir := filepath.Join(dir, config.ProjectDirName, "agents")
		if info, err := os.Stat(agentsDir); err == nil && info.IsDir() {
			return agentsDir, nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break // Reached root
		}
		dir = parent
	}
	return "", os.ErrNotExist
}

// loadAgentsWithSource loads a directory of agent files and stamps each one
// with the source it came from ("user" or "project").
func loadAgentsWithSource(dir, source string) ([]AgentConfig, error) {
	agents, err := LoadAgentsFromDir(dir)
	if err != nil {
		return nil, err
	}
	for i := range agents {
		agents[i].Source = source
	}
	return agents, nil
}

// mergeAgentsByName appends agents to all, replacing in place any entry already
// present under the same name. seen maps agent name → index in all and is
// updated as entries are appended, so successive calls override earlier ones.
func mergeAgentsByName(all []AgentConfig, seen map[string]int, agents []AgentConfig) []AgentConfig {
	for _, agent := range agents {
		if idx, ok := seen[agent.Name]; ok {
			all[idx] = agent
			continue
		}
		seen[agent.Name] = len(all)
		all = append(all, agent)
	}
	return all
}

// DiscoverAgents loads agents from bundled, user, and project directories.
// Priority: project > user > bundled (later sources override earlier ones by name).
func DiscoverAgents(cwd string, scope AgentScope) (*AgentDiscoveryResult, error) {
	result := &AgentDiscoveryResult{}

	// Load bundled agents
	bundledAgents, err := LoadBundledAgents()
	if err != nil {
		return nil, fmt.Errorf("loading bundled agents: %w", err)
	}
	result.Bundled = bundledAgents

	// Load user agents (~/.pirate/agents/)
	if _, homeErr := os.UserHomeDir(); homeErr == nil {
		userAgents, err := loadAgentsWithSource(filepath.Join(config.PirateHome(), "agents"), "user")
		if err != nil {
			return nil, fmt.Errorf("loading user agents: %w", err)
		}
		result.User = userAgents
	}

	// Load project agents (.pirate/agents/ in nearest ancestor)
	if projectDir, findErr := findNearestProjectAgentsDir(cwd); findErr == nil {
		projectAgents, err := loadAgentsWithSource(projectDir, "project")
		if err != nil {
			return nil, fmt.Errorf("loading project agents: %w", err)
		}
		result.Project = projectAgents
	}

	// Merge all agents with priority: project > user > bundled — each pass
	// overrides same-named entries left by the passes before it.
	seen := make(map[string]int) // name → index in All
	result.All = mergeAgentsByName(result.All, seen, result.Bundled)
	result.All = mergeAgentsByName(result.All, seen, result.User)
	result.All = mergeAgentsByName(result.All, seen, result.Project)

	// Filter based on scope
	switch scope {
	case ScopeBundled:
		result.All = result.Bundled
	case ScopeProject:
		result.All = append(result.Project, result.User...)
	}

	return result, nil
}

// FindAgent looks up an agent by name from a discovery result.
func FindAgent(result *AgentDiscoveryResult, name string) (AgentConfig, bool) {
	for _, agent := range result.All {
		if agent.Name == name {
			return agent, true
		}
	}
	return AgentConfig{}, false
}
