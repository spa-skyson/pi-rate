package tui

import (
	"context"

	"github.com/spa-skyson/pi-rate/internal/agent"
	"github.com/spa-skyson/pi-rate/internal/config"
	"github.com/spa-skyson/pi-rate/internal/extension"
	"github.com/spa-skyson/pi-rate/internal/logger"
	"github.com/spa-skyson/pi-rate/internal/palace"
	"github.com/spa-skyson/pi-rate/internal/permission"
	pisession "github.com/spa-skyson/pi-rate/internal/session"
	"github.com/spa-skyson/pi-rate/internal/subagent"
	"github.com/spa-skyson/pi-rate/internal/tools"

	llmmodel "google.golang.org/adk/v2/model"
	adktool "google.golang.org/adk/v2/tool"
)

// Config holds configuration for the TUI.
type Config struct {
	Agent        *agent.Agent
	LLM          llmmodel.LLM // The active LLM, used by /ping.
	SessionID    string
	AppVersion   string
	ModelName    string
	ProviderName string
	// ThinkingLevel is the configured reasoning effort: "none", "low",
	// "medium", "high" or "max", or empty for the provider's own default.
	//
	// It is read at startup by the CLI, which threads it into
	// provider.NewLLM, and shown by the sidebar under the model name. The two
	// are deliberately separate: this field reports what was configured, while
	// a provider that ignores the level still accepts it, so the row describes
	// the request rather than a guarantee.
	ThinkingLevel  string
	ActiveRole     string
	Roles          map[string]config.RoleConfig
	SessionService *pisession.FileService
	WorkDir        string
	Orchestrator   *subagent.Orchestrator
	// GenerateCommitMsg is called by /commit to generate a conventional commit message from diffs.
	// If nil, /commit is disabled.
	GenerateCommitMsg func(ctx context.Context, diffs string) (string, error)
	// Logger is the session logger. If nil, logging is disabled.
	Logger *logger.Logger
	// Skills is loaded from skill directories for command completion.
	Skills []extension.Skill
	// SkillDirs are the directories to re-scan for skills on each completion.
	SkillDirs []string
	// AgentEventCh receives subagent events from the agent tool for live display.
	AgentEventCh <-chan AgentSubEvent
	// SystemNoticeCh receives short system messages that must be shown in the
	// chat — auto-compaction outcomes, for instance. Compaction discards
	// history, so it must never happen silently.
	SystemNoticeCh <-chan string
	// ApprovalCh receives tool-approval requests from the permission gate
	// (permission ask rules), tagged with the session that raised them. Each
	// request blocks the agent loop until the dialog answers it via
	// request.Reply. Nil disables the dialog: sessions without it fall back
	// to the non-interactive denial of ask.
	ApprovalCh <-chan SessionApproval
	// QuestionCh receives question requests from the question tool, tagged
	// with the session that asked. Each request blocks the tool call until
	// the dialog answers it via request.Reply. Nil keeps the tool in its
	// headless mode: it returns canceled immediately instead of blocking.
	QuestionCh <-chan SessionQuestion
	// NewSessionAgent, when non-nil, forks a fresh session on demand:
	// CreateSession on the shared SessionService plus agent.New over the
	// shared tool instances, with asker closures tagged to the new session.
	// Ctrl+B uses it to open a brand-new session in the active slot while
	// the running one moves whole into the background slot. Nil disables the
	// fork — Ctrl+B refuses with a notice instead of backgrounding.
	NewSessionAgent NewSessionAgentFunc
	// Attention gates the terminal attention signals (bell, OSC 777
	// notification) sent when an approval dialog opens or a long turn
	// completes. Nil turns attention off entirely — no sequences, and focus
	// reporting stays off too. The CLI passes the resolved config section;
	// tests that build Config directly get the quiet default.
	Attention *config.AttentionConfig
	// ContextBreakdown attributes fixed context overhead (system prompt, tool
	// definitions, rules, skills, MCP tools, subagents) to its origins, so the
	// gauge can show what is filling the window rather than only how much.
	// Nil renders the gauge in a single severity color.
	ContextBreakdown *ContextBreakdown
	// TokenTracker tracks daily token usage and enforces limits. May be nil.
	TokenTracker TokenTracker
	// CompactMetrics tracks output compaction statistics. May be nil.
	CompactMetrics CompactStatsProvider
	// ThemeName is the configured theme name from config. Empty or "default" uses tokyo-night.
	ThemeName string
	// PlanAutoFix enables the automatic plan-repair loop: when a /plan session
	// ends with a spec that fails the PDD contract, the blocking findings are
	// fed back to the planner as its next prompt instead of waiting for a human
	// turn. Bounded by maxPlanFixCycles.
	PlanAutoFix bool
	// MidTurnAttempts is the resolved mid-turn replay budget (issues #41,
	// #43): how many times a turn may run in total when a transient failure
	// cuts it short mid-reply. The CLI resolves it once at startup (env →
	// config.json → default, config.ResolveMidTurnAttempts); nil runs the
	// default, which is what Configs built directly in tests get.
	MidTurnAttempts *int
	// LifecycleHooks are shell-command hooks fired on agent lifecycle events
	// (turn_complete, user_input_required). Each carries the event name and a
	// small data payload as JSON on stdin. Nil disables lifecycle hooks.
	LifecycleHooks []extension.HookConfig
	// Palace is the resolved palace config (paths plus embedder choice) the
	// CLI computes once via palaceConfigFromCLI. The sidebar status tick opens
	// the palace with it, so the embedder it reports is the configured backend
	// rather than a default the user never asked for. Nil keeps the defaults —
	// what Configs built directly in tests get.
	Palace *palace.PalaceConfig

	// DeferredInit, if non-nil, is a channel of InitEvent messages.
	// When set, the TUI starts immediately in loading state and receives
	// initialization progress updates. The final event carries the fully
	// initialized subsystems in its Result field.
	DeferredInit <-chan InitEvent

	// MCPToolsets holds the live MCP toolsets, used by /mcp to show status.
	MCPToolsets []adktool.Toolset
	// MCPServers holds the configured MCP server definitions.
	MCPServers []extension.MCPServerConfig
	// A2A holds the configured A2A agent endpoints, shown in the sidebar.
	A2A *config.A2AConfig

	// TodoCh receives todo/plan state updates after every todo_write call.
	// Buffered, non-blocking send; the TUI keeps the latest state. Nil = no
	// todo tracking.
	TodoCh <-chan tools.TodoState

	// ModelSwitcher creates a new LLM instance for the given model name,
	// updates the token tracker's context window size, and returns the
	// wrapped LLM, resolved model name, and provider. Used by /model <name>.
	// If nil, model switching via /model is disabled.
	ModelSwitcher func(ctx context.Context, modelName string) (llmmodel.LLM, string, string, error)

	// ModelCandidates seeds the /model picker popup: models declared under
	// config.json providers ("<provider>/<model>") plus the configured roles,
	// each entry an executable /model argument. Built by cli at startup.
	ModelCandidates []SearchItem
	// ModelCandidatesRefresh, when non-nil, produces the fuller candidate
	// list in the background when the models popup opens: each configured
	// named provider's catalog, from its cache or a live fetch. The result
	// replaces the open popup's entries; an empty result leaves the list as
	// it is (errors are the implementation's to swallow). Nil disables the
	// background refresh.
	ModelCandidatesRefresh func(ctx context.Context) []SearchItem

	// SubagentStatuses, when non-nil, returns the orchestrator's view of the
	// session's subagents for the /subagents monitor popup: who is running,
	// who finished, and for how long. The cli closes over Orchestrator.List().
	// The stream itself comes from the transcript cards, stitched to these
	// statuses by agentID; nil degrades the monitor to cards only, with
	// statuses inferred from the cards' results.
	SubagentStatuses func() []subagent.AgentStatus

	// SteerSubagent sends text to a running subagent as a follow-up message
	// (the monitor's `s` key). The cli closes over Orchestrator.Steer; nil
	// disables steering, and the key says so via a notice.
	SteerSubagent func(agentID, text string) error

	// PrimaryAgents lists the agents the main session can switch into
	// (frontmatter `mode: primary` or `all`), sorted by name. Filled by
	// deferred init; empty until then and when no agent declares a mode.
	PrimaryAgents []subagent.AgentConfig
	// AgentSwitcher applies a primary agent ("" = the built-in default) to
	// the main session: it builds the target's LLM — nil when no model
	// resolves, meaning "keep the current one" — and returns the system
	// prompt and the rebuilt tool-callback chains (the agent's permission
	// rules merged over the global ones, a step limiter for its budget).
	// modelOverride, when non-empty, is a session /model recorded for this
	// agent and beats its `model:`/`role:`; "" lets the agent's own
	// specification stand. Used by /agent and Shift+Tab. If nil, agent
	// switching is disabled.
	AgentSwitcher func(ctx context.Context, agentName string, modelOverride string) (AgentSwitch, error)

	// CheckUpdate, when non-nil, fetches the newest release and returns its
	// tag when it is newer than the running build ("" = up to date). Powers
	// the startup notice and /update; nil disables both. A disabled check
	// (dev builds) is an error so /update can say why.
	CheckUpdate func(ctx context.Context) (string, error)
	// ApplyUpdate runs the official install script with its output captured —
	// the TUI owns the terminal, so the script must never write to it. The
	// error carries the tail of the captured output. Used by /update after
	// the y/n confirmation; nil disables it.
	ApplyUpdate func(ctx context.Context) error
}

// NewSessionAgentFunc builds a fresh agent and session for a new active
// slot: it creates a session on the shared SessionService, then agent.New
// over the shared tool instances with per-session asker closures tagged to
// the new sessionID, and the autocompact pre-turn hook installed. The llm
// argument is the active session's current model — TUI state (/model,
// /agent) the CLI cannot snapshot at startup — so the fork lands on the
// model the user was actually talking to.
type NewSessionAgentFunc func(ctx context.Context, llm llmmodel.LLM) (ag *agent.Agent, sessionID, title string, err error)

// SessionApproval is an approval request tagged with the session that raised
// it. ApprovalCh is global while sessions are not: with a background session
// running next to the active one, two live loops can block on approvals at
// the same time, and the tag is the only thing that tells the TUI whose
// dialog to open. An empty tag is a factory bug — the request is denied and
// logged, never routed by guesswork.
type SessionApproval struct {
	Req       permission.ApprovalRequest
	SessionID string
}

// SessionQuestion is SessionApproval's twin for the question tool.
type SessionQuestion struct {
	Req       tools.QuestionRequest
	SessionID string
}

// AgentSwitch is the payload AgentSwitcher returns for one switch target.
type AgentSwitch struct {
	// Name is the target's agent identity, stamped onto the rebuilt runner
	// so its events render under the agent's own name. Empty for the
	// built-in default agent (restores AgentName).
	Name string
	// LLM is the model to run the session on. Nil keeps the current one:
	// the target named no `model:`/`role:` model.
	LLM       llmmodel.LLM
	ModelName string
	Provider  string
	// Instruction is the system prompt for the target (the session's rules
	// and skills sections are kept by the CLI when building it).
	Instruction string
	// Callback chains rebuilt for the target. Assigned to the agent as
	// given — nil clears them.
	BeforeTool []agent.BeforeToolCallback
	AfterTool  []agent.AfterToolCallback
}

// InitEvent reports progress from deferred initialization.
type InitEvent struct {
	Item   string      // subsystem name (e.g. "lsp", "memory", "mcp")
	Done   bool        // true when this item finished loading
	Total  int         // planned subsystem count when known; 0 means derive from seen items
	Result *InitResult // set on the final event when all init is complete
	Err    error       // fatal initialization error
}

// InitResult holds the fully initialized subsystems delivered by deferred init.
type InitResult struct {
	Agent     *agent.Agent
	SessionID string
	// SessionTitle is the default title the agent applied to the new session
	// (git repo name, or CWD basename). The TUI seeds its terminal window/tab
	// title with it so the user sees a label before the first user prompt
	// arrives. Empty if the agent has no title namer or no CWD to derive from.
	SessionTitle string
	// Resumed reports that SessionID names an existing session rather than one
	// created for this run, so the TUI knows to rebuild its transcript from the
	// session's events instead of opening on the welcome splash.
	Resumed           bool
	SessionService    *pisession.FileService
	Orchestrator      *subagent.Orchestrator
	Logger            *logger.Logger
	Skills            []extension.Skill
	SkillDirs         []string
	GenerateCommitMsg func(context.Context, string) (string, error)
	AgentEventCh      <-chan AgentSubEvent
	SystemNoticeCh    <-chan string
	ContextBreakdown  *ContextBreakdown
	TokenTracker      TokenTracker
	CompactMetrics    CompactStatsProvider
	GitBranch         string
	DiffAdded         int
	DiffRemoved       int
	// MCPToolsets holds the live MCP toolsets for /mcp status display.
	MCPToolsets []adktool.Toolset
	// MCPServers holds the configured MCP server definitions.
	MCPServers []extension.MCPServerConfig
	// LLM, when non-nil, replaces the startup LLM: defaultAgent restarted
	// the session on the agent's own model. ModelName and ProviderName then
	// describe it; empty names keep the startup values.
	LLM          llmmodel.LLM
	ModelName    string
	ProviderName string
	// ActiveAgent names the primary agent the session started in via
	// defaultAgent ("" = the built-in default agent).
	ActiveAgent string
	// PrimaryAgents lists the switchable agents (mode: primary or all),
	// sorted by name, for /agent and Shift+Tab.
	PrimaryAgents []subagent.AgentConfig
	// NewSessionAgent carries the session factory (see Config.NewSessionAgent).
	NewSessionAgent NewSessionAgentFunc
}

// CompactStatsProvider provides compaction statistics for TUI display.
type CompactStatsProvider interface {
	FormatStats() string
}

// TokenTracker provides read access to daily token usage for the status bar.
type TokenTracker interface {
	Limit() int64
	Remaining() int64     // -1 if unlimited
	PercentUsed() float64 // 0-100+
	TotalUsed() int64     // total tokens consumed today

	// Session context window tracking.
	LastPromptTokens() int64     // most recent prompt tokens from LLM response
	ContextWindowSize() int64    // model's context window size (0 = unknown)
	ContextPercentUsed() float64 // context window usage 0-100+

	// SetLastPromptTokens overrides the last-prompt baseline. Compaction
	// passes call this with the post-pass token count so the context gauge
	// reflects the new window immediately rather than waiting for the next
	// LLM response. Implementations also reset the cached-prefix baseline
	// (the new window has a new prefix by definition).
	SetLastPromptTokens(n int64)

	// ResetContextWindow zeroes the per-window baselines (last prompt,
	// cached-prefix and last-cached counts) so the gauge reads empty. /clear
	// calls this after the session's events are gone. SetLastPromptTokens(0)
	// is deliberately not a substitute: it keeps a non-zero prompt count on
	// purpose, so it can never empty the gauge. Daily usage totals are not
	// affected — clearing a conversation does not un-spend tokens.
	ResetContextWindow()

	// Prompt-cache tracking. Prompt tokens are billed at a steep discount when
	// served from cache, so LastPromptTokens alone overstates cost.
	LastCachedTokens() int64    // cache reads on the most recent response
	CachedTokensToday() int64   // cache reads accumulated today
	CacheHitRateToday() float64 // share of today's prompt tokens cached, 0-100
	BodyTokens() int64          // tokens accumulated after the cached prefix
	CachePrefixTokens() int64   // stable cached prefix for this window
}

// BashEventKind namespaces a live shell-command event so it can share the
// subagent event channel without being mistaken for subagent activity.
// Producers outside this package must route bash events through it rather than
// hard-coding the prefix.
func BashEventKind(kind string) string {
	return bashEventPrefix + kind
}

// AgentSubEvent carries a subagent event from the agent tool to the TUI.
type AgentSubEvent struct {
	AgentID    string
	Kind       string // "tool_call", "tool_result", "text_delta", etc.
	Content    string
	PipelineID string // groups agents in same call
	Mode       string // "single", "parallel", "chain"
	Step       int    // 1-based position in pipeline
	Total      int    // total agents in pipeline
	Background bool   // true for spawn/done of a background agent
}
