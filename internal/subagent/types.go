package subagent

import (
	"fmt"
	"time"

	"github.com/spa-skyson/pi-rate/internal/session"
)

// SpawnInput is the input to spawn a subagent with an AgentConfig.
type SpawnInput struct {
	Agent        AgentConfig `json:"agent"`                   // Agent configuration
	Prompt       string      `json:"prompt"`                  // Task prompt for the agent
	Worktree     *bool       `json:"worktree,omitempty"`      // Override worktree setting
	WorktreeName string      `json:"worktree_name,omitempty"` // Optional worktree/branch name prefix
	// WorktreeBase is the commit the worktree's branch starts at (a branch, tag,
	// sha, or `HEAD~2`). Empty means HEAD, which is the behavior every existing
	// caller gets. It exists for readers that need to see a committed change: a
	// worktree branched from HEAD shows an empty diff, so a review of an
	// already-merged PR has nothing to read. Ignored when WorkDir is set.
	WorktreeBase string   `json:"worktree_base,omitempty"`
	WorkDir      string   `json:"work_dir,omitempty"`     // Override working directory (e.g. existing worktree path)
	Background   bool     `json:"background,omitempty"`   // Run in background
	SkipCleanup  bool     `json:"skip_cleanup,omitempty"` // Don't auto-cleanup worktree on completion
	Env          []string `json:"env,omitempty"`          // Additional environment variables
	MaxRetries   int      `json:"max_retries,omitempty"`  // Max retry attempts on crash (default 0, max 3)
	Timeout      int      `json:"timeout,omitempty"`      // Absolute timeout override in milliseconds
	// ModelOverride replaces the resolved model for this spawn; used by
	// SpawnWithRetry to restart on a fallback model after a fatal provider
	// error. Empty means the agent model/role chain decides.
	ModelOverride string `json:"model_override,omitempty"`

	// Attribution records where the spawned agent sits in a run tree. The
	// orchestrator forwards it through the environment; the child writes it
	// onto its own session metadata. Nil for spawns that are not part of one.
	Attribution *session.AgentContext `json:"attribution,omitempty"`
}

// AgentInput is the legacy input to spawn a subagent (deprecated, use SpawnInput).
// It is kept for backward compatibility with existing callers.
// Orchestrator.SpawnWithInput converts it against the full agent registry
// (project > user > bundled); the standalone ToSpawnInput sees bundled
// agents only.
type AgentInput struct {
	Type         string `json:"type"`                    // Agent type name
	Prompt       string `json:"prompt"`                  // Task prompt for the agent
	Worktree     *bool  `json:"worktree,omitempty"`      // Override worktree setting
	WorktreeName string `json:"worktree_name,omitempty"` // Optional worktree/branch name prefix
	// WorktreeBase is the commit the worktree's branch starts at. Empty means
	// HEAD; see SpawnInput.WorktreeBase.
	WorktreeBase string `json:"worktree_base,omitempty"`
	WorkDir      string `json:"work_dir,omitempty"`     // Override working directory (e.g. existing worktree path)
	Background   bool   `json:"background,omitempty"`   // Run in background
	SkipCleanup  bool   `json:"skip_cleanup,omitempty"` // Deprecated: worktree cleanup is always deferred to the caller or shutdown
	Timeout      int    `json:"timeout,omitempty"`      // Absolute timeout override in milliseconds

	// Attribution records where the spawned agent sits in a run tree.
	Attribution *session.AgentContext `json:"attribution,omitempty"`
}

// ToSpawnInput converts a legacy AgentInput to the new SpawnInput format.
// It looks up the agent config from bundled agents ONLY — user and project
// agents are invisible here. For the full registry use
// Orchestrator.SpawnWithInput, which resolves through LookupAgent.
func (a AgentInput) ToSpawnInput() (SpawnInput, error) {
	// Look up the agent from bundled agents
	bundled, err := LoadBundledAgents()
	if err != nil {
		return SpawnInput{}, fmt.Errorf("loading bundled agents: %w", err)
	}

	var agent AgentConfig
	found := false
	for _, cfg := range bundled {
		if cfg.Name == a.Type {
			agent = cfg
			found = true
			break
		}
	}
	if !found {
		return SpawnInput{}, fmt.Errorf("unknown agent type %q; valid types: captain, first-mate, skipper, cabin-boy, memory-compressor", a.Type)
	}

	return SpawnInput{
		Agent:        agent,
		Prompt:       a.Prompt,
		Worktree:     a.Worktree,
		WorktreeName: a.WorktreeName,
		WorktreeBase: a.WorktreeBase,
		WorkDir:      a.WorkDir,
		Background:   a.Background,
		SkipCleanup:  a.SkipCleanup,
		Timeout:      a.Timeout,
		Attribution:  a.Attribution,
	}, nil
}

// AgentOutput is the result of a completed subagent.
type AgentOutput struct {
	AgentID  string `json:"agent_id"`
	Type     string `json:"type"`
	Result   string `json:"result"`
	Error    string `json:"error,omitempty"`
	Duration string `json:"duration"`
}

// AgentStatus represents the current state of a subagent.
type AgentStatus struct {
	AgentID   string    `json:"agent_id"`
	Type      string    `json:"type"`
	Status    string    `json:"status"` // "running", "completed", "failed", "canceled", "killed"
	Prompt    string    `json:"prompt"`
	StartedAt time.Time `json:"started_at"`
	Duration  string    `json:"duration,omitempty"`
}

// Event is a streaming event from a subagent process.
type Event struct {
	Type       string `json:"type"`                 // "text_delta", "tool_call", "tool_result", "message_end", "error"
	Content    string `json:"content,omitempty"`    // Text content for text_delta; tool name for tool_call; JSON for tool_result
	Error      string `json:"error,omitempty"`      // Error message for error events
	SessionID  string `json:"session_id,omitempty"` // Subprocess session ID (from message_start)
	StopReason string `json:"stopReason,omitempty"` // ACP stopReason on message_end
	ToolArgs   any    `json:"tool_input,omitempty"` // Tool arguments for tool_call (from pi --mode json)
	Status     string `json:"status,omitempty"`     // Final status for run_done events
}

// EventFallback is synthesized by the SpawnWithInputFallback stream between
// two attempts: the previous attempt died of a fatal provider error and the
// next runs on the fallback model. Content carries the human-readable notice;
// it is not an error event, and consumers must not count it as a failure.
const EventFallback = "fallback"
