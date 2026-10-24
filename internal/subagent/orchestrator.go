package subagent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/spa-skyson/pi-rate/internal/config"
	"github.com/spa-skyson/pi-rate/internal/permission"
	"github.com/spa-skyson/pi-rate/internal/retry"
	"github.com/spa-skyson/pi-rate/internal/session"
)

// DefaultPoolSize is the default maximum number of concurrent subagents.
//
// Three rather than five, because the budget is per process and a spawned
// coordinator builds its own pool: with the Coordinator/Worker SOP the totals
// multiply with nesting. Measured over six hours of runs, 39% of sessions died
// on the provider's per-minute token limit at a peak of eight agents in
// flight. Raise it with PI_SUBAGENT_CONCURRENCY where the account has headroom
// — see ConcurrencyFromEnv.
const DefaultPoolSize = 3

// recentTaskTTL is how long a completed subagent result is kept before being evicted.
const recentTaskTTL = 30 * time.Minute

// maxCompletedAgents is the maximum number of completed/failed agent states
// retained in the in-memory agents map. Once this limit is exceeded, the
// oldest completed entries are evicted.
const maxCompletedAgents = 50

// recentTask tracks a recently completed subagent result for deduplication.
type recentTask struct {
	CompletedAt time.Time
	Summary     string // short summary of the result (first line or truncated)
	Status      string // "completed", "failed"
}

// Orchestrator composes Pool, Spawner, and WorktreeManager to manage subagent lifecycle.
type Orchestrator struct {
	pool     *Pool
	spawner  *Spawner
	worktree *WorktreeManager
	cfg      *config.Config
	repoRoot string                 // git repo root; used as default WorkDir for subagents
	registry map[string]AgentConfig // agent name → config (from discovery)
	agents   map[string]*agentState
	mu       sync.Mutex
	closed   bool // set by Shutdown to reject new Spawn calls

	// recentTasks tracks completed subagent research tasks for deduplication.
	// Key is a normalized version of the task prompt.
	recentTasks   map[string]recentTask
	recentTasksMu sync.RWMutex
	pruneTicker   *time.Ticker // periodic cleanup of expired recentTasks
	pruneTickerMu sync.Mutex

	// LLM provider settings passed to child subagents
	BaseURL  string   // LLM API base URL
	Insecure bool     // Skip TLS verification
	Headers  []string // Extra HTTP headers

	// ACP event log — when set, events from ACP subagents (claude, gemini) are
	// appended as JSONL to this path.
	acpLogPath string
	acpLogMu   sync.Mutex
}

// acpAgentNames is the set of bundled agent names that are ACP subprocess
// adapters; their event streams are tee'd to the session's acp.jsonl when an
// ACP log path is configured on the orchestrator.
var acpAgentNames = map[string]struct{}{
	"claude":  {},
	"gemini":  {},
	"cursor":  {},
	"copilot": {},
	"agy":     {},
}

// isACPAgent reports whether the named agent is an ACP subprocess adapter.
func isACPAgent(name string) bool {
	_, ok := acpAgentNames[name]
	return ok
}

// agentState tracks the runtime state of a subagent.
type agentState struct {
	ID          string
	Type        string
	Prompt      string
	StartedAt   time.Time
	FinishedAt  time.Time // set when status changes from "running"
	Process     *Process
	Worktree    bool   // whether a worktree was created
	SkipCleanup bool   // don't auto-cleanup worktree on completion (for gate validation)
	Status      string // "running", "completed", "failed", "canceled", "killed"
}

// NewOrchestrator creates an Orchestrator from config.
// repoRoot is the git repository root (empty string disables worktree support).
// agentConfigs are the discovered agent definitions (from DiscoverAgents + bundled).
func NewOrchestrator(cfg *config.Config, repoRoot string, agentConfigs []AgentConfig) *Orchestrator {
	var wm *WorktreeManager
	if repoRoot != "" {
		wm = NewWorktreeManager(repoRoot)
	}

	registry := make(map[string]AgentConfig, len(agentConfigs))
	for _, ac := range agentConfigs {
		registry[ac.Name] = ac
	}

	return &Orchestrator{
		pool:        NewPool(ConcurrencyFromEnv()),
		spawner:     NewSpawner(""),
		worktree:    wm,
		cfg:         cfg,
		repoRoot:    repoRoot,
		registry:    registry,
		agents:      make(map[string]*agentState),
		recentTasks: make(map[string]recentTask),
	}
}

// SetProviderOptions configures the LLM provider settings that will be passed
// to child subagents. Call this after NewOrchestrator to propagate CLI flags.
func (o *Orchestrator) SetProviderOptions(baseURL string, insecure bool, headers []string) {
	o.BaseURL = baseURL
	o.Insecure = insecure
	o.Headers = headers
}

// SetPiBinary overrides the binary used to spawn pi subagents.
//
// NewSpawner("") defaults to os.Executable(), which in a test harness is the
// test binary itself — spawning that would re-exec the test instead of a real
// pi worker. The eval harness resolves the real binary and points the
// orchestrator at it here.
func (o *Orchestrator) SetPiBinary(path string) {
	o.spawner.PiBinary = path
}

// SetACPLogPath configures where ACP subagent events (claude, gemini) are
// captured as JSONL. Pass an empty string to disable.
func (o *Orchestrator) SetACPLogPath(path string) {
	o.acpLogMu.Lock()
	defer o.acpLogMu.Unlock()
	o.acpLogPath = path
}

// writeACPEvent appends a single event entry to the configured acp.jsonl.
// Each entry wraps the underlying Event with agent metadata and a timestamp
// so a single file can hold events from multiple ACP subagent runs.
func (o *Orchestrator) writeACPEvent(agentID, agentType string, ev Event) {
	o.acpLogMu.Lock()
	path := o.acpLogPath
	if path == "" {
		o.acpLogMu.Unlock()
		return
	}
	entry := struct {
		Timestamp time.Time `json:"timestamp"`
		AgentID   string    `json:"agent_id"`
		Agent     string    `json:"agent"`
		Event     Event     `json:"event"`
	}{time.Now(), agentID, agentType, ev}
	data, err := json.Marshal(entry)
	if err != nil {
		o.acpLogMu.Unlock()
		return
	}
	data = append(data, '\n')
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		o.acpLogMu.Unlock()
		return
	}
	_, _ = f.Write(data)
	_ = f.Close()
	o.acpLogMu.Unlock()
}

// ensurePruneLoop starts the periodic cleanup goroutine if not already running.
// It's called lazily on first Spawn to avoid starting goroutines in tests.
func (o *Orchestrator) ensurePruneLoop() {
	o.pruneTickerMu.Lock()
	defer o.pruneTickerMu.Unlock()
	if o.pruneTicker == nil {
		o.pruneTicker = time.NewTicker(recentTaskTTL)
		go func() {
			for range o.pruneTicker.C {
				o.pruneRecentTasks()
			}
		}()
	}
}

// RegisterAgents replaces the agent registry with the given configs.
func (o *Orchestrator) RegisterAgents(configs []AgentConfig) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.registry = make(map[string]AgentConfig, len(configs))
	for _, ac := range configs {
		o.registry[ac.Name] = ac
	}
}

// AgentNames returns the names of all registered agents.
func (o *Orchestrator) AgentNames() []string {
	o.mu.Lock()
	defer o.mu.Unlock()
	names := make([]string, 0, len(o.registry))
	for name := range o.registry {
		names = append(names, name)
	}
	return names
}

// normalizeTaskKey produces a stable lowercase key from a task prompt for deduplication.
func normalizeTaskKey(prompt string) string {
	// Normalize: lowercase, collapse whitespace, truncate to 200 chars.
	// This captures semantic similarity without triggering on minor differences.
	norm := prompt
	norm = strings.ToLower(norm)
	norm = strings.Join(strings.Fields(norm), " ")
	if len(norm) > 200 {
		norm = norm[:200]
	}
	return norm
}

// RecentTaskResult holds the result of a recently completed subagent task.
type RecentTaskResult struct {
	CompletedAt time.Time
	Summary     string
	Status      string
}

// pruneRecentTasks removes expired entries from recentTasks map.
// Called periodically or on shutdown to prevent unbounded growth.
func (o *Orchestrator) pruneRecentTasks() {
	cutoff := time.Now().Add(-recentTaskTTL)
	o.recentTasksMu.Lock()
	defer o.recentTasksMu.Unlock()
	for k, v := range o.recentTasks {
		if v.CompletedAt.Before(cutoff) {
			delete(o.recentTasks, k)
		}
	}
}

// FindRecentTask checks if a task matching the given prompt was completed recently.
// Returns the result if found and not expired, nil otherwise.
func (o *Orchestrator) FindRecentTask(prompt string) *RecentTaskResult {
	o.recentTasksMu.RLock()
	defer o.recentTasksMu.RUnlock()

	key := normalizeTaskKey(prompt)
	task, ok := o.recentTasks[key]
	if !ok {
		return nil
	}
	if time.Since(task.CompletedAt) > recentTaskTTL {
		return nil
	}
	return &RecentTaskResult{
		CompletedAt: task.CompletedAt,
		Summary:     task.Summary,
		Status:      task.Status,
	}
}

// RecordTask marks a completed subagent task for deduplication.
func (o *Orchestrator) RecordTask(prompt string, summary, status string) {
	o.recentTasksMu.Lock()
	defer o.recentTasksMu.Unlock()

	key := normalizeTaskKey(prompt)
	// Truncate summary to first line, max 200 chars.
	summary = strings.TrimSpace(summary)
	if len(summary) > 200 {
		summary = summary[:200]
	}
	o.recentTasks[key] = recentTask{
		CompletedAt: time.Now(),
		Summary:     summary,
		Status:      status,
	}
	// Note: pruneRecentTasks should be called periodically or on shutdown
}

// LookupAgent returns the AgentConfig for the given name, or an error if not found.
func (o *Orchestrator) LookupAgent(name string) (AgentConfig, error) {
	o.mu.Lock()
	defer o.mu.Unlock()
	ac, ok := o.registry[name]
	if !ok {
		names := make([]string, 0, len(o.registry))
		for n := range o.registry {
			names = append(names, n)
		}
		return AgentConfig{}, fmt.Errorf("unknown agent %q; available: %v", name, names)
	}
	return ac, nil
}

// AgentModel returns the model string the named agent spawns on — the same
// raw resolution the Spawn path applies (frontmatter `model:` wins, otherwise
// the agent's role), so callers can show the child's model without
// duplicating the rules. Empty for an unknown agent or an unresolvable role.
func (o *Orchestrator) AgentModel(name string) string {
	agent, err := o.LookupAgent(name)
	if err != nil {
		return ""
	}
	model, err := agentSpawnModel(o.cfg, agent)
	if err != nil {
		return ""
	}
	return model
}

// SpawnWithRetry spawns a subagent with automatic retry on crash (up to
// maxRetries) and, when an attempt ends in a fatal provider error (a terminal
// error event with text - the child WithRetry has already exhausted the
// transient ones), restarts it on the next model of the agent fallback chain:
// frontmatter fallback-models, else the role fallbackModels, capped at
// maxAgentFallbacks. Each fallback restart buys one extra attempt on top of
// the crash budget; crash retries never change the model. The whole chain —
// the first attempt and every restart — shares one timeout budget, taken when
// the first attempt spawns, so a restart inherits only what the earlier
// attempts left of it. ACP and codex agents are exempt: their runners
// deliberately ignore SpawnOpts.Model (spawner_acp.go, spawner_codex.go), so
// a fallback restart could not reach them. Returns the final events channel,
// agentID, and error (nil on success).
func (o *Orchestrator) SpawnWithRetry(ctx context.Context, input SpawnInput) (<-chan Event, string, error) {
	return o.spawnWithRetry(ctx, input, nil, nil)
}

// chainBudget is the chain-wide timeout budget for one SpawnWithRetry call:
// one WithTimeout covering the first attempt and every restart, each
// inheriting whatever is left of it. It is taken at the first spawn — a
// budget taken at the first restart would hand the chain the agent timeout
// twice over.
type chainBudget struct {
	ctx    context.Context
	cancel context.CancelFunc
}

// start creates the budget: a context derived from parent with the chain's
// absolute timeout. It runs exactly once per chain, before the first attempt.
func (b *chainBudget) start(parent context.Context, agentTimeoutMs int) {
	b.ctx, b.cancel = context.WithTimeout(parent, ResolveTimeout(agentTimeoutMs).Absolute)
}

// effectiveSpawnTimeout returns the absolute spawn timeout in effect for this
// input: an explicit SpawnInput.Timeout overrides the agent's own declaration.
// Spawn (the child's own absolute limit) and the fallback chain budget must
// answer this identically — a chain budget derived from anything else would
// hand a restart a different ceiling than the attempt it restarts, silently
// zeroing the agent's declared timeout on the second try.
func effectiveSpawnTimeout(agent AgentConfig, input SpawnInput) int {
	if input.Timeout > 0 {
		return input.Timeout
	}
	return agent.Timeout
}

// streamIdleEnv renders an agent's frontmatter `streamIdleTimeout:` as the
// child env override the config resolver reads (EnvStreamIdleTimeoutMS).
// Nil frontmatter renders nothing — absence is inherit, "0" is off — and
// append accepts the empty slice, so the caller needs no nil check.
func streamIdleEnv(agent AgentConfig) []string {
	if agent.StreamIdleTimeout == nil {
		return nil
	}
	return []string{fmt.Sprintf("%s=%d", config.EnvStreamIdleTimeoutMS, *agent.StreamIdleTimeout)}
}

// mcpEnv renders an agent's frontmatter `mcp:` as the child env override the
// CLI reads (config.EnvMCP, the --mcp fallback). Empty frontmatter renders
// nothing — the child keeps the one-shot default of lifting no MCP server
// (issue #68) unless its own flags or environment name one. buildCommand
// lets this entry override an inherited PI_MCP rather than race it as a
// duplicate.
func mcpEnv(agent AgentConfig) []string {
	if len(agent.MCP) == 0 {
		return nil
	}
	return []string{config.EnvMCP + "=" + strings.Join(agent.MCP, ",")}
}

// personaEnv renders the persona budget the child applies to its own
// whole-prompt gate: the instruction this spawn passes as --system IS the
// agent's persona, and the child — a process that never sees the agent file
// — must know its size to resolve the persona-aware limit
// (agent.Config.PersonaBudgetBytes, issue #85). An agent without an
// instruction renders nothing: the child keeps the plain 32K gate.
func personaEnv(agent AgentConfig) []string {
	if agent.Instruction == "" {
		return nil
	}
	return []string{fmt.Sprintf("%s=%d", config.EnvPersonaBytes, len(agent.Instruction))}
}

// streamStart is the synchronous handshake of SpawnWithInputFallback: the
// first attempt that produced a live event stream, or the failure when no
// attempt ever did.
type streamStart struct {
	agentID string
	err     error
}

// spawnWithRetry is SpawnWithRetry's loop. tap, when non-nil, receives every
// event of every attempt the loop observes (see SpawnWithInputFallback); an
// attempt it abandons contributes nothing past its terminal event. first
// carries the streamStart handshake exactly once.
func (o *Orchestrator) spawnWithRetry(ctx context.Context, input SpawnInput, tap chan<- Event, first chan<- streamStart) (<-chan Event, string, error) {
	maxRetries := clampRetries(input.MaxRetries)
	// External runners (ACP, codex) cannot take a fallback model, so their
	// error events never steer the chain; awaitAttemptOutcome classifies them
	// by tracked process state instead, exactly as before this path existed.
	externalRunner := isExternalRunnerAgent(input.Agent.Name)

	// The model the current attempt runs on, for the restart notice. A
	// resolution failure leaves it empty: the notice then names no model,
	// which is cosmetic, not load-bearing.
	currentModel, _ := agentSpawnModel(o.cfg, input.Agent)

	// Fallback state, resolved lazily on the first provider error: agents
	// that never hit one pay nothing for the chain.
	var fallbacks []string
	fallbacksResolved := false
	fallbackUsed := 0
	// The chain budget starts here, at the first spawn — not at the first
	// restart — so attempt 0's time is subtracted too and the attempts
	// together never exceed one effectiveSpawnTimeout.
	chain := &chainBudget{}
	chain.start(ctx, effectiveSpawnTimeout(input.Agent, input))
	// failExit tears the chain budget down on the exits that leave nothing
	// running: every attempt's stream has been drained to close, so no
	// process still needs the context. The success exits deliberately skip
	// it — they hand back a live stream whose attempt runs under the same
	// budget, and canceling here would kill a healthy child mid-flight ("pi
	// process failed: context canceled" right after its result). A budget
	// left standing expires on its own, exactly like a plain Spawn's timeout.
	// It also completes the streamStart handshake when no attempt ever
	// produced a stream.
	failExit := func(agentID string, err error) (<-chan Event, string, error) {
		// A false startFallback on entries still unused means the chain
		// budget, not the chain, ran out — say so, or the error is
		// indistinguishable from "no fallbacks configured".
		if err != nil && chain.ctx.Err() != nil && fallbackUsed < len(fallbacks) {
			err = fmt.Errorf("%w: chain budget exhausted", err)
		}
		chain.cancel()
		if first != nil {
			first <- streamStart{err: err}
		}
		return nil, agentID, err
	}

	// startFallback resolves the chain on first use and, when an unused entry
	// and chain budget remain, advances to the next model and reports whether
	// the chain continues. errText is the failure being fled from, for the
	// restart notice.
	startFallback := func(errText string) bool {
		if !fallbacksResolved {
			fallbacks = agentFallbackModels(o.cfg, input.Agent)
			fallbacksResolved = true
		}
		if fallbackUsed >= len(fallbacks) {
			return false
		}
		if chain.ctx.Err() != nil {
			// The chain budget is spent: a restart would begin past its own
			// deadline. Reporting false ends the chain, and the failExit
			// wrap below appends "chain budget exhausted" — the fallbacks
			// remaining here would otherwise make this indistinguishable
			// from "no fallbacks configured".
			return false
		}
		next := fallbacks[fallbackUsed]
		fallbackUsed++
		if tap != nil {
			tap <- Event{Type: EventFallback, Content: fmt.Sprintf(
				"model %s failed (%s); restarting on %s",
				currentModel, firstLine(errText), next)}
		}
		input.ModelOverride = next
		currentModel = next
		return true
	}

	var finalAgentID string
	var finalEvents <-chan Event

	for attempt := 0; ; attempt++ {
		events, agentID, err := o.Spawn(chain.ctx, input)
		if err != nil {
			if attempt >= maxRetries {
				return failExit("", fmt.Errorf("spawn failed after %d attempts: %w", attempt+1, err))
			}
			continue
		}
		if first != nil {
			first <- streamStart{agentID: agentID}
			first = nil
		}

		finalAgentID = agentID
		finalEvents = events

		// If no retries configured, return immediately - the caller gets
		// the stream undrained.
		if maxRetries == 0 {
			return finalEvents, finalAgentID, nil
		}

		outcome, errText := o.awaitAttemptOutcome(events, agentID, externalRunner, tap)
		switch {
		case outcome == attemptHealthy, outcome == attemptSilent && attempt >= maxRetries:
			return finalEvents, finalAgentID, nil
		case outcome == attemptProviderError:
			if startFallback(errText) {
				continue
			}
			// Chain exhausted or none configured: the provider error is
			// final. Crash retries must not burn attempts on a dead account.
			return failExit(agentID, fmt.Errorf("subagent %s failed: %s", agentID, errText))
		case outcome == attemptStopped:
			// Timed out or otherwise stopped for good: neither a crash worth
			// re-spawning nor a model problem a fallback could fix.
			return failExit(agentID, fmt.Errorf("subagent %s stopped: %s", agentID, errText))
		case outcome == attemptCrashed:
			// A crash whose last error text still reads as a windowless
			// provider wall — a 503 the child could not ride out, reaching the
			// parent through the child's own error event or the exit-error
			// synthetic (issue #39) — is the model failing, not the process.
			// crashIsModelFailure classifies it here, on the outcome, so the
			// fallback happens before any crash retry burns on the dead model;
			// it also refuses external runners (ACP, codex): they ignore
			// ModelOverride (isExternalRunnerAgent), so a restart never reaches
			// a different model and their error texts must not spend the chain.
			if crashIsModelFailure(errText, externalRunner) {
				if startFallback(errText) {
					continue
				}
			}
			// Not a model problem, or the chain is spent: the plain crash
			// path — re-spawn on the same model while attempts remain, end
			// the chain once the budget is spent. The budget check lives in
			// this case rather than in a separate attempt >= maxRetries arm:
			// crashes were the only outcome that ever reached that arm, and
			// the transient check above already diverts every model-shaped
			// crash out of here before the budget can matter.
			if attempt >= maxRetries {
				return failExit(agentID, fmt.Errorf("subagent %s crashed after %d attempts", agentID, attempt+1))
			}
		}
		// Silent, or a crash the chain could not answer, with attempts left:
		// re-spawn on the same model.
	}
}

// crashIsModelFailure reports whether a crash's last error text describes the
// model failing — a failure a different model could survive — and therefore
// whether spawnWithRetry's crash case should spend the fallback chain on it
// (issue #39). Three guards keep process-shaped text out:
//
//   - externalRunner (ACP, codex) is refused outright: those runners ignore
//     SpawnOpts.Model (isExternalRunnerAgent), so a restart could not land on
//     a different model, and their error texts — transient-looking or not —
//     must not consume the chain.
//   - the spawner's own synthetic exit error ("pi process failed: …",
//     spawner.go's childExitError) describes the process dying, not the model:
//     a child killed before it emitted a verdict of its own must not have its
//     stderr noise read as one. This is deliberately prefix-shaped, not a
//     "agent run:" requirement — a bare "503" without any wrapper (the
//     raw-5xx path) has to keep steering the chain.
//   - an empty text is nothing to classify.
//
// IsExhaustedTransient is deliberately absent here although the child's
// exhausted budget ("transient error after N retries: …") is the other
// model-shaped crash: awaitAttemptOutcome's gate intercepts every
// marker-bearing error event — synthetic included — as attemptProviderError
// before the crash classification runs, so no text reaching this helper can
// carry the marker. A server-named window ("retry in 59s") is excluded inside
// crashTransientNoWindow: that failure clears by waiting, so no model switch
// is due.
func crashIsModelFailure(errText string, externalRunner bool) bool {
	if errText == "" || externalRunner {
		return false
	}
	if strings.HasPrefix(errText, "pi process failed:") {
		return false
	}
	return crashTransientNoWindow(errText)
}

// crashTransientNoWindow reports whether a crash's final error text is a
// transient provider failure the server gave no retry window for (issue #39,
// the raw-503 path): the child burned its own retry budget on it, so waiting
// longer on this model buys nothing a different model would not.
func crashTransientNoWindow(errText string) bool {
	err := errors.New(errText)
	return retry.IsTransient(err) && !retry.HasServerWindow(err)
}

// firstLine trims a provider error to its first line for a notice: the
// payloads arrive multi-line and a subscriber renders the text verbatim.
func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	return s
}

// SpawnWithInputFallback is SpawnWithInput with the fallback mechanics of
// SpawnWithRetry and a stream that survives the restarts: the returned
// channel carries every event of every attempt — a failed attempt's stream up
// to and including its terminal error, a synthetic "fallback" notice before
// each restart, and the final attempt's stream to completion, its run_done
// included. A failed attempt's trailing events (the spawner's exit noise and
// its own run_done) are kept off the stream, so a subscriber can never
// mistake an abandoned attempt for the end.
//
// The agent id returned synchronously is the first attempt's; a restart runs
// under a new id. MaxRetries below 1 is raised to 1: the retry loop
// classifies attempts only when it may re-spawn, and without that a provider
// error could never reach the fallback chain.
func (o *Orchestrator) SpawnWithInputFallback(ctx context.Context, input AgentInput) (<-chan Event, string, error) {
	spawnInput, err := o.resolveAgentInput(input)
	if err != nil {
		return nil, "", err
	}
	if spawnInput.MaxRetries < 1 {
		spawnInput.MaxRetries = 1
	}

	out := make(chan Event, 64)
	start := make(chan streamStart, 1)
	go func() {
		defer close(out)
		events, _, err := o.spawnWithRetry(ctx, spawnInput, out, start)
		if err != nil {
			// No attempt produced a usable stream, or the chain died after
			// its restarts: surface the failure as the stream's terminal
			// event, so a subscriber reading only events still sees it.
			out <- Event{Type: "error", Error: err.Error()}
			return
		}
		// The winning attempt's stream continues past the terminal event the
		// loop consumed — run_done and whatever preceded it. Forward the
		// remainder: the subscriber's stream ends the way a plain Spawn's
		// does.
		for ev := range events {
			out <- ev
		}
	}()

	s := <-start
	if s.err != nil {
		return nil, "", s.err
	}
	return out, s.agentID, nil
}

// clampRetries pins a requested retry budget to the supported range: never
// negative, never more than three re-spawns.
func clampRetries(maxRetries int) int {
	if maxRetries < 0 {
		return 0
	}
	if maxRetries > 3 {
		return 3
	}
	return maxRetries
}

// attemptOutcome is how one SpawnWithRetry attempt ended, as observed on the
// agent's event stream.
type attemptOutcome int

const (
	// attemptHealthy: a terminal event arrived and the agent was not in a
	// crashed state — the stream belongs to the caller now.
	attemptHealthy attemptOutcome = iota
	// attemptCrashed: a terminal event arrived while the agent was tracked as
	// "failed" or "killed".
	attemptCrashed
	// attemptSilent: the stream closed without ever producing a terminal event.
	attemptSilent

	// attemptProviderError: the child emitted a terminal error event with
	// text that reads as a fatal provider failure (quota, 402, auth, an
	// exhausted transient budget — the shared retry classification, on a
	// pi runner only: an external runner's error events never classify as
	// this). Distinct from a crash: classified from the event, not from
	// process state. This is the only outcome that steers the fallback chain.
	attemptProviderError

	// attemptStopped: an error event whose text is not a provider wall, and
	// the agent did not crash — a timeout, a cancel, a clean exit after an
	// error. Final like a provider error (no re-spawn buys anything), but a
	// model switch could not fix it.
	attemptStopped
)

// awaitAttemptOutcome consumes events until the first terminal one
// ("message_end" or "error") and reports how the attempt ended. Events before
// that are forwarded to tap when it is non-nil — that is how the streaming
// wrapper's subscriber sees a failed attempt's stream — and dropped otherwise,
// which is why the maxRetries==0 shortcut in spawnWithRetry skips this
// entirely. The second return value is the provider error text when the
// outcome is attemptProviderError, the stop text for attemptStopped, and the
// last error event's text for attemptCrashed (empty when the attempt never
// emitted one) — the crash path classifies from it in spawnWithRetry.
//
// externalRunner routes an external runner's error events away from the
// provider-error classification: a pi child's text is pi's own provider
// verdict, but an ACP/codex runner's is its CLI's, and those runners cannot
// take a fallback model anyway.
func (o *Orchestrator) awaitAttemptOutcome(events <-chan Event, agentID string, externalRunner bool, tap chan<- Event) (attemptOutcome, string) {
	// lastErrText remembers the latest error event's text: a crash verdict is
	// classified from it, since the crash path itself carries no text (issue
	// #39 — the crash-path classification in spawnWithRetry reads it). It can
	// never carry the exhausted-transient marker: the gate below catches every
	// marker-bearing error event first.
	lastErrText := ""
	for ev := range events {
		if tap != nil {
			tap <- ev
		}
		if ev.Type == "error" && ev.Error != "" {
			lastErrText = ev.Error
		}
		if ev.Type != "message_end" && ev.Type != "error" {
			continue
		}
		// A provider error is classified from the event text itself, before
		// any process-state check: the status write races the event
		// forwarding, but the event is authoritative. Only a failure the
		// shared retry classification calls terminal may steer the fallback
		// chain — a crashed child's synthesized exit error ("pi process
		// failed: exit status 1: ...") must not, or a crash would be treated
		// as a model problem and burn the chain on it.
		if ev.Type == "error" && ev.Error != "" {
			// An exhausted transient budget (the child's WithRetryContext
			// wrapper, possibly re-wrapped by the spawner's exit-error
			// synthetic) reads as terminal too: the child already spent its
			// whole retry budget, so re-waiting on this model buys nothing
			// (issue #39). A windowed rate limit is excluded inside
			// IsExhaustedTransient — that one clears on its own.
			if !externalRunner && (retry.IsTerminal(errors.New(ev.Error)) || retry.IsExhaustedTransient(errors.New(ev.Error))) {
				o.drainAttempt(events)
				return attemptProviderError, ev.Error
			}
			// Everything else — a crash's exit error, a timeout, an external
			// runner's failure — is classified by tracked process state,
			// exactly as before the fallback path existed. Drain first: the
			// forwarder writes the terminal status only after the stream
			// closes, so consuming the rest orders that write before the
			// agentCrashed read below instead of racing it.
			o.drainAttempt(events)
			if o.agentCrashed(agentID) {
				return attemptCrashed, lastErrText
			}
			if externalRunner {
				// An external runner that failed without crashing is handed
				// back as a success, as before the fallback path existed.
				return attemptHealthy, ""
			}
			return attemptStopped, ev.Error
		}
		if o.agentCrashed(agentID) {
			return attemptCrashed, lastErrText
		}
		return attemptHealthy, ""
	}
	return attemptSilent, ""
}

// drainAttempt consumes the rest of an attempt's event stream without
// forwarding it anywhere. A failed attempt's trailing events — the spawner's
// synthetic exit error, the forwarder's run_done — describe an attempt the
// retry loop is about to abandon; leaking the run_done in particular would
// make the subscriber treat the whole chain as finished. Draining to close
// also orders the forwarder's terminal status write before the caller's next
// agentCrashed read.
func (o *Orchestrator) drainAttempt(events <-chan Event) {
	for range events {
	}
}

// agentCrashed reports whether the tracked agent is in a state worth
// re-spawning. An agent that has already been evicted from the map is not
// treated as crashed.
func (o *Orchestrator) agentCrashed(agentID string) bool {
	o.mu.Lock()
	defer o.mu.Unlock()
	state := o.agents[agentID]
	if state == nil {
		return false
	}
	return state.Status == "failed" || state.Status == "killed"
}

// Spawn starts a new subagent and returns an event channel.
// It acquires a pool slot, optionally creates a worktree, and spawns the pi process.
func (o *Orchestrator) Spawn(ctx context.Context, input SpawnInput) (<-chan Event, string, error) {
	// Start the prune loop lazily on first use
	o.ensurePruneLoop()

	o.mu.Lock()
	if o.closed {
		o.mu.Unlock()
		return nil, "", fmt.Errorf("orchestrator is shut down")
	}
	o.mu.Unlock()

	// Validate agent config.
	agent := input.Agent
	if agent.Name == "" {
		return nil, "", fmt.Errorf("agent config must have a name")
	}

	// Resolve the model this agent runs on: a frontmatter `model:` wins
	// outright, otherwise the agent's role is resolved.
	model, err := agentSpawnModel(o.cfg, agent)
	if err != nil {
		return nil, "", err
	}

	// A fallback restart (SpawnWithRetry) overrides the resolved model: the
	// frontmatter model/role chain decided only the first attempt.
	if input.ModelOverride != "" {
		model = input.ModelOverride
	}

	// Acquire a pool slot.
	if err := o.pool.Acquire(ctx); err != nil {
		return nil, "", fmt.Errorf("acquiring pool slot: %w", err)
	}

	// Generate agent ID.
	agentID := fmt.Sprintf("%s-%d", agent.Name, uniqueNano())

	// Determine if worktree is needed.
	useWorktree := resolveWorktreeUsage(agent.Worktree, input.Worktree, input.WorkDir, input.WorktreeBase)

	workDir, err := o.resolveWorkDir(agentID, input, useWorktree)
	if err != nil {
		o.pool.Release()
		return nil, "", err
	}

	// An explicit spawn timeout overrides the agent definition; otherwise use
	// the timeout declared by the resolved agent.
	timeout := effectiveSpawnTimeout(agent, input)

	// Build spawn options shared by the pi spawner and the ACP dispatcher.
	// ThinkingLevel is normalized here rather than at parse time so the raw
	// frontmatter value stays inspectable and unknown spellings warn once per
	// spawn, next to everything else the orchestrator validates.
	spawnOpts := SpawnOpts{
		AgentID:       agentID,
		Model:         model,
		WorkDir:       workDir,
		Prompt:        input.Prompt,
		Instruction:   agent.Instruction,
		Timeout:       timeout,
		Env:           o.spawnEnv(input.Env, workDir, attributionFor(input, agentID, workDir)),
		BaseURL:       agentSpawnBaseURL(o.cfg, o.BaseURL, model),
		Insecure:      o.Insecure,
		Headers:       o.Headers,
		LSP:           agent.LSP,
		Tools:         agent.Tools,
		Temperature:   agent.Temperature,
		ThinkingLevel: NormalizeReasoningEffort(agent.ReasoningEffort),
		Steps:         agent.Steps,
	}

	// The child gates its own tool calls with this agent's permission rules.
	// They ride the environment because the child is a separate pi process:
	// opts.Env lands after the filtered parent env in cmd.Env, so a nested
	// spawn's rules correctly replace the ones this process inherited rather
	// than stacking on them. Marshal cannot fail for this shape; the warn
	// keeps an unexpected failure from turning into silent ungating.
	if !agent.Permission.Empty() {
		if blob, err := json.Marshal(agent.Permission); err == nil {
			spawnOpts.Env = append(spawnOpts.Env, permission.EnvVar+"="+string(blob))
		} else {
			slog.Warn("subagent: could not serialize permission rules; child runs without them",
				"agent", agent.Name, "error", err)
		}
	}

	// The child's stream-idle budget: the frontmatter value rides the
	// environment — nil means absent (the child keeps its own config or the
	// default), 0 means explicitly off — because it governs the child's own
	// LLM streams and the child is a separate pi process that never sees
	// this agent's frontmatter.
	spawnOpts.Env = append(spawnOpts.Env, streamIdleEnv(agent)...)

	// The child's MCP servers: same ride, same reason (issue #68). A child
	// is a one-shot json run and lifts no MCP server by default; the agent's
	// frontmatter `mcp:` names the ones it instructs the model to use.
	spawnOpts.Env = append(spawnOpts.Env, mcpEnv(agent)...)

	// The child's whole-prompt budget: the persona rides --system, and its
	// size rides the environment (config.EnvPersonaBytes) — the child is a
	// separate process that never sees the agent file, and its 32K whole-prompt
	// gate would reject a persona the file gate deliberately allows (issue
	// #85). Unset when the spawn carries no persona instruction, so a plain
	// child keeps the plain gate.
	spawnOpts.Env = append(spawnOpts.Env, personaEnv(agent)...)

	proc, err := o.dispatchSpawn(ctx, spawnOpts, agent.Name)
	if err != nil {
		o.abandonSpawn(agentID, useWorktree)
		return nil, "", fmt.Errorf("spawning agent: %w", err)
	}

	state := &agentState{
		ID:          agentID,
		Type:        agent.Name,
		Prompt:      input.Prompt,
		StartedAt:   time.Now(),
		Process:     proc,
		Worktree:    useWorktree && o.worktree != nil,
		SkipCleanup: input.SkipCleanup,
		Status:      "running",
	}

	if !o.trackAgent(state) {
		// Orchestrator shut down while we were setting up — clean up and bail.
		proc.Cancel()
		o.abandonSpawn(agentID, useWorktree)
		return nil, "", fmt.Errorf("orchestrator is shut down")
	}

	// Create a forwarding channel that handles cleanup on completion.
	events := make(chan Event, 64)
	// Codex subagents are logged to acp.jsonl alongside the ACP ones: the log
	// records external-CLI subagent event streams, and the Event shape is the
	// same regardless of which protocol produced it.
	logACP := isACPAgent(agent.Name) || isCodexAgent(agent.Name)
	go o.forwardAgentEvents(events, proc, state, logACP)

	return events, agentID, nil
}

// maxAgentFallbacks caps how many fallback models a subagent may try. It
// mirrors the TUI's own maxModelFallbacks; the two are intentionally not
// shared to avoid a tui→subagent dependency.
const maxAgentFallbacks = 3

// agentSpawnModel resolves the model a subagent runs on. A frontmatter
// `model:` on the agent wins outright — the name is forwarded as is and the
// child validates it. Otherwise the agent's role is resolved; when that
// yields a bare model name (no "/") on a declared provider, the provider's
// name is prefixed so the child can route the request from the model name
// alone, without knowing the parent's roles.
func agentSpawnModel(cfg *config.Config, agent AgentConfig) (string, error) {
	if agent.Model != "" {
		return agent.Model, nil
	}
	model, prov, _, _, _, err := cfg.ResolveRole(agent.Role)
	if err != nil {
		return "", fmt.Errorf("resolving role %q for agent %q: %w", agent.Role, agent.Name, err)
	}
	return normalizeSpawnModelName(cfg, prov, model), nil
}

// normalizeSpawnModelName prefixes a bare model name (no "/") with prov when
// that provider is declared, so the child can route the request from the
// model name alone. It is also applied to fallback-model names, where the
// provider is unknown: a bare fallback gets cfg.DefaultProvider as a best
// effort, and names that already carry a "/" or have no declared default
// provider pass through unchanged.
func normalizeSpawnModelName(cfg *config.Config, prov, model string) string {
	if strings.Contains(model, "/") {
		return model
	}
	if _, declared := cfg.Providers[prov]; declared {
		return prov + "/" + model
	}
	return model
}

// agentFallbackModels returns the fallback chain for a subagent: the agent's
// own frontmatter `fallback-models:` when set, otherwise the ones configured
// on the agent's role, falling back to the default role's. The chain is
// capped at maxAgentFallbacks; an empty result is returned as nil so callers
// can treat "no fallbacks" uniformly.
func agentFallbackModels(cfg *config.Config, agent AgentConfig) []string {
	models := agent.FallbackModels
	if len(models) == 0 {
		models = cfg.ResolveFallbackModels(agent.Role)
	}
	if len(models) == 0 {
		return nil
	}
	if len(models) > maxAgentFallbacks {
		models = models[:maxAgentFallbacks]
	}
	// A bare fallback name has no provider of its own; it rides the provider
	// of the model it is a fallback FOR — same account, same endpoint. Only
	// when the primary model itself carries no declared-provider prefix does
	// cfg.DefaultProvider step in as a best effort. The result is a fresh
	// slice: models may alias the parsed agent config, which is shared.
	prov := cfg.DefaultProvider
	if primary, err := agentSpawnModel(cfg, agent); err == nil {
		if name, _, ok := cfg.NamedProviderPrefix(primary); ok {
			prov = name
		}
	}
	out := make([]string, len(models))
	for i, m := range models {
		out[i] = normalizeSpawnModelName(cfg, prov, m)
	}
	return out
}

// agentSpawnBaseURL decides which --url the child gets. A named-provider
// model carries its endpoint and key in config.json, which the child reads
// itself; the parent's explicit --url describes the parent model's endpoint,
// and passing it through would override the provider's own (in the child the
// flag beats the provider's baseURL). Non-named models keep the parent URL.
func agentSpawnBaseURL(cfg *config.Config, parentURL, model string) string {
	if _, _, named := cfg.NamedProviderPrefix(model); named {
		return ""
	}
	return parentURL
}

// resolveWorkDir picks the directory the subagent runs in, creating its
// worktree when one is called for. A base ref branches the worktree from that
// commit instead of HEAD, which is what lets a read-only agent see an
// already-committed change.
func (o *Orchestrator) resolveWorkDir(agentID string, input SpawnInput, useWorktree bool) (string, error) {
	if input.WorkDir != "" {
		return input.WorkDir, nil
	}
	if useWorktree && o.worktree != nil {
		wtPath, err := o.worktree.CreateAt(agentID, input.WorktreeBase, input.WorktreeName)
		if err != nil {
			return "", fmt.Errorf("creating worktree: %w", err)
		}
		return wtPath, nil
	}
	return o.repoRoot, nil // default to project root so subagents run in the right directory
}

// spawnEnv extends the caller's env with the roots the subagent needs: the repo
// root so its sandbox covers the full repo and not just the worktree directory,
// and the worktree path so it can normalize relative paths.
// attributionFor fills in the fields the orchestrator knows — the agent's ID,
// its type, and the worktree it was given — over whatever the caller supplied.
// The caller owns the run-level fields (run ID, spec, slice, cycle) because
// only it knows them.
func attributionFor(input SpawnInput, agentID, workDir string) *session.AgentContext {
	var ctx session.AgentContext
	if input.Attribution != nil {
		ctx = *input.Attribution
	}
	if ctx.AgentID == "" {
		ctx.AgentID = agentID
	}
	if ctx.AgentType == "" {
		ctx.AgentType = input.Agent.Name
	}
	if ctx.Worktree == "" {
		ctx.Worktree = workDir
	}
	return &ctx
}

func (o *Orchestrator) spawnEnv(env []string, workDir string, attribution *session.AgentContext) []string {
	out := append([]string(nil), env...)
	if o.worktree != nil {
		out = append(out, "PI_SANDBOX_ROOT="+o.worktree.RepoRoot())
		if workDir != "" {
			out = append(out, "PI_WORKTREE_ROOT="+workDir)
		}
	}
	// Attribution rides the same channel. The child records it on its own
	// session so the run tree is recoverable from meta.json rather than by
	// grouping sessions by working directory and guessing at roles.
	out = append(out, attribution.Env()...)
	return out
}

// isExternalRunnerAgent reports whether the named agent runs through an
// external runner (an ACP adapter or the codex app-server). Those runners
// deliberately ignore SpawnOpts.Model (spawner_acp.go, spawner_codex.go), so
// a model fallback restart could not reach them.
func isExternalRunnerAgent(name string) bool {
	return isACPAgent(name) || isCodexAgent(name)
}

// dispatchSpawn launches the agent through the right runner. ACP-bundled agents
// (claude/gemini/cursor/copilot) launch their own CLI binary via the ACP
// adapter; codex agents launch `codex app-server` and speak its JSON-RPC
// protocol directly; everyone else runs as a child pi --mode json.
func (o *Orchestrator) dispatchSpawn(ctx context.Context, spawnOpts SpawnOpts, agentName string) (*Process, error) {
	switch {
	case isACPAgent(agentName):
		return dispatchACP(ctx, spawnOpts, agentName)
	case isCodexAgent(agentName):
		return dispatchCodex(ctx, spawnOpts, agentName)
	default:
		return o.spawner.Spawn(ctx, spawnOpts)
	}
}

// abandonSpawn gives back the resources taken for a spawn that never reached
// the running state.
func (o *Orchestrator) abandonSpawn(agentID string, useWorktree bool) {
	if useWorktree && o.worktree != nil {
		_ = o.worktree.Cleanup(agentID)
	}
	o.pool.Release()
}

// trackAgent registers a running agent, reporting false if the orchestrator was
// shut down while the spawn was being set up.
func (o *Orchestrator) trackAgent(state *agentState) bool {
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.closed {
		return false
	}
	o.agents[state.ID] = state
	return true
}

// forwardAgentEvents republishes the process's events on out, then records the
// terminal status and publishes it as a final run_done event.
func (o *Orchestrator) forwardAgentEvents(out chan Event, proc *Process, state *agentState, logACP bool) {
	defer close(out)
	defer o.pool.Release()

	for ev := range proc.Events() {
		if logACP {
			o.writeACPEvent(state.ID, state.Type, ev)
		}
		out <- ev
	}

	// Process done — update state.
	_, waitErr := proc.Wait()

	o.mu.Lock()
	if state.Status == "running" {
		state.Status = terminalStatus(waitErr)
		state.FinishedAt = time.Now()
	}
	finalStatus := state.Status
	o.evictCompletedAgentsLocked()
	o.mu.Unlock()

	// Publish the terminal status before closing the channel. Consumers must
	// use this snapshot rather than querying the evictable status map.
	out <- Event{Type: "run_done", Status: finalStatus}

	// Worktree cleanup is intentionally NOT done here. Deletion is
	// deferred to the caller (e.g. the TUI /run flow calls
	// wm.Cleanup after a successful merge) or to Shutdown via
	// CleanupAll. Auto-cleaning on process exit raced with the
	// merge step, causing "no worktree found" merge failures.
	_ = state.SkipCleanup // kept for API stability; no longer gated here
}

// terminalStatus distinguishes a timeout from a crash from an exit-code
// failure. A timeout is reported first because it also arrives as a signal
// kill — the spawner SIGKILLs the group — and would otherwise be
// indistinguishable from an OOM.
func terminalStatus(waitErr error) string {
	switch {
	case errors.Is(waitErr, ErrSubagentTimeout):
		return "timeout"
	case waitErr != nil && isKilledBySignal(waitErr):
		return "killed"
	case waitErr != nil:
		return "failed"
	default:
		return "completed"
	}
}

// isKilledBySignal returns true if the error indicates the process was killed by a signal
// (e.g., SIGKILL from timeout or OOM), as opposed to an exit code failure.
func isKilledBySignal(err error) bool {
	if err == nil {
		return false
	}
	// exec.Error is returned when cmd.Wait() encounters a signal-terminated process.
	// Check the error using errors.As for proper wrapped error handling.
	if execErr, ok := errors.AsType[*exec.ExitError](err); ok {
		// ExitError with no code typically means killed by signal.
		return !execErr.Success()
	}
	// For other error types, check if the message indicates a signal.
	errStr := err.Error()
	return len(errStr) >= 6 && errStr[len(errStr)-6:] == "killed"
}

// evictCompletedAgentsLocked removes the oldest completed/failed/killed/canceled
// agent states from the in-memory map when the number of non-running entries
// exceeds maxCompletedAgents. Caller must hold o.mu.
func (o *Orchestrator) evictCompletedAgentsLocked() {
	completed := 0
	for _, s := range o.agents {
		if s.Status != "running" {
			completed++
		}
	}
	if completed <= maxCompletedAgents {
		return
	}
	// Evict the oldest finished entries by FinishedAt.
	type entry struct {
		id   string
		time time.Time
	}
	finished := make([]entry, 0, completed)
	for id, s := range o.agents {
		if s.Status != "running" && !s.FinishedAt.IsZero() {
			finished = append(finished, entry{id, s.FinishedAt})
		}
	}
	// Sort by FinishedAt ascending (oldest first).
	sort.Slice(finished, func(i, j int) bool {
		return finished[i].time.Before(finished[j].time)
	})
	toRemove := completed - maxCompletedAgents
	for i := 0; i < toRemove && i < len(finished); i++ {
		delete(o.agents, finished[i].id)
	}
}

// List returns the status of all tracked agents.
func (o *Orchestrator) List() []AgentStatus {
	o.mu.Lock()
	defer o.mu.Unlock()

	statuses := make([]AgentStatus, 0, len(o.agents))
	for _, s := range o.agents {
		dur := ""
		if s.Status != "running" && !s.FinishedAt.IsZero() {
			dur = s.FinishedAt.Sub(s.StartedAt).Truncate(time.Millisecond).String()
		}
		statuses = append(statuses, AgentStatus{
			AgentID:   s.ID,
			Type:      s.Type,
			Status:    s.Status,
			Prompt:    s.Prompt,
			StartedAt: s.StartedAt,
			Duration:  dur,
		})
	}
	return statuses
}

// SetStatusForTest is a test-only helper that records a synthetic
// agent state so callers can exercise post-spawn verification paths
// without spawning a real subprocess. The agent is marked "running"
// until SetStatusForTest is called with a final status. Returns false
// if the orchestrator was already shut down. Not for production use.
func (o *Orchestrator) SetStatusForTest(agentID, status string) bool {
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.closed {
		return false
	}
	now := time.Now()
	o.agents[agentID] = &agentState{
		ID:         agentID,
		Type:       "test",
		StartedAt:  now,
		FinishedAt: now,
		Status:     status,
	}
	return true
}

// Get returns the status of a single tracked agent by ID.
// Returns (AgentStatus{}, false) when the agent is unknown. Useful for
// post-spawn verification (e.g. the /run flow checks exit status before
// moving on to gate validation).
func (o *Orchestrator) Get(agentID string) (AgentStatus, bool) {
	o.mu.Lock()
	defer o.mu.Unlock()

	state, ok := o.agents[agentID]
	if !ok {
		return AgentStatus{}, false
	}
	dur := ""
	if state.Status != "running" && !state.FinishedAt.IsZero() {
		dur = state.FinishedAt.Sub(state.StartedAt).Truncate(time.Millisecond).String()
	}
	return AgentStatus{
		AgentID:   state.ID,
		Type:      state.Type,
		Status:    state.Status,
		Prompt:    state.Prompt,
		StartedAt: state.StartedAt,
		Duration:  dur,
	}, true
}

// resolveWorktreeUsage determines whether a worktree should be created for a spawn.
//
// A non-empty worktreeBase forces one: the base only means something inside a
// worktree, so honoring it on the repo root would silently answer a different
// question than the caller asked. This is what makes a base ref work for a
// read-only agent, whose own definition sets no worktree — the caller asks for
// the diff, and the worktree is the only way to see it.
func resolveWorktreeUsage(agentDefault bool, inputOverride *bool, workDir, worktreeBase string) bool {
	if workDir != "" {
		return false // explicit work dir — no worktree
	}
	if inputOverride != nil {
		return *inputOverride
	}
	if worktreeBase != "" {
		return true
	}
	return agentDefault
}

// validateAgentForCancel checks whether an agent can be canceled.
func validateAgentForCancel(state *agentState, agentID string) error {
	if state == nil {
		return fmt.Errorf("agent %q not found", agentID)
	}
	if state.Status != "running" {
		return fmt.Errorf("agent %q is not running (status: %s)", agentID, state.Status)
	}
	return nil
}

// Cancel cancels a running agent by ID.
func (o *Orchestrator) Cancel(agentID string) error {
	o.mu.Lock()
	defer o.mu.Unlock()

	state := o.agents[agentID]
	if err := validateAgentForCancel(state, agentID); err != nil {
		return err
	}

	state.Process.Cancel()
	state.Status = "canceled"
	state.FinishedAt = time.Now()

	return nil
}

// Steer sends text to a running subagent as a follow-up message. The child
// queues it and runs it as an extra turn in its session once the current turn
// finishes; see Process.Steer. Works for background agents too — they are the
// same Process.
func (o *Orchestrator) Steer(agentID, text string) error {
	o.mu.Lock()
	state, ok := o.agents[agentID]
	o.mu.Unlock()
	if !ok {
		return fmt.Errorf("agent %q not found", agentID)
	}
	// The write happens outside o.mu: Steer writes to a pipe, and a child
	// wedged with a full pipe must not block the orchestrator's other calls.
	return state.Process.Steer(text)
}

// Concurrency reports how many subagents this process may run at once. It is
// the pool size, which is what actually gates a spawn — not maxParallelTasks,
// which only caps how many tasks one call may name. A batch larger than this
// does not run in parallel; it queues, and the call takes proportionally
// longer.
func (o *Orchestrator) Concurrency() int {
	if o == nil || o.pool == nil {
		return 0
	}
	return o.pool.Size()
}

// Worktree returns the WorktreeManager (may be nil if worktrees are disabled).
func (o *Orchestrator) Worktree() *WorktreeManager {
	return o.worktree
}

// Shutdown cancels all running agents and cleans up worktrees.
// For production use, prefer ShutdownWithTimeout.
func (o *Orchestrator) Shutdown() {
	o.ShutdownWithTimeout(5 * time.Second)
}

// ShutdownWithTimeout gracefully cancels all running agents with the given timeout,
// then forces cleanup of worktrees. The timeout applies to the entire shutdown
// operation, not individual agents.
func (o *Orchestrator) ShutdownWithTimeout(timeout time.Duration) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	// Stop the prune ticker if it was started and do final cleanup
	o.pruneTickerMu.Lock()
	if o.pruneTicker != nil {
		o.pruneTicker.Stop()
	}
	o.pruneTickerMu.Unlock()
	o.pruneRecentTasks()

	// First: graceful cancellation of running agents
	var hadRunning bool
	o.mu.Lock()
	o.closed = true
	for _, state := range o.agents {
		if state.Status == "running" {
			state.Process.Cancel()
			state.Status = "canceled"
			state.FinishedAt = time.Now()
			hadRunning = true
		}
	}
	o.mu.Unlock()

	// Only wait for the graceful timeout if we actually canceled something.
	if hadRunning {
		<-ctx.Done()
	}

	// Force cleanup of worktrees
	if o.worktree != nil {
		_ = o.worktree.CleanupAll()
	}
}

// SpawnWithInput is the legacy method that accepts AgentInput for backward compatibility.
// It resolves the agent through the orchestrator's registry (project > user >
// bundled) and calls Spawn.
// Deprecated: Use Spawn with SpawnInput directly.
func (o *Orchestrator) SpawnWithInput(ctx context.Context, input AgentInput) (<-chan Event, string, error) {
	spawnInput, err := o.resolveAgentInput(input)
	if err != nil {
		return nil, "", err
	}
	return o.Spawn(ctx, spawnInput)
}

// resolveAgentInput converts a legacy AgentInput to a SpawnInput, looking the
// agent up in the orchestrator's registry rather than in the bundled set
// alone: user and project agents are visible to the subagent tool's
// validation and to a bundled-only lookup is what made them pass validation
// and then fail to spawn.
func (o *Orchestrator) resolveAgentInput(input AgentInput) (SpawnInput, error) {
	agent, err := o.LookupAgent(input.Type)
	if err != nil {
		return SpawnInput{}, err
	}
	return SpawnInput{
		Agent:        agent,
		Prompt:       input.Prompt,
		Worktree:     input.Worktree,
		WorktreeName: input.WorktreeName,
		WorktreeBase: input.WorktreeBase,
		WorkDir:      input.WorkDir,
		Background:   input.Background,
		SkipCleanup:  input.SkipCleanup,
		Timeout:      input.Timeout,
		Attribution:  input.Attribution,
	}, nil
}

// lastAgentNano is the timestamp the previous agent ID was minted from.
var lastAgentNano atomic.Int64

// uniqueNano returns a strictly increasing nanosecond timestamp for agent IDs.
//
// The ID is "<name>-<nanos>", and the orchestrator keys its agent table on it,
// so two spawns minted from the same clock reading collide and the second
// silently replaces the first. On Linux that needs two spawns inside the same
// nanosecond; on Windows the monotonic clock ticks in 100ns steps -- and
// coarser still on older kernels -- so a retry after an instant crash, as
// SpawnWithRetry performs, reproduced it.
func uniqueNano() int64 {
	for {
		now := time.Now().UnixNano()
		last := lastAgentNano.Load()
		if now <= last {
			now = last + 1
		}
		if lastAgentNano.CompareAndSwap(last, now) {
			return now
		}
	}
}
