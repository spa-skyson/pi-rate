package tools

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool"

	"github.com/spa-skyson/pi-rate/internal/subagent"
)

// backgroundEntry holds the collected result of a background agent.
type backgroundEntry struct {
	Agent   string        `json:"agent"`
	AgentID string        `json:"agent_id"`
	Status  string        `json:"status"` // "running", "completed", "failed", "timeout"
	Result  string        `json:"result"`
	Error   string        `json:"error,omitempty"`
	done    chan struct{} // closed when status is terminal; read-only outside lock
}

// backgroundRegistry tracks spawned background agents by agentID.
type backgroundRegistry struct {
	mu sync.Mutex
	m  map[string]*backgroundEntry
}

func newBackgroundRegistry() *backgroundRegistry {
	return &backgroundRegistry{m: make(map[string]*backgroundEntry)}
}

func (r *backgroundRegistry) start(agentID, agent string) *backgroundEntry {
	r.mu.Lock()
	defer r.mu.Unlock()
	entry := &backgroundEntry{
		Agent:   agent,
		AgentID: agentID,
		Status:  "running",
		done:    make(chan struct{}),
	}
	r.m[agentID] = entry
	return entry
}

func (r *backgroundRegistry) finish(agentID, status, result, errMsg string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	entry, ok := r.m[agentID]
	if !ok {
		return
	}
	entry.Status = status
	entry.Result = result
	entry.Error = errMsg
	close(entry.done)
}

// get returns a copy of the entry and its done channel, both read under the
// lock so there is no race between reading status and reading the channel.
func (r *backgroundRegistry) get(agentID string) (Status string, Result string, Error string, done chan struct{}, ok bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	e, ok := r.m[agentID]
	if !ok {
		return "", "", "", nil, false
	}
	return e.Status, e.Result, e.Error, e.done, true
}

func (r *backgroundRegistry) list() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	ids := make([]string, 0, len(r.m))
	for id := range r.m {
		ids = append(ids, id)
	}
	return ids
}

// resolveContext extracts a context.Context from agent.Context, defaulting to context.Background().
func resolveContext(ctx agent.Context) context.Context {
	if ctx != nil {
		return ctx
	}
	return context.Background()
}

// maxParallelTasks is the maximum number of tasks allowed in parallel mode.
const maxParallelTasks = 8

// maxChainSteps is the maximum number of steps allowed in chain mode.
const maxChainSteps = 8

// SubagentInput defines the parameters for the subagent tool.
type SubagentInput struct {
	// Single mode: agent name to spawn.
	Agent string `json:"agent,omitempty"`
	// Single mode: task prompt for the agent.
	Task string `json:"task,omitempty"`
	// Single mode: commit the agent's worktree branches from — a branch, tag,
	// sha, or revision like main~3. Supplying it creates a worktree for this
	// spawn even when the agent's own definition sets none, which is what makes
	// an already-committed change reviewable: the agent runs in a tree where the
	// change is still a pending diff. Requires single mode.
	Base string `json:"base,omitempty"`

	// Background, when true, spawns the agent in the background and returns
	// immediately. Only supported in single mode. The model collects the result
	// with the agent_result tool.
	Background bool `json:"background,omitempty"`

	// Parallel mode: list of tasks to run concurrently.
	Tasks []TaskItem `json:"tasks,omitempty"`

	// Chain mode: sequential pipeline of agents.
	Chain []ChainItem `json:"chain,omitempty"`
}

// TaskItem defines a single task in parallel mode.
type TaskItem struct {
	Agent string `json:"agent"`
	Task  string `json:"task"`
}

// ChainItem defines a single step in chain mode.
type ChainItem struct {
	Agent string `json:"agent"`
	Task  string `json:"task"` // supports {previous} and {previous_json}
}

// SubagentOutput is the result from a completed subagent call.
type SubagentOutput struct {
	Mode    string        `json:"mode"`
	Results []AgentResult `json:"results"`
	Summary string        `json:"summary"`
}

// AgentResult holds the result from a single agent execution.
type AgentResult struct {
	Agent     string `json:"agent"`
	AgentID   string `json:"agent_id"`
	Status    string `json:"status"` // "completed", "failed", "timeout"
	Result    string `json:"result"`
	Error     string `json:"error,omitempty"`
	Duration  string `json:"duration"`
	SessionID string `json:"session_id,omitempty"`
}

// SubagentEvent extends agent events with pipeline metadata for the TUI.
type SubagentEvent struct {
	AgentID    string `json:"agent_id"`
	Kind       string `json:"kind"` // "spawn", "text_delta", "tool_call", "tool_result", "error", "done"
	Content    string `json:"content"`
	PipelineID string `json:"pipeline_id"` // groups agents in same call
	Mode       string `json:"mode"`        // "single", "parallel", "chain"
	Step       int    `json:"step"`        // 1-based position
	Total      int    `json:"total"`       // total agents in pipeline
	// Background marks spawn and done events for background agents.
	Background bool `json:"background,omitempty"`
}

// SubagentEventCallback is called for each subagent event with pipeline metadata.
type SubagentEventCallback func(event SubagentEvent)

// NewSubagentTool creates the subagent ADK tool wired to an Orchestrator.
func NewSubagentTool(orch *subagent.Orchestrator, onEvent SubagentEventCallback, bg *backgroundRegistry) (tool.Tool, error) {
	desc := buildSubagentDescription(orch)

	return newTool("subagent", desc,
		func(ctx agent.Context, input SubagentInput) (SubagentOutput, error) {
			return subagentHandler(ctx, orch, input, onEvent, bg)
		},
		// Common LLM parameter name mistakes
		map[string]string{
			"type":    "agent", // LLM sends "type" instead of "agent"
			"prompt":  "task",  // LLM sends "prompt" instead of "task"
			"message": "task",  // LLM sends "message" instead of "task"
			"items":   "tasks", // LLM sends "items" instead of "tasks"
			"steps":   "chain", // LLM sends "steps" instead of "chain"
		},
	)
}

// agentResultInput defines the parameters for the agent_result tool.
type agentResultInput struct {
	AgentID     string  `json:"agent_id"`
	WaitSeconds float64 `json:"wait_seconds,omitempty"`
}

// agentResultOutput is the result of an agent_result call.
type agentResultOutput struct {
	Status  string `json:"status"` // "running", "completed", "failed", "timeout"
	Result  string `json:"result,omitempty"`
	Error   string `json:"error,omitempty"`
	Summary string `json:"summary,omitempty"`
	// Hint is set when the agent is still running and no wait was requested.
	Hint string `json:"hint,omitempty"`
}

// SubagentTools returns tools containing the subagent tool and the agent_result tool.
func SubagentTools(orch *subagent.Orchestrator, onEvent SubagentEventCallback) ([]tool.Tool, error) {
	bg := newBackgroundRegistry()

	sub, err := NewSubagentTool(orch, onEvent, bg)
	if err != nil {
		return nil, err
	}

	ar, err := newTool("agent_result", `Collect the result of a background subagent spawned with background:true.

Input:
  agent_id: string — the agent_id returned by subagent when background:true was set
  wait_seconds: float — optional, wait up to this many seconds for the agent to finish (max 30)

When the agent is still running:
  - Without wait_seconds: returns {status:"running", hint:"call again with wait_seconds"}
  - With wait_seconds: waits up to min(wait_seconds, 30) seconds, returns {status:"running"} on timeout

When the agent is done: returns {status, result, error} with the full result.`,
		func(ctx agent.Context, input agentResultInput) (agentResultOutput, error) {
			return agentResultHandler(bg, input)
		},
	)
	if err != nil {
		return nil, err
	}

	return []tool.Tool{sub, ar}, nil
}

// buildSubagentDescription generates the tool description with available agent names.
func buildSubagentDescription(orch *subagent.Orchestrator) string {
	var b strings.Builder
	b.WriteString(`Spawn a subagent to perform a task autonomously.

Modes:
- Single: {agent: "<name>", task: "<prompt>"} — spawn one agent
- Single + background: {agent: "<name>", task: "<prompt>", background: true} — spawn and return immediately; collect with agent_result(agent_id)
- Parallel: {tasks: [{agent: "<name>", task: "<prompt>"}, ...]} — run multiple agents concurrently (max 8)
- Chain: {chain: [{agent: "<name>", task: "<prompt>"}, ...]} — run agents sequentially, passing results forward

Use background for long-running, independent tasks that do not block further work. The agent_result tool retrieves the result: call agent_result({agent_id: "<id>"}) to check status, or agent_result({agent_id: "<id>", wait_seconds: 10}) to wait up to 10 seconds for completion.

Worktree agents:
- Agents marked [worktree] edit an isolated git worktree; their edits are not applied to the current tree. Ask for an exact patch/file list to apply, or use a non-worktree editing agent.

Reviewing a committed change (single mode, base):
- A worktree is created at HEAD, so an agent sent to review an already-committed change sees a clean tree and an empty diff. Pass base to compare against that commit instead:
  {agent: "captain", task: "...", base: "main"}
- The worktree holds the current tip with its index at base, so the agent sees base..HEAD as a pending diff via git diff, and reads post-change file content.
- This works for read-only agents too: base creates the worktree even when the agent's own definition sets none, which is what makes a read-only reviewer usable.
- Reviewing a branch before merging: pass its merge base. For a pull request merged long ago, base alone compares against the current tip, so check that commit out first if you need exactly that diff.

`)

	b.WriteString("Available agents:\n")
	for _, name := range orch.AgentNames() {
		ac, err := orch.LookupAgent(name)
		if err != nil {
			continue
		}
		// `mode: primary` agents are reserved for the main session
		// (Shift+Tab / /agent); they are not advertised as spawnable.
		// Spawning one directly still works — this is a listing filter,
		// not a gate.
		if ac.PrimaryOnly() {
			continue
		}
		marker := ""
		if ac.Worktree {
			marker = " [worktree]"
		}
		fmt.Fprintf(&b, "- %s%s: %s\n", ac.Name, marker, ac.Description)
	}

	// Report the pool size, not just the per-call cap. They are different
	// numbers and the model needs the smaller one: maxParallelTasks limits how
	// many tasks a single call may name, while the pool is what actually gates
	// a spawn. Naming more tasks than the pool allows does not run them in
	// parallel — they queue, and the one tool call takes proportionally longer,
	// which is how a "parallel" batch turns into a timeout.
	concurrency := orch.Concurrency()
	fmt.Fprintf(&b, "\nAt most %d task(s) per call. This process runs %d subagent(s) at a time",
		maxParallelTasks, concurrency)
	if concurrency <= 1 {
		b.WriteString(" — parallel mode gives no speed-up here, so prefer one task per call")
	} else {
		fmt.Fprintf(&b, ", so batches larger than %d queue rather than overlap", concurrency)
	}
	b.WriteString(". Each agent runs as a separate process.")
	return b.String()
}

// subagentHandler dispatches to the appropriate mode handler.
func subagentHandler(ctx agent.Context, orch *subagent.Orchestrator, input SubagentInput, onEvent SubagentEventCallback, bg *backgroundRegistry) (SubagentOutput, error) {
	mode := detectMode(input)

	// Background is single-mode only.
	if input.Background && mode != "single" {
		return SubagentOutput{
			Mode:    mode,
			Summary: "background is single-mode only",
			Results: []AgentResult{{
				Agent:  mode,
				Status: "failed",
				Error:  fmt.Sprintf("background is single-mode only, not supported in %s mode", mode),
			}},
		}, nil
	}

	switch mode {
	case "single":
		if input.Background {
			return singleModeBackgroundHandler(ctx, orch, input, onEvent, bg)
		}
		return singleModeHandler(ctx, orch, input, onEvent)
	case "parallel":
		return parallelModeHandler(ctx, orch, input, onEvent)
	case "chain":
		return chainModeHandler(ctx, orch, input, onEvent)
	default:
		return SubagentOutput{}, fmt.Errorf(
			"could not detect mode: provide exactly ONE of:\n"+
				"  Single: {agent: \"\\u003cname\\u003e\", task: \"\\u003cprompt\\u003e\"} — spawn one agent\n"+
				"  Parallel: {tasks: [{agent: \"\\u003cname\\u003e\", task: \"\\u003cprompt\\u003e\"}, ...]} — run multiple agents concurrently\n"+
				"  Chain: {chain: [{agent: \"\\u003cname\\u003e\", task: \"\\u003cprompt\\u003e\"}, ...]} — run agents sequentially\n"+
				"Received: agent=%q, task=%q, tasks=%d, chain=%d",
			input.Agent, input.Task, len(input.Tasks), len(input.Chain))
	}
}

// detectMode determines the execution mode from the input fields.
// Chain takes priority over parallel, which takes priority over single.
func detectMode(input SubagentInput) string {
	// Chain mode takes priority
	if len(input.Chain) > 0 {
		return "chain"
	}
	// Parallel mode
	if len(input.Tasks) > 0 {
		return "parallel"
	}
	// Single mode: allow if either agent or task is present
	// This is lenient - we accept partial input to help recover from LLM mistakes
	if input.Agent != "" || input.Task != "" {
		return "single"
	}
	return ""
}

// singleModeHandler spawns a single agent and collects its result.
func singleModeHandler(ctx agent.Context, orch *subagent.Orchestrator, input SubagentInput, onEvent SubagentEventCallback) (SubagentOutput, error) {
	start := time.Now()
	pipelineID := fmt.Sprintf("pipe-%d", time.Now().UnixNano())

	// Validate agent exists.
	if _, err := orch.LookupAgent(input.Agent); err != nil {
		return SubagentOutput{
			Mode: "single",
			Results: []AgentResult{{
				Agent:    input.Agent,
				Status:   "failed",
				Error:    err.Error(),
				Duration: time.Since(start).Truncate(time.Millisecond).String(),
			}},
			Summary: fmt.Sprintf("unknown agent %q", input.Agent),
		}, nil
	}

	// Spawn the agent. The fallback-capable spawn keeps the stream whole
	// across a model restart: a fatal provider error on the primary model
	// surfaces as its error event, then a fallback notice, then the restart's
	// events — nothing is lost between attempts.
	events, agentID, err := orch.SpawnWithInputFallback(resolveContext(ctx), subagent.AgentInput{
		Type:         input.Agent,
		Prompt:       input.Task,
		WorktreeBase: input.Base,
	})
	if err != nil {
		return SubagentOutput{
			Mode: "single",
			Results: []AgentResult{{
				Agent:    input.Agent,
				Status:   "failed",
				Error:    err.Error(),
				Duration: time.Since(start).Truncate(time.Millisecond).String(),
			}},
			Summary: fmt.Sprintf("failed to spawn %s: %s", input.Agent, err),
		}, nil
	}

	// Emit spawn event.
	emitEvent(onEvent, SubagentEvent{
		AgentID:    agentID,
		Kind:       "spawn",
		Content:    input.Agent,
		PipelineID: pipelineID,
		Mode:       "single",
		Step:       1,
		Total:      1,
	})

	// Consume events, forward to TUI, accumulate result.
	resultText, status, errMsg, subSessionID := forwardSingleModeEvents(events, onEvent, agentID, pipelineID)

	emitEvent(onEvent, SubagentEvent{
		AgentID: agentID, Kind: "done",
		PipelineID: pipelineID, Mode: "single", Step: 1, Total: 1,
	})

	duration := time.Since(start).Truncate(time.Millisecond).String()

	return SubagentOutput{
		Mode: "single",
		Results: []AgentResult{{
			Agent:     input.Agent,
			AgentID:   agentID,
			Status:    status,
			Result:    resultText,
			Error:     errMsg,
			Duration:  duration,
			SessionID: subSessionID,
		}},
		Summary: fmt.Sprintf("%s %s in %s", input.Agent, status, duration),
	}, nil
}

// singleModeBackgroundHandler spawns a subagent in the background and returns
// immediately. The result is collected later via agent_result.
func singleModeBackgroundHandler(ctx agent.Context, orch *subagent.Orchestrator, input SubagentInput, onEvent SubagentEventCallback, bg *backgroundRegistry) (SubagentOutput, error) {
	pipelineID := fmt.Sprintf("pipe-%d", time.Now().UnixNano())

	// Validate agent exists.
	if _, err := orch.LookupAgent(input.Agent); err != nil {
		return SubagentOutput{
			Mode: "single",
			Results: []AgentResult{{
				Agent:  input.Agent,
				Status: "failed",
				Error:  err.Error(),
			}},
			Summary: fmt.Sprintf("unknown agent %q", input.Agent),
		}, nil
	}

	// Spawn the agent; see singleModeHandler for why the fallback-capable
	// spawn. The background reader below consumes the merged stream, so a
	// restart is transparent to agent_result.
	events, agentID, err := orch.SpawnWithInputFallback(resolveContext(ctx), subagent.AgentInput{
		Type:         input.Agent,
		Prompt:       input.Task,
		WorktreeBase: input.Base,
	})
	if err != nil {
		return SubagentOutput{
			Mode: "single",
			Results: []AgentResult{{
				Agent:  input.Agent,
				Status: "failed",
				Error:  err.Error(),
			}},
			Summary: fmt.Sprintf("failed to spawn %s: %s", input.Agent, err),
		}, nil
	}

	// Register in the background registry before emitting the spawn event.
	_ = bg.start(agentID, input.Agent) // entry stored in registry; result used by agent_result

	// Emit spawn event with background flag.
	emitEvent(onEvent, SubagentEvent{
		AgentID:    agentID,
		Kind:       "spawn",
		Content:    input.Agent,
		PipelineID: pipelineID,
		Mode:       "single",
		Step:       1,
		Total:      1,
		Background: true,
	})

	// Read events in a goroutine, forwarding each to the TUI, then
	// record the result in the registry.
	go func() {
		resultText, status, errMsg, _ := forwardSingleModeEvents(events, onEvent, agentID, pipelineID)
		bg.finish(agentID, status, resultText, errMsg)
		emitEvent(onEvent, SubagentEvent{
			AgentID:    agentID,
			Kind:       "done",
			PipelineID: pipelineID,
			Mode:       "single",
			Step:       1,
			Total:      1,
			Background: true,
		})
	}()

	return SubagentOutput{
		Mode: "single",
		Results: []AgentResult{{
			Agent:   input.Agent,
			AgentID: agentID,
			Status:  "background",
		}},
		Summary: fmt.Sprintf("%s spawned in background (agent_id: %s) — collect with agent_result(%q)", input.Agent, agentID, agentID),
	}, nil
}

// agentResultHandler collects the result of a background agent.
func agentResultHandler(bg *backgroundRegistry, input agentResultInput) (agentResultOutput, error) {
	status, result, errMsg, done, ok := bg.get(input.AgentID)
	if !ok {
		ids := bg.list()
		return agentResultOutput{
			Status:  "unknown",
			Summary: fmt.Sprintf("unknown agent_id %q; known background IDs: %v", input.AgentID, ids),
		}, nil
	}

	if status == "running" {
		if input.WaitSeconds > 0 {
			wait := time.Duration(min(input.WaitSeconds, 30) * float64(time.Second))
			timer := time.NewTimer(wait)
			defer timer.Stop()
			select {
			case <-done:
				// Agent finished; re-read the result.
				status, result, errMsg, _, _ = bg.get(input.AgentID)
			case <-timer.C:
				return agentResultOutput{
					Status:  "running",
					Summary: fmt.Sprintf("agent %s still running after waiting %.1fs", input.AgentID, input.WaitSeconds),
				}, nil
			}
		} else {
			return agentResultOutput{
				Status:  "running",
				Hint:    "call again with wait_seconds",
				Summary: fmt.Sprintf("agent %s still running", input.AgentID),
			}, nil
		}
	}

	// Terminal state.
	out := agentResultOutput{
		Status:  status,
		Result:  result,
		Summary: result,
	}
	if errMsg != "" {
		out.Error = errMsg
	}
	if len(out.Summary) > 200 {
		out.Summary = out.Summary[:197] + "..."
	}
	return out, nil
}

// forwardSingleModeEvents drains the single agent's event channel, forwarding
// each event to the TUI, and returns the accumulated output, terminal status,
// error message and sub-session id.
//
// It stays separate from forwardSubagentEvents because single mode reads one
// more event kind: run_done, whose status distinguishes a timeout from a
// crash. Parallel and chain do not, and folding the two paths together would
// change what they report.
func forwardSingleModeEvents(events <-chan subagent.Event, onEvent SubagentEventCallback, agentID, pipelineID string) (resultText, status, errMsg, sessID string) {
	var result strings.Builder
	status = "completed"
	for ev := range events {
		evContent := ev.Content
		if ev.Type == "error" && evContent == "" {
			evContent = ev.Error
		}
		emitEvent(onEvent, SubagentEvent{
			AgentID: agentID, Kind: ev.Type, Content: evContent,
			PipelineID: pipelineID, Mode: "single", Step: 1, Total: 1,
		})
		switch ev.Type {
		case "text_delta":
			result.WriteString(ev.Content)
		case "error":
			status = "failed"
			errMsg = ev.Error
		case subagent.EventFallback:
			// The restart notice means the previous attempt's failure was
			// recovered by a model switch: a fresh attempt owns the outcome
			// now, so an earlier attempt's error must not fail the result.
			status = "completed"
			errMsg = ""
		case "message_start":
			if ev.SessionID != "" {
				sessID = ev.SessionID
			}
		case "run_done":
			// The orchestrator's terminal status is more specific than the
			// "failed" inferred from an error event: it distinguishes a
			// timeout from a crash. That distinction is actionable for the
			// model — a timed-out agent is worth retrying with a narrower
			// task, a crashed one is not — and until now it was computed
			// and then dropped on the floor here.
			if ev.Status == "timeout" {
				status = "timeout"
			}
		}
	}
	return truncateOutput(result.String()), status, errMsg, sessID
}

// subagentTask is one agent-and-prompt pair, the shape parallel and chain mode
// have in common once TaskItem and ChainItem are past the JSON boundary.
type subagentTask struct {
	Agent string
	Task  string
}

// subagentPipelineSpec is what differs between the two multi-agent modes: the
// mode name reported back, the noun used in the over-limit message, and the
// per-call limit itself.
type subagentPipelineSpec struct {
	Mode  string // "parallel" or "chain"
	Noun  string // "parallel tasks" or "chain steps"
	Limit int
}

// subagentStepMeta is the pipeline position stamped onto every event emitted
// for one step, so the TUI can group and order them.
type subagentStepMeta struct {
	PipelineID string
	Mode       string
	Step       int // 1-based
	Total      int
}

// checkSubagentPipeline enforces the per-call limit and validates that every
// named agent exists, before anything is spawned. It returns the failure
// output and false when the call cannot run.
func checkSubagentPipeline(orch *subagent.Orchestrator, spec subagentPipelineSpec, tasks []subagentTask, start time.Time) (SubagentOutput, bool) {
	total := len(tasks)
	if total > spec.Limit {
		return SubagentOutput{
			Mode:    spec.Mode,
			Summary: fmt.Sprintf("too many %s: %d (max %d)", spec.Noun, total, spec.Limit),
			Results: []AgentResult{{
				Agent:    spec.Mode,
				Status:   "failed",
				Error:    fmt.Sprintf("too many %s: %d exceeds maximum of %d", spec.Noun, total, spec.Limit),
				Duration: time.Since(start).Truncate(time.Millisecond).String(),
			}},
		}, false
	}

	for _, task := range tasks {
		if _, err := orch.LookupAgent(task.Agent); err != nil {
			return SubagentOutput{
				Mode: spec.Mode,
				Results: []AgentResult{{
					Agent:    task.Agent,
					Status:   "failed",
					Error:    err.Error(),
					Duration: time.Since(start).Truncate(time.Millisecond).String(),
				}},
				Summary: fmt.Sprintf("validation failed: unknown agent %q", task.Agent),
			}, false
		}
	}
	return SubagentOutput{}, true
}

// runSubagentStep spawns one agent, forwards its events to the callback and
// returns the finished result. A spawn failure comes back as a failed
// AgentResult, never as an error: the model can act on the former.
func runSubagentStep(ctx context.Context, orch *subagent.Orchestrator, onEvent SubagentEventCallback, meta subagentStepMeta, agentName, prompt string) AgentResult {
	stepStart := time.Now()

	// Spawn with the fallback-capable path: the merged stream carries a
	// restart as [failed attempt events, fallback notice, retry events], so
	// parallel and chain steps survive a provider wall mid-pipeline.
	events, agentID, err := orch.SpawnWithInputFallback(ctx, subagent.AgentInput{
		Type:   agentName,
		Prompt: prompt,
	})
	if err != nil {
		return AgentResult{
			Agent:    agentName,
			Status:   "failed",
			Error:    err.Error(),
			Duration: time.Since(stepStart).Truncate(time.Millisecond).String(),
		}
	}

	emitEvent(onEvent, SubagentEvent{
		AgentID:    agentID,
		Kind:       "spawn",
		Content:    agentName,
		PipelineID: meta.PipelineID,
		Mode:       meta.Mode,
		Step:       meta.Step,
		Total:      meta.Total,
	})

	resultText, status, errMsg, sessID := forwardSubagentEvents(events, onEvent, agentID, meta)

	emitEvent(onEvent, SubagentEvent{
		AgentID:    agentID,
		Kind:       "done",
		PipelineID: meta.PipelineID,
		Mode:       meta.Mode,
		Step:       meta.Step,
		Total:      meta.Total,
	})

	return AgentResult{
		Agent:     agentName,
		AgentID:   agentID,
		Status:    status,
		Result:    resultText,
		Error:     errMsg,
		Duration:  time.Since(stepStart).Truncate(time.Millisecond).String(),
		SessionID: sessID,
	}
}

// forwardSubagentEvents drains one agent's event channel, forwarding each
// event to the callback with pipeline metadata attached, and returns the
// accumulated output, terminal status, error message and sub-session id.
func forwardSubagentEvents(events <-chan subagent.Event, onEvent SubagentEventCallback, agentID string, meta subagentStepMeta) (resultText, status, errMsg, sessID string) {
	var result strings.Builder
	status = "completed"

	for ev := range events {
		evContent := ev.Content
		if ev.Type == "error" && evContent == "" {
			evContent = ev.Error
		}
		emitEvent(onEvent, SubagentEvent{
			AgentID:    agentID,
			Kind:       ev.Type,
			Content:    evContent,
			PipelineID: meta.PipelineID,
			Mode:       meta.Mode,
			Step:       meta.Step,
			Total:      meta.Total,
		})

		switch ev.Type {
		case "text_delta":
			result.WriteString(ev.Content)
		case "error":
			status = "failed"
			errMsg = ev.Error
		case subagent.EventFallback:
			// A restart recovered the previous attempt's failure; see
			// forwardSubagentEvents.
			status = "completed"
			errMsg = ""
		case "message_start":
			if ev.SessionID != "" {
				sessID = ev.SessionID
			}
		}
	}
	return truncateOutput(result.String()), status, errMsg, sessID
}

// parallelModeHandler spawns multiple agents concurrently and collects all results.
func parallelModeHandler(ctx agent.Context, orch *subagent.Orchestrator, input SubagentInput, onEvent SubagentEventCallback) (SubagentOutput, error) {
	start := time.Now()
	pipelineID := fmt.Sprintf("pipe-%d", time.Now().UnixNano())

	tasks := make([]subagentTask, len(input.Tasks))
	for i, t := range input.Tasks {
		tasks[i] = subagentTask(t)
	}

	spec := subagentPipelineSpec{Mode: "parallel", Noun: "parallel tasks", Limit: maxParallelTasks}
	if out, ok := checkSubagentPipeline(orch, spec, tasks, start); !ok {
		return out, nil
	}

	// Spawn all agents concurrently and collect results.
	total := len(tasks)
	results := make([]AgentResult, total)
	var wg sync.WaitGroup
	spawnCtx := resolveContext(ctx)

	for i, task := range tasks {
		wg.Add(1)
		go func(idx int, t subagentTask) {
			defer wg.Done()
			meta := subagentStepMeta{PipelineID: pipelineID, Mode: spec.Mode, Step: idx + 1, Total: total}
			results[idx] = runSubagentStep(spawnCtx, orch, onEvent, meta, t.Agent, t.Task)
		}(i, task)
	}

	wg.Wait()

	duration := time.Since(start).Truncate(time.Millisecond).String()
	return SubagentOutput{
		Mode:    spec.Mode,
		Results: results,
		Summary: buildParallelSummary(results, total, duration),
	}, nil
}

// chainModeHandler runs agents sequentially, passing each result to the next step.
// Task prompts support {previous} (text result) and {previous_json} (JSON-escaped) placeholders.
func chainModeHandler(ctx agent.Context, orch *subagent.Orchestrator, input SubagentInput, onEvent SubagentEventCallback) (SubagentOutput, error) {
	start := time.Now()
	pipelineID := fmt.Sprintf("pipe-%d", time.Now().UnixNano())

	tasks := make([]subagentTask, len(input.Chain))
	for i, step := range input.Chain {
		tasks[i] = subagentTask(step)
	}

	spec := subagentPipelineSpec{Mode: "chain", Noun: "chain steps", Limit: maxChainSteps}
	if out, ok := checkSubagentPipeline(orch, spec, tasks, start); !ok {
		return out, nil
	}

	// Execute steps sequentially, passing results forward.
	total := len(tasks)
	results := make([]AgentResult, 0, total)
	spawnCtx := resolveContext(ctx)
	previousResult := ""

	for idx, step := range tasks {
		meta := subagentStepMeta{PipelineID: pipelineID, Mode: spec.Mode, Step: idx + 1, Total: total}
		// Expand template placeholders in the task prompt.
		res := runSubagentStep(spawnCtx, orch, onEvent, meta, step.Agent, expandChainTemplate(step.Task, previousResult))
		results = append(results, res)

		// Chain stops on the first failure, spawn failures included.
		if res.Status != "completed" {
			break
		}

		// Pass result to next step.
		previousResult = res.Result
	}

	duration := time.Since(start).Truncate(time.Millisecond).String()
	return SubagentOutput{
		Mode:    spec.Mode,
		Results: results,
		Summary: buildChainSummary(results, total, duration),
	}, nil
}

// expandChainTemplate replaces {previous} and {previous_json} placeholders in the task prompt.
func expandChainTemplate(task, previousResult string) string {
	if previousResult == "" {
		return task
	}
	result := strings.ReplaceAll(task, "{previous}", previousResult)
	// JSON-escape: escape backslashes, quotes, and newlines for embedding in JSON strings.
	jsonEscaped := strings.ReplaceAll(previousResult, `\`, `\\`)
	jsonEscaped = strings.ReplaceAll(jsonEscaped, `"`, `\"`)
	jsonEscaped = strings.ReplaceAll(jsonEscaped, "\n", `\n`)
	jsonEscaped = strings.ReplaceAll(jsonEscaped, "\r", `\r`)
	jsonEscaped = strings.ReplaceAll(jsonEscaped, "\t", `\t`)
	result = strings.ReplaceAll(result, "{previous_json}", jsonEscaped)
	return result
}

// emitEvent safely calls the event callback if non-nil.
func emitEvent(cb SubagentEventCallback, ev SubagentEvent) {
	if cb != nil {
		cb(ev)
	}
}

// consumeAgentEvents reads events from a channel, accumulates text, and detects errors.
// Returns the accumulated result text, final status, and error message.
func consumeAgentEvents(events <-chan subagent.Event) (resultText, status, errMsg string) {
	var result strings.Builder
	status = "completed"
	for ev := range events {
		switch ev.Type {
		case "text_delta":
			result.WriteString(ev.Content)
		case "error":
			status = "failed"
			errMsg = ev.Error
		}
	}
	resultText = truncateOutput(result.String())
	return
}

// buildParallelSummary formats the summary line for parallel mode.
func buildParallelSummary(results []AgentResult, total int, duration string) string {
	completed, failed := 0, 0
	for _, r := range results {
		if r.Status == "completed" {
			completed++
		} else {
			failed++
		}
	}
	if failed > 0 {
		return fmt.Sprintf("parallel: %d/%d completed, %d failed in %s", completed, total, failed, duration)
	}
	return fmt.Sprintf("parallel: %d/%d completed in %s", completed, total, duration)
}

// buildChainSummary formats the summary line for chain mode.
func buildChainSummary(results []AgentResult, total int, duration string) string {
	completed := 0
	for _, r := range results {
		if r.Status == "completed" {
			completed++
		}
	}
	if completed < total {
		return fmt.Sprintf("chain: stopped at step %d/%d in %s", len(results), total, duration)
	}
	return fmt.Sprintf("chain: %d/%d steps completed in %s", completed, total, duration)
}
