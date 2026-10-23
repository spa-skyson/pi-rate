package compare

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	llmagent "google.golang.org/adk/v2/agent/llmagent"
	llmmodel "google.golang.org/adk/v2/model"
	"google.golang.org/genai"

	"github.com/spa-skyson/pi-rate/internal/agent"
	"github.com/spa-skyson/pi-rate/internal/eval"
	"github.com/spa-skyson/pi-rate/internal/extension"
	"github.com/spa-skyson/pi-rate/internal/logger"
	"github.com/spa-skyson/pi-rate/internal/provider"
	pisession "github.com/spa-skyson/pi-rate/internal/session"
	"github.com/spa-skyson/pi-rate/internal/tools"
)

// ErrNoProvider reports that no live provider is available: the environment
// offers no candidate (key+endpoint) that resolves and answers a probe
// round-trip. RunArm wraps it so a caller — the live test above all — can
// distinguish "no one to talk to" from a real execution failure and skip
// instead of failing.
var ErrNoProvider = errors.New("compare: no live provider available")

// liveStepLimit bounds one arm's run at ≤6 model round-trips: five tool
// rounds plus the final text — the same budget the harness A/B spike drove.
const liveStepLimit = 5

// liveRunTimeout caps one run so a wedged provider cannot hold the driver
// past the test's own timeout.
const liveRunTimeout = 5 * time.Minute

// defaultLiveTask is the scenario the arms run when Task and the variant
// label are empty: the short, tool-forcing task the harness A/B spike drove.
const defaultLiveTask = "Count the files in the current directory and tell me the number."

// LiveRunner implements Runner over the real tool loop: each RunArm builds a
// full agent.Config for the variant — a seeded workspace, core tools, the
// step limit and the arm's own sessions directory — drives the scenario task
// through agent.Run, and aggregates the arm's result from what the run wrote
// (ATIF trajectories, per-tool counts, the context-inventory line).
//
// The workspace is seeded once per Runner and copied into every arm run, so
// control and variant start from identical bytes (the shared seeding both
// hands see); sessions are unique per arm run, so the driver reads exactly
// what that run produced. Model is resolved once per Runner via NewLiveModel;
// a Runner with no reachable provider reports ErrNoProvider from RunArm.
type LiveRunner struct {
	// Root is the driver's scratch directory: the workspace seed, one
	// workspace copy and one sessions dir per arm run live under it.
	// Empty = created on first use and left for the caller to clean up.
	Root string
	// Model is the live model every arm runs on, resolved by NewLiveModel.
	// Nil = resolved on the first RunArm call.
	Model llmmodel.LLM
	// ModelName is the resolved model's name, for logs and reports.
	ModelName string
	// Task is the scenario prompt every arm runs; empty = the variant's
	// label, falling back to the default file-counting task.
	Task string
	// Runs overrides the per-RunArm run count when the RunArm argument is
	// not positive. Zero = default to one run.
	Runs int
}

// RunArm implements Runner: `runs` full agent runs of the variant, aggregated
// into one ArmResult. A provider the Runner could not resolve fails the arm
// with ErrNoProvider; an agent-build failure fails the arm; a run whose
// provider died mid-turn degrades into a failed RunResult, not an arm error.
func (lr *LiveRunner) RunArm(ctx context.Context, v Variant, runs int) (ArmResult, error) {
	if runs <= 0 {
		runs = lr.Runs
	}
	if runs <= 0 {
		runs = 1
	}
	if lr.Model == nil {
		mdl, name, err := NewLiveModel(ctx)
		if err != nil {
			return ArmResult{}, err
		}
		lr.Model, lr.ModelName = mdl, name
	}

	tmpl, err := lr.workspaceTemplate()
	if err != nil {
		return ArmResult{}, err
	}

	arm := ArmResult{Variant: v.Name}
	for i := 0; i < runs; i++ {
		// Per-run isolation: every run gets its own workspace copy and its
		// own sessions subdir, so trajectories and inventory lines never
		// cross runs — the driver reads exactly what this run wrote.
		ws := filepath.Join(lr.Root, fmt.Sprintf("%s-run%d-ws", armDirName(v.Name), i+1))
		if err := copyDir(ws, tmpl); err != nil {
			return ArmResult{}, fmt.Errorf("compare: seating workspace for run %d: %w", i+1, err)
		}
		sessionsDir := filepath.Join(lr.Root, fmt.Sprintf("%s-run%d-sessions", armDirName(v.Name), i+1))

		rr, total, err := lr.runOnce(ctx, v, ws, sessionsDir)
		if err != nil {
			return ArmResult{}, fmt.Errorf("compare: arm %s run %d: %w", v.Name, i+1, err)
		}
		arm.Runs = append(arm.Runs, rr)

		loaded, err := eval.LoadTrajectories(sessionsDir)
		if err != nil {
			return ArmResult{}, fmt.Errorf("compare: arm %s run %d: loading trajectories: %w", v.Name, i+1, err)
		}
		arm.Metrics = mergeMetrics(arm.Metrics, eval.ComputeTrajectoryMetrics(loaded))
		calls, err := ToolCallsFromDir(sessionsDir)
		if err != nil {
			return ArmResult{}, fmt.Errorf("compare: arm %s run %d: counting tool calls: %w", v.Name, i+1, err)
		}
		arm.ToolCalls = MergeToolCalls(arm.ToolCalls, calls)
		arm.PromptTotalBytes += total
	}
	return arm, nil
}

// runOnce drives one full agent run of the variant over ws, persisting the
// session under sessionsDir. The returned total is the run's prompt inventory
// size parsed from the session log the run's logger wrote.
func (lr *LiveRunner) runOnce(ctx context.Context, v Variant, ws, sessionsDir string) (RunResult, int, error) {
	cfg, log, err := lr.buildConfig(v, ws, sessionsDir)
	if err != nil {
		return RunResult{}, 0, err
	}
	a, err := agent.New(cfg)
	if err != nil {
		_ = log.Close()
		return RunResult{}, 0, fmt.Errorf("agent.New: %w", err)
	}
	defer func() { _ = log.Close() }()

	task := lr.Task
	if task == "" {
		task = v.Label
	}
	if task == "" {
		task = defaultLiveTask
	}

	ctx, cancel := context.WithTimeout(ctx, liveRunTimeout)
	defer cancel()
	sid, _, err := a.CreateSession(ctx)
	if err != nil {
		return RunResult{}, 0, fmt.Errorf("CreateSession: %w", err)
	}

	rr := RunResult{Scenario: "live", TrajDir: sessionsDir}
	var reply strings.Builder
	var runErr error
	for ev, err := range a.Run(ctx, sid, task) {
		if err != nil {
			runErr = err
			break
		}
		// A provider failure arrives as an event error code with no Go
		// error; surfacing it keeps a dead arm from looking like a pass.
		if err := agent.EventError(ev); err != nil {
			runErr = err
			break
		}
		if ev != nil && ev.Content != nil {
			for _, p := range ev.Content.Parts {
				if p != nil && p.Text != "" {
					reply.WriteString(p.Text)
				}
			}
		}
	}
	rr.Status = eval.StatusPass
	if runErr != nil || strings.TrimSpace(reply.String()) == "" {
		rr.Status = eval.StatusFail
	}
	// The context-inventory line lands in the session log this run wrote —
	// the file the logger itself names (its HOME-scoped path), not anything
	// derived from the sessions dir.
	total, err := InventoryTotalFromLog(log.Path())
	if err != nil {
		return RunResult{}, 0, err
	}
	return rr, total, nil
}

// buildConfig assembles the agent config for one variant: a sandbox over the
// arm's workspace copy, the core tool set, the step-limit callback and a
// FileService rooted at the arm's own sessions directory. Parts nil = the
// file-driven assembly the production drivers run (control); non-nil = the
// injected parts (variant). The logger comes back separately: it owns the
// session log path the inventory line is read from.
func (lr *LiveRunner) buildConfig(v Variant, ws, sessionsDir string) (agent.Config, *logger.Logger, error) {
	sb, err := tools.NewSandbox(ws)
	if err != nil {
		return agent.Config{}, nil, fmt.Errorf("NewSandbox(%s): %w", ws, err)
	}
	coreTools, err := tools.CoreTools(sb)
	if err != nil {
		return agent.Config{}, nil, fmt.Errorf("CoreTools: %w", err)
	}
	chain := extension.ComposeAfterToolChain([]llmagent.AfterToolCallback{agent.NewStepLimitCallback(liveStepLimit)})
	svc, err := pisession.NewFileService(sessionsDir)
	if err != nil {
		return agent.Config{}, nil, fmt.Errorf("NewFileService(%s): %w", sessionsDir, err)
	}

	var parts agent.InstructionParts
	if v.Parts != nil {
		parts = *v.Parts
	} else {
		// Control arm: the file-driven assembly, resolved for the arm's own
		// workspace so its AGENTS.md rules are what the model sees.
		parts = agent.LoadInstructionPartsFor(agent.SystemInstruction, ws)
	}

	log, err := logger.New()
	if err != nil {
		return agent.Config{}, nil, fmt.Errorf("logger.New: %w", err)
	}

	return agent.Config{
		Model:              lr.Model,
		Tools:              coreTools,
		WorkingDir:         ws,
		SessionService:     svc,
		AfterToolCallbacks: chain,
		Logger:             log,
		InstructionParts:   &parts,
	}, log, nil
}

// workspaceTemplate seeds (once) and returns the shared workspace source
// directory: a minimal AGENTS.md — the file-driven control arm's rules — and
// a few files to count. Every arm run starts from a copy of these bytes.
func (lr *LiveRunner) workspaceTemplate() (string, error) {
	if lr.Root == "" {
		root, err := os.MkdirTemp("", "compare-live-")
		if err != nil {
			return "", fmt.Errorf("compare: creating runner root: %w", err)
		}
		lr.Root = root
	}
	if err := os.MkdirAll(lr.Root, 0o755); err != nil {
		return "", fmt.Errorf("compare: creating runner root: %w", err)
	}

	tmpl := filepath.Join(lr.Root, "workspace-seed")
	if _, err := os.Stat(filepath.Join(tmpl, "AGENTS.md")); err == nil {
		return tmpl, nil
	}
	if err := os.MkdirAll(tmpl, 0o755); err != nil {
		return "", fmt.Errorf("compare: creating workspace seed: %w", err)
	}
	rules := "# Project Rules\n\nAnswer in at most three sentences.\n"
	if err := os.WriteFile(filepath.Join(tmpl, "AGENTS.md"), []byte(rules), 0o600); err != nil {
		return "", fmt.Errorf("compare: seeding AGENTS.md: %w", err)
	}
	for _, name := range []string{"alpha.txt", "beta.txt", "gamma.txt"} {
		if err := os.WriteFile(filepath.Join(tmpl, name), []byte(name+"\n"), 0o600); err != nil {
			return "", fmt.Errorf("compare: seeding %s: %w", name, err)
		}
	}
	return tmpl, nil
}

// armDirName maps a variant name onto a filesystem-safe directory component.
func armDirName(name string) string {
	return "arm-" + strings.NewReplacer("/", "-", "\\", "-", " ", "-").Replace(name)
}

// copyDir copies src into dst (a fresh directory), so each arm run starts
// from identical workspace bytes.
func copyDir(dst, src string) error {
	if err := os.MkdirAll(dst, 0o755); err != nil {
		return err
	}
	entries, err := os.ReadDir(src)
	if err != nil {
		return err
	}
	for _, e := range entries {
		s, d := filepath.Join(src, e.Name()), filepath.Join(dst, e.Name())
		if e.IsDir() {
			if err := copyDir(d, s); err != nil {
				return err
			}
			continue
		}
		data, err := os.ReadFile(s)
		if err != nil {
			return err
		}
		if err := os.WriteFile(d, data, 0o600); err != nil {
			return err
		}
	}
	return nil
}

// mergeMetrics folds one run's trajectory metrics into the arm aggregate.
func mergeMetrics(dst, src eval.TrajectoryMetrics) eval.TrajectoryMetrics {
	dst.Sessions = append(dst.Sessions, src.Sessions...)
	dst.TotalSteps += src.TotalSteps
	dst.TotalToolCalls += src.TotalToolCalls
	dst.NestedAgentCalls += src.NestedAgentCalls
	if src.MaxDepth > dst.MaxDepth {
		dst.MaxDepth = src.MaxDepth
	}
	return dst
}

// NewLiveModel resolves the live model the arms run on: the first candidate
// the environment offers that resolves and answers a one-word probe. The
// order mirrors the harness A/B spike — anthropic, then the OpenAI-compatible
// endpoint the env names — so a 5xx gateway (the known anthropic 503)
// degrades to the next candidate instead of failing the run. ErrNoProvider
// when no candidate works; key values are never logged.
func NewLiveModel(ctx context.Context) (llmmodel.LLM, string, error) {
	type candidate struct {
		name    string
		baseURL string
		apiKey  string
		resolve func() (provider.Info, error)
	}
	candidates := []candidate{
		{
			name:    "claude-sonnet-4-6",
			baseURL: os.Getenv("ANTHROPIC_BASE_URL"),
			apiKey:  os.Getenv("ANTHROPIC_API_KEY"),
			resolve: func() (provider.Info, error) { return provider.Resolve("claude-sonnet-4-6") },
		},
		{
			name:    "gpt-oss:20b",
			baseURL: os.Getenv("OPENAI_BASE_URL"),
			apiKey:  os.Getenv("OPENAI_API_KEY"),
			resolve: func() (provider.Info, error) {
				return provider.ResolveWithBaseURL("gpt-oss:20b", os.Getenv("OPENAI_BASE_URL"))
			},
		},
	}

	var firstErr error
	for _, c := range candidates {
		if c.apiKey == "" {
			continue
		}
		info, err := c.resolve()
		if err != nil {
			if firstErr == nil {
				firstErr = err
			}
			continue
		}
		llm, err := provider.NewLLM(ctx, info, c.apiKey, c.baseURL, "none", nil)
		if err != nil {
			if firstErr == nil {
				firstErr = err
			}
			continue
		}
		// One probe round-trip per candidate: a 5xx gateway must read as
		// "candidate down", not as a wiring failure with an empty result.
		if err := probeLiveModel(ctx, llm); err != nil {
			if firstErr == nil {
				firstErr = err
			}
			continue
		}
		return llm, c.name, nil
	}
	if firstErr == nil {
		firstErr = errors.New("no candidate: set ANTHROPIC_API_KEY or OPENAI_API_KEY (+ OPENAI_BASE_URL)")
	}
	return nil, "", fmt.Errorf("%w: resolving a live provider: %w", ErrNoProvider, firstErr)
}

// probeLiveModel makes one minimal live round-trip to check the provider is
// actually serving: a bare request, a fixed prompt, no tools.
func probeLiveModel(ctx context.Context, llm llmmodel.LLM) error {
	req := &llmmodel.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("Reply with the single word OK.", genai.RoleUser)},
	}
	pctx, cancel := context.WithTimeout(ctx, 90*time.Second)
	defer cancel()
	for _, err := range llm.GenerateContent(pctx, req, false) {
		if err != nil {
			return fmt.Errorf("probe call: %w", err)
		}
	}
	return nil
}

// InventoryTotalFromLog parses the total= number from the LAST "context
// inventory:" line of a session log, the way the harness A/B spike's
// readInventory did. An empty path is not an error: a run that never reached
// the model carries no inventory.
func InventoryTotalFromLog(logPath string) (int, error) {
	if logPath == "" {
		return 0, nil
	}
	data, err := os.ReadFile(logPath)
	if err != nil {
		return 0, fmt.Errorf("reading session log %s: %w", logPath, err)
	}
	return parseInventoryTotal(string(data))
}

// parseInventoryTotal extracts total= from the last context-inventory line of
// a session log. The lines are JSON records whose content field carries the
// human-readable inventory; non-JSON lines are skipped.
func parseInventoryTotal(data string) (int, error) {
	total := -1
	for _, l := range strings.Split(data, "\n") {
		if !strings.Contains(l, "context inventory:") {
			continue
		}
		var rec struct {
			Content string `json:"content"`
		}
		if err := json.Unmarshal([]byte(l), &rec); err != nil {
			continue
		}
		if !strings.Contains(rec.Content, "total=") {
			continue
		}
		_, tail, ok := strings.Cut(rec.Content, "total=")
		if !ok {
			continue
		}
		var n int
		if _, err := fmt.Sscanf(tail, "%d", &n); err != nil {
			continue
		}
		total = n // keep scanning: the last line is the most recent build
	}
	if total < 0 {
		return 0, fmt.Errorf("context inventory line missing from session log")
	}
	return total, nil
}
