// Package agent_test holds the harness A/B proof spike for #79
// (InstructionParts injection). Unlike the in-package tests it imports
// internal/eval — which itself imports internal/agent — so it must live in
// the external test package to avoid an import cycle.
package agent_test

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	llmagent "google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"

	"github.com/spa-skyson/pi-rate/internal/agent"
	"github.com/spa-skyson/pi-rate/internal/eval"
	"github.com/spa-skyson/pi-rate/internal/extension"
	"github.com/spa-skyson/pi-rate/internal/logger"
	"github.com/spa-skyson/pi-rate/internal/provider"
	pisession "github.com/spa-skyson/pi-rate/internal/session"
	"github.com/spa-skyson/pi-rate/internal/testenv"
	"github.com/spa-skyson/pi-rate/internal/tools"
)

// proofCapturingLLM records every LLMRequest it receives (not just the last)
// and answers with a fixed text response. The external test package cannot
// reuse the in-package capturingLLM, so this is the same shape with a
// request log.
type proofCapturingLLM struct {
	name     string
	response string

	mu       sync.Mutex
	requests []*model.LLMRequest
}

func (m *proofCapturingLLM) Name() string { return m.name }

func (m *proofCapturingLLM) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	m.mu.Lock()
	m.requests = append(m.requests, req)
	m.mu.Unlock()
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{
			Content: genai.NewContentFromText(m.response, genai.RoleModel),
		}, nil)
	}
}

// capturedSystemInstruction flattens the system instruction of the most
// recent request into one string, the way the in-package helper does.
func (m *proofCapturingLLM) capturedSystemInstruction() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.requests) == 0 {
		return ""
	}
	req := m.requests[len(m.requests)-1]
	if req.Config == nil || req.Config.SystemInstruction == nil {
		return ""
	}
	var sb strings.Builder
	for _, p := range req.Config.SystemInstruction.Parts {
		if p != nil && p.Text != "" {
			sb.WriteString(p.Text)
		}
	}
	return sb.String()
}

// runOneTurn drives one user turn through the agent so the capturing LLM
// records the system instruction the runner actually sent.
func runOneTurn(t *testing.T, a *agent.Agent) {
	t.Helper()
	ctx := context.Background()
	sessionID, _, err := a.CreateSession(ctx)
	if err != nil {
		t.Fatalf("CreateSession() error: %v", err)
	}
	for _, err := range a.Run(ctx, sessionID, "hello") {
		if err != nil {
			t.Fatalf("Run() yielded error: %v", err)
		}
	}
}

// readInventory parses the "context inventory:" line out of the session log
// an agent with a Logger wrote. It returns the raw line so the report carries
// the exact base=… rules=… total=… numbers.
func readInventory(t *testing.T, log *logger.Logger) (raw string, total int) {
	t.Helper()
	data, err := os.ReadFile(log.Path())
	if err != nil {
		t.Fatalf("reading session log: %v", err)
	}
	for _, l := range strings.Split(string(data), "\n") {
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
		raw, total = rec.Content, 0
		if _, tail, ok := strings.Cut(rec.Content, "total="); ok {
			if _, err := fmt.Sscanf(tail, "%d", &total); err != nil {
				t.Fatalf("parsing total= from inventory line %q: %v", rec.Content, err)
			}
		}
		// Keep scanning: a log file may hold several inventory lines, and
		// the last one is the most recent build.
	}
	if raw == "" {
		t.Fatalf("context inventory line missing from session log:\n%s", data)
	}
	return raw, total
}

// TestHarnessABProofWiring is proof [A]: the wiring is honest. Two agents on
// the same working directory — one file-driven (nil InstructionParts), one
// with a supplied variant — must produce observably different prompts at the
// model, with the inventory line describing each, and the file harness
// untouched in both arms.
//
//   - control: nil InstructionParts → the AGENTS.md marker AB-PROOF-FILEMARK
//     reaches the prompt, the inventory line is written, total = X.
//   - variant: {Base: AB-PROOF-BASE, Rules: …} → FILEMARK is gone, both
//     markers are present, the recomposed Runtime Environment block closes
//     the prompt, total = Y ≠ X.
func TestHarnessABProofWiring(t *testing.T) {
	testenv.SetHome(t, t.TempDir())

	// One working directory shared by both arms, with a context file whose
	// marker only the file-driven assembly can see.
	cwd := t.TempDir()
	const fileMarker = "AB-PROOF-FILEMARK"
	agentsMD := "# Project Rules\n\n" + fileMarker + "\nAlways answer concisely.\n"
	agentsPath := filepath.Join(cwd, "AGENTS.md")
	if err := os.WriteFile(agentsPath, []byte(agentsMD), 0o600); err != nil {
		t.Fatal(err)
	}

	const (
		variantBase  = "AB-PROOF-BASE"
		variantRules = "AB-PROOF-RULES: answer in exactly one word."
	)

	// No skills menu in either arm: the A proof is about Base/Rules/Env, and
	// an empty skills dir keeps the assembly deterministic.
	t.Setenv("PI_SKILLS_DIRS", filepath.Join(t.TempDir(), "no-skills"))

	type arm struct {
		name   string
		prompt string
		invRaw string
		invTot int
	}

	runArm := func(name string, parts *agent.InstructionParts) arm {
		t.Helper()
		// A HOME per arm: logger.New() names the session log by the second,
		// so two arms inside one second would share a file and the inventory
		// reads would cross.
		testenv.SetHome(t, t.TempDir())
		llm := &proofCapturingLLM{name: "m", response: "ok"}
		log, err := logger.New()
		if err != nil {
			t.Fatalf("%s: logger.New() error: %v", name, err)
		}
		t.Cleanup(func() { _ = log.Close() })

		// Control (parts nil): the file-driven assembly the production
		// drivers run — LoadInstructionPartsFor reads AGENTS.md and the
		// skills menu for cwd and hands New a plain instruction. Variant:
		// the supplied parts replace exactly that assembly.
		instruction := agent.LoadInstructionPartsFor(agent.SystemInstruction, cwd).String()
		cfg := agent.Config{
			Model:       llm,
			Instruction: instruction,
			WorkingDir:  cwd,
			Logger:      log,
		}
		if parts != nil {
			p := *parts
			cfg.InstructionParts = &p
		}
		a, err := agent.New(cfg)
		if err != nil {
			t.Fatalf("%s: agent.New() error: %v", name, err)
		}
		runOneTurn(t, a)

		prompt := llm.capturedSystemInstruction()
		if prompt == "" {
			t.Fatalf("%s: no system instruction reached the model", name)
		}
		raw, total := readInventory(t, log)

		// The harness promise: injection never touches the disk. Compare the
		// context file after the run with what was seeded; a drift means the
		// run wrote to the file harness.
		data, err := os.ReadFile(agentsPath)
		if err != nil {
			t.Fatalf("%s: reading AGENTS.md after the run: %v", name, err)
		}
		if string(data) != agentsMD {
			t.Errorf("%s: AGENTS.md was modified by the run:\n got %q\nwant %q", name, data, agentsMD)
		}
		return arm{name: name, prompt: prompt, invRaw: raw, invTot: total}
	}

	control := runArm("control", nil)
	// Variant: the same working directory, the file assembly replaced by the
	// supplied parts.
	variantParts := &agent.InstructionParts{
		Base:  variantBase,
		Rules: "\n\n# Project Rules\n\n" + variantRules + "\n",
	}
	variant := runArm("variant", variantParts)

	// --- control assertions -------------------------------------------------
	if !strings.Contains(control.prompt, fileMarker) {
		t.Errorf("control: prompt must carry the file-backed marker %s, got:\n%.400s", fileMarker, control.prompt)
	}
	if strings.Contains(control.prompt, variantBase) || strings.Contains(control.prompt, variantRules) {
		t.Errorf("control: prompt must not carry the variant markers, got:\n%.400s", control.prompt)
	}
	if control.invRaw == "" || control.invTot <= 0 {
		t.Errorf("control: context inventory line not recorded: %q", control.invRaw)
	}
	t.Logf("[A] control  inventory: %s", control.invRaw)

	// --- variant assertions -------------------------------------------------
	if strings.Contains(variant.prompt, fileMarker) {
		t.Errorf("variant: the file-backed marker %s must NOT reach the prompt, got:\n%.400s", fileMarker, variant.prompt)
	}
	if !strings.Contains(variant.prompt, variantBase) {
		t.Errorf("variant: prompt must carry the supplied Base %s, got:\n%.400s", variantBase, variant.prompt)
	}
	if !strings.Contains(variant.prompt, variantRules) {
		t.Errorf("variant: prompt must carry the supplied Rules, got:\n%.400s", variant.prompt)
	}
	// Env is always recomposed from the live process environment and stays
	// the last block of the prompt.
	envMarker := "\n\n# Runtime Environment\n\n"
	if !strings.Contains(variant.prompt, envMarker) {
		t.Errorf("variant: recomposed Runtime Environment block missing:\n%.400s", variant.prompt)
	}
	if i := strings.Index(variant.prompt, envMarker); i >= 0 {
		tail := variant.prompt[i+len(envMarker):]
		if strings.Contains(tail, "# Project Rules") || strings.Contains(tail, variantBase) {
			t.Errorf("variant: Runtime Environment block is not the last section:\n%.400s", variant.prompt)
		}
		if !strings.Contains(tail, "- HOME=") {
			t.Errorf("variant: recomposed Env block lacks the environment entries:\n%.400s", variant.prompt)
		}
	}
	if variant.invRaw == "" || variant.invTot <= 0 {
		t.Errorf("variant: context inventory line not recorded: %q", variant.invRaw)
	}
	t.Logf("[A] variant  inventory: %s", variant.invRaw)

	// --- the diff behavior --------------------------------------------------
	if variant.invTot == control.invTot {
		t.Errorf("the two arms must produce different prompt totals, both are %d", control.invTot)
	}
	if control.prompt == variant.prompt {
		t.Error("the two arms produced the same prompt — the injection did not reach the model")
	}

	x, y := control.invTot, variant.invTot
	t.Logf("[A] PROOF: prompt total control X=%d vs variant Y=%d (diff %+d bytes)", x, y, y-x)
	t.Logf("[A] PROOF: FILEMARK in control prompt=%t, in variant prompt=%t",
		strings.Contains(control.prompt, fileMarker), strings.Contains(variant.prompt, fileMarker))
	t.Logf("[A] PROOF: file harness untouched (AGENTS.md byte-identical after both arms)")
}

// proofScriptedLLM is a scripted model for the mock B-alternative: the first
// maxToolRounds calls of a run issue one fixed function call each, then a
// text answer ends the turn. Each arm gets its own instance, so the per-arm
// LLM round-trip count is directly observable.
type proofScriptedLLM struct {
	name          string
	call          *genai.FunctionCall
	maxToolRounds int

	mu    sync.Mutex
	calls int
}

func (m *proofScriptedLLM) Name() string { return m.name }

func (m *proofScriptedLLM) GenerateContent(ctx context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	m.mu.Lock()
	m.calls++
	n := m.calls
	m.mu.Unlock()
	return func(yield func(*model.LLMResponse, error) bool) {
		if err := ctx.Err(); err != nil {
			yield(nil, err)
			return
		}
		if n <= m.maxToolRounds {
			yield(&model.LLMResponse{
				Content: &genai.Content{
					Role:  genai.RoleModel,
					Parts: []*genai.Part{{FunctionCall: m.call}},
				},
			}, nil)
			return
		}
		yield(&model.LLMResponse{
			Content: &genai.Content{
				Role:  genai.RoleModel,
				Parts: []*genai.Part{{Text: "done"}},
			},
		}, nil)
	}
}

func (m *proofScriptedLLM) callCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.calls
}

// abArmResult is what one B-arm run reports: the trajectory metrics the eval
// harness computes, the per-tool call counts the trajectory shows, and — mock
// arms only — the number of model round-trips the arm made.
type abArmResult struct {
	metrics    eval.TrajectoryMetrics
	toolCalls  map[string]int // function name → calls, across all sessions
	modelCalls int
}

// toolCallsByTool folds the per-tool call counts out of the loaded
// trajectories — the tool-set diff the variant's no-bash rule should show.
func toolCallsByTool(loaded []*eval.LoadedTrajectory) map[string]int {
	out := map[string]int{}
	for _, lt := range loaded {
		if lt == nil || lt.Traj == nil {
			continue
		}
		for i := range lt.Traj.Steps {
			for _, tc := range lt.Traj.Steps[i].ToolCalls {
				out[tc.FunctionName]++
			}
		}
	}
	return out
}

// seedABWorkspace writes the working directory a B-arm runs over: a minimal
// AGENTS.md (the file-driven control arm's rules) and a few files to count.
func seedABWorkspace(t *testing.T) string {
	t.Helper()
	cwd := t.TempDir()
	rules := "# Project Rules\n\nAnswer in at most three sentences.\n"
	if err := os.WriteFile(filepath.Join(cwd, "AGENTS.md"), []byte(rules), 0o600); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"alpha.txt", "beta.txt", "gamma.txt"} {
		if err := os.WriteFile(filepath.Join(cwd, name), []byte(name+"\n"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	return cwd
}

// formatToolCalls renders a per-tool count map deterministically for logs.
func formatToolCalls(calls map[string]int) string {
	names := make([]string, 0, len(calls))
	for name := range calls {
		names = append(names, name)
	}
	sortStrings(names)
	parts := make([]string, 0, len(names))
	for _, name := range names {
		parts = append(parts, fmt.Sprintf("%s=%d", name, calls[name]))
	}
	if len(parts) == 0 {
		return "(none)"
	}
	return strings.Join(parts, ", ")
}

// sortStrings insertion-sorts a tiny slice in place.
func sortStrings(s []string) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j] < s[j-1]; j-- {
			s[j], s[j-1] = s[j-1], s[j]
		}
	}
}

// runToolLoopArm drives one B-arm run: a real ADK tool loop (core tools,
// step-limit callback) over a seeded working directory, with the arm's
// sessions persisted through a FileService rooted at its own directory, so
// LoadTrajectories reads exactly this arm's ATIF record. parts nil = the
// file-driven assembly; llm is the arm's model (scripted or live).
func runToolLoopArm(
	t *testing.T,
	name string,
	parts *agent.InstructionParts,
	cwd, sessionsDir string,
	llm model.LLM,
	checkProviderEvents bool,
) abArmResult {
	t.Helper()

	sb, err := tools.NewSandbox(cwd)
	if err != nil {
		t.Fatalf("%s: NewSandbox(%s): %v", name, cwd, err)
	}
	t.Cleanup(func() { _ = sb.Close() })
	coreTools, err := tools.CoreTools(sb)
	if err != nil {
		t.Fatalf("%s: CoreTools() error: %v", name, err)
	}

	// The step budget bounds the arm at ≤6 model round-trips: five tool
	// rounds plus the final text — the same callback the --steps flag wires.
	const stepLimit = 5
	chain := extension.ComposeAfterToolChain([]llmagent.AfterToolCallback{agent.NewStepLimitCallback(stepLimit)})

	svc, err := pisession.NewFileService(sessionsDir)
	if err != nil {
		t.Fatalf("%s: NewFileService(%s): %v", name, sessionsDir, err)
	}

	cfg := agent.Config{
		Model:              llm,
		Tools:              coreTools,
		Instruction:        agent.SystemInstruction,
		WorkingDir:         cwd,
		SessionService:     svc,
		AfterToolCallbacks: chain,
	}
	if parts != nil {
		p := *parts
		cfg.InstructionParts = &p
	}
	a, err := agent.New(cfg)
	if err != nil {
		t.Fatalf("%s: agent.New() error: %v", name, err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	sid, _, err := a.CreateSession(ctx)
	if err != nil {
		t.Fatalf("%s: CreateSession() error: %v", name, err)
	}
	for ev, err := range a.Run(ctx, sid, "Count the files in the current directory and tell me the number.") {
		if err != nil {
			t.Fatalf("%s: Run() yielded error: %v", name, err)
		}
		// A provider failure arrives as an event error code with no Go
		// error; surfacing it keeps a dead live arm from looking like a
		// wiring result with an empty trajectory.
		if checkProviderEvents && ev != nil {
			if err := agent.EventError(ev); err != nil {
				t.Fatalf("%s: provider error on event: %v", name, err)
			}
		}
	}

	loaded, err := eval.LoadTrajectories(sessionsDir)
	if err != nil {
		t.Fatalf("%s: LoadTrajectories(%s): %v", name, sessionsDir, err)
	}
	if len(loaded) == 0 {
		t.Fatalf("%s: no ATIF trajectory under %s", name, sessionsDir)
	}
	return abArmResult{
		metrics:   eval.ComputeTrajectoryMetrics(loaded),
		toolCalls: toolCallsByTool(loaded),
	}
}

// TestHarnessABProofMockLoop is the B-alternative on mocks: the same short
// task (count the files in the working directory) driven through a real ADK
// tool loop twice — control on the file-driven prompt, variant on an injected
// InstructionParts whose scripted calls use a different tool — produces two
// trajectories with different tool sets and metrics, both visible to
// LoadTrajectories + ComputeTrajectoryMetrics (the exact layers make
// eval-run reports through). Each arm is a separate session under its own
// session dir; the step limit bounds each arm at ≤6 model round-trips.
//
// This is the wiring+metrics half of the live A/B: injected parts change what
// the loop does, the ATIF record carries it, the metric layer sees it. The
// live arm (TestHarnessABProofLive) covers the same path with a real provider
// when a key is present.
func TestHarnessABProofMockLoop(t *testing.T) {
	testenv.SetHome(t, t.TempDir())
	// No skills menu: the loop proof is about Rules moving tool choice, and
	// an empty skills dir keeps the two prompts byte-comparable apart from it.
	t.Setenv("PI_SKILLS_DIRS", filepath.Join(t.TempDir(), "no-skills"))

	// Control: the scripted model counts files with ls — the tool a
	// file-driven prompt naturally allows.
	newControlLLM := func(dir string) *proofScriptedLLM {
		return &proofScriptedLLM{
			name:          "mock-control",
			maxToolRounds: 2,
			call: &genai.FunctionCall{
				ID:   "c-ls",
				Name: "ls",
				Args: map[string]any{"path": dir},
			},
		}
	}
	// Variant: the same loop, but the scripted calls use read — the injected
	// Rules ban the tool the control used, so the trajectories must differ.
	newVariantLLM := func(dir string) *proofScriptedLLM {
		return &proofScriptedLLM{
			name:          "mock-variant",
			maxToolRounds: 2,
			call: &genai.FunctionCall{
				ID:   "c-read",
				Name: "read",
				Args: map[string]any{"file_path": filepath.Join(dir, "alpha.txt")},
			},
		}
	}

	variantParts := &agent.InstructionParts{
		Base:  agent.SystemInstruction,
		Rules: "\n\n# Project Rules\n\nHARD CONSTRAINT: never use the bash tool or the ls tool in this session. Use only the read tool: read each file to count.\n",
	}

	// Each arm seeds its own workspace and its own session dir; the scripted
	// function call names that workspace's path. Only the prompt and the
	// scripted tool differ — the task text, step limit and tool registry are
	// the same.
	runMock := func(name string, parts *agent.InstructionParts, newLLM func(dir string) *proofScriptedLLM) abArmResult {
		dir := seedABWorkspace(t)
		llm := newLLM(dir)
		r := runToolLoopArm(t, name, parts, dir, filepath.Join(t.TempDir(), "sessions"), llm, false)
		r.modelCalls = llm.callCount()
		return r
	}

	control := runMock("control", nil, newControlLLM)
	variant := runMock("variant", variantParts, newVariantLLM)

	if control.modelCalls == 0 || variant.modelCalls == 0 {
		t.Fatalf("model round-trips: control=%d variant=%d, both arms must run", control.modelCalls, variant.modelCalls)
	}
	if control.modelCalls > 6 || variant.modelCalls > 6 {
		t.Errorf("the step limit must keep each arm at ≤6 model calls: control=%d variant=%d",
			control.modelCalls, variant.modelCalls)
	}

	logArm := func(name string, r abArmResult) {
		t.Logf("[B-mock] %-7s {TotalSteps=%d TotalToolCalls=%d MaxDepth=%d sessions=%d modelCalls=%d}",
			name, r.metrics.TotalSteps, r.metrics.TotalToolCalls, r.metrics.MaxDepth, len(r.metrics.Sessions), r.modelCalls)
		t.Logf("[B-mock] %-7s tools: %s", name, formatToolCalls(r.toolCalls))
		for _, s := range r.metrics.Sessions {
			t.Logf("[B-mock] %-7s session=%s steps=%d tool_calls=%d depth=%d",
				name, s.SessionID, s.Steps, s.ToolCalls, s.Depth)
		}
	}
	logArm("control", control)
	logArm("variant", variant)

	// The metric layer must have seen real trajectories in both arms.
	if control.metrics.TotalSteps == 0 || variant.metrics.TotalSteps == 0 {
		t.Errorf("both arms must produce steps: control=%d variant=%d",
			control.metrics.TotalSteps, variant.metrics.TotalSteps)
	}
	if control.metrics.TotalToolCalls == 0 || variant.metrics.TotalToolCalls == 0 {
		t.Errorf("both arms must produce tool calls: control=%d variant=%d",
			control.metrics.TotalToolCalls, variant.metrics.TotalToolCalls)
	}

	// The tool-set diff: control used ls, variant used read, and neither
	// crossed over — the injected Rules changed what the loop did.
	if control.toolCalls["ls"] == 0 {
		t.Errorf("control arm must call ls, got: %s", formatToolCalls(control.toolCalls))
	}
	if variant.toolCalls["read"] == 0 {
		t.Errorf("variant arm must call read, got: %s", formatToolCalls(variant.toolCalls))
	}
	if variant.toolCalls["ls"] != 0 {
		t.Errorf("variant arm must not call ls (the injected rules ban it), got: %s", formatToolCalls(variant.toolCalls))
	}

	t.Logf("[B-mock] PROOF: LoadTrajectories saw both arms' ATIF sessions; metrics differ: control {%d steps, %d tool calls} vs variant {%d steps, %d tool calls}",
		control.metrics.TotalSteps, control.metrics.TotalToolCalls,
		variant.metrics.TotalSteps, variant.metrics.TotalToolCalls)
}

// harnessABLiveModel returns a live model for the B-live arms: the first
// candidate the environment offers that answers a one-word probe. Providers
// are tried in order — anthropic, then the OpenAI-compatible endpoint the
// env names — so a 5xx gateway degrades to the next candidate instead of
// failing the arm. It skips with an explicit reason when no candidate works;
// key values are never logged.
func harnessABLiveModel(t *testing.T) (model.LLM, string) {
	t.Helper()

	buildCtx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

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
			t.Logf("[B-live] candidate %s does not resolve: %v", c.name, err)
			continue
		}
		llm, err := provider.NewLLM(buildCtx, info, c.apiKey, c.baseURL, "none", nil)
		if err != nil {
			t.Logf("[B-live] candidate %s: client build failed: %v", c.name, err)
			continue
		}
		// One probe round-trip per candidate: a 5xx gateway must read as
		// "candidate down", not as a wiring failure with an empty trajectory.
		if err := probeLiveModel(llm); err != nil {
			if firstErr == nil {
				firstErr = err
			}
			t.Logf("[B-live] candidate %s is not serving: %v", c.name, err)
			continue
		}
		return llm, c.name
	}

	t.Skipf("live arm not executed: no live provider candidate answered (last error: %v); the wiring is proven by TestHarnessABProofWiring and the loop+metrics by TestHarnessABProofMockLoop", firstErr)
	return nil, ""
}

// TestHarnessABProofLive is the live half of proof [B]: the same short task
// through the same loop as the mock arm, on a real provider model. Each arm
// is bounded by the step limit (≤6 LLM calls); the injected variant rules
// ban bash, so a compliant run shows no bash call in its trajectory.
//
// Like the eval harness (PI_EVAL_RUN=1, make eval-run) this half is opt-in:
// it skips unless PI_HARNESS_AB_LIVE=1, because it makes real model calls.
// The wiring ([A]) and mock-loop ([B-mock]) halves above stay unconditional.
// It further skips with an explicit reason when no usable key or provider
// answers — the flag alone must not require the network by default.
func TestHarnessABProofLive(t *testing.T) {
	if os.Getenv("PI_HARNESS_AB_LIVE") != "1" {
		t.Skipf("set PI_HARNESS_AB_LIVE=1 to run live A/B (makes real model calls)")
	}
	if testing.Short() {
		t.Skip("live A/B skipped in -short mode")
	}
	// The live model and its credential come from the first candidate the
	// environment offers that actually answers; no key value is ever logged.
	llm, liveModelName := harnessABLiveModel(t)

	testenv.SetHome(t, t.TempDir())

	// The variant bans bash outright: a compliant live run must not show a
	// single bash call in its trajectory, whatever else it does.
	variantParts := &agent.InstructionParts{
		Base:  agent.SystemInstruction,
		Rules: "\n\n# Project Rules\n\nHARD CONSTRAINT: never use the bash tool in this session. Use only the read, ls, find and tree tools. Count the files, then answer with the number.\n",
	}

	run := func(name string, parts *agent.InstructionParts) abArmResult {
		dir := seedABWorkspace(t)
		return runToolLoopArm(t, name, parts, dir, filepath.Join(t.TempDir(), "sessions"), llm, true)
	}

	control := run("control", nil)
	variant := run("variant", variantParts)

	logArm := func(name string, r abArmResult) {
		t.Logf("[B-live] %-7s {TotalSteps=%d TotalToolCalls=%d MaxDepth=%d sessions=%d}",
			name, r.metrics.TotalSteps, r.metrics.TotalToolCalls, r.metrics.MaxDepth, len(r.metrics.Sessions))
		t.Logf("[B-live] %-7s tools: %s", name, formatToolCalls(r.toolCalls))
		for _, s := range r.metrics.Sessions {
			t.Logf("[B-live] %-7s session=%s model=%s steps=%d tool_calls=%d",
				name, s.SessionID, s.Model, s.Steps, s.ToolCalls)
		}
	}
	logArm("control", control)
	logArm("variant", variant)

	t.Logf("[B-live] live model: %s (each arm bounded by the step limit, ≤6 LLM calls)", liveModelName)

	// Live runs are weather, not physics: report the shape, do not assert on
	// it — except the one rule the variant states as a hard constraint. A
	// bash call in the variant's trajectory means the injected rules did not
	// bind the model, which is exactly the wiring defect this spike hunts.
	if variant.toolCalls["bash"] > 0 {
		t.Errorf("variant trajectory shows %d bash call(s) despite the injected no-bash rule; tools: %s",
			variant.toolCalls["bash"], formatToolCalls(variant.toolCalls))
	}
	if control.metrics.TotalToolCalls == 0 && variant.metrics.TotalToolCalls == 0 {
		t.Error("neither arm produced a tool call — the live runs did not act on the task")
	}
}

// probeLiveModel makes one minimal live round-trip to check the provider is
// actually serving: a bare agent, a fixed prompt, no tools.
func probeLiveModel(llm model.LLM) error {
	a, err := agent.New(agent.Config{Model: llm, Instruction: "Reply with the single word OK."})
	if err != nil {
		return fmt.Errorf("building probe agent: %w", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	sid, _, err := a.CreateSession(ctx)
	if err != nil {
		return fmt.Errorf("probe session: %w", err)
	}
	for ev, err := range a.Run(ctx, sid, "Reply with the single word OK.") {
		if err != nil {
			return fmt.Errorf("probe call: %w", err)
		}
		if err := agent.EventError(ev); err != nil {
			return fmt.Errorf("probe call: %w", err)
		}
	}
	return nil
}
