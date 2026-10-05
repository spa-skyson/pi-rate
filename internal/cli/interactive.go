package cli

import (
	"cmp"
	"context"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	adkagent "google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	adkmodel "google.golang.org/adk/v2/model"
	adktool "google.golang.org/adk/v2/tool"

	"github.com/spa-skyson/pi-rate/internal/agent"
	"github.com/spa-skyson/pi-rate/internal/autocompact"
	"github.com/spa-skyson/pi-rate/internal/config"
	"github.com/spa-skyson/pi-rate/internal/ctxwindow"
	"github.com/spa-skyson/pi-rate/internal/extension"
	"github.com/spa-skyson/pi-rate/internal/guardrail"
	"github.com/spa-skyson/pi-rate/internal/httplog"
	"github.com/spa-skyson/pi-rate/internal/logger"
	"github.com/spa-skyson/pi-rate/internal/lsp"
	"github.com/spa-skyson/pi-rate/internal/memory"
	"github.com/spa-skyson/pi-rate/internal/notice"
	"github.com/spa-skyson/pi-rate/internal/permission"
	"github.com/spa-skyson/pi-rate/internal/provider"
	pisession "github.com/spa-skyson/pi-rate/internal/session"
	"github.com/spa-skyson/pi-rate/internal/subagent"
	"github.com/spa-skyson/pi-rate/internal/tools"
	"github.com/spa-skyson/pi-rate/internal/tui"
)

// initResources tracks resources created during deferred init for cleanup.
type initResources struct {
	sandbox    *tools.Sandbox
	lspMgr     *lsp.Manager
	orch       *subagent.Orchestrator
	memStore   memory.Store
	memWorker  *memory.Worker
	sessionLog *logger.Logger
	sessionID  string // captured for resume hint on exit
	bashSup    *tools.BashSupervisor

	// cbIn and agentConfigs back the TUI's AgentSwitcher, which fires only
	// after deferred init has completed — the InitEvent send on the init
	// channel is the happens-before edge, so plain reads here are race-free.
	// cbIn carries the shared deduper and compaction metrics, so a switch
	// rebuild keeps working with the instances the auto-compact hook and the
	// context gauge already hold.
	cbIn         callbackInputs
	agentConfigs []subagent.AgentConfig

	// memSummarizer, with memSessionID and memProject, describes the
	// end-of-session summary written during cleanup. All three are empty when
	// memory is off, which is what makes the closer's summary step a no-op.
	memSummarizer *memory.SessionSummarizer
	memSessionID  string
	memProject    string
}

func (r *initResources) cleanup() {
	// Stop backgrounded shell commands first. Nothing else owns them, so a
	// command still running when the session ends is a leaked process — the
	// exact failure this supervisor was introduced to prevent.
	if r.bashSup != nil {
		r.bashSup.KillAll()
	}
	if r.sessionLog != nil {
		// Detach before closing: a late trace from an in-flight streaming body
		// would otherwise write into a closed file.
		httplog.SetSink(nil)
		_ = r.sessionLog.Close()
	}
	if r.memWorker != nil {
		ctx, cancel := context.WithTimeout(context.Background(), memoryDrainTimeout)
		drainErr := r.memWorker.Shutdown(ctx)
		cancel()

		// Summarize between the drain and the close, for the reason the piagent
		// closer documents: the drain is what moves the session's queued tool
		// calls into the store, so a summary read before it would describe a
		// prefix of the session and read as complete. The store then has to
		// outlive the summary, because that is where the summary is written.
		if r.memSummarizer != nil {
			summarizeSessionAfterDrain(summarizeParams{
				store:      r.memStore,
				summarizer: r.memSummarizer,
				sessionID:  r.memSessionID,
				project:    r.memProject,
				// Interactive keeps the full budget: exit latency is not
				// billed to anyone here, and the session tail is the product.
				drainBudget: memoryDrainTimeout,
				log:         slog.Default(),
			}, drainErr, true)
		}
	}
	if r.memStore != nil {
		_ = r.memStore.Close()
	}
	if r.lspMgr != nil {
		r.lspMgr.Shutdown()
	}
	if r.orch != nil {
		r.orch.Shutdown()
	}
	if r.sandbox != nil {
		_ = r.sandbox.Close()
	}
}

// runInteractive starts the TUI immediately and performs heavy initialization
// in a background goroutine, reporting progress via InitEvent channel.
// headerSessionID is the session ID already baked into the LLM client's
// ${SESSION_ID} headers (empty when none were needed): the TUI's session is
// created under it, so the header names the conversation the logs do.
func runInteractive(
	ctx context.Context,
	cfg config.Config,
	llm adkmodel.LLM,
	info provider.Info,
	tokenTracker *guardrail.Tracker,
	activeRole, cwd, sandboxRoot, worktreeDir, headerSessionID string,
) error {
	initCh := make(chan tui.InitEvent, 32)

	// Extension notices — a skipped MCP server, a rerouted docs source, an
	// OAuth re-login, a blocked skill — must land in the chat, not on the
	// terminal. The TUI paints its frame with direct cursor control, so a
	// stderr write from a background init goroutine lands inside the layout
	// and stays there until the next full repaint. The send is non-blocking:
	// a notice raised while the TUI is busy is dropped rather than stalling
	// the agent turn behind it. The channel is handed to the TUI below as
	// well as in the InitResult, so it is drained from the first frame and
	// the buffer does not have to hold every startup notice — a run that
	// blocks a large number of skills would otherwise lose the tail.
	noticeCh := make(chan string, 64)
	prevSink := notice.SetSink(func(msg string) {
		select {
		case noticeCh <- msg:
		default:
		}
	})
	defer func() { notice.SetSink(prevSink) }()

	// Started only now that notices are routed to the TUI: an update banner
	// written to os.Stderr would land inside the painted frame.
	go checkForUpdate(ctx, Version, "/update")

	// Likewise for anything config load decided: it ran before the sink
	// existed, and the terminal reset before the first frame would have
	// erased a message written then.
	config.NotifyReroutedLLMS(cfg)

	// Todo updates are sent from the todo_write tool to the TUI via a
	// buffered channel. Non-blocking send; the TUI keeps the latest state.
	todoCh := make(chan tools.TodoState, 1)

	// The todo tools need the session ID at CoreTools build time, which runs
	// in deferred init's first phase — before resolveDeferredSession creates
	// the session. A fresh run therefore gets a pre-generated ID here (the
	// same generator FileService uses internally), so the session, its LLM
	// headers and todos.json all name the same conversation. A --session or
	// --continue ID is already final by this point and is kept as-is.
	if headerSessionID == "" {
		headerSessionID = pisession.GenerateSessionID()
	}

	// A user is present and a browser is reachable, so an MCP server that
	// answers 401 can be re-authorized interactively. Headless modes leave
	// this off and skip the server instead of blocking on an approval nobody
	// will see.
	extension.SetInteractiveOAuth(true)
	defer extension.SetInteractiveOAuth(false)

	var res initResources
	initDone := make(chan struct{})

	// Create a child context so deferred init is canceled when the TUI exits.
	initCtx, initCancel := context.WithCancel(ctx)

	// Tool-approval bridge for the interactive session: the permission gate
	// sends ask requests here and the TUI dialog answers them. The channel is
	// created unconditionally, but the gate only consults it when the global
	// config rules actually contain ask directives.
	approvalCh := make(chan permission.ApprovalRequest)

	// Question bridge, same shape: the question tool sends its request here
	// and the TUI dialog answers it. Buffer 1 plus the TUI's parked reader
	// keeps the tool's send non-blocking; a question arriving while one is
	// already up displaces it (the TUI answers the stale one canceled).
	questionCh := make(chan tools.QuestionRequest, 1)

	go func() {
		defer close(initDone)
		defer close(initCh)
		deferredInit(initCtx, cfg, llm, info.Provider, info.Model, info.BaseURL, tokenTracker, cwd, sandboxRoot, worktreeDir, headerSessionID, initCh, noticeCh, approvalCh, questionCh, todoCh, &res)
	}()

	// The TUI owns the terminal from here until tui.Run returns: it renders on
	// the normal screen and anything written to stdout/stderr while the
	// tea.Program is live paints raw text over the drawn frame, and the
	// renderer never learns those cells changed (AGENTS.md, TUI output
	// safety). Everything below — the exit epilogue, cleanup diagnostics —
	// therefore runs only after Run has returned and the terminal state is
	// restored. printSessionEpilogue takes its writer as a parameter precisely
	// so this invariant is unit-testable.
	// The mid-turn replay budget resolves once here so the TUI reads a plain
	// value; env override and out-of-range handling live in the resolver
	// (issue #43).
	midTurnAttempts := cfg.ResolveMidTurnAttempts()
	// Same source as setupPalace: the sidebar opens the palace with the
	// embedder the user actually configured.
	palaceCfg := palaceConfigFromCLI(&cfg)
	tuiErr := tui.Run(ctx, tui.Config{
		PlanAutoFix:     planAutoFixEnabled(cfg),
		MidTurnAttempts: &midTurnAttempts,
		LLM:             llm,
		AppVersion:      versionString(),
		ModelName:       llm.Name(),
		ProviderName:    info.Provider,
		ThinkingLevel:   effectiveThinkingLevel(cfg),
		ActiveRole:      activeRole,
		Roles:           cfg.Roles,
		WorkDir:         cwd,
		ThemeName:       cfg.Theme,
		TokenTracker:    tokenTracker,
		LifecycleHooks:  convertHooks(cfg.Hooks),
		// Attention carries the resolved config section (nil pointers inside mean
		// "on"); nil section means the signals stay off. Consumed by
		// internal/tui/attention.go.
		Attention: cfg.Attention,
		// The resolved palace config (paths plus embedder choice), so the
		// sidebar status reports the configured backend instead of a default.
		Palace:         &palaceCfg,
		DeferredInit:   initCh,
		SystemNoticeCh: noticeCh,
		ApprovalCh:     approvalCh,
		QuestionCh:     questionCh,
		TodoCh:         todoCh,
		ModelSwitcher: func(switchCtx context.Context, modelName string) (adkmodel.LLM, string, string, error) {
			return buildSwitchedLLM(switchCtx, cfg, tokenTracker, modelName, headerSessionID)
		},
		// ModelCandidates seeds the /model popup; the refresh fills in each
		// named provider's catalog (cache first, then a live fetch) while the
		// popup is open.
		ModelCandidates: modelCandidates(cfg),
		ModelCandidatesRefresh: func(refreshCtx context.Context) []tui.SearchItem {
			return refreshModelCandidates(refreshCtx, cfg)
		},
		// SubagentStatuses backs the /subagents monitor: the orchestrator is
		// built during deferred init and only exists from then on, so the
		// closure reads it through res at call time — the same InitEvent
		// happens-before edge the AgentSwitcher below relies on. A user can
		// only open the monitor long after init has delivered.
		SubagentStatuses: func() []subagent.AgentStatus {
			if res.orch == nil {
				return nil
			}
			return res.orch.List()
		},
		// SteerSubagent sends a follow-up message to a running subagent (the
		// monitor's `s` key). Same deferred-init closure as SubagentStatuses:
		// the orchestrator only exists from the InitEvent onward.
		SteerSubagent: func(agentID, text string) error {
			if res.orch == nil {
				return fmt.Errorf("orchestrator is not ready yet")
			}
			return res.orch.Steer(agentID, text)
		},
		// AgentSwitcher fires only after deferred init has filled res (the
		// InitEvent send is the happens-before edge): it rebuilds the
		// session's callback chains from the shared inputs and builds the
		// target agent's LLM from its frontmatter, or from the modelOverride
		// the TUI recorded for it via /model.
		AgentSwitcher: func(switchCtx context.Context, agentName string, modelOverride string) (tui.AgentSwitch, error) {
			return agentSwitch(switchCtx, cfg, tokenTracker, headerSessionID, &res.cbIn, res.agentConfigs, agentName, modelOverride)
		},
		// Update checking and installing for the startup notice and /update.
		// The checker guards dev builds itself; the installer captures the
		// script's output — the TUI owns the terminal.
		CheckUpdate: newUpdateChecker(),
		ApplyUpdate: newUpdateInstaller(),
		A2A:         cfg.A2A,
	})

	initCancel() // signal deferred init to stop
	<-initDone

	// Print session ID and resume command on exit — after Run, per the
	// invariant above.
	printSessionEpilogue(os.Stderr, res.sessionID)

	res.cleanup()
	return tuiErr
}

// printSessionEpilogue writes the exit banner — session ID and the command to
// resume the session — to w. It is called only after tui.Run has returned and
// the tea.Program no longer owns the terminal: while the TUI is live, any
// write to the terminal corrupts the drawn frame (AGENTS.md, TUI output
// safety). The writer is a parameter, not a package-level sink, so a unit
// test can pin both the output bytes and the fact that the helper itself
// never prints anywhere on its own.
func printSessionEpilogue(w io.Writer, sessionID string) {
	if sessionID == "" {
		return
	}
	fmt.Fprintf(w, "\nSession: %s\nResume:  pirate --session %s\n", sessionID, sessionID)
}

// deferredInit performs all heavy initialization, sending progress via ch.
// Resources that need cleanup are stored in res. approvalCh is the bridge the
// permission gate sends ask requests over; questionCh is the bridge the
// question tool sends its requests over; nil keeps each non-interactive
// fallback.
func deferredInit(
	ctx context.Context,
	cfg config.Config,
	llm adkmodel.LLM,
	providerName string,
	modelName string,
	baseURL string,
	tokenTracker *guardrail.Tracker,
	cwd, sandboxRoot, worktreeDir string,
	headerSessionID string,
	ch chan<- tui.InitEvent,
	noticeCh chan string,
	approvalCh chan permission.ApprovalRequest,
	questionCh chan tools.QuestionRequest,
	todoCh chan tools.TodoState,
	res *initResources,
) {
	initTotal := deferredInitTotal(cfg)
	send := func(item string, done bool) {
		ch <- tui.InitEvent{Item: item, Done: done, Total: initTotal}
	}
	fail := func(err error) {
		ch <- tui.InitEvent{Err: err}
	}

	// --- Phase 1: Core tools (fast, needed by everything) ---
	send("tools", false)

	coreTools, err := deferredInitCoreTools(sandboxRoot, worktreeDir, headerSessionID, todoCh, questionCh, res)
	if err != nil {
		fail(err)
		return
	}
	sandbox, bashSup := res.sandbox, res.bashSup

	send("tools", true)

	// --- Phase 2: Parallel subsystems ---
	ps := runDeferredInitPhase2(ctx, cfg, cwd, send)

	// --- Phase 3: Sequential finalization ---
	send("agent", false)

	// Store cleanup resources.
	res.lspMgr = ps.lspMgr

	// Build orchestrator (needs git results).
	orch := subagent.NewOrchestrator(&cfg, ps.repoRoot, ps.agentConfigs)
	orch.SetProviderOptions(flagURL, flagInsecure, flagHeaders)
	res.orch = orch

	// Build agent event channel and tools.
	agentEventCh := make(chan tui.AgentSubEvent, 128)
	agentEventCB := func(ev tools.SubagentEvent) {
		select {
		case agentEventCh <- tui.AgentSubEvent{
			AgentID:    ev.AgentID,
			Kind:       ev.Kind,
			Content:    ev.Content,
			PipelineID: ev.PipelineID,
			Mode:       ev.Mode,
			Step:       ev.Step,
			Total:      ev.Total,
			Background: ev.Background,
		}:
		default:
		}
	}
	agentTools, _ := tools.SubagentTools(orch, agentEventCB)
	coreTools = append(coreTools, agentTools...)

	// Stream live shell output to the same channel the subagent cards use. The
	// prefix keeps the two streams apart; the non-blocking send in agentEventCB
	// is what keeps a slow UI from stalling a running command.
	bashSup.SetSink(func(execID, kind, content string) {
		agentEventCB(tools.SubagentEvent{AgentID: execID, Kind: tui.BashEventKind(kind), Content: content})
	})

	// Append LSP tools.
	if ps.lspTools != nil {
		coreTools = append(coreTools, ps.lspTools...)
	}

	coreTools, memStore, memRecorder := appendDeferredMemoryTools(cfg, cwd, coreTools)

	// Build system instruction. The parts are kept so the context gauge can
	// attribute overhead to each section; composing them here is what keeps the
	// breakdown honest — instruction is literally parts.String().
	instructionParts := buildDeferredInstructionParts()

	// The callback inputs are shared with the runtime agent switcher: a
	// /agent switch rebuilds the chains from the same sandbox, LSP manager,
	// memory recorder, deduper and compaction metrics, changing only the
	// permission rules and the step budget.
	cbIn := callbackInputs{
		cfg:            cfg,
		providerName:   providerName,
		sandbox:        sandbox,
		lspMgr:         ps.lspMgr,
		memRecorder:    memRecorder,
		approvalCh:     approvalCh,
		deduper:        tools.NewResultDeduper(),
		compactMetrics: tools.NewCompactMetrics(),
	}
	res.cbIn = cbIn
	res.agentConfigs = ps.agentConfigs

	cbs := cbIn.build(globalPermissionRules(cfg), flagSteps)

	// defaultAgent: start the session inside a primary agent (opencode
	// parity). Unknown or non-primary names are a soft fallback to the
	// built-in agent; an explicit --model keeps the startup LLM and takes
	// only the agent's prompt, permission rules and step budget. The
	// agent's instruction replaces the prompt's base — the rules and skills
	// sections are session-level and survive the swap.
	activeAgent := ""
	agentModel, agentProvider := "", ""
	if cfg.DefaultAgent != "" {
		ac, ok, fallback := resolveDefaultAgent(ps.agentConfigs, cfg.DefaultAgent)
		if fallback != "" {
			softNotice(noticeCh, fallback)
		}
		if ok {
			activeAgent = ac.Name
			instructionParts.Base = ac.Instruction
			cbs = cbIn.build(permission.Merge(globalPermissionRules(cfg), ac.Permission), effectiveAgentSteps(ac.Steps))
			if flagModel == "" {
				if swLLM, swName, swProvider, err := agentSwitchLLM(ctx, cfg, tokenTracker, headerSessionID, &ac, ""); err != nil {
					softNotice(noticeCh, fmt.Sprintf("model for agent %q unavailable (%v); keeping the startup model", ac.Name, err))
				} else if swLLM != nil {
					llm = swLLM
					agentModel, agentProvider = swName, swProvider
				}
			}
		}
	}
	instruction := instructionParts.String()

	// Session service.
	sessionsPath, sessionSvc, err := openSessionService()
	if err != nil {
		fail(err)
		return
	}

	mcpToolsets := ps.mcpToolsets
	// llms.txt documentation sources are cheap to build (no network), so they
	// attach synchronously here rather than through the deferred loader that
	// handles MCP servers. Without this the fetch_docs tool exists in one-shot
	// and piagent modes but never in the interactive TUI. It goes into
	// coreTools rather than mcpToolsets: the context gauge's toolsetBytes only
	// counts *extension.resilientToolset entries, so a local toolset there
	// would be live for the model yet invisible in the breakdown, and the MCP
	// panel would list a non-MCP source.
	if llms := cfg.LLMSSources(); llms != nil {
		coreTools = append(coreTools, tools.LLMSTools(tools.NewLLMSCachedToolset(llms))...)
	}
	// Gemini search grounding (see agent.GeminiGroundingTool doc).
	//
	// APPEND — never replace. Assigning coreTools = []adktool.Tool{gTool} here
	// leaves the agent with *no* tools at all: bash, read, write, edit, grep,
	// ls, subagent, LSP and memory all vanish, and the model, given no function
	// declarations, invents names like "execute_command" and gets back
	// "tool not found. Available tools: " with an empty list.
	//
	// The built-in search coexists with function declarations: geminitool's
	// ProcessRequest appends to req.Config.Tools rather than overwriting it, and
	// a single Gemini turn will happily call `read` and `google_search` both.
	if gTool, ok := agent.GeminiGroundingTool(providerName, modelName); ok {
		coreTools = append(coreTools, gTool)
	}

	// --tools allow list. Applied after every append above (subagent, LSP,
	// memory, LLMS, grounding) so nothing in the flat list escapes it; MCP
	// toolsets were filtered by server name where they were built
	// (mcpToolsetsForRun).
	coreTools = applyToolAllowlist(coreTools)

	// Session logger. Created before the agent so it can capture the agent's
	// non-fatal diagnostics (e.g. unresolved instruction placeholders) in the
	// session log instead of leaking them to stderr and corrupting the TUI.
	// SessionStart is deferred until the session ID is resolved below.
	sessionLog, logErr := logger.New()
	if logErr == nil {
		res.sessionLog = sessionLog
		// --trace-http entries are dropped until the log file exists; the
		// transport is built before this point. See the matching note in
		// cli.go. Detached on shutdown where sessionLog is closed.
		httplog.SetSink(logger.HTTPSink(sessionLog))
	}

	// Create agent.
	ag, err := agent.New(agent.Config{
		Name:                 activeAgent,
		Model:                llm,
		Tools:                coreTools,
		Toolsets:             mcpToolsets,
		Instruction:          instruction,
		SessionService:       sessionSvc,
		Version:              versionString(),
		BeforeToolCallbacks:  cbs.beforeTool,
		AfterToolCallbacks:   cbs.afterTool,
		BeforeModelCallbacks: cbs.beforeModel,
		AfterModelCallbacks:  cbs.afterModel,
		Logger:               sessionLog,
	})
	if err != nil {
		fail(fmt.Errorf("creating agent: %w", err))
		return
	}

	sessionID, defaultTitle, resumed, err := resolveDeferredSession(ctx, ag, sessionSvc, llm, providerName, baseURL, headerSessionID)
	if err != nil {
		fail(err)
		return
	}

	// Store session ID for resume hint on exit.
	res.sessionID = sessionID

	// Two-stage auto-compaction, installed as a pre-turn hook so history is
	// only ever rewritten between turns. It shares the caller's notice channel
	// — a buffered, non-blocking send, so a compaction notice never blocks the
	// turn if the TUI is momentarily busy.
	if hook := autocompact.BuildHook(autocompact.Deps{
		SessionSvc:    sessionSvc,
		Tracker:       tokenTracker,
		Deduper:       cbs.deduper,
		Cfg:           autocompact.ConfigFrom(cfg),
		Log:           sessionLog,
		SummarizerLLM: llm,
		Notify: func(msg string) {
			select {
			case noticeCh <- msg:
			default:
			}
		},
	}); hook != nil {
		ag.SetPreTurnHook(hook)
	}

	// Capture ACP subagent events (claude, gemini) under the session dir.
	res.orch.SetACPLogPath(filepath.Join(sessionsPath, sessionID, "acp.jsonl"))

	// Session logger was created above; record the session start now that the
	// session ID is known.
	if logErr == nil {
		sessionLog.SessionStart(sessionID, llm.Name(), providerName, provider.BackendName(provider.Info{Provider: providerName}, config.APIKeys()[providerName], baseURL), baseURL, "interactive")
	}

	// Commit message function.
	commitMsgFn := buildCommitMsgFunc(ctx, cfg)

	send("agent", true)

	// Send final result.
	ch <- tui.InitEvent{
		Done: true,
		Result: &tui.InitResult{
			Agent:             ag,
			SessionID:         sessionID,
			SessionTitle:      defaultTitle,
			Resumed:           resumed,
			SessionService:    sessionSvc,
			Orchestrator:      orch,
			Logger:            sessionLog,
			Skills:            ps.skills,
			SkillDirs:         ps.skillDirs,
			GenerateCommitMsg: commitMsgFn,
			AgentEventCh:      agentEventCh,
			SystemNoticeCh:    noticeCh,
			ContextBreakdown: buildContextBreakdown(
				instructionParts, coreTools, mcpToolsets, ps.skills, ps.agentConfigs),
			TokenTracker:   tokenTracker,
			CompactMetrics: cbs.compactMetrics,
			GitBranch:      ps.gitBranch,
			DiffAdded:      ps.diffAdded,
			DiffRemoved:    ps.diffRemoved,
			MCPToolsets:    ps.mcpToolsets,
			MCPServers:     buildMCPServerConfigs(cfg),
			// defaultAgent: reflect the agent's model in the TUI when it
			// replaced the startup LLM, and hand over the switchable list.
			LLM:           llm,
			ModelName:     cmp.Or(agentModel, modelName),
			ProviderName:  cmp.Or(agentProvider, providerName),
			ActiveAgent:   activeAgent,
			PrimaryAgents: primaryAgentsFor(ps.agentConfigs),
		},
	}

	if memStore != nil {
		initMemoryAfterUI(ctx, cfg, cwd, sessionID, orch, llm, memStore, memRecorder, res)
	}
}

// deferredInitCoreTools builds the sandbox, the bash supervisor and the core
// tool set. Both the sandbox and the supervisor are recorded on res as soon as
// they exist, so a later failure here still leaves them for cleanup to close.
//
// headerSessionID registers the todo tools (todo_write/todo_read); todoCh
// receives the aggregated state after each todo_write so the sidebar and the
// /todos popup stay live. The send is non-blocking — a state update dropped
// while the TUI is busy is self-correcting, because the next todo_write
// carries the full list again.
//
// questionCh wires the question tool to the TUI dialog; nil keeps the tool's
// headless mode (immediate canceled).
func deferredInitCoreTools(sandboxRoot, worktreeDir, headerSessionID string, todoCh chan tools.TodoState, questionCh chan tools.QuestionRequest, res *initResources) ([]adktool.Tool, error) {
	sandbox, err := tools.NewSandbox(sandboxRoot, worktreeDir)
	if err != nil {
		return nil, fmt.Errorf("creating sandbox: %w", err)
	}
	res.sandbox = sandbox

	// Allow agent tools to access the Pi-rate home (logs, sessions, config).
	_ = sandbox.AddExtraDir(config.PirateHome())

	// The supervisor is built here, before the UI event channel exists, because
	// the bash tool needs it at construction time. Its sink is attached later,
	// once there is somewhere to stream to.
	bashSup := tools.NewBashSupervisor()
	res.bashSup = bashSup

	var todoNotifier func(tools.TodoState)
	if todoCh != nil {
		todoNotifier = func(s tools.TodoState) {
			select {
			case todoCh <- s:
			default:
			}
		}
	}
	// The question notifier must not block: a blocked send here parks the
	// tool call forever (production hang #32). The buffer-1 channel plus the
	// TUI's parked reader make the fast path non-blocking; if the slot is
	// still full (a delivered-but-unshown request), the stale one is
	// displaced with a canceled answer — nobody is waiting to show it, and
	// its tool call stops waiting on a Reply nobody owns.
	var questionNotifier func(tools.QuestionRequest)
	if questionCh != nil {
		questionNotifier = func(req tools.QuestionRequest) {
			select {
			case questionCh <- req:
				return
			default:
			}
			select {
			case stale := <-questionCh:
				// Reply is buffered to one: the send never blocks, even
				// when the abandoned call has stopped reading.
				stale.Reply <- tools.QuestionAnswer{Selected: "canceled"}
			default:
			}
			questionCh <- req
		}
	}
	coreTools, err := tools.CoreTools(sandbox, coreToolOptions(bashSup, headerSessionID, todoNotifier, questionNotifier)...)
	if err != nil {
		return nil, fmt.Errorf("creating core tools: %w", err)
	}
	bashCtlTools, err := tools.BashControlTools(bashSup)
	if err != nil {
		return nil, fmt.Errorf("creating bash control tools: %w", err)
	}
	return append(coreTools, bashCtlTools...), nil
}

// deferredParallelState collects what the phase-2 goroutines discover. Fields
// written by more than one goroutine are guarded by mu.
type deferredParallelState struct {
	mu sync.Mutex

	// Git + subagents
	repoRoot     string
	agentConfigs []subagent.AgentConfig
	gitBranch    string
	diffAdded    int
	diffRemoved  int

	// LSP
	lspMgr   *lsp.Manager
	lspTools []adktool.Tool

	// MCP
	mcpToolsets []adktool.Toolset

	// Skills
	skills    []extension.Skill
	skillDirs []string
}

// runDeferredInitPhase2 discovers git state, LSP servers, MCP toolsets and
// skills concurrently and returns once all four are done.
func runDeferredInitPhase2(ctx context.Context, cfg config.Config, cwd string, send func(item string, done bool)) *deferredParallelState {
	var ps deferredParallelState
	var wg sync.WaitGroup

	// Git + subagent discovery
	wg.Add(1)
	go func() {
		defer wg.Done()
		send("git", false)
		ps.repoRoot = detectGitRoot(ctx, cwd)
		discovery, _ := subagent.DiscoverAgents(cwd, subagent.ScopeBoth)
		if discovery != nil {
			ps.agentConfigs = discovery.All
		}
		ps.gitBranch = detectBranch(ctx, cwd)
		ps.diffAdded, ps.diffRemoved = computeDiffStats(ctx, cwd)
		send("git", true)
	}()

	// LSP
	wg.Add(1)
	go func() {
		defer wg.Done()
		send("lsp", false)
		mgr := lsp.NewManager(nil)
		// Only advertise the LSP tools when a server can actually answer them —
		// see the matching note in cli.go. The manager itself is always kept:
		// the after-tool callback and diagnostics plumbing cost nothing idle.
		var lt []adktool.Tool
		if mgr.AnyAvailable() {
			lt, _ = tools.LSPToolsFor(mgr, resolveLSPMode())
		}
		ps.mu.Lock()
		ps.lspMgr = mgr
		ps.lspTools = lt
		ps.mu.Unlock()
		send("lsp", true)
	}()

	// MCP
	wg.Add(1)
	go func() {
		defer wg.Done()
		if cfg.MCP == nil || len(cfg.MCP.Servers) == 0 {
			return
		}
		send("mcp", false)
		ts := mcpToolsetsForRun(cfg)
		ps.mu.Lock()
		ps.mcpToolsets = ts
		ps.mu.Unlock()
		send("mcp", true)
	}()

	// Skills
	wg.Add(1)
	go func() {
		defer wg.Done()
		send("skills", false)
		dirs := extension.DefaultSkillDirsIn(cwd, cfg)
		sk, _ := extension.LoadSkills(dirs...)
		ps.mu.Lock()
		ps.skills = sk
		ps.skillDirs = dirs
		ps.mu.Unlock()
		send("skills", true)
	}()

	wg.Wait()
	return &ps
}

// appendDeferredMemoryTools adds the memory tools when memory is enabled and
// returns the lazy store and observation recorder they run against. Both are
// nil when memory is off, which is what gates the post-UI memory init.
func appendDeferredMemoryTools(cfg config.Config, cwd string, coreTools []adktool.Tool) ([]adktool.Tool, *lazyMemoryStore, *deferredMemoryRecorder) {
	if !deferredMemoryEnabled(cfg) {
		return coreTools, nil, nil
	}
	memStore := newLazyMemoryStore()
	if memTools, memErr := tools.MemoryTools(memStore); memErr == nil {
		coreTools = append(coreTools, memTools...)
	}
	return coreTools, memStore, newDeferredMemoryRecorder(cfg, cwd)
}

// buildDeferredInstructionParts returns the system instruction as its parts:
// --system replaces the base outright, otherwise the built-in set is loaded.
func buildDeferredInstructionParts() agent.InstructionParts {
	if flagSystem != "" {
		return agent.InstructionParts{Base: flagSystem}
	}
	return agent.LoadInstructionParts(agent.SystemInstruction)
}

// deferredCallbacks is the callback wiring for the interactive agent, plus the
// two objects the caller still needs a handle on: the deduper the auto-compact
// hook shares, and the metrics the context gauge reads.
type deferredCallbacks struct {
	beforeTool     []llmagent.BeforeToolCallback
	afterTool      []llmagent.AfterToolCallback
	beforeModel    []llmagent.BeforeModelCallback
	afterModel     []llmagent.AfterModelCallback
	deduper        *tools.ResultDeduper
	compactMetrics *tools.CompactMetrics
}

// callbackInputs collects what the interactive callback chains are built
// against. The same inputs rebuild the chains when the main session moves to
// a primary agent — only the permission rules and the step budget change —
// so the deduper and the compaction metrics stay shared across switches: the
// auto-compact hook and the context gauge hold the original instances.
type callbackInputs struct {
	cfg            config.Config
	providerName   string
	sandbox        *tools.Sandbox
	lspMgr         *lsp.Manager
	memRecorder    *deferredMemoryRecorder
	approvalCh     chan permission.ApprovalRequest
	deduper        *tools.ResultDeduper
	compactMetrics *tools.CompactMetrics
}

// build assembles the tool and model callback chains in the order they must
// run. rules gate tool calls; steps caps iterations (0 = unlimited). When
// approvalCh is non-nil, ask shows the TUI approval dialog instead of the
// hard denial.
func (in callbackInputs) build(rules permission.Rules, steps int) deferredCallbacks {
	if in.deduper == nil {
		in.deduper = tools.NewResultDeduper()
	}
	if in.compactMetrics == nil {
		in.compactMetrics = tools.NewCompactMetrics()
	}
	compactorCfg := compactorConfigFrom(in.cfg)
	compactorCB := tools.BuildCompactorCallback(compactorCfg, in.compactMetrics)
	resultDeduper := in.deduper

	hooks := convertHooks(in.cfg.Hooks)
	beforeCBs := extension.BuildBeforeToolCallbacks(hooks)

	// Permission rules gate every tool call before execution. The default
	// session contributes only the global config rules; a primary agent's
	// frontmatter rules arrive merged on top of them. With an approval
	// bridge wired, ask shows the TUI dialog instead of denying; subagent
	// processes keep the hard denial.
	if !rules.Empty() {
		if in.approvalCh != nil {
			beforeCBs = append(beforeCBs, agent.NewAskingPermissionCallback(rules, approvalAsker(in.approvalCh)))
		} else {
			beforeCBs = append(beforeCBs, agent.NewPermissionCallback(rules))
		}
	}
	afterCBs := extension.BuildAfterToolCallbacks(hooks)

	// Always add OTEL tracing callbacks so all tool calls are traced.
	tracingBefore, tracingAfter := extension.BuildTracingCallbacks()
	beforeCBs = append(beforeCBs, tracingBefore...)
	afterCBs = append(afterCBs, tracingAfter...)
	if in.lspMgr != nil {
		afterCBs = append(afterCBs, lsp.BuildLSPAfterToolCallback(in.lspMgr))
	}
	// Dedup runs BEFORE the compactor: its hash must cover the bytes the tool
	// produced, not the truncated form. Compaction is lossy, so hashing after it
	// makes two different results collide and the second is wrongly reported as
	// "content is unchanged". See the note in cli.go where the same chain is
	// wired for non-interactive runs.
	afterCBs = append(afterCBs, tools.BuildDedupCallback(resultDeduper), compactorCB)

	// LLM tracing: before/after model callbacks emit spans per LLM invocation.
	llmBefore, llmAfter := extension.BuildLLMTracingCallbacks(in.providerName)

	// Inject image bytes (screenshots) as visible InlineData parts for the model.
	llmBefore = append(llmBefore, extension.BuildReadImageCallback(in.sandbox, in.providerName))

	if in.memRecorder != nil {
		afterCBs = append(afterCBs, in.memRecorder.afterTool)
	}

	// steps caps the tool-call iterations; 0 disables it. Must ride the
	// composed chain below — a separate slice entry would never run (ADK
	// stops at the first callback that returns a result).
	if steps > 0 {
		afterCBs = append(afterCBs, agent.NewStepLimitCallback(steps))
	}

	// Fold the after-tool chain into the single callback ADK runs. ADK's
	// Flow.invokeAfterToolCallbacks returns at the first callback that yields a
	// non-nil result, and every callback above returns the result map, so
	// passing the slice would run only the first and skip the rest — which is
	// how dedup, the compactor and memory recording were all dead in production.
	return deferredCallbacks{
		beforeTool:     beforeCBs,
		afterTool:      extension.ComposeAfterToolChain(afterCBs),
		beforeModel:    llmBefore,
		afterModel:     llmAfter,
		deduper:        resultDeduper,
		compactMetrics: in.compactMetrics,
	}
}

// buildDeferredCallbacks assembles the callback chains for the default
// (built-in agent) session: global permission rules and the --steps budget.
func buildDeferredCallbacks(
	cfg config.Config,
	providerName string,
	sandbox *tools.Sandbox,
	lspMgr *lsp.Manager,
	memRecorder *deferredMemoryRecorder,
	approvalCh chan permission.ApprovalRequest,
) deferredCallbacks {
	return callbackInputs{
		cfg:            cfg,
		providerName:   providerName,
		sandbox:        sandbox,
		lspMgr:         lspMgr,
		memRecorder:    memRecorder,
		approvalCh:     approvalCh,
		deduper:        tools.NewResultDeduper(),
		compactMetrics: tools.NewCompactMetrics(),
	}.build(globalPermissionRules(cfg), flagSteps)
}

// approvalAsker returns the blocking bridge the permission gate parks on for
// each ask decision. It hands the request to the TUI dialog over ch and waits
// for the user's answer. ctx is the agent turn's context: when the turn is
// canceled (Ctrl+C, session exit) the wait must end in a denial rather than
// leave the gate parked on a dialog nobody will answer. An already-canceled
// context never reaches the dialog at all — a dead tool call must not open a
// window. A late answer is harmless — Reply is buffered to one and simply
// dropped.
func approvalAsker(ch chan<- permission.ApprovalRequest) func(context.Context, permission.ApprovalRequest) permission.ApprovalResult {
	return func(ctx context.Context, req permission.ApprovalRequest) permission.ApprovalResult {
		if ctx == nil {
			ctx = context.Background()
		}
		if ctx.Err() != nil {
			// A pre-select check, not just a select arm: with both channels
			// ready, select picks randomly, and a canceled turn must not
			// enqueue a dialog for a call that is already dead.
			return permission.ApprovalResult{}
		}
		select {
		case ch <- req:
		case <-ctx.Done():
			return permission.ApprovalResult{}
		}
		select {
		case res := <-req.Reply:
			return res
		case <-ctx.Done():
			return permission.ApprovalResult{}
		}
	}
}

// resolveDeferredSession returns the session to run in — the one named by
// --continue/--session if there is one, otherwise a fresh one — and records the
// model and backend it is running under.
func resolveDeferredSession(
	ctx context.Context,
	ag *agent.Agent,
	sessionSvc *pisession.FileService,
	llm adkmodel.LLM,
	providerName, baseURL string,
	headerSessionID string,
) (sessionID, defaultTitle string, resumed bool, err error) {
	// --continue is resolved in the fast path, which sets flagSession.
	sessionID = flagSession
	resumed = sessionID != ""
	if sessionID == "" {
		// The header ID wins when one was baked in: the LLM client is already
		// sending it, so the session must carry the same name.
		sessionID, defaultTitle, err = ag.CreateSessionWithID(ctx, headerSessionID)
		if err != nil {
			return "", "", false, fmt.Errorf("creating session: %w", err)
		}
	}

	// Keep the recorded model honest. meta.Model used to be written once, at
	// creation, so a session resumed under a different model (via --model) kept
	// advertising the old one — and the next resume would restore that instead
	// of what the session actually last ran with.
	if resumed {
		_ = sessionSvc.SetSessionModel(sessionID, llm.Name()) // best-effort metadata
	}
	// Record the backend for every session, resumed or fresh. The model name on
	// its own does not identify what actually served the request, which is the
	// first thing anyone needs when reading a transcript back.
	_ = sessionSvc.SetSessionProvider(sessionID, providerName, baseURL) // best-effort metadata

	return sessionID, defaultTitle, resumed, nil
}

type lazyMemoryStore struct {
	mu    sync.RWMutex
	ready chan struct{}
	store memory.Store
	err   error
}

func newLazyMemoryStore() *lazyMemoryStore {
	return &lazyMemoryStore{ready: make(chan struct{})}
}

func (s *lazyMemoryStore) setReady(store memory.Store, err error) {
	s.mu.Lock()
	s.store = store
	s.err = err
	s.mu.Unlock()
	close(s.ready)
}

func (s *lazyMemoryStore) wait(ctx context.Context) (memory.Store, error) {
	select {
	case <-s.ready:
		s.mu.RLock()
		defer s.mu.RUnlock()
		if s.err != nil {
			return nil, s.err
		}
		if s.store == nil {
			return nil, fmt.Errorf("memory store unavailable")
		}
		return s.store, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (s *lazyMemoryStore) CreateSession(ctx context.Context, sess *memory.Session) error {
	store, err := s.wait(ctx)
	if err != nil {
		return err
	}
	return store.CreateSession(ctx, sess)
}

func (s *lazyMemoryStore) CompleteSession(ctx context.Context, sessionID string) error {
	store, err := s.wait(ctx)
	if err != nil {
		return err
	}
	return store.CompleteSession(ctx, sessionID)
}

func (s *lazyMemoryStore) InsertObservation(ctx context.Context, obs *memory.Observation) error {
	store, err := s.wait(ctx)
	if err != nil {
		return err
	}
	return store.InsertObservation(ctx, obs)
}

func (s *lazyMemoryStore) GetObservations(ctx context.Context, ids []int64) ([]*memory.Observation, error) {
	store, err := s.wait(ctx)
	if err != nil {
		return nil, err
	}
	return store.GetObservations(ctx, ids)
}

func (s *lazyMemoryStore) RecentObservations(ctx context.Context, project string, limit int) ([]*memory.Observation, error) {
	store, err := s.wait(ctx)
	if err != nil {
		return nil, err
	}
	return store.RecentObservations(ctx, project, limit)
}

func (s *lazyMemoryStore) SessionObservations(ctx context.Context, sessionID string) ([]*memory.Observation, error) {
	store, err := s.wait(ctx)
	if err != nil {
		return nil, err
	}
	return store.SessionObservations(ctx, sessionID)
}

func (s *lazyMemoryStore) HasObservations(ctx context.Context, sessionID string) (bool, error) {
	store, err := s.wait(ctx)
	if err != nil {
		return false, err
	}
	return store.HasObservations(ctx, sessionID)
}

func (s *lazyMemoryStore) UpsertSummary(ctx context.Context, sum *memory.SessionSummary) error {
	store, err := s.wait(ctx)
	if err != nil {
		return err
	}
	return store.UpsertSummary(ctx, sum)
}

func (s *lazyMemoryStore) RecentSummaries(ctx context.Context, project string, limit int) ([]*memory.SessionSummary, error) {
	store, err := s.wait(ctx)
	if err != nil {
		return nil, err
	}
	return store.RecentSummaries(ctx, project, limit)
}

func (s *lazyMemoryStore) Search(ctx context.Context, q memory.SearchQuery) (*memory.SearchResult, error) {
	store, err := s.wait(ctx)
	if err != nil {
		return nil, err
	}
	return store.Search(ctx, q)
}

func (s *lazyMemoryStore) Timeline(ctx context.Context, anchorID int64, before, after int) ([]*memory.Observation, error) {
	store, err := s.wait(ctx)
	if err != nil {
		return nil, err
	}
	return store.Timeline(ctx, anchorID, before, after)
}

func (s *lazyMemoryStore) Close() error {
	return nil
}

type deferredMemoryRecorder struct {
	mu            sync.RWMutex
	project       string
	excludedTools map[string]bool
	sessionID     string
	worker        *memory.Worker
}

func newDeferredMemoryRecorder(cfg config.Config, project string) *deferredMemoryRecorder {
	excluded := make(map[string]bool)
	if cfg.Memory != nil {
		for _, name := range cfg.Memory.ExcludedTools {
			excluded[name] = true
		}
	}
	return &deferredMemoryRecorder{
		project:       project,
		excludedTools: excluded,
	}
}

func (r *deferredMemoryRecorder) setReady(sessionID string, worker *memory.Worker) {
	r.mu.Lock()
	r.sessionID = sessionID
	r.worker = worker
	r.mu.Unlock()
}

func (r *deferredMemoryRecorder) afterTool(_ adkagent.Context, t adktool.Tool, args, result map[string]any, toolErr error) (map[string]any, error) {
	if toolErr != nil {
		return result, nil
	}

	name := t.Name()
	r.mu.RLock()
	worker := r.worker
	sessionID := r.sessionID
	excluded := r.excludedTools[name]
	project := r.project
	r.mu.RUnlock()

	if worker == nil || sessionID == "" || excluded {
		return result, nil
	}

	worker.Enqueue(memory.RawObservation{
		SessionID:  sessionID,
		Project:    project,
		ToolName:   name,
		ToolInput:  args,
		ToolOutput: result,
		Timestamp:  time.Now(),
	})
	return result, nil
}

func initMemoryAfterUI(
	ctx context.Context,
	cfg config.Config,
	cwd string,
	sessionID string,
	orch *subagent.Orchestrator,
	llm adkmodel.LLM,
	store *lazyMemoryStore,
	recorder *deferredMemoryRecorder,
	res *initResources,
) {
	memCfg := deferredMemoryConfig(cfg)
	dbPath := deferredMemoryDBPath(memCfg)
	if dbPath == "" {
		store.setReady(nil, fmt.Errorf("memory init: home directory unavailable"))
		return
	}

	memDB, err := memory.OpenDB(dbPath)
	if err != nil {
		store.setReady(nil, fmt.Errorf("memory init: %w", err))
		return
	}

	memStore := memory.NewSQLiteStore(memDB)
	_ = memStore.CreateSession(ctx, &memory.Session{
		SessionID: sessionID,
		Project:   cwd,
		StartedAt: time.Now(),
		Status:    "active",
	})

	compressorName, known := cfg.ResolveCompressor()
	if !known {
		slog.Warn("memory: unknown compressor in config, using default",
			"configured", cfg.Memory.Compressor, "using", compressorName)
	}

	worker := memory.NewWorker(memStore, memory.NewCompressor(compressorName, orch), memCfg.MaxPending)
	worker.Start(ctx)

	// The summary is written by initResources.cleanup after the worker has
	// drained, so it is held here rather than run now: at this point no tool
	// call has been made, and a summary written here would describe an empty
	// session.
	res.memSessionID = sessionID
	res.memProject = cwd
	res.memSummarizer = memory.NewSessionSummarizer(memStore, llm)

	res.memStore = memStore
	res.memWorker = worker
	if recorder != nil {
		recorder.setReady(sessionID, worker)
	}
	store.setReady(memStore, nil)
}

func deferredMemoryConfig(cfg config.Config) config.MemoryConfig {
	memCfg := config.MemoryDefaults()
	if cfg.Memory == nil {
		return memCfg
	}
	if cfg.Memory.DBPath != "" {
		memCfg.DBPath = cfg.Memory.DBPath
	}
	if cfg.Memory.TokenBudget > 0 {
		memCfg.TokenBudget = cfg.Memory.TokenBudget
	}
	if cfg.Memory.MaxPending > 0 {
		memCfg.MaxPending = cfg.Memory.MaxPending
	}
	if cfg.Memory.LookbackHours > 0 {
		memCfg.LookbackHours = cfg.Memory.LookbackHours //nolint:govet // reserved for future use
	}
	return memCfg
}

func deferredMemoryDBPath(memCfg config.MemoryConfig) string {
	if memCfg.DBPath != "" {
		return memCfg.DBPath
	}
	return filepath.Join(config.PirateHome(), "memory", "claude-mem.db")
}

func deferredInitTotal(cfg config.Config) int {
	total := 5 // tools, git, lsp, skills, agent
	if cfg.MCP != nil && len(cfg.MCP.Servers) > 0 {
		total++
	}
	if deferredMemoryEnabled(cfg) {
		total++
	}
	return total
}

func deferredMemoryEnabled(cfg config.Config) bool {
	if flagMemoryOff {
		return false
	}
	return cfg.Memory == nil || cfg.Memory.Enabled == nil || *cfg.Memory.Enabled
}

// detectBranch returns the current git branch name.
func detectBranch(ctx context.Context, workDir string) string {
	ctx, cancel := context.WithTimeout(ctx, gitCmdTimeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, "git", "rev-parse", "--abbrev-ref", "HEAD")
	if workDir != "" {
		cmd.Dir = workDir
	}
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

// computeDiffStats returns added and removed line counts from git diff,
// including lines from untracked files.
func computeDiffStats(ctx context.Context, cwd string) (added, removed int) {
	diffCtx, cancel := context.WithTimeout(ctx, gitCmdTimeout)
	defer cancel()
	cmd := exec.CommandContext(diffCtx, "git", "diff", "--numstat", "HEAD")
	cmd.Dir = cwd
	out, err := cmd.Output()
	if err != nil {
		return 0, 0
	}
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		if line == "" {
			continue
		}
		var a, r int
		if _, err := fmt.Sscanf(line, "%d\t%d\t", &a, &r); err == nil {
			added += a
			removed += r
		}
	}
	added += countUntrackedLines(ctx, cwd)
	return added, removed
}

// countUntrackedLines counts total lines across untracked files. The whole
// operation (ls-files plus one wc per file) shares a single bounded timeout
// so a large or stalled untracked-file set can't hang the init pipeline.
func countUntrackedLines(ctx context.Context, cwd string) int {
	ctx, cancel := context.WithTimeout(ctx, gitCmdTimeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, "git", "ls-files", "--others", "--exclude-standard")
	cmd.Dir = cwd
	out, err := cmd.Output()
	if err != nil {
		return 0
	}
	total := 0
	for _, file := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		if file == "" {
			continue
		}
		wc := exec.CommandContext(ctx, "wc", "-l", file)
		wc.Dir = cwd
		wcOut, err := wc.Output()
		if err != nil {
			continue
		}
		var lines int
		if _, err := fmt.Sscanf(strings.TrimSpace(string(wcOut)), "%d", &lines); err == nil {
			total += lines
		}
	}
	return total
}

// buildMCPServerConfigs converts config.MCPServer slice to extension.MCPServerConfig slice.
func buildMCPServerConfigs(cfg config.Config) []extension.MCPServerConfig {
	if cfg.MCP == nil {
		return nil
	}
	out := make([]extension.MCPServerConfig, len(cfg.MCP.Servers))
	for i, s := range cfg.MCP.Servers {
		out[i] = extension.MCPServerConfig{
			Name:    s.Name,
			Command: s.Command,
			Args:    s.Args,
			URL:     s.URL,
			Headers: s.Headers,
			OAuth:   s.OAuth,
		}
	}
	return out
}

// buildSwitchedLLM creates a new LLM instance for the given model name using
// the current config and token tracker. It resolves the provider, validates
// the model, creates the LLM, updates the token tracker's context window size,
// and wraps it with the guardrail. Used by the TUI /model <name> command.
// headerSessionID keeps the ${SESSION_ID} headers pointed at the conversation
// the client was originally built for — /model switches models, not sessions.
func buildSwitchedLLM(ctx context.Context, cfg config.Config, tokenTracker *guardrail.Tracker, modelName, headerSessionID string) (adkmodel.LLM, string, string, error) {
	return buildSwitchedLLMWith(ctx, cfg, tokenTracker, modelName, headerSessionID, temperatureFlagOpt(), effectiveThinkingLevel(cfg))
}

// buildSwitchedLLMWith is buildSwitchedLLM with explicit temperature and
// thinking level, so a primary agent's frontmatter can override both.
func buildSwitchedLLMWith(ctx context.Context, cfg config.Config, tokenTracker *guardrail.Tracker, modelName, headerSessionID string, temperature *float64, thinking string) (adkmodel.LLM, string, string, error) {
	providerName := ""
	if rc, ok := cfg.Roles["default"]; ok && rc.Provider != "" {
		providerName = rc.Provider
	}

	info, baseURL, apiKey, err := resolveSwitchedModel(cfg, modelName, providerName)
	if err != nil {
		return nil, "", "", err
	}

	llmOpts := &provider.LLMOptions{
		ExtraHeaders: providerExtraHeaders(cfg, info.Provider, headerSessionID, flagHeaders),
		Temperature:  temperature,
	}
	applyTransportOptions(llmOpts, cfg, info)
	llm, err := provider.NewLLM(ctx, info, apiKey, baseURL, thinking, llmOpts)
	if err != nil {
		return nil, "", "", fmt.Errorf("creating LLM: %w", err)
	}

	tokenTracker.SetContextWindowSize(switchContextWindowSize(ctx, cfg, info, baseURL))
	llm = guardrail.WrapModel(llm, tokenTracker)

	return llm, switchedModelName(cfg, info), info.Provider, nil
}

// agentSwitch builds the tui.AgentSwitch that moves the main session onto a
// primary agent (name "" = the built-in default): the system prompt, the
// callback chains rebuilt over the agent's permission rules and step budget,
// and the agent's LLM when its `model:`/`role:` — or modelOverride — names
// one. A non-empty modelOverride (a /model typed while the agent was active)
// beats the agent's own model; a nil LLM means "keep the model that is
// running".
func agentSwitch(
	ctx context.Context,
	cfg config.Config,
	tokenTracker *guardrail.Tracker,
	headerSessionID string,
	in *callbackInputs,
	configs []subagent.AgentConfig,
	name string,
	modelOverride string,
) (tui.AgentSwitch, error) {
	var ac *subagent.AgentConfig
	if name != "" {
		found, ok := findAgentConfig(configs, name)
		if !ok {
			return tui.AgentSwitch{}, fmt.Errorf("unknown agent %q", name)
		}
		if !found.IsPrimary() {
			return tui.AgentSwitch{}, fmt.Errorf("agent %q is not a primary agent (frontmatter mode: primary)", name)
		}
		ac = &found
	}

	sw := tui.AgentSwitch{}
	rules := globalPermissionRules(cfg)
	steps := flagSteps
	if ac != nil {
		sw.Instruction = ac.Instruction
		rules = permission.Merge(rules, ac.Permission)
		steps = effectiveAgentSteps(ac.Steps)
	} else {
		sw.Instruction = buildDeferredInstructionParts().String()
	}
	cbs := in.build(rules, steps)
	sw.BeforeTool, sw.AfterTool = cbs.beforeTool, cbs.afterTool
	// The switch target's identity rides along so the rebuilt runner stamps
	// its events with the agent's own name (empty restores the built-in one).
	sw.Name = name

	llm, modelName, providerName, err := agentSwitchLLM(ctx, cfg, tokenTracker, headerSessionID, ac, modelOverride)
	if err != nil {
		return tui.AgentSwitch{}, err
	}
	sw.LLM, sw.ModelName, sw.Provider = llm, modelName, providerName
	return sw, nil
}

// agentSwitchLLM resolves the model a switch target runs on: modelOverride
// (a session /model typed while the agent was active) when set, otherwise the
// agent's `model:` or its `role:`'s model, otherwise — for the default target
// — the default role. The agent's temperature and reasoningEffort override
// the flag/config defaults either way. Nothing resolvable returns a nil LLM —
// the caller keeps whatever is running.
func agentSwitchLLM(ctx context.Context, cfg config.Config, tokenTracker *guardrail.Tracker, headerSessionID string, ac *subagent.AgentConfig, modelOverride string) (adkmodel.LLM, string, string, error) {
	var modelName string
	switch {
	case modelOverride != "":
		modelName = modelOverride
	case ac != nil:
		modelName = ac.Model
		if modelName == "" && ac.Role != "" {
			if rc, ok := cfg.Roles[ac.Role]; ok {
				modelName = rc.Model
			}
		}
	default:
		if rc, ok := cfg.Roles["default"]; ok {
			modelName = rc.Model
		}
	}
	if modelName == "" {
		return nil, "", "", nil
	}

	temperature := temperatureFlagOpt()
	thinking := effectiveThinkingLevel(cfg)
	if ac != nil {
		if ac.Temperature > 0 {
			t := ac.Temperature
			temperature = &t
		}
		if th := subagent.NormalizeReasoningEffort(ac.ReasoningEffort); th != "" {
			thinking = th
		}
	}
	return buildSwitchedLLMWith(ctx, cfg, tokenTracker, modelName, headerSessionID, temperature, thinking)
}

// findAgentConfig looks up an agent by name in a discovery result slice.
func findAgentConfig(configs []subagent.AgentConfig, name string) (subagent.AgentConfig, bool) {
	for _, ac := range configs {
		if ac.Name == name {
			return ac, true
		}
	}
	return subagent.AgentConfig{}, false
}

// resolveDefaultAgent checks config.json's defaultAgent against the
// discovered agents. ok reports a usable primary agent; notice carries the
// soft-fallback message for an unknown or non-primary name ("" when ok).
func resolveDefaultAgent(configs []subagent.AgentConfig, name string) (ac subagent.AgentConfig, ok bool, notice string) {
	found, foundOk := findAgentConfig(configs, name)
	if !foundOk {
		return subagent.AgentConfig{}, false, fmt.Sprintf("defaultAgent %q not found; starting with the default agent", name)
	}
	if !found.IsPrimary() {
		return subagent.AgentConfig{}, false, fmt.Sprintf("defaultAgent %q is not a primary agent (frontmatter mode: primary); starting with the default agent", name)
	}
	return found, true, ""
}

// effectiveAgentSteps resolves the step budget for an agent-driven session:
// the agent's own cap when set, else the --steps flag.
func effectiveAgentSteps(agentSteps int) int {
	if agentSteps > 0 {
		return agentSteps
	}
	return flagSteps
}

// primaryAgentsFor filters the discovery result down to the switchable
// agents (frontmatter mode: primary or all), sorted by name.
func primaryAgentsFor(configs []subagent.AgentConfig) []subagent.AgentConfig {
	var out []subagent.AgentConfig
	for _, ac := range configs {
		if ac.IsPrimary() {
			out = append(out, ac)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// softNotice routes a non-fatal startup message to the TUI notice channel.
// Non-blocking: a notice raised while the TUI is busy is dropped rather than
// stalling startup.
func softNotice(ch chan<- string, msg string) {
	select {
	case ch <- msg:
	default:
	}
}

// resolveSwitchedModel resolves modelName to a validated model info plus the
// endpoint and API key to reach it: provider auto-detection from config's
// default role, base URL resolution, model validation, and Ollama endpoint
// fallback all happen here.
func resolveSwitchedModel(cfg config.Config, modelName, providerName string) (provider.Info, string, string, error) {
	// A model served by a declared provider (config.json "providers") resolves
	// through that provider's endpoint and key, by its prefixed name alone —
	// before the role's provider and the built-in detection can mis-route it.
	if info, apiKey, baseURL, ok := namedModelInfo(cfg, modelName, providerName); ok {
		if err := provider.ValidateModel(info); err != nil {
			return provider.Info{}, "", "", fmt.Errorf("model validation: %w", err)
		}
		info.BaseURL = baseURL
		return info, baseURL, apiKey, nil
	}

	// A model named with an explicit provider prefix ("openrouter/gemma-4")
	// keeps the provider it named: the role's provider is only a default for a
	// bare name, and letting it win would send the request to the wrong
	// backend — a switch away from the default role's provider could then never
	// succeed, which is what `/model` hit once the role had a provider written
	// into it.
	if _, _, prefixed := provider.ProviderFromPrefix(modelName); prefixed {
		providerName = ""
	}

	baseURL := flagURL
	if baseURL == "" && providerName != "" {
		baseURLs := cfg.ResolveBaseURLs()
		baseURL = baseURLs[providerName]
	}

	info, err := provider.ResolveWithBaseURL(modelName, baseURL)
	if err != nil {
		return provider.Info{}, "", "", fmt.Errorf("resolving model: %w", err)
	}
	if providerName != "" {
		info.Provider = providerName
		info.Custom = baseURL != ""
	}
	if baseURL == "" {
		baseURLs := cfg.ResolveBaseURLs()
		baseURL = baseURLs[info.Provider]
		if baseURL != "" {
			info.Custom = true
		}
	}
	if err := provider.ValidateModel(info); err != nil {
		return provider.Info{}, "", "", fmt.Errorf("model validation: %w", err)
	}

	// ResolveAPIKeys covers the built-in env vars and each declared provider's
	// key; the named branch above already returned, so this reads a built-in
	// provider's key.
	apiKey := cfg.ResolveAPIKeys()[info.Provider]

	if info.Ollama {
		baseURL = provider.ResolveOllamaEndpoint(provider.OllamaRouting{
			Model:      info.Model,
			BaseURL:    baseURL,
			APIKey:     apiKey,
			ForceLocal: info.LocalOllama,
		})
	}

	// Record the endpoint actually chosen, after every fallback above has had
	// its say, so session metadata names the backend rather than leaving the
	// model name to be interpreted.
	info.BaseURL = baseURL
	return info, baseURL, apiKey, nil
}

// switchContextWindowSize determines the context window for the switched
// model. The rules live in ctxwindow.Resolve — the same resolution the
// startup model gets — so a /model or agent switch and a restart can never
// disagree about one model's window.
func switchContextWindowSize(ctx context.Context, cfg config.Config, info provider.Info, baseURL string) int64 {
	return ctxwindow.Resolve(ctx, cfg, info, baseURL)
}

// switchedModelName hands back a name that re-resolves to the same endpoint.
// Resolve strips the provider prefix from info.Model, and this answer is what
// the TUI stores as the current model, persists to the default role, and
// re-resolves on the next switch — so returning the bare name would drop the
// one part of the spelling that identifies the backend. For ollama/ that is
// what says "local", and a cloud-tagged model would move to api.ollama.com
// behind the user's back the moment a key was set; for the rest it is the
// difference between one gateway and another.
//
// Re-adding the prefix is what keeps the name self-sufficient, which is what
// lets the next switch resolve it without a provider recorded alongside it.
func switchedModelName(cfg config.Config, info provider.Info) string {
	if info.LocalOllama {
		return "ollama/" + info.Model
	}
	// A declared provider (config.json "providers") has no built-in prefix,
	// and a bare name can never re-resolve onto it — Resolve only returns
	// built-in providers — so the name is always part of a self-sufficient
	// spelling. Without this a switch to corp-claude/foo would persist the
	// bare "foo" and the next switch would land on the built-in provider that
	// prefix-detects it.
	if info.Custom && info.Protocol != "" {
		if _, declared := cfg.Providers[info.Provider]; declared {
			return info.Provider + "/" + info.Model
		}
	}
	// A bare name that already resolves to this provider needs no prefix, and
	// adding one would rewrite a spelling the catalog and the user recognize
	// ("claude-opus-5" → "anthropic/claude-opus-5").
	if resolved, err := provider.Resolve(info.Model); err == nil && resolved.Provider == info.Provider {
		return info.Model
	}
	if prefix, ok := provider.PrefixFor(info.Provider); ok {
		return prefix + info.Model
	}
	return info.Model
}

// planAutoFixEnabled resolves the tri-state planAutoFix config value. The
// default is on: a plan that fails the PDD contract should repair itself rather
// than wait for a human to notice. An explicit false in config.json turns it
// off.
func planAutoFixEnabled(cfg config.Config) bool {
	if cfg.PlanAutoFix == nil {
		return true
	}
	return *cfg.PlanAutoFix
}
