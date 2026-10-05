package cli

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"log/slog"
	"maps"
	"net/http"
	_ "net/http/pprof" // registers pprof HTTP handlers on /debug/pprof
	"os"
	"os/signal"
	"path/filepath"
	"runtime/pprof"
	"strconv"
	"strings"
	"sync"
	"time"

	adkagent "google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	adkmodel "google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	adktool "google.golang.org/adk/v2/tool"

	"github.com/spa-skyson/pi-rate/internal/agent"
	"github.com/spa-skyson/pi-rate/internal/autocompact"
	"github.com/spa-skyson/pi-rate/internal/config"
	"github.com/spa-skyson/pi-rate/internal/ctxwindow"
	"github.com/spa-skyson/pi-rate/internal/extension"
	"github.com/spa-skyson/pi-rate/internal/gitroot"
	"github.com/spa-skyson/pi-rate/internal/guardrail"
	"github.com/spa-skyson/pi-rate/internal/httplog"
	"github.com/spa-skyson/pi-rate/internal/jsonrpc"
	"github.com/spa-skyson/pi-rate/internal/logger"
	"github.com/spa-skyson/pi-rate/internal/lsp"
	"github.com/spa-skyson/pi-rate/internal/memory"
	"github.com/spa-skyson/pi-rate/internal/otel"
	"github.com/spa-skyson/pi-rate/internal/palace"
	"github.com/spa-skyson/pi-rate/internal/permission"
	"github.com/spa-skyson/pi-rate/internal/pirpc"
	"github.com/spa-skyson/pi-rate/internal/provider"
	"github.com/spa-skyson/pi-rate/internal/ratelimit"
	"github.com/spa-skyson/pi-rate/internal/retry"
	pisession "github.com/spa-skyson/pi-rate/internal/session"
	"github.com/spa-skyson/pi-rate/internal/subagent"
	"github.com/spa-skyson/pi-rate/internal/tools"
	"github.com/spa-skyson/pi-rate/internal/tui"

	"github.com/spf13/cobra"
)

var (
	flagModel   string
	flagMode    string
	flagSession string
	flagSocket  string
	flagURL     string
	flagHeaders []string

	// flagJSONDeltas selects how --mode json emits streamed assistant text:
	// "group" (default) coalesces it into one event per sentence, "full" emits
	// one event per model chunk.
	flagJSONDeltas string

	// flagSocketChanged records whether --socket was passed explicitly, so
	// the deprecated `--mode rpc --socket` spelling can be distinguished
	// from the default value. Set by runRoot.
	flagSocketChanged bool

	flagContinue     bool
	flagInsecure     bool
	flagCACert       string
	flagSmol         bool
	flagSlow         bool
	flagPlan         bool
	flagMemoryOff    bool
	flagWebSearch    bool
	flagLSP          string
	flagTools        string
	flagTemperature  float64
	flagThinking     string
	flagSteps        int
	flagSystem       string
	flagPprof        string
	flagPprofPort    string
	flagMetrics      bool
	flagMetricsPort  string
	flagCPUProfile   string
	flagTraceHTTP    bool
	flagA2AAddr      string
	flagA2AReadyAddr string

	// lastSessionFileOverride redirects lastSessionFile() in tests. The path
	// itself is resolved lazily (see lastSessionFile) so it honors PIRATE_HOME /
	// PI_GO_HOME set after package init — the legacy-home migration runs inside
	// Execute, after init.
	lastSessionFileOverride string
)

// lastSessionFile returns the path of the last-session metadata file under the
// Pi-rate home. Resolved on call, not at package init, so env overrides set
// after init are honored.
func lastSessionFile() string {
	if lastSessionFileOverride != "" {
		return lastSessionFileOverride
	}
	return filepath.Join(config.PirateHome(), "last-session.json")
}

// lastSessionData is written to lastSessionFile on each print-mode start.
type lastSessionData struct {
	Timestamp time.Time `json:"timestamp"`
	SessionID string    `json:"session_id"`
	WorkDir   string    `json:"work_dir"`
	Model     string    `json:"model"`
}

// Version and BuildTag are set at build time via -ldflags.
var (
	Version  = "dev"
	BuildTag = ""
)

func versionString() string {
	if BuildTag == "" {
		return Version
	}
	return Version + "+" + BuildTag
}

func newRootCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "pirate [prompt]",
		Short: "Pi-rate coding agent",
		Long: `A Go coding agent with multi-provider LLM support, tool calling, and interactive TUI.

Run with no prompt for the interactive TUI; pass a prompt to answer once and exit.

The provider is inferred from the model name, so --model is usually the only
routing you need:

  claude-*                 Anthropic       ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN)
  gpt-*                    OpenAI          OPENAI_API_KEY
  gemini-*                 Google Gemini   GEMINI_API_KEY (or GOOGLE_API_KEY)
  mistral-*, magistral-*   Mistral         MISTRAL_API_KEY
  grok-*                   xAI             XAI_API_KEY
  openrouter/<model>       OpenRouter      OPENROUTER_API_KEY
  agentgateway/<model>     agentgateway    none; http://localhost:4000
  ollama/<model>           Ollama, local   none; http://localhost:11434
  <model>:cloud            Ollama Cloud    OLLAMA_API_KEY; https://api.ollama.com
                                           without a key: the local daemon
  azure/<deployment>       Azure OpenAI    AZUREOPENAI_API_KEY
  opencode/<model>         OpenCode        OPENCODE_API_KEY

A name with no recognized prefix is rejected rather than guessed at — reach for
the ollama/ prefix or the :cloud suffix to name an Ollama model explicitly.

Set a default in ~/.pirate/config.json so --model is only needed to deviate;
--smol, --slow and --plan switch between the roles configured there.`,
		Example: `  # Anthropic
  pirate --model claude-sonnet-5 "explain what this repo does"

  # OpenAI
  pirate --model gpt-5.2 "add a table-driven test for the parser"

  # Google Gemini
  pirate --model gemini-3.5-pro "review the diff on this branch"

  # Mistral
  pirate --model mistral-large-latest "summarize the changelog"

  # xAI
  pirate --model grok-4.6 "trace where this request handler blocks"

  # OpenRouter — any model in the OpenRouter catalog, vendor-prefixed ID
  pirate --model openrouter/google/gemini-3.7-flash "compare these two APIs"

  # Ollama against a local daemon — no API key needed
  pirate --model ollama/gemma4:e4b "rename this symbol everywhere"

  # Ollama Cloud — the :cloud tag routes to api.ollama.com with OLLAMA_API_KEY
  # set; without one it falls back to the local daemon, which serves cloud
  # models on your "ollama signin" identity. The ollama/ prefix forces local.
  pirate --model minimax-m3:cloud "port this module to generics"

  # Azure OpenAI — the deployment name follows azure/
  pirate --model azure/my-gpt5-deployment "draft release notes"

  # OpenCode
  pirate --model opencode/claude-sonnet-5 "find the goroutine leak"

  # agentgateway — a local OpenAI-compatible gateway, no API key needed
  pirate --model agentgateway/deepseek-v4-flash:0731-cloud "draft release notes"

  # Any OpenAI-compatible gateway, with an extra header and a corporate CA
  pirate --url https://llm.corp.internal/v1 --model gpt-5.2 \
     --header X-Team=platform --ca-cert /etc/ssl/corp.pem "run the tests"

  # One-shot answer instead of the TUI, and resuming a session
  pirate --mode print "what changed in the last commit?"
  pirate --continue
  pirate --session 01JQ8Z... "carry on where we left off"

  # Diagnosing a provider
  pirate ping                                 # DNS/TCP/TLS/HTTP trace, curl -v style
  pirate ping --model minimax-m3:cloud        # check one model end to end
  pirate --trace-http "why was that rejected?"  # full request/response in the session log`,
		Version: versionString(),
		Args:    cobra.ArbitraryArgs,
		// Start pprof here rather than in runRoot: subcommands (`pirate memory mine`,
		// `pirate audit`, ...) have their own RunE and never reach runRoot, so
		// profiling them was impossible. PersistentPreRun runs for the root and
		// every subcommand alike.
		PersistentPreRun: func(*cobra.Command, []string) {
			startPprofServer()
			startMetricsServer()
			startCPUProfile()
		},
		PersistentPostRun: func(*cobra.Command, []string) {
			stopCPUProfile()
		},
		RunE: runRoot,
	}

	cmd.Flags().StringVar(&flagModel, "model", "", "LLM model to use (e.g. claude-sonnet-5, gpt-5.2, gemini-3.5-pro, ollama/gemma4:e4b, minimax-m3:cloud)")
	cmd.Flags().StringVar(&flagMode, "mode", "", "Output mode: interactive, print, json, socket, rpc")
	// Grouping is the default because the ungrouped stream is one event per SSE
	// chunk — a few characters each — which buries the tool calls and results a
	// JSON-mode consumer is usually after.
	cmd.Flags().StringVar(&flagJSONDeltas, "json-deltas", "group",
		"JSON mode streamed-text granularity: group (one event per sentence) or full (one event per model chunk)")
	cmd.Flags().StringVar(&flagSocket, "socket", "/tmp/pi-go.sock", "Unix socket path for socket mode")
	// pi-acp unconditionally spawns `pi --mode rpc --no-themes`. pi-go has no
	// themes to disable, but rejecting the flag kills the child on spawn and
	// the adapter reports only "stream was destroyed", so accept and ignore.
	cmd.Flags().Bool("no-themes", false, "Accepted for pi CLI compatibility; ignored")
	_ = cmd.Flags().MarkHidden("no-themes")
	cmd.Flags().StringVar(&flagSession, "session", "", "Session ID to resume")
	cmd.Flags().StringVar(&flagURL, "url", "", "Alternative base URL for the LLM API endpoint")
	cmd.Flags().BoolVar(&flagContinue, "continue", false, "Continue last session")
	cmd.Flags().BoolVar(&flagSmol, "smol", false, "Use the smol role (fast/cheap model)")
	cmd.Flags().BoolVar(&flagSlow, "slow", false, "Use the slow role (powerful model)")
	cmd.Flags().BoolVar(&flagPlan, "plan", false, "Use the plan role (planning model)")
	cmd.Flags().StringVar(&flagSystem, "system", "", "System instruction (overrides default)")
	cmd.Flags().StringArrayVar(&flagHeaders, "header", nil, "Extra HTTP header for LLM requests (key=value, repeatable)")
	// Allow bare --header so a following flag is not consumed as a header value.
	if f := cmd.Flags().Lookup("header"); f != nil {
		f.NoOptDefVal = ""
	}
	cmd.Flags().BoolVar(&flagInsecure, "insecure", false, "Skip TLS certificate verification for LLM API calls")
	cmd.Flags().StringVar(&flagCACert, "ca-cert", "", "PEM bundle to trust for LLM API calls, in addition to the system roots")
	cmd.Flags().BoolVar(&flagMemoryOff, "memory-off", false, "Disable the persistent memory system for this session")
	cmd.Flags().BoolVar(&flagWebSearch, "web-search-enabled", false,
		"Register the web_search tool (needs a running Ollama daemon, or OLLAMA_API_KEY). Off by default; PI_WEB_SEARCH=1 does the same, and propagates to subagents")
	cmd.Flags().StringVar(&flagLSP, "lsp", "min", "Language-server tools: off, min (symbols+diagnostics), or full (all seven)")
	cmd.Flags().StringVar(&flagTools, "tools", "", "Comma-separated tool names to keep (e.g. read,bash); MCP servers match by server name; default keeps all")
	cmd.Flags().Float64Var(&flagTemperature, "temperature", 0, "LLM sampling temperature (0 keeps the provider default)")
	cmd.Flags().StringVar(&flagThinking, "thinking", "", "Reasoning effort: none, low, medium, high, or max (overrides config thinking level)")
	cmd.Flags().IntVar(&flagSteps, "steps", 0, "Max tool-call iterations per run (0 = no limit)")
	// Persistent, not local: `pirate memory mine . --pprof true` and every other
	// subcommand must accept these too. As local flags they were rejected with
	// "unknown flag: --pprof" the moment a subcommand was used.
	cmd.PersistentFlags().StringVar(&flagPprof, "pprof", "", "Enable pprof profiling (serves /debug/pprof; any non-empty value enables it)")
	cmd.PersistentFlags().StringVar(&flagPprofPort, "pprof-port", "6060", "Port for the pprof HTTP server")
	// A separate flag and server from --pprof on purpose: pprof is a profiling
	// tool a user turns on for one debugging session, while rate-limit metrics
	// are an ops signal someone may want scraped continuously. Tying it to
	// --pprof would mean either enabling CPU/heap profiling overhead just to
	// see quota headroom, or never exposing the quota view without it; a
	// dedicated mux also keeps /debug/pprof off this port when only metrics
	// were asked for. 9464 is OpenTelemetry's own default Prometheus exporter
	// port, so a scraper config aimed at "the usual place" finds it.
	cmd.PersistentFlags().BoolVar(&flagMetrics, "metrics", false, "Serve rate-limit usage metrics in Prometheus text format at /metrics (see --metrics-port)")
	cmd.PersistentFlags().StringVar(&flagMetricsPort, "metrics-port", "9464", "Port for the rate-limit metrics HTTP server")
	// --cpuprofile writes a runtime CPU profile to the given path for the whole
	// process lifetime. This is the profile PGO consumes: `go build` reads a CPU
	// pprof profile (default.pgo in the main package dir, or -pgo=<path>) to
	// guide inlining and layout. Collect it from a representative workload —
	// the eval-tools suite (`make record-pgo`) — not from a microbenchmark.
	cmd.PersistentFlags().StringVar(&flagCPUProfile, "cpuprofile", "", "Write a CPU profile to this path for the process lifetime (PGO input)")
	// Persistent for the same reason as --pprof: `pirate ping --trace-http` and the
	// other subcommands that reach a provider all need it.
	cmd.PersistentFlags().BoolVar(&flagTraceHTTP, "trace-http", false,
		"Log full LLM request/response headers and bodies to the session log and OTel spans (credentials masked; prompts are not)")

	// Append the resolved role table to `pi --help`. A help func set on the
	// root is inherited by every subcommand, so this reproduces the default
	// output and only adds the footer when help was asked for the root itself
	// — `pirate audit --help` has no use for it.
	defaultHelp := cmd.HelpFunc()
	cmd.SetHelpFunc(func(c *cobra.Command, args []string) {
		defaultHelp(c, args)
		if c == cmd {
			writeRoleSummary(c.OutOrStdout())
		}
	})

	cmd.AddCommand(newPingCmd())
	cmd.AddCommand(newAuditCmd())
	cmd.AddCommand(newServeCmd())
	cmd.AddCommand(newMemoryCmd())
	cmd.AddCommand(newModelCmd())
	cmd.AddCommand(newLoginCmd())
	cmd.AddCommand(newSetupCmd())
	cmd.AddCommand(newACPServerCmd())
	cmd.AddCommand(newA2AServerCmd())
	cmd.AddCommand(newUpgradeCmd())
	cmd.AddCommand(newSessionStatsCmd())
	cmd.AddCommand(newVerifyCmd())
	cmd.AddCommand(newPluginCmd())

	return cmd
}

type rootRuntime struct {
	cfg          config.Config
	llm          adkmodel.LLM
	info         provider.Info
	tokenTracker *guardrail.Tracker
	activeRole   string
	mode         string
	prompt       string
	cwd          string
	sandboxRoot  string
	worktreeDir  string
	// headerSessionID is the session ID already baked into the LLM client's
	// ${SESSION_ID} headers (empty when no header needed one). A fresh
	// session must be created under it so the header names the conversation
	// the logs do.
	headerSessionID string
}

func resolveActiveRole() string {
	activeRole := "default"
	switch {
	case flagSmol:
		activeRole = "smol"
	case flagSlow:
		activeRole = "slow"
	case flagPlan:
		activeRole = "plan"
	}
	return activeRole
}

func resolveMode() string {
	if flagMode != "" {
		return flagMode
	}
	return detectMode()
}

func loadRootConfig() (config.Config, error) {
	cfg, err := config.Load()
	if err != nil {
		return config.Config{}, fmt.Errorf("loading config: %w", err)
	}
	if flagModel != "" {
		cfg.Roles["default"] = config.RoleConfig{Model: flagModel}
	}
	return cfg, nil
}

// resolveRuntimeModel turns the role's model and provider names into a
// validated provider.Info plus the base URL actually used to reach it. An
// explicit --url wins; otherwise the config's per-provider base URL is
// consulted, first under the role's provider name and then under the provider
// the model itself resolved to.
func resolveRuntimeModel(cfg config.Config, modelName, providerName string) (provider.Info, string, error) {
	return resolveRuntimeModelForRole(cfg, modelName, providerName, "")
}

// namedModelInfo resolves a model that belongs to a user-declared provider —
// an entry in config.json's "providers" section. Either the model name itself
// carries the provider prefix ("corp-claude/claude-opus-5" — the spelling a
// /model switch persists) or the role names such a provider for a bare model.
// A resumed session is neither: its meta records the bare llm.Name() plus the
// provider separately, so resolveRuntimeModelForRole re-runs this with the
// recorded provider name. ok is false when neither applies and the built-in
// resolution should run.
//
// The declared provider's endpoint is used unless --url overrides it, and the
// key comes from the provider entry (literal or env). internal/config and
// internal/provider must not import each other, so the type→protocol mapping
// lives on config.ProviderConfig and the Info is assembled here, at the one
// seam that touches both packages.
func namedModelInfo(cfg config.Config, modelName, providerName string) (info provider.Info, apiKey, baseURL string, ok bool) {
	name, rest, prefixed := cfg.NamedProviderPrefix(modelName)
	if !prefixed {
		// A role naming a declared provider only claims a bare model: an
		// explicit built-in prefix on the name is the stronger statement,
		// same rule the ProviderFromPrefix guard applies to role providers.
		if _, _, builtin := provider.ProviderFromPrefix(modelName); builtin {
			return provider.Info{}, "", "", false
		}
		if _, declared := cfg.Providers[providerName]; declared {
			name, rest = providerName, modelName
		} else {
			return provider.Info{}, "", "", false
		}
	}
	pc := cfg.Providers[name]
	baseURL = flagURL
	if baseURL == "" {
		baseURL = pc.BaseURL
	}
	info = provider.Info{
		Provider: name,
		Model:    rest,
		Custom:   true,
		Protocol: pc.Protocol(),
		BaseURL:  baseURL,
	}
	return info, cfg.ResolveAPIKeys()[name], baseURL, true
}

func resolveRuntimeModelForRole(cfg config.Config, modelName, providerName, activeRole string) (provider.Info, string, error) {
	// A model served by a declared provider routes by its name alone — before
	// the built-in prefix machinery and before the role's provider can
	// overwrite it. A resumed session's model persists the prefixed spelling,
	// so the same detect covers resume.
	if info, _, baseURL, ok := namedModelInfo(cfg, modelName, providerName); ok {
		if err := provider.ValidateModel(info); err != nil {
			return provider.Info{}, "", fmt.Errorf("model validation: %w", err)
		}
		return info, baseURL, nil
	}
	baseURL := flagURL
	resumedProvider := ""
	if baseURL == "" && flagSession != "" && flagModel == "" && activeRole == "default" {
		if dir, err := sessionsDir(); err == nil {
			if rp, resumedURL, ok := pisession.SessionBackend(dir, flagSession); ok {
				if rp != "" {
					providerName = rp
					resumedProvider = rp
				}
				baseURL = resumedURL
			}
		}
	}
	// A session backed by a user-declared provider persists the bare model
	// name (meta.Model is llm.Name(), which carries no "provider/" prefix),
	// so the namedModelInfo call above missed it: the name is bare and
	// providerName is still the config's default, not the recorded one.
	// Re-run the named resolution against the provider the session recorded —
	// without it the fallback below built an Info with no Protocol and
	// NewLLM failed with "unsupported provider: zai-coding-plan". A recorded
	// built-in provider (ollama, agentgateway, openai) is not declared in
	// config, misses here, and keeps the fallback below.
	if resumedProvider != "" {
		if info, _, declaredURL, ok := namedModelInfo(cfg, modelName, resumedProvider); ok {
			if err := provider.ValidateModel(info); err != nil {
				return provider.Info{}, "", fmt.Errorf("model validation: %w", err)
			}
			// The endpoint recorded with the session is what actually served
			// it; the provider's declared URL only fills a blank.
			if baseURL == "" {
				baseURL = declaredURL
			}
			info.BaseURL = baseURL
			return info, baseURL, nil
		}
	}
	// An explicit provider prefix on the model name wins over the role's
	// provider, which is only a default for a bare name. Not applied to a
	// resumed session: there the recorded provider is the authority for which
	// backend actually served the model.
	if resumedProvider == "" {
		if _, _, prefixed := provider.ProviderFromPrefix(modelName); prefixed {
			providerName = ""
		}
	}
	if baseURL == "" && providerName != "" {
		baseURLs := cfg.ResolveBaseURLs()
		baseURL = baseURLs[providerName]
	}
	info, err := provider.ResolveWithBaseURL(modelName, baseURL)
	if err != nil {
		// A resumed session's model may be a virtual name that only its
		// recorded provider understands — e.g. an agentgateway virtual model
		// like "ollama-deepseek", whose dash spelling carries no provider
		// prefix and so cannot be resolved from the name alone. The provider
		// recorded in the session metadata is the authority for which backend
		// served it, so fall back to it rather than failing the resume.
		if resumedProvider != "" {
			info = provider.Info{Provider: resumedProvider, Model: modelName}
		} else {
			return provider.Info{}, "", fmt.Errorf("resolving model: %w", err)
		}
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
		return provider.Info{}, "", fmt.Errorf("model validation: %w", err)
	}
	return info, baseURL, nil
}

// requireRuntimeAPIKey rejects a provider that needs a key when none is
// available. A custom base URL, or a provider that authenticates some other
// way, is exempt.
func requireRuntimeAPIKey(info provider.Info, apiKey, baseURL string) error {
	if apiKey == "" && baseURL == "" && info.Provider != "gemini" && info.Provider != "ollama" && info.Provider != "azure" && info.Provider != "agentgateway" && !info.Ollama {
		envVar := providerEnvVar(info.Provider)
		return fmt.Errorf("no API key found for provider %q (set %s)", info.Provider, envVar)
	}
	return nil
}

// applyRuntimeOllamaEndpoint picks the Ollama daemon for the model, records it
// on info, and health-checks a local daemon before anything depends on it. It
// returns the base URL to use; non-Ollama models pass through untouched.
func applyRuntimeOllamaEndpoint(info *provider.Info, apiKey, baseURL string) (string, error) {
	if info.Ollama {
		// The model's tag decides the daemon, not whether OLLAMA_API_KEY
		// happens to be exported: a key set for some :cloud model used to send
		// locally pulled names like qwen3.8:27b-mlx to api.ollama.com, which
		// answers 404 for a model only this machine has.
		baseURL = provider.ResolveOllamaEndpoint(provider.OllamaRouting{
			Model:      info.Model,
			BaseURL:    baseURL,
			APIKey:     apiKey,
			ForceLocal: info.LocalOllama,
		})
		// Record the endpoint actually chosen so session metadata names the
		// backend instead of leaving the model name to be interpreted.
		info.BaseURL = baseURL
	}
	if info.Ollama && apiKey == "" && !provider.IsOllamaCloudEndpoint(baseURL) {
		if err := provider.CheckOllama(baseURL); err != nil {
			return "", fmt.Errorf("ollama health check: %w", err)
		}
	}
	return baseURL, nil
}

func buildRootRuntime(ctx context.Context, args []string) (rootRuntime, error) {
	cfg, err := loadRootConfig()
	if err != nil {
		return rootRuntime{}, err
	}

	// Resolve which session we are resuming before the model is picked: the
	// model restore below reads that session's metadata, and --continue has to
	// become a concrete ID for the lookup to happen at all.
	if err := resolveResumeSession(); err != nil {
		return rootRuntime{}, err
	}

	activeRole := resolveActiveRole()
	applyResumedModel(&cfg, activeRole)

	modelName, providerName, advisorModel, advisorMaxUses, advisorCaching, err := cfg.ResolveRole(activeRole)
	if err != nil {
		return rootRuntime{}, fmt.Errorf("resolving model role: %w", err)
	}

	mode := resolveMode()
	info, baseURL, err := resolveRuntimeModelForRole(cfg, modelName, providerName, activeRole)
	if err != nil {
		return rootRuntime{}, err
	}
	info.BaseURL = baseURL

	// ResolveAPIKeys covers the built-in env vars and each declared provider's
	// key, so a named provider reaches its endpoint with its own credential.
	keys := cfg.ResolveAPIKeys()
	apiKey := keys[info.Provider]
	if err := requireRuntimeAPIKey(info, apiKey, baseURL); err != nil {
		return rootRuntime{}, err
	}

	baseURL, err = applyRuntimeOllamaEndpoint(&info, apiKey, baseURL)
	if err != nil {
		return rootRuntime{}, err
	}

	// A ${SESSION_ID} header needs the session ID before the LLM client is
	// built — headers freeze into it. A resumed session already has its final
	// ID (resolveResumeSession resolved --continue into flagSession); a fresh
	// one gets a pre-generated ID that the session is then created under, so
	// the header and the logs name the same conversation.
	headerSessionID := flagSession
	if headerSessionID == "" && headersNeedSessionID(cfg, info.Provider) {
		headerSessionID = pisession.GenerateSessionID()
	}
	// The ID is no longer only for ${SESSION_ID} headers: the session-scoped
	// tools (todo_write/todo_read) register through it in every mode, and
	// resolveSessionID below creates the session under this same ID. Without
	// this, a print/json run with no such headers would silently lose the
	// todo tools.
	if headerSessionID == "" {
		headerSessionID = pisession.GenerateSessionID()
	}

	llmOpts := &provider.LLMOptions{
		ExtraHeaders:   providerExtraHeaders(cfg, info.Provider, headerSessionID, flagHeaders),
		AdvisorModel:   advisorModel,
		AdvisorMaxUses: advisorMaxUses,
		AdvisorCaching: advisorCaching,
		Temperature:    temperatureFlagOpt(),
	}
	applyTransportOptions(llmOpts, cfg, info)
	llm, err := provider.NewLLM(ctx, info, apiKey, baseURL, effectiveThinkingLevel(cfg), llmOpts)
	if err != nil {
		return rootRuntime{}, fmt.Errorf("creating LLM provider: %w", err)
	}

	tokenTracker := guardrail.New(cfg.MaxDailyTokens)
	tokenTracker.SetContextWindowSize(ctxwindow.Resolve(ctx, cfg, info, baseURL))
	llm = guardrail.WrapModel(llm, tokenTracker)

	cwd, err := os.Getwd()
	if err != nil {
		return rootRuntime{}, fmt.Errorf("getting working directory: %w", err)
	}
	sandboxRoot := os.Getenv("PI_SANDBOX_ROOT")
	if sandboxRoot == "" {
		sandboxRoot = cwd
	}
	worktreeDir := os.Getenv("PI_WORKTREE_ROOT")

	checkForRapidRestartAndWarn(cwd)
	_ = writeLastSession(cwd, info.Provider, llm.Name())

	return rootRuntime{
		cfg:             cfg,
		llm:             llm,
		info:            info,
		tokenTracker:    tokenTracker,
		activeRole:      activeRole,
		mode:            mode,
		prompt:          strings.Join(args, " "),
		cwd:             cwd,
		sandboxRoot:     sandboxRoot,
		worktreeDir:     worktreeDir,
		headerSessionID: headerSessionID,
	}, nil
}

// pprofOnce guards the pprof listener. PersistentPreRun fires once per command,
// but runRoot may also be reached directly in tests; starting twice would fail
// with "address already in use".
var pprofOnce sync.Once

// cpuProfileOnce guards the CPU profile writer. Like pprofOnce it exists so a
// command reached both through PersistentPreRun and directly in tests does not
// start two writers on the same file.
var (
	cpuProfileOnce sync.Once
	cpuProfileMu   sync.Mutex
	cpuProfileFile *os.File
)

// startCPUProfile begins writing a runtime CPU profile to flagCPUProfile when
// set. The profile is the input PGO consumes, so it must cover a representative
// workload (see `make record-pgo`). It is stopped and closed by stopCPUProfile.
func startCPUProfile() {
	if flagCPUProfile == "" {
		return
	}
	cpuProfileOnce.Do(func() {
		f, err := os.Create(flagCPUProfile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "cpuprofile: create %s: %v\n", flagCPUProfile, err)
			return
		}
		if err := pprof.StartCPUProfile(f); err != nil {
			fmt.Fprintf(os.Stderr, "cpuprofile: start: %v\n", err)
			_ = f.Close()
			return
		}
		cpuProfileMu.Lock()
		cpuProfileFile = f
		cpuProfileMu.Unlock()
		fmt.Fprintf(os.Stderr, "cpuprofile: writing to %s\n", flagCPUProfile)
	})
}

// stopCPUProfile flushes and closes the CPU profile started by startCPUProfile.
// It is safe to call when no profile was started or multiple times.
func stopCPUProfile() {
	cpuProfileMu.Lock()
	defer cpuProfileMu.Unlock()
	if cpuProfileFile != nil {
		pprof.StopCPUProfile()
		_ = cpuProfileFile.Close()
		cpuProfileFile = nil
	}
}

// startPprofServer serves net/http/pprof on --pprof-port when --pprof is set to
// any non-empty value. Profiles are then collected over HTTP
// (http://localhost:<port>/debug/pprof), so no profile-specific setup is needed
// here — the value is only echoed back so the user can see what they asked for.
//
// Diagnostics go through slog, not fmt: this starts from PersistentPreRun and
// the interactive TUI may already own the terminal by the time the listener
// fails, and a raw stdout or stderr write would render as garbage over the
// alternate screen (see AGENTS.md).
func startPprofServer() {
	if flagPprof == "" {
		return
	}
	pprofOnce.Do(func() {
		addr := ":" + flagPprofPort
		go func() {
			slog.Info("pprof server listening",
				"addr", addr,
				"profile", flagPprof,
				"collect_with", "go tool pprof http://localhost:"+flagPprofPort+"/debug/pprof/heap")
			if err := http.ListenAndServe(addr, nil); err != nil {
				slog.Error("pprof server error", "error", err)
			}
		}()
	})
}

// metricsOnce guards the metrics listener the same way pprofOnce guards the
// pprof one: PersistentPreRun fires once per command, including subcommands.
var metricsOnce sync.Once

// startMetricsServer serves internal/ratelimit's Prometheus-format /metrics
// endpoint on --metrics-port when --metrics is set.
//
// It runs on its own ServeMux rather than sharing pprof's DefaultServeMux
// (or listening on the same address): the two flags are independent, and a
// user who asked only for --metrics should not also get /debug/pprof for
// free, or vice versa. Diagnostics go through slog rather than fmt/stdout —
// this can start while the interactive TUI already owns the terminal, and a
// raw stdout write from here would corrupt its display (see AGENTS.md).
func startMetricsServer() {
	if !flagMetrics {
		return
	}
	metricsOnce.Do(func() {
		addr := ":" + flagMetricsPort
		mux := http.NewServeMux()
		mux.Handle("/metrics", ratelimit.MetricsHandler())
		go func() {
			slog.Info("rate-limit metrics server listening", "addr", addr, "path", "/metrics")
			if err := http.ListenAndServe(addr, mux); err != nil {
				slog.Error("rate-limit metrics server error", "error", err)
			}
		}()
	})
}

func runRoot(cmd *cobra.Command, args []string) error {
	// --socket has a non-empty default, so its value alone cannot tell us
	// whether the caller asked for it. Record explicit use here so
	// dispatchMode can honor the pre-rename `--mode rpc --socket` spelling.
	flagSocketChanged = cmd.Flags().Changed("socket")
	// Same for --temperature: its zero value means "provider default", so
	// only Changed distinguishes a passed 0 from an absent flag. Captured
	// here because buildRootRuntime below reads the result.
	flagTemperatureChanged = cmd.Flags().Changed("temperature")

	// Load API keys from ~/.pirate/.env (set by /login command).
	loadDotEnv()

	// Normally started by the root's PersistentPreRun; harmless if already up.
	startPprofServer()
	startCPUProfile()
	// Flush the CPU profile on the way out. runRoot is the only path that
	// reaches the agent loop, so deferring here (rather than in main) keeps the
	// profile covering exactly the work the process did.
	defer stopCPUProfile()

	runtime, err := buildRootRuntime(cmd.Context(), args)
	if err != nil {
		return err
	}

	if runtime.mode == "interactive" {
		// The update check runs inside runInteractive so its notice is
		// delivered after the TUI has claimed the notice sink. Started out
		// here it would race the sink installation and could still land on
		// os.Stderr, in the middle of the painted frame.
		return runInteractive(
			cmd.Context(),
			runtime.cfg,
			runtime.llm,
			runtime.info,
			runtime.tokenTracker,
			runtime.activeRole,
			runtime.cwd,
			runtime.sandboxRoot,
			runtime.worktreeDir,
			runtime.headerSessionID,
		)
	}

	go checkForUpdate(cmd.Context(), Version, "`pirate upgrade`")
	config.NotifyReroutedLLMS(runtime.cfg)

	return runNonInteractive(
		cmd.Context(),
		cmd,
		runtime.cfg,
		runtime.llm,
		runtime.info,
		runtime.tokenTracker,
		runtime.cwd,
		runtime.sandboxRoot,
		runtime.worktreeDir,
		runtime.mode,
		runtime.prompt,
		runtime.headerSessionID,
	)
}

type nonInteractiveRuntime struct {
	sandbox      *tools.Sandbox
	coreTools    []adktool.Tool
	orch         *subagent.Orchestrator
	agentEventCh chan tui.AgentSubEvent
	bashSup      *tools.BashSupervisor
}

func initNonInteractiveRuntime(ctx context.Context, cfg *config.Config, cwd, sandboxRoot, worktreeDir, headerSessionID string) (*nonInteractiveRuntime, error) {
	sandbox, err := tools.NewSandbox(sandboxRoot, worktreeDir)
	if err != nil {
		return nil, fmt.Errorf("creating sandbox: %w", err)
	}

	if aErr := sandbox.AddExtraDir(config.PirateHome()); aErr != nil {
		fmt.Fprintf(os.Stderr, "pirate: warning: could not add %s to sandbox: %v\n", config.PirateHome(), aErr)
	}

	bashSup := tools.NewBashSupervisor()
	coreTools, err := tools.CoreTools(sandbox, coreToolOptions(bashSup, headerSessionID, nil, nil)...)
	if err != nil {
		_ = sandbox.Close()
		return nil, fmt.Errorf("creating core tools: %w", err)
	}
	bashCtlTools, err := tools.BashControlTools(bashSup)
	if err != nil {
		_ = sandbox.Close()
		return nil, fmt.Errorf("creating bash control tools: %w", err)
	}
	coreTools = append(coreTools, bashCtlTools...)

	repoRoot := detectGitRoot(ctx, cwd)
	discovery, err := subagent.DiscoverAgents(cwd, subagent.ScopeBoth)
	if err != nil {
		fmt.Fprintf(os.Stderr, "pirate: warning: agent discovery failed: %v\n", err)
	}
	var agentConfigs []subagent.AgentConfig
	if discovery != nil {
		agentConfigs = discovery.All
	}
	orch := subagent.NewOrchestrator(cfg, repoRoot, agentConfigs)
	orch.SetProviderOptions(flagURL, flagInsecure, flagHeaders)

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
	agentTools, err := tools.SubagentTools(orch, agentEventCB)
	if err != nil {
		orch.Shutdown()
		_ = sandbox.Close()
		return nil, fmt.Errorf("creating agent tools: %w", err)
	}
	coreTools = append(coreTools, agentTools...)

	bashSup.SetSink(func(execID, kind, content string) {
		agentEventCB(tools.SubagentEvent{AgentID: execID, Kind: tui.BashEventKind(kind), Content: content})
	})

	return &nonInteractiveRuntime{
		sandbox:      sandbox,
		coreTools:    coreTools,
		orch:         orch,
		agentEventCh: agentEventCh,
		bashSup:      bashSup,
	}, nil
}

func (r *nonInteractiveRuntime) close() {
	if r == nil {
		return
	}
	// Backgrounded commands have no owner but this supervisor; leaving them
	// running past the run is a leaked process tree.
	if r.bashSup != nil {
		r.bashSup.KillAll()
	}
	if r.orch != nil {
		r.orch.Shutdown()
	}
	if r.sandbox != nil {
		_ = r.sandbox.Close()
	}
}

// runNonInteractive performs synchronous initialization and runs print/json/rpc modes.
func runNonInteractive(
	parentCtx context.Context,
	cmd *cobra.Command,
	cfg config.Config,
	llm adkmodel.LLM,
	info provider.Info,
	tokenTracker *guardrail.Tracker,
	cwd, sandboxRoot, worktreeDir, mode, prompt string,
	headerSessionID string,
) error {
	runtime, err := initNonInteractiveRuntime(parentCtx, &cfg, cwd, sandboxRoot, worktreeDir, headerSessionID)
	if err != nil {
		return err
	}
	defer runtime.close()

	coreTools := runtime.coreTools
	orch := runtime.orch

	// memSessionID is only known once the session is created below; both the
	// observation recorder and the end-of-session summary read it at call time,
	// so recording starts from that point rather than from wiring.
	var memSessionID string

	memStore, memWorker, closeMemory := setupMemory(parentCtx, cfg, orch, llm, &memSessionID, cwd, mode)
	defer closeMemory()

	coreTools = appendNonInteractiveMemoryTools(coreTools, memStore)

	palaceTools, palaceContext, closePalace := setupPalace(cfg, memWorker)
	defer closePalace()
	coreTools = append(coreTools, palaceTools...)

	instruction := buildNonInteractiveInstruction(palaceContext)

	hooks := convertHooks(cfg.Hooks)
	beforeCBs := extension.BuildBeforeToolCallbacks(hooks)

	// Permission rules gate every tool call before execution: the global
	// config rules, plus the spawning agent's own rules when this process IS
	// a subagent (handed over in PI_AGENT_PERMISSION by the orchestrator).
	permRules := globalPermissionRules(cfg)
	if envRules, err := permission.FromEnv(); err != nil {
		return fmt.Errorf("reading %s: %w", permission.EnvVar, err)
	} else if !envRules.Empty() {
		permRules = permission.Merge(permRules, envRules)
	}
	if !permRules.Empty() {
		beforeCBs = append(beforeCBs, agent.NewPermissionCallback(permRules))
	}
	afterCBs := extension.BuildAfterToolCallbacks(hooks)

	tracingBefore, tracingAfter := extension.BuildTracingCallbacks()
	beforeCBs = append(beforeCBs, tracingBefore...)
	afterCBs = append(afterCBs, tracingAfter...)
	llmBefore, llmAfter := extension.BuildLLMTracingCallbacks(info.Provider)
	llmBefore = append(llmBefore, extension.BuildReadImageCallback(runtime.sandbox, info.Provider))

	lspMgr := lsp.NewManager(nil)
	defer lspMgr.Shutdown()
	// Dedup runs BEFORE the compactor, so its equality test compares the bytes
	// the tool actually produced. Compaction is lossy: two different results can
	// truncate to the same head, and if dedup hashed that truncated form it would
	// report a changed file as "content is unchanged". The deduper's contract is
	// that the model is never served stale bytes, which only holds when the hash
	// covers the pre-compaction content.
	resultDeduper := tools.NewResultDeduper()
	afterCBs = append(afterCBs,
		lsp.BuildLSPAfterToolCallback(lspMgr),
		tools.BuildDedupCallback(resultDeduper),
		tools.BuildCompactorCallback(compactorConfigFrom(cfg), tools.NewCompactMetrics()))

	// The recorder reads memSessionID at call time, so recording starts from
	// the point the session is created below.
	if memWorker != nil {
		afterCBs = append(afterCBs, memoryObservationCallback(memWorker, cfg, cwd, &memSessionID))
	}

	// --steps caps the tool-call iterations (the subagent budget a parent
	// passes down). Appended before the fold so it rides the composed chain:
	// handed to ADK as a separate slice entry it would never run.
	if flagSteps > 0 {
		afterCBs = append(afterCBs, agent.NewStepLimitCallback(flagSteps))
	}

	// Fold the whole after-tool chain into the single callback ADK runs.
	// ADK's Flow.invokeAfterToolCallbacks returns at the first callback that
	// yields a non-nil result, and every callback above returns the result map —
	// so handing ADK the slice ran only the first entry and silently skipped
	// dedup, the compactor and memory recording. Composing preserves each
	// stage's effect while presenting ADK one callback to invoke.
	afterCBs = extension.ComposeAfterToolChain(afterCBs)

	// LSP tool declarations are billed on every request, and with no server
	// installed every call they enable fails — so the model pays tokens for
	// capability it cannot use. Gate on a server existing, then advertise only
	// as much surface as the mode asks for. The after-tool callback stays wired
	// either way; it is free when no server starts.
	coreTools, err = appendNonInteractiveLSPTools(coreTools, lspMgr)
	if err != nil {
		return err
	}

	allToolsets := buildToolsets(cfg)

	loadNonInteractiveSkills(mode, cfg, cwd)
	instruction += memoryInstructionContext(parentCtx, memStore, cfg, cwd)

	sessionsPath, sessionSvc, err := openSessionService()
	if err != nil {
		return err
	}

	// Gemini search grounding. Always on for the Gemini provider; kill
	// switch via PI_NO_GROUNDING=1 (propagates to subagent pirate processes via
	// FilterEnv's PI_ prefix allowlist).
	//
	// APPEND — never replace. See the matching note in interactive.go: replacing
	// coreTools here strips every real tool and every MCP toolset, leaving the
	// model with nothing to call. The built-in search and function declarations
	// coexist fine.
	if gTool, ok := agent.GeminiGroundingTool(info.Provider, info.Model); ok {
		coreTools = append(coreTools, gTool)
	}

	// --tools allow list. Applied here, after the last core-tool append
	// (memory, palace, LSP and grounding above), so the flat list the agent
	// sees is exactly what --tools allows; MCP toolsets were already filtered
	// by server name inside buildToolsets.
	coreTools = applyToolAllowlist(coreTools)

	// Session logger. Created before the agent so it can capture the agent's
	// non-fatal diagnostics (e.g. unresolved instruction placeholders) in the
	// session log instead of leaking them to stderr. SessionStart is recorded
	// below once the session ID is resolved.
	sessionLog, err := logger.New()
	if err != nil {
		fmt.Fprintf(os.Stderr, "pirate: warning: could not create session log: %v\n", err)
	}
	// --trace-http entries are dropped until this point, because the transport
	// is built well before the log file exists. In practice the only requests
	// that precede it are the ollama health check and model listing.
	httplog.SetSink(logger.HTTPSink(sessionLog))
	defer func() {
		httplog.SetSink(nil)
		_ = sessionLog.Close()
	}()

	ag, err := agent.New(agent.Config{
		Model:                llm,
		Tools:                coreTools,
		Toolsets:             allToolsets,
		Instruction:          instruction,
		SessionService:       sessionSvc,
		Version:              versionString(),
		BeforeToolCallbacks:  beforeCBs,
		AfterToolCallbacks:   afterCBs,
		BeforeModelCallbacks: llmBefore,
		AfterModelCallbacks:  llmAfter,
		Logger:               sessionLog,
	})
	if err != nil {
		return fmt.Errorf("creating agent: %w", err)
	}

	ctx, stop := signal.NotifyContext(parentCtx, os.Interrupt)
	defer stop()

	sessionID, err := resolveSessionID(ctx, ag, sessionSvc, headerSessionID)
	if err != nil {
		return err
	}

	// Two-stage auto-compaction: shed superseded tool results at the lower
	// threshold, summarize at the upper one. Installed as a pre-turn hook so it
	// only ever rewrites history between turns.
	if hook := autocompact.BuildHook(autocompact.Deps{
		SessionSvc:    sessionSvc,
		Tracker:       tokenTracker,
		Deduper:       resultDeduper,
		Cfg:           autocompact.ConfigFrom(cfg),
		Log:           sessionLog,
		SummarizerLLM: llm,
		Notify:        func(msg string) { fmt.Fprintf(os.Stderr, "pirate: %s\n", msg) },
	}); hook != nil {
		ag.SetPreTurnHook(hook)
	}

	// Capture ACP subagent events (claude, gemini) under the session dir.
	orch.SetACPLogPath(filepath.Join(sessionsPath, sessionID, "acp.jsonl"))

	armMemoryObservationSession(ctx, memStore, sessionID, cwd, &memSessionID)

	sessionLog.SessionStart(sessionID, llm.Name(), info.Provider, provider.BackendName(info, config.APIKeys()[info.Provider], info.BaseURL), info.BaseURL, mode)
	return dispatchMode(ctx, mode, prompt, ag, sessionID, sessionLog, llm.Name(), cfg, tokenTracker, info.Provider)
}

// appendNonInteractiveMemoryTools adds the memory tools when a store is
// configured. Memory is optional: a tool-construction failure is a warning and
// the run continues without them.
func appendNonInteractiveMemoryTools(coreTools []adktool.Tool, memStore memory.Store) []adktool.Tool {
	if memStore == nil {
		return coreTools
	}
	memTools, memErr := tools.MemoryTools(memStore)
	if memErr != nil {
		fmt.Fprintf(os.Stderr, "pirate: warning: memory tools disabled: %v\n", memErr)
		return coreTools
	}
	if memTools != nil {
		coreTools = append(coreTools, memTools...)
	}
	return coreTools
}

// buildNonInteractiveInstruction assembles the system instruction: the built-in
// one unless --system replaces it, with the palace memory context appended.
func buildNonInteractiveInstruction(palaceContext string) string {
	instruction := agent.LoadInstruction(agent.SystemInstruction)
	if flagSystem != "" {
		instruction = flagSystem
	}
	if palaceContext != "" {
		instruction += "\n\n## Palace Memory Context\n\n" + palaceContext
	}
	return instruction
}

// appendNonInteractiveLSPTools adds LSP tools only when a language server is
// actually installed.
//
// LSP tool declarations are billed on every request, and with no server
// installed every call they enable fails — so the model pays tokens for
// capability it cannot use. Gate on a server existing, then advertise only
// as much surface as the mode asks for. The after-tool callback stays wired
// either way; it is free when no server starts.
func appendNonInteractiveLSPTools(coreTools []adktool.Tool, lspMgr *lsp.Manager) ([]adktool.Tool, error) {
	if !lspMgr.AnyAvailable() {
		return coreTools, nil
	}
	lspTools, lspErr := tools.LSPToolsFor(lspMgr, resolveLSPMode())
	if lspErr != nil {
		return nil, fmt.Errorf("creating LSP tools: %w", lspErr)
	}
	return append(coreTools, lspTools...), nil
}

// openSessionService opens the on-disk session store, returning both the
// directory it lives in and the service over it.
func openSessionService() (string, *pisession.FileService, error) {
	sessionsPath, err := sessionsDir()
	if err != nil {
		return "", nil, err
	}
	sessionSvc, err := pisession.NewFileService(sessionsPath)
	if err != nil {
		return "", nil, fmt.Errorf("creating session service: %w", err)
	}
	return sessionsPath, sessionSvc, nil
}

// armMemoryObservationSession opens the memory session row and publishes the
// session ID the observation callback records under. Until memSessionID is set
// the callback is inert, so this is what arms it.
func armMemoryObservationSession(ctx context.Context, memStore memory.Store, sessionID, project string, memSessionID *string) {
	if memStore == nil {
		return
	}
	*memSessionID = sessionID
	_ = memStore.CreateSession(ctx, &memory.Session{
		SessionID: sessionID,
		Project:   project,
		StartedAt: time.Now(),
		Status:    "active",
	})
}

// loadNonInteractiveSkills loads the skill set and reports what it found.
// Skills are optional: a load failure is a warning, not a fatal error.
func loadNonInteractiveSkills(mode string, cfg config.Config, cwd string) {
	skills, err := extension.LoadSkills(extension.DefaultSkillDirsIn(cwd, cfg)...)
	if mode == "print" {
		fmt.Fprint(os.Stderr, formatPrintSkillLoad(len(skills), err))
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "pirate: warning: skills disabled: %v\n", err)
	}
}

// memoryInstructionContext generates the recalled-memory block to append to the
// system instruction, or an empty string when there is no memory to add.
func memoryInstructionContext(ctx context.Context, store memory.Store, cfg config.Config, project string) string {
	if store == nil {
		return ""
	}
	budget := config.MemoryDefaults().TokenBudget
	if cfg.Memory != nil && cfg.Memory.TokenBudget > 0 {
		budget = cfg.Memory.TokenBudget
	}

	memContext, err := memory.NewContextGenerator(store, budget).Generate(ctx, project)
	if err != nil {
		fmt.Fprintf(os.Stderr, "pirate: warning: memory context generation failed: %v\n", err)
		return ""
	}
	if memContext == "" {
		return ""
	}
	return "\n\n" + memContext
}

// sessionsDir is the directory FileService keeps one subdirectory per session
// in. Startup reaches for it before any session service exists, so it cannot
// be asked of the service itself. PI_SESSIONS_DIR overrides the default of
// $PIRATE_HOME/sessions — a server whose home is not durable storage points
// it at a directory that is.
func sessionsDir() (string, error) {
	if dir := strings.TrimSpace(os.Getenv("PI_SESSIONS_DIR")); dir != "" {
		return dir, nil
	}
	return filepath.Join(config.PirateHome(), "sessions"), nil
}

// resolveResumeSession turns --continue into an explicit --session value, so
// everything downstream — the model restore, the agent, the TUI's transcript
// restore — reads one resolved session ID instead of handling two ways of
// asking for the same thing.
func resolveResumeSession() error {
	if !flagContinue {
		return nil
	}
	dir, err := sessionsDir()
	if err != nil {
		return err
	}
	sessionSvc, err := pisession.NewFileService(dir)
	if err != nil {
		return fmt.Errorf("creating session service: %w", err)
	}
	lastID := sessionSvc.LastSessionID(agent.AppName, agent.DefaultUserID)
	if lastID == "" {
		return fmt.Errorf("no previous session found to continue")
	}
	flagSession = lastID
	return nil
}

// applyResumedModel restores the model a resumed session last ran under.
//
// Sessions record their model in meta.json, but startup used to resolve the
// model from config alone: `pi --session <id>` continued a conversation held
// with one model under whatever the default role happened to point at, with
// nothing on screen admitting the swap. Since the whole transcript is replayed
// to the new model, the switch is silent and total.
//
// Explicit intent still wins. --model names a model outright, and --smol /
// --slow / --plan pick a role for a reason, so each leaves the session's
// recorded model alone.
func applyResumedModel(cfg *config.Config, activeRole string) {
	if flagSession == "" || flagModel != "" || activeRole != "default" || cfg.Roles == nil {
		return
	}
	dir, err := sessionsDir()
	if err != nil {
		return
	}
	name := pisession.SessionModel(dir, flagSession)
	if name == "" {
		return
	}
	rc := cfg.Roles["default"]
	if rc.Model == name {
		return
	}
	rc.Model = name
	// The configured provider belongs to the configured model. Carrying it
	// over would route the session's model to the wrong API — an anthropic
	// role serving a gpt-* model — so let it be re-detected from the name.
	rc.Provider = ""
	cfg.Roles["default"] = rc
}

// resolveSessionID picks the session to run in: an explicit --session, the
// most recent one under --continue, or a freshly created session. headerID is
// the ID already baked into the LLM client's ${SESSION_ID} headers — a fresh
// session is created under it, so header and logs name the same conversation.
func resolveSessionID(ctx context.Context, ag *agent.Agent, sessionSvc *pisession.FileService, headerID string) (string, error) {
	// buildRootRuntime already resolved --continue into flagSession, but this
	// branch stays authoritative: --continue with nothing to continue is an
	// error, and must never fall through to opening a brand-new session.
	if flagContinue {
		lastID := sessionSvc.LastSessionID(agent.AppName, agent.DefaultUserID)
		if lastID == "" {
			return "", fmt.Errorf("no previous session found to continue")
		}
		fmt.Fprintf(os.Stderr, "pirate: continuing session %s\n", lastID)
		return lastID, nil
	}
	if flagSession != "" {
		return flagSession, nil
	}

	sessionID, _, err := ag.CreateSessionWithID(ctx, headerID)
	if err != nil {
		return "", fmt.Errorf("creating session: %w", err)
	}
	return sessionID, nil
}

// dispatchMode runs the agent in the requested non-interactive mode. The
// server modes serve requests instead of a prompt; the others need one.
//
// Two server modes exist and they are not interchangeable:
//
//   - "socket": pi-go's own JSON-RPC 2.0 over a Unix socket, for editor/IDE
//     integration. This was spelled "rpc" before the rename.
//   - "rpc": the stdio NDJSON protocol that `pi-acp` drives, wire-compatible
//     with upstream pi's `--mode rpc`.
func dispatchMode(ctx context.Context, mode, prompt string, ag *agent.Agent, sessionID string, sessionLog *logger.Logger, modelName string, cfg config.Config, tokenTracker *guardrail.Tracker, providerName string) error {
	// Pre-rename spelling: `--mode rpc --socket <path>` meant the Unix socket
	// server. Honor it with a warning rather than silently starting the
	// stdio server and leaving the caller's socket client hanging.
	if mode == "rpc" && flagSocketChanged {
		fmt.Fprintln(os.Stderr,
			"pirate: `--mode rpc --socket` is deprecated and will be removed; use `--mode socket`.")
		mode = "socket"
	}
	if mode == "socket" {
		return jsonrpc.NewServer(jsonrpc.Config{
			Agent:      ag,
			SocketPath: flagSocket,
		}).Run(ctx)
	}
	if mode == "rpc" {
		return pirpc.NewServer(pirpc.Config{
			Agent:     ag,
			SessionID: sessionID,
			In:        os.Stdin,
			Out:       os.Stdout,
			Log:       sessionLog,
			Model:     modelName,
			ModelSwitcher: func(switchCtx context.Context, name, providerHint string) (adkmodel.LLM, string, string, error) {
				// buildSwitchedLLM seeds the provider from the default role
				// and then overrides detection with it, so a stale pin would
				// route e.g. an OpenAI model through Ollama. Substitute the
				// provider the ACP client named; when it names none, clearing
				// the pin lets pi-go detect from the model name.
				switchCfg := cfg
				switchCfg.Roles = maps.Clone(cfg.Roles)
				if switchCfg.Roles == nil {
					switchCfg.Roles = map[string]config.RoleConfig{}
				}
				rc := switchCfg.Roles["default"]
				rc.Provider = providerHint
				switchCfg.Roles["default"] = rc
				return buildSwitchedLLM(switchCtx, switchCfg, tokenTracker, name, sessionID)
			},
		}).Run(ctx)
	}
	if prompt == "" {
		fmt.Fprintf(os.Stderr, "pirate: no prompt provided (model: %s, mode: %s)\n", modelName, mode)
		return nil
	}
	if mode == "json" {
		return runJSON(ctx, ag, sessionID, prompt, os.Stdin, sessionLog)
	}
	return runPrint(ctx, ag, sessionID, prompt, sessionLog, providerName)
}

// memoryEnabled reports whether the observation memory subsystem is on.
// It defaults to on: only an explicit false in config, or --memory-off,
// disables it.
func memoryEnabled(cfg config.Config) bool {
	return !flagMemoryOff && (cfg.Memory == nil || cfg.Memory.Enabled == nil || *cfg.Memory.Enabled)
}

// palaceIsEnabled reports whether the memory palace is on, with the same
// default-on semantics as memoryEnabled.
func palaceIsEnabled(cfg config.Config) bool {
	return !flagMemoryOff && (cfg.Palace == nil || cfg.Palace.Enabled == nil || *cfg.Palace.Enabled)
}

// coreToolOptions builds the CoreTools options every entry point shares, so a
// new opt-in gate is wired once rather than at each construction site
// (interactive, print/JSON, ACP and the eval inventory).
//
// Only the flag is read here. PI_WEB_SEARCH is read inside tools.CoreTools
// instead, because internal/acp/server builds the same tool set and cannot
// import this package — a gate that lived only here would leave ACP without the
// tool its session asked for.
//
// sessionID, when non-empty, registers todo_write/todo_read tools. todoNotifier
// is called after each successful todo_write. questionNotifier bridges the
// question tool to the interactive TUI dialog. Pass nil when not needed — a
// nil questionNotifier keeps the tool headless (immediate canceled).
func coreToolOptions(sup *tools.BashSupervisor, sessionID string, todoNotifier func(tools.TodoState), questionNotifier func(tools.QuestionRequest)) []tools.CoreOption {
	opts := []tools.CoreOption{tools.WithBashSupervisor(sup)}
	if flagWebSearch {
		opts = append(opts, tools.WithWebSearch())
	}
	if sessionID != "" {
		opts = append(opts, tools.WithSessionID(sessionID))
		if todoNotifier != nil {
			opts = append(opts, tools.WithTodoNotifier(todoNotifier))
		}
	}
	if questionNotifier != nil {
		opts = append(opts, tools.WithQuestionNotifier(questionNotifier))
	}
	return opts
}

// setupMemory opens the observation store and starts its background worker.
// Memory is best-effort: every failure downgrades to "no memory" with a
// warning, and the returned closer is always safe to defer.
//
// On close it drains the worker, then writes one session summary for sessionID
// and closes the store. The order is load-bearing: draining is what moves the
// session's queued tool calls into the store, so a summary read before it would
// describe a prefix of the session — which reads as complete and is the hardest
// kind of wrong to notice. The store must therefore outlive the summary.
//
// llm writes the summary. A nil model, or an empty sessionID, skips the summary
// and only drains: an embedder or a subagent may have no session to summarize,
// and a summary is worth less than a clean shutdown. project is the key the
// summary is filed under, matching the one observations were recorded with.
//
// sessionID is read through a pointer because the session is created after this
// call — the closer runs at exit, by which point it is set. mode picks the
// drain ceiling: one-shot modes (print/json) get the reduced budget, everything
// else the full one — see memoryDrainBudget.
func setupMemory(ctx context.Context, cfg config.Config, orch *subagent.Orchestrator, llm adkmodel.LLM, sessionID *string, project, mode string) (memory.Store, *memory.Worker, func()) {
	noop := func() {}
	if !memoryEnabled(cfg) {
		return nil, nil, noop
	}

	memCfg := deferredMemoryConfig(cfg)
	dbPath := memCfg.DBPath
	if dbPath == "" {
		dbPath = filepath.Join(config.PirateHome(), "memory", "claude-mem.db")
	}

	memDB, err := memory.OpenDB(dbPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "pirate: warning: memory system disabled: %v\n", err)
		return nil, nil, noop
	}

	compressorName, known := cfg.ResolveCompressor()
	if !known {
		slog.Warn("memory: unknown compressor in config, using default",
			"configured", cfg.Memory.Compressor, "using", compressorName)
	}

	store := memory.NewSQLiteStore(memDB)
	worker := memory.NewWorker(store, memory.NewCompressor(compressorName, orch), memCfg.MaxPending)
	worker.Start(ctx)

	summarizer := memory.NewSessionSummarizer(store, llm)

	return store, worker, func() {
		drainBudget := memoryDrainBudget(mode)
		shutdownCtx, cancel := context.WithTimeout(context.Background(), drainBudget)
		drainErr := worker.Shutdown(shutdownCtx)
		cancel()

		summarizeSessionAfterDrain(summarizeParams{
			store:       store,
			summarizer:  summarizer,
			sessionID:   derefString(sessionID),
			project:     project,
			drainBudget: drainBudget,
			log:         slog.Default(),
		}, drainErr, llm != nil)

		_ = store.Close()
	}
}

// summarizeParams bundles the inputs to summarizeSessionAfterDrain.
type summarizeParams struct {
	store      memory.Store
	summarizer *memory.SessionSummarizer
	sessionID  string
	project    string
	// drainBudget is the budget the drain actually ran with, reported when a
	// drain timeout skips the summary — the full and one-shot ceilings differ,
	// so the log names the one that applied.
	drainBudget time.Duration
	log         *slog.Logger
}

// derefString reads a string through a pointer, treating nil as empty. The
// session ID is unknown when the memory subsystem is wired and known by the
// time its closer runs, so the closer reads it through a pointer.
func derefString(p *string) string {
	if p == nil {
		return ""
	}
	return *p
}

// summarizeSessionAfterDrain writes the end-of-session summary, but only when
// the worker drained cleanly.
//
// A drain timeout means the worker may still be storing, so a summary taken now
// would describe a prefix of the session and read as complete. That is reported
// and skipped.
//
// A session that recorded nothing is the common shape for short one-shot runs:
// it is probed on the index before any summarizer machinery runs, and reported
// at Debug rather than as a failure. The model is never called over an empty
// session — the probe is the first guarantee, the summarizer's
// [memory.ErrNoObservations] the second.
func summarizeSessionAfterDrain(p summarizeParams, drainErr error, modelAvailable bool) {
	if drainErr != nil {
		p.log.Warn("memory: drain timed out; skipping session summary",
			"error", drainErr, "budget", p.drainBudget)
		return
	}
	if !modelAvailable || p.sessionID == "" {
		return
	}

	if p.store != nil {
		probeCtx, cancel := context.WithTimeout(context.Background(), hasObservationsProbeTimeout)
		has, err := p.store.HasObservations(probeCtx, p.sessionID)
		cancel()
		if err == nil && !has {
			p.log.Debug("memory: skipping session summary, no observations",
				"session", p.sessionID)
			return
		}
		// A probe error falls through to the summarizer, which reports its own
		// read failure — silently dropping the summary would be worse.
	}

	ctx, cancel := context.WithTimeout(context.Background(), sessionSummaryBudget())
	defer cancel()
	if err := p.summarizer.SummarizeSession(ctx, p.sessionID, p.project); err != nil {
		if errors.Is(err, memory.ErrNoObservations) {
			p.log.Debug("memory: skipping session summary, no observations",
				"session", p.sessionID)
			return
		}
		// Best-effort: a provider that did not answer inside the budget must
		// not fail the shutdown that follows.
		p.log.Warn("memory: session summary failed",
			"session", p.sessionID, "error", err)
		return
	}
	p.log.Info("memory: session summary written", "session", p.sessionID)
}

// resolveLSPMode turns the --lsp flag into a mode, warning once on a value it
// does not recognize rather than silently picking one. A subagent inherits the
// parent's choice through the same flag on its command line, which is how an
// agent that needs the wide surface gets it without every session paying for it.
func resolveLSPMode() tools.LSPMode {
	mode, ok := tools.ParseLSPMode(flagLSP)
	if !ok {
		fmt.Fprintf(os.Stderr, "pirate: warning: unknown --lsp value %q; using %q\n", flagLSP, mode)
	}
	return mode
}

// parseToolAllowlist splits the --tools value into a normalized allow list:
// comma-separated entries, trimmed and lowercased, empties dropped. Empty or
// absent input returns nil, which the filter treats as "keep everything".
func parseToolAllowlist(s string) []string {
	var out []string
	for _, t := range strings.Split(s, ",") {
		if t = strings.ToLower(strings.TrimSpace(t)); t != "" {
			out = append(out, t)
		}
	}
	return out
}

// applyToolAllowlist narrows a tool list to the --tools allow list and warns
// about names that matched nothing. With no --tools it is a no-op. Each run
// path calls it once, after the last core-tool append (memory, LSP, LLMS and
// grounding included), so the flat list the agent sees is exactly what
// --tools allows; MCP toolsets are filtered separately, whole, by server
// name — see mcpToolsetsForRun.
func applyToolAllowlist(list []adktool.Tool) []adktool.Tool {
	allow := parseToolAllowlist(flagTools)
	if len(allow) == 0 {
		return list
	}
	kept, unknown := tools.FilterToolsByName(list, allow)
	if len(unknown) > 0 {
		available := make([]string, len(list))
		for i, t := range list {
			available[i] = t.Name()
		}
		fmt.Fprintf(os.Stderr, "pirate: warning: --tools %v matched nothing; available: %s\n",
			unknown, strings.Join(available, ", "))
	}
	return kept
}

// mcpToolsetsForRun builds the MCP toolsets for this run. Under an active
// --tools allow list a server is connected only when its name appears in the
// list (case-insensitive). MCP tools sit behind the Toolset interface and
// cannot be filtered one by one, so a server whose tools would all be
// filtered away is not started at all; A2A and LLMS toolsets are not
// affected by the allow list.
func mcpToolsetsForRun(cfg config.Config) []adktool.Toolset {
	servers := mcpServersForAllowlist(buildMCPServerConfigs(cfg), parseToolAllowlist(flagTools))
	ts, _ := extension.BuildMCPToolsets(servers)
	return ts
}

// mcpServersForAllowlist keeps only the servers whose name appears in allow,
// which parseToolAllowlist has already normalized to lowercase; an empty
// allow list keeps every server. Pure, so the server choice is testable
// without constructing toolsets.
func mcpServersForAllowlist(servers []extension.MCPServerConfig, allow []string) []extension.MCPServerConfig {
	if len(allow) == 0 {
		return servers
	}
	in := make(map[string]bool, len(allow))
	for _, n := range allow {
		in[n] = true
	}
	kept := make([]extension.MCPServerConfig, 0, len(servers))
	for _, s := range servers {
		if in[strings.ToLower(s.Name)] {
			kept = append(kept, s)
		}
	}
	return kept
}

// palaceHasContent reports whether the palace holds at least one drawer.
// A count error is treated as "no content": the tools would fail anyway, and
// the caller's job is to decide whether advertising them is worth the tokens.
func palaceHasContent(p *palace.Palace) bool {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	status, err := p.Status(ctx)
	if err != nil {
		slog.Warn("palace: drawer count failed, not advertising palace tools", "error", err)
		return false
	}
	return status.DrawerCount > 0
}

// setupPalace opens the memory palace when one already exists on disk, and
// returns its tools plus the wake-up context to inject into the system prompt.
// A missing palace is not an error — it simply contributes nothing.
func setupPalace(cfg config.Config, memWorker *memory.Worker) ([]adktool.Tool, string, func()) {
	noop := func() {}
	if !palaceIsEnabled(cfg) {
		return nil, "", noop
	}
	palaceCfg := palaceConfigFromCLI(&cfg)
	if _, err := os.Stat(palaceCfg.DBPath); err != nil {
		return nil, "", noop
	}

	p, err := palace.New(palace.WithConfig(palaceCfg))
	if err != nil {
		fmt.Fprintf(os.Stderr, "pirate: warning: palace tools disabled: %v\n", err)
		return nil, "", noop
	}
	closePalace := func() { _ = p.Close() }

	// Eleven palace tool declarations cost ~1.6k tokens on every request. An
	// empty palace has nothing for them to find, so searching it is a wasted
	// call and the tokens buy nothing — the same trade the LSP gate makes. An
	// existing file is not evidence of content: `pirate memory init` creates one
	// with zero drawers. Gate on drawers, not on the file.
	//
	// The palace is still opened when empty: the bridge below fills it, and the
	// tools appear on the next session once it has content.
	if !palaceHasContent(p) {
		if memWorker != nil {
			memWorker.OnAfterStore(palace.NewObservationBridge(p).ConvertAndStore)
		}
		return nil, "", closePalace
	}

	// A tool-building failure still leaves a usable palace for the bridge and
	// the wake-up context below, so it only costs the tools.
	palaceTools, err := palace.PalaceTools(p)
	if err != nil {
		fmt.Fprintf(os.Stderr, "pirate: warning: palace tools disabled: %v\n", err)
		palaceTools = nil
	}

	// Wire the observation bridge: auto-file observations as palace drawers.
	if memWorker != nil {
		memWorker.OnAfterStore(palace.NewObservationBridge(p).ConvertAndStore)
	}

	var wakeUpContext string
	if wakeUp, wErr := p.WakeUp(context.Background(), ""); wErr == nil {
		wakeUpContext = wakeUp
	}
	return palaceTools, wakeUpContext, closePalace
}

// compactorConfigFrom overlays the configured compactor settings on the
// defaults, ignoring zero values so an unset field keeps its default.
func compactorConfigFrom(cfg config.Config) tools.CompactorConfig {
	compactorCfg := tools.DefaultCompactorConfig()
	if cfg.Compactor == nil {
		return compactorCfg
	}
	if cfg.Compactor.Enabled != nil {
		compactorCfg.Enabled = *cfg.Compactor.Enabled
	}
	if cfg.Compactor.SourceCodeFiltering != "" {
		compactorCfg.SourceCodeFiltering = cfg.Compactor.SourceCodeFiltering
	}
	if cfg.Compactor.MaxChars > 0 {
		compactorCfg.MaxChars = cfg.Compactor.MaxChars
	}
	if cfg.Compactor.MaxLines > 0 {
		compactorCfg.MaxLines = cfg.Compactor.MaxLines
	}
	return compactorCfg
}

// memoryObservationCallback records each successful tool call as a raw
// observation. sessionID is read through the pointer because the session is
// only created after the callbacks are wired.
func memoryObservationCallback(worker *memory.Worker, cfg config.Config, project string, sessionID *string) llmagent.AfterToolCallback {
	var excluded map[string]bool
	if cfg.Memory != nil && len(cfg.Memory.ExcludedTools) > 0 {
		excluded = make(map[string]bool, len(cfg.Memory.ExcludedTools))
		for _, name := range cfg.Memory.ExcludedTools {
			excluded[name] = true
		}
	}

	return func(_ adkagent.Context, t adktool.Tool, args, result map[string]any, toolErr error) (map[string]any, error) {
		if toolErr != nil || *sessionID == "" {
			return result, nil
		}
		name := t.Name()
		if excluded[name] {
			return result, nil
		}
		worker.Enqueue(memory.RawObservation{
			SessionID:  *sessionID,
			Project:    project,
			ToolName:   name,
			ToolInput:  args,
			ToolOutput: result,
			Timestamp:  time.Now(),
		})
		return result, nil
	}
}

// buildToolsets assembles the external toolsets — MCP servers and A2A agents —
// configured for this run. MCP servers are filtered by an active --tools allow
// list (see mcpToolsetsForRun); A2A and LLMS toolsets are not affected by it.
func buildToolsets(cfg config.Config) []adktool.Toolset {
	var toolsets []adktool.Toolset
	if ts := mcpToolsetsForRun(cfg); len(ts) > 0 {
		toolsets = append(toolsets, ts...)
	}
	if cfg.A2A != nil && len(cfg.A2A.Agents) > 0 {
		toolsets = append(toolsets, tools.NewA2AToolset(cfg.A2A))
	}
	if llms := cfg.LLMSSources(); llms != nil {
		toolsets = append(toolsets, tools.NewLLMSCachedToolset(llms))
	}
	return toolsets
}

// providerEnvVar delegates to the provider package so the CLI and the public
// pimodels package cannot drift on where a key comes from.
func providerEnvVar(p string) string {
	return provider.APIKeyEnvVar(p)
}

// globalPermissionRules returns the config.json `permission` rules, or the
// zero rules when none are configured.
func globalPermissionRules(cfg config.Config) permission.Rules {
	if cfg.Permission == nil {
		return permission.Rules{}
	}
	return *cfg.Permission
}

// detectMode returns the default output mode based on terminal state.
// If stdin is not a terminal, defaults to "print" for piped input.
func detectMode() string {
	if fi, err := os.Stdin.Stat(); err == nil {
		if (fi.Mode() & os.ModeCharDevice) == 0 {
			return "print"
		}
	}
	return "interactive"
}

// writeLastSession persists session start metadata for rapid-restart detection.
func writeLastSession(workDir, provider, model string) error {
	data := lastSessionData{
		Timestamp: time.Now(),
		WorkDir:   workDir,
		Model:     model,
	}
	blob, err := json.Marshal(data)
	if err != nil {
		return err
	}
	dir := filepath.Dir(lastSessionFile())
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	return os.WriteFile(lastSessionFile(), blob, 0o600)
}

// readLastSession reads the last session metadata, or nil if unavailable.
func readLastSession() (*lastSessionData, error) {
	data := &lastSessionData{}
	blob, err := os.ReadFile(lastSessionFile())
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	if err := json.Unmarshal(blob, data); err != nil {
		return nil, err
	}
	return data, nil
}

// checkForRapidRestartAndWarn detects if print mode is restarting repeatedly.
// If the same workdir started a session within 3 seconds, it shows the warning.
func checkForRapidRestartAndWarn(workDir string) {
	prev, err := readLastSession()
	if err != nil || prev == nil || prev.WorkDir != workDir {
		return
	}
	elapsed := time.Since(prev.Timestamp)
	if elapsed < 3*time.Second {
		fmt.Fprintf(os.Stderr,
			"pirate: warning: rapid restart detected (%.0fs since last session). "+
				"If init keeps failing, check ~/.pirate/log/ for errors.\n",
			elapsed.Seconds())
		path, msg, readErr := lastLoggedError()
		switch {
		case readErr != nil:
			fmt.Fprintf(os.Stderr, "pirate: warning: failed to inspect session logs: %v\n", readErr)
		case msg != "":
			fmt.Fprintf(os.Stderr, "pirate: last logged error (%s): %s\n", path, msg)
		default:
			fmt.Fprintln(os.Stderr, "pirate: no recent logged errors found.")
		}
	}
}

// lastLoggedError returns the most recent "error" entry from session logs.
func lastLoggedError() (path, msg string, err error) {
	logRoot := filepath.Join(config.PirateHome(), "log")
	dateDirs, err := os.ReadDir(logRoot)
	if err != nil {
		if os.IsNotExist(err) {
			return "", "", nil
		}
		return "", "", err
	}
	// Log directories sort ascending by date, so walking backwards reaches the
	// most recent one first.
	for i := len(dateDirs) - 1; i >= 0; i-- {
		d := dateDirs[i]
		if !d.IsDir() {
			continue
		}
		if p, m := lastLoggedErrorInDateDir(filepath.Join(logRoot, d.Name())); m != "" {
			return p, m, nil
		}
	}
	return "", "", nil
}

// lastLoggedErrorInDateDir scans one date directory's session logs newest-first
// and returns the path and message of the first error entry found. A directory
// that cannot be listed yields no result rather than an error: this whole path
// is a best-effort diagnostic hint.
func lastLoggedErrorInDateDir(datePath string) (path, msg string) {
	files, listErr := os.ReadDir(datePath)
	if listErr != nil {
		return "", ""
	}
	for j := len(files) - 1; j >= 0; j-- {
		f := files[j]
		if f.IsDir() || !strings.HasPrefix(f.Name(), "session-") || !strings.HasSuffix(f.Name(), ".log") {
			continue
		}
		p := filepath.Join(datePath, f.Name())
		if m := lastLoggedErrorInLogFile(p); m != "" {
			return p, m
		}
	}
	return "", ""
}

// lastLoggedErrorInLogFile returns the content of the last non-empty "error"
// entry in one session log, or "" when it holds none or cannot be read.
func lastLoggedErrorInLogFile(logPath string) string {
	blob, readErr := os.ReadFile(logPath)
	if readErr != nil {
		return ""
	}
	lines := strings.Split(string(blob), "\n")
	for k := len(lines) - 1; k >= 0; k-- {
		line := strings.TrimSpace(lines[k])
		if line == "" {
			continue
		}
		var entry logger.Entry
		if err := json.Unmarshal([]byte(line), &entry); err != nil {
			continue
		}
		if entry.Type == "error" && entry.Content != "" {
			return entry.Content
		}
	}
	return ""
}

func toolArgsPreview(args map[string]any) string {
	if len(args) == 0 {
		return ""
	}
	data, err := json.Marshal(args)
	if err != nil {
		data = []byte(fmt.Sprintf("%v", args))
	}
	preview := string(data)
	const maxToolArgsPreviewLen = 100
	runes := []rune(preview)
	if len(runes) <= maxToolArgsPreviewLen {
		return preview
	}
	return string(runes[:maxToolArgsPreviewLen])
}

const (
	printToolCallColor = "\033[36m"
	printToolDoneColor = "\033[32m"
	printToolDimColor  = "\033[2m"
	printToolReset     = "\033[0m"
)

// derivePrintTitle extracts a short, single-line session title from a user
// prompt. Mirrors the TUI's deriveSessionTitle so /sessions lists look
// consistent regardless of which mode created the session.
func derivePrintTitle(prompt string) string {
	title := strings.TrimSpace(prompt)
	if title == "" {
		return ""
	}
	if i := strings.IndexByte(title, '\n'); i >= 0 {
		title = title[:i]
	}
	const max = 200
	if len(title) > max {
		title = title[:max-1] + "…"
	}
	return title
}

func formatPrintToolCall(name string, args map[string]any) string {
	if preview := toolArgsPreview(args); preview != "" {
		return fmt.Sprintf("%s🛠️  ⚙ tool: %s %s%s%s\n", printToolCallColor, name, printToolDimColor, preview, printToolReset)
	}
	return fmt.Sprintf("%s🛠️  ⚙ tool: %s%s\n", printToolCallColor, name, printToolReset)
}

func formatPrintToolDone(name string) string {
	return fmt.Sprintf("%s✅ ✓ tool: %s done%s\n", printToolDoneColor, name, printToolReset)
}

func formatPrintSkillLoad(count int, err error) string {
	if err != nil {
		return fmt.Sprintf("%s⚠ skills: failed%s\n", printToolDimColor, printToolReset)
	}
	return fmt.Sprintf("%s✅ ✓ skills: loaded %d%s\n", printToolDoneColor, count, printToolReset)
}

// runPrint runs the agent and prints text responses to stdout.
// Tool calls are shown as status lines on stderr.
func runPrint(ctx context.Context, ag *agent.Agent, sessionID, prompt string, log *logger.Logger, providerName string) error {
	ctx, span := otel.Tracer("pi-go").Start(ctx, "agent.prompt")
	defer span.End()
	span.SetAttributes(otel.AttributeInt("prompt.length", len(prompt)))

	log.UserMessage(prompt)
	// Auto-set the session title from the user prompt. Print mode is
	// non-interactive, so we don't emit OSC 0 — there may be no terminal, or
	// the terminal may be a script's stdout. The title is metadata for
	// /sessions listing and the meta.json file, both of which still benefit.
	if title := derivePrintTitle(prompt); title != "" {
		_ = ag.SetSessionTitle(sessionID, title)
	}
	retryCfg := agent.DefaultRetryConfig()
	// Say when a request is being re-sent, so a backoff pause is not mistaken
	// for a hung run. stderr keeps it out of the captured reply.
	ctx = retry.WithNotifier(ctx, func(a retry.Attempt) {
		fmt.Fprintln(os.Stderr, a.String())
	})
	// GroundingMetadata repeats on every chunk of the response it grounds;
	// report each search once.
	groundedSeen := map[string]bool{}
	// SSE delivers the reply as deltas and then once more as an aggregate;
	// without this the whole answer prints twice.
	var dedup agent.StreamDedup
	for ev, err := range agent.WithRetryContext(ctx, retryCfg, func() iter.Seq2[*session.Event, error] {
		return ag.RunStreaming(ctx, sessionID, prompt)
	}) {
		if err != nil {
			if ctx.Err() != nil {
				fmt.Fprintln(os.Stderr, "\ninterrupted")
				return nil
			}
			log.Error(err.Error())
			return fmt.Errorf("agent run: %w", err)
		}
		if ev == nil {
			continue
		}
		printGroundingEvent(ev, providerName, groundedSeen, log)
		// Without this, a provider failure exits 0 having printed nothing.
		// See agent.EventError.
		if evErr := agent.EventError(ev); evErr != nil {
			log.Error(evErr.Error())
			return fmt.Errorf("agent run: %w", evErr)
		}
		if ev.Content == nil {
			continue
		}
		dedup.BeginEvent(ev)
		printEventParts(ev, &dedup, log)
	}
	fmt.Println()
	return nil
}

// printGroundingEvent reports a server-side search as a tool call.
//
// Gemini grounding and OpenAI's built-in web_search both run server-side and
// emit no FunctionCall, so they would otherwise be invisible. GroundingMetadata
// repeats on every chunk of the response it grounds, so groundedSeen keeps each
// search to one report.
//
// providerName decides the label: the two searches hit different indexes, so
// reporting an OpenAI search as google_search would misstate where the facts
// came from. Empty falls back to the Gemini name, which is the older shape.
func printGroundingEvent(ev *session.Event, providerName string, groundedSeen map[string]bool, log *logger.Logger) {
	gm := ev.GroundingMetadata
	if gm == nil || len(gm.WebSearchQueries) == 0 {
		return
	}
	key := agent.GroundingQueryKey(gm.WebSearchQueries)
	if groundedSeen[key] {
		return
	}
	groundedSeen[key] = true
	toolName := agent.GroundingToolNameFor(providerName)
	args := map[string]any{"query": agent.GroundingQuery(gm)}
	fmt.Fprint(os.Stderr, formatPrintToolCall(toolName, args))
	log.ToolCall("grounding", toolName, args)
	for _, src := range strings.Split(agent.GroundingSummary(gm), "\n") {
		fmt.Fprintf(os.Stderr, "%s   %s%s\n", printToolDimColor, src, printToolReset)
	}
	fmt.Fprint(os.Stderr, formatPrintToolDone(toolName))
	log.ToolResult("grounding", toolName, agent.GroundingSources(gm))
}

// printEventParts writes one event's parts: reply text to stdout, thinking and
// tool activity to stderr. The caller must have called dedup.BeginEvent for ev.
func printEventParts(ev *session.Event, dedup *agent.StreamDedup, log *logger.Logger) {
	for _, part := range ev.Content.Parts {
		if part.Text != "" && ev.Content.Role == "thinking" {
			fmt.Fprintf(os.Stderr, "\033[2m%s\033[0m", part.Text)
			log.Thinking(ev.Author, part.Text)
			continue
		}
		if part.Text != "" {
			if dedup.SkipText(ev) {
				continue
			}
			fmt.Print(part.Text)
			log.LLMText(ev.Author, part.Text)
		}
		if part.FunctionCall != nil {
			fmt.Fprint(os.Stderr, formatPrintToolCall(part.FunctionCall.Name, part.FunctionCall.Args))
			log.ToolCall(ev.Author, part.FunctionCall.Name, part.FunctionCall.Args)
		}
		if part.FunctionResponse != nil {
			fmt.Fprint(os.Stderr, formatPrintToolDone(part.FunctionResponse.Name))
			log.ToolResult(ev.Author, part.FunctionResponse.Name, fmt.Sprintf("%v", part.FunctionResponse.Response))
		}
	}
}

// jsonEvent represents a JSONL event for JSON output mode.
// Event types follow the spec: message_start, text_delta, tool_call, tool_result, message_end.
type jsonEvent struct {
	Type      string `json:"type"`
	Agent     string `json:"agent,omitempty"`
	Role      string `json:"role,omitempty"`
	Delta     string `json:"delta,omitempty"`
	Content   string `json:"content,omitempty"`
	ToolName  string `json:"tool_name,omitempty"`
	ToolInput any    `json:"tool_input,omitempty"`
	SessionID string `json:"session_id,omitempty"`
	Error     string `json:"error,omitempty"`
}

// runJSON runs the agent and emits JSONL events to stdout.
// Events: message_start (once), text_delta (one per sentence, or per text chunk
// under --json-deltas full), tool_call, tool_result, message_end (once).
//
// Steer input: lines read from stdin are follow-up messages for the same
// conversation. The protocol is deliberately minimal — one line of UTF-8 text
// per steer, no JSON envelope; empty lines are ignored. A steer received while
// a turn runs (or between turns) queues a full extra turn on the same session
// once the current turn finishes: message_start through message_end, exactly
// like the first one. This is the child half of the parent's subagent-steer
// channel (see subagent.Process.Steer).
func runJSON(ctx context.Context, ag *agent.Agent, sessionID, prompt string, stdin io.Reader, log *logger.Logger) error {
	log.UserMessage(prompt)
	// Auto-set the session title for JSON mode too. The first jsonEvent
	// carries session_id, so the title is just metadata to keep meta.json
	// in sync with the user prompt — consumers can use it to label sessions.
	// Steer turns must not retitle the session, so this stays on the first
	// prompt only.
	if title := derivePrintTitle(prompt); title != "" {
		_ = ag.SetSessionTitle(sessionID, title)
	}
	raw, err := jsonRawDeltas()
	if err != nil {
		return err
	}
	em := newJSONEmitter(json.NewEncoder(os.Stdout), log, raw)
	retryCfg := agent.DefaultRetryConfig()
	steers := startJSONSteer(ctx, stdin, em)

	queue := []string{prompt}
	for len(queue) > 0 {
		msg := queue[0]
		queue = queue[1:]
		if err := runJSONTurn(ctx, ag, sessionID, msg, em, retryCfg, log); err != nil {
			return err
		}
		// Drain steers that arrived while the turn ran; each one queues the
		// next turn. Non-blocking: when nothing is queued the process is done
		// — stdin is never waited on, so EOF without steers behaves exactly
		// as before.
		for {
			select {
			case s, ok := <-steers:
				if ok {
					queue = append(queue, s)
					continue
				}
				steers = nil // EOF: no further steer can arrive
			default:
			}
			break
		}
	}
	return nil
}

// startJSONSteer scans stdin lines as steer messages, announcing each with a
// steer_queued event. It never blocks the run: lines land in a bounded channel
// the turn loop drains between turns, and context cancellation stops the
// scanner. A nil reader yields an immediately-closed channel.
func startJSONSteer(ctx context.Context, r io.Reader, em *jsonEmitter) <-chan string {
	ch := make(chan string, 16)
	if r == nil {
		close(ch)
		return ch
	}
	go func() {
		defer close(ch)
		sc := bufio.NewScanner(r)
		for sc.Scan() {
			line := strings.TrimSpace(sc.Text())
			if line == "" {
				continue
			}
			em.emit(jsonEvent{Type: "steer_queued"})
			select {
			case ch <- line:
			case <-ctx.Done():
				return
			}
		}
	}()
	return ch
}

// runJSONTurn streams one turn — one RunStreaming pass — and emits its full
// event cycle. Each turn is self-contained (own StreamDedup, own message_start
// and message_end), so a steer turn reads exactly like the first one.
func runJSONTurn(ctx context.Context, ag *agent.Agent, sessionID, msg string, em *jsonEmitter, retryCfg agent.RetryConfig, log *logger.Logger) error {
	log.UserMessage(msg)
	// Keep-alive while waiting on the LLM: the provider's stream watch ticks
	// this hook (see provider.WithStreamHeartbeat), and each tick writes one
	// keep-alive line. That is what keeps the parent's inactivity watchdog
	// from reading a long time-to-first-token as a wedged child (issue #37).
	// Only this JSON-mode loop installs the hook, so nothing is written in
	// TUI or print mode.
	ctx = provider.WithStreamHeartbeat(ctx, em.keepalive)
	started := false
	// SSE delivers the reply as deltas and then once more as an aggregate;
	// without this every text_delta is emitted twice. One dedup per turn: it
	// tracks one event stream, and each turn is its own stream.
	var dedup agent.StreamDedup

	for ev, err := range agent.WithRetryContext(ctx, retryCfg, func() iter.Seq2[*session.Event, error] {
		return ag.RunStreaming(ctx, sessionID, msg)
	}) {
		if err != nil {
			if ctx.Err() != nil {
				em.flush()
				em.emit(jsonEvent{Type: "message_end"})
				return nil
			}
			log.Error(err.Error())
			return fmt.Errorf("agent run: %w", err)
		}
		if ev == nil {
			continue
		}
		// Emit provider failures as an explicit `error` event so consumers can
		// tell a failed run from an empty one. See agent.EventError.
		if evErr := agent.EventError(ev); evErr != nil {
			log.Error(evErr.Error())
			em.flush()
			em.emit(jsonEvent{Type: "error", Agent: ev.Author, Error: evErr.Error()})
			return fmt.Errorf("agent run: %w", evErr)
		}
		// A tool failure can be the thing that ended the turn: the step budget
		// (agent.NewStepLimitCallback) fails the over-budget call and sets
		// SkipSummarization, so the function response carrying the error is the
		// turn's last event and no model text follows. Emit it as an explicit
		// `error` event too — without this the failure lives only inside a
		// tool_result payload nobody reads as an error, and the parent takes
		// the accumulated partial text for a finished report (issue #51).
		if frErr := turnEndingToolError(ev); frErr != nil {
			log.Error(frErr.Error())
			em.flush()
			em.emit(jsonEvent{Type: "error", Agent: ev.Author, Error: frErr.Error()})
			// Deliberately not a process error: the turn still ends through
			// the normal message_end below and runJSON exits 0. A non-zero
			// exit would read as a crash to the parent's fallback chain
			// (spawnWithRetry re-spawns crashed children on the same model),
			// re-running the whole task the budget just stopped; the clean
			// exit with an error event routes to attemptStopped — final, no
			// re-spawn. The error event is the verdict.
		}
		if ev.Content == nil {
			continue
		}

		// Emit message_start on the first event from the assistant.
		if !started {
			em.emit(jsonEvent{
				Type:      "message_start",
				Agent:     ev.Author,
				Role:      ev.Content.Role,
				SessionID: sessionID,
			})
			started = true
		}

		dedup.BeginEvent(ev)
		em.parts(ev, &dedup)
	}
	em.flush()
	if !started {
		const warn = "pirate: warning: no assistant events received before message_end"
		fmt.Fprintln(os.Stderr, warn)
		log.Error(warn)
	}
	em.emit(jsonEvent{Type: "message_end"})
	return nil
}

// turnEndingToolError extracts a tool error that ended the turn, or nil.
//
// ADK turns every tool-callback error into a function response the model can
// recover from, so a failing tool is normally not a turn event at all. The
// exception is a response on the turn's final event — the failing callback set
// SkipSummarization (agent.NewStepLimitCallback does; RequestConfirmation sets
// the same flag, but its event carries a call awaiting approval, not an error
// response) — there the turn ends on the error and no model text follows. That
// is the one case a consumer of the JSON stream must see as an error.
func turnEndingToolError(ev *session.Event) error {
	if !ev.Actions.SkipSummarization || ev.Content == nil {
		return nil
	}
	for _, part := range ev.Content.Parts {
		if part.FunctionResponse == nil {
			continue
		}
		if msg, ok := part.FunctionResponse.Response["error"].(string); ok && msg != "" {
			return errors.New(msg)
		}
	}
	return nil
}

// buildCommitMsgFunc creates the GenerateCommitMsg callback for /commit.
// It resolves the "commit" role (falling back to "default") and creates a one-shot LLM.
func buildCommitMsgFunc(ctx context.Context, cfg config.Config) func(context.Context, string) (string, error) {
	// Resolve commit role, fall back to default.
	commitModel, commitProvider, _, _, _, err := cfg.ResolveRole("commit")
	if err != nil {
		commitModel, commitProvider, _, _, _, err = cfg.ResolveRole("default")
		if err != nil {
			return nil // no model available
		}
	}

	// A model served by a declared provider resolves through its own endpoint
	// and key; provider.Resolve below only knows the built-in names and would
	// fail — or worse, mis-route a bare name — for one of these.
	if info, apiKey, baseURL, ok := namedModelInfo(cfg, commitModel, commitProvider); ok {
		if err := provider.ValidateModel(info); err != nil {
			return nil
		}
		llm, err := provider.NewLLM(ctx, info, apiKey, baseURL, "none", &provider.LLMOptions{
			ExtraHeaders:    cfg.ExtraHeaders,
			InsecureSkipTLS: cfg.InsecureSkipTLS,
		})
		if err != nil {
			return nil
		}
		return tui.GenerateCommitMsgFunc(llm)
	}

	info, err := provider.Resolve(commitModel)
	if err != nil {
		return nil
	}
	if commitProvider != "" {
		info.Provider = commitProvider
	}
	if err := provider.ValidateModel(info); err != nil {
		return nil
	}

	keys := cfg.ResolveAPIKeys()
	apiKey := keys[info.Provider]
	// Resolve base URL: --url flag takes precedence over env var, then Ollama default.
	baseURL := flagURL
	if baseURL == "" {
		baseURLs := cfg.ResolveBaseURLs()
		baseURL = baseURLs[info.Provider]
	}
	if info.Ollama {
		baseURL = provider.ResolveOllamaEndpoint(provider.OllamaRouting{
			Model:      info.Model,
			BaseURL:    baseURL,
			APIKey:     apiKey,
			ForceLocal: info.LocalOllama,
		})
	}

	if info.Ollama && !provider.IsOllamaCloudEndpoint(baseURL) {
		if err := provider.CheckOllama(baseURL); err != nil {
			return nil
		}
	}

	llm, err := provider.NewLLM(ctx, info, apiKey, baseURL, "none", &provider.LLMOptions{
		ExtraHeaders:    cfg.ExtraHeaders,
		InsecureSkipTLS: cfg.InsecureSkipTLS,
	})
	if err != nil {
		return nil
	}

	return tui.GenerateCommitMsgFunc(llm)
}

// mergeExtraHeaders merges config extraHeaders with CLI --header flags.
// CLI flags override config values on key conflict.
func mergeExtraHeaders(cfgHeaders map[string]string, cliHeaders []string) map[string]string {
	if len(cfgHeaders) == 0 && len(cliHeaders) == 0 {
		return nil
	}
	merged := make(map[string]string)
	for k, v := range cfgHeaders {
		merged[k] = v
	}
	for _, h := range cliHeaders {
		key, val, ok := strings.Cut(h, "=")
		if ok {
			merged[strings.TrimSpace(key)] = strings.TrimSpace(val)
		}
	}
	if len(merged) == 0 {
		return nil
	}
	return merged
}

// providerExtraHeaders assembles the ExtraHeaders an LLM client is built
// with. Precedence, weakest to strongest: the config's global extraHeaders,
// the active named provider's headers (more specific than the globals — they
// override shared keys), an explicit --header flag (the user's direct
// instruction beats everything). ${SESSION_ID} inside a provider header is
// replaced with sessionID; a value still holding the placeholder when no
// session ID is known is dropped — an empty conversation header is worse
// than no header.
func providerExtraHeaders(cfg config.Config, providerName, sessionID string, cliHeaders []string) map[string]string {
	providerHeaders := map[string]string{}
	if p, ok := cfg.Providers[providerName]; ok {
		for k, v := range p.Headers {
			if resolved, ok := resolveSessionIDHeader(v, sessionID); ok {
				providerHeaders[k] = resolved
			}
		}
	}

	// mergeExtraHeaders copies its first argument then overlays the second;
	// the same two steps here give the precedence order documented above.
	merged := mergeExtraHeaders(cfg.ExtraHeaders, nil)
	if merged == nil {
		merged = map[string]string{}
	}
	maps.Copy(merged, providerHeaders)
	maps.Copy(merged, mergeExtraHeaders(nil, cliHeaders))
	if len(merged) == 0 {
		return nil
	}
	return merged
}

// resolveSessionIDHeader substitutes the ${SESSION_ID} placeholder, reporting
// false when the placeholder is present but no session ID is known yet.
func resolveSessionIDHeader(value, sessionID string) (string, bool) {
	if !strings.Contains(value, config.SessionIDPlaceholder) {
		return value, true
	}
	if sessionID == "" {
		return "", false
	}
	return strings.ReplaceAll(value, config.SessionIDPlaceholder, sessionID), true
}

// headersNeedSessionID reports whether any header destined for the named
// provider's LLM client carries ${SESSION_ID} — the signal to fix the session
// ID before the client is built, because headers freeze into it.
func headersNeedSessionID(cfg config.Config, providerName string) bool {
	p, ok := cfg.Providers[providerName]
	if !ok {
		return false
	}
	for _, v := range p.Headers {
		if strings.Contains(v, config.SessionIDPlaceholder) {
			return true
		}
	}
	return false
}

// flagTemperatureChanged records whether --temperature was passed. The flag's
// zero value doubles as "unset" (0 keeps the provider default), so only
// cobra's Changed can tell a passed 0 from an absent flag; runRoot captures it
// the same way flagSocketChanged is.
var flagTemperatureChanged bool

// temperatureFlagOpt returns the --temperature value as *float64, nil when the
// flag was not passed or holds a negative value (a provider would reject it;
// warn and keep the default rather than failing every request).
func temperatureFlagOpt() *float64 {
	if !flagTemperatureChanged {
		return nil
	}
	if flagTemperature < 0 {
		slog.Warn("ignoring negative --temperature", "value", flagTemperature)
		return nil
	}
	t := flagTemperature
	return &t
}

// effectiveThinkingLevel resolves the run's reasoning effort: the --thinking
// flag wins over config.json's thinking level when both name one.
func effectiveThinkingLevel(cfg config.Config) string {
	if v := strings.TrimSpace(flagThinking); v != "" {
		return strings.ToLower(v)
	}
	return cfg.ThinkingLevel
}

// applyTransportOptions layers the --insecure/--ca-cert/--trace-http flags over
// the transport settings from config. The flags are additive: none of them can
// turn a config-enabled setting back off, matching how --header behaves.
func applyTransportOptions(opts *provider.LLMOptions, cfg config.Config, info provider.Info) {
	opts.MaxOutputTokens = cfg.MaxOutputTokens
	opts.InsecureSkipTLS = cfg.InsecureSkipTLS || flagInsecure
	opts.CACertPath = cfg.CACertPath
	if flagCACert != "" {
		opts.CACertPath = flagCACert
	}
	opts.DisableSystemCAs = cfg.DisableSystemCAs

	opts.TraceHTTP = cfg.TraceHTTP || flagTraceHTTP
	// The transport reads this through httplog rather than from opts, so that
	// a client built before the flag was parsed still honors it. Setting it
	// here keeps the two in step for every path that builds an LLM.
	if opts.TraceHTTP {
		httplog.SetEnabled(true)
	}

	// Pacing is resolved here, alongside the other transport settings, because
	// it is installed the same way — as a RoundTripper by BuildTransport — and
	// because every path that builds an LLM already funnels through this
	// function. A provider added to the switch in NewLLM without a stop here
	// would silently send unpaced.
	opts.RateLimit = cfg.ResolveRateLimits(info.Provider, info.Model)

	// Same reasoning, same funnel: the stream-idle watch is built into every
	// model NewLLM returns, so the budget has to ride the options or the
	// watch stays off everywhere.
	opts.StreamIdleTimeout = cfg.ResolveStreamIdleTimeout()
}

// convertHooks converts config.HookConfig to extension.HookConfig.
func convertHooks(cfgHooks []config.HookConfig) []extension.HookConfig {
	hooks := make([]extension.HookConfig, len(cfgHooks))
	for i, h := range cfgHooks {
		hooks[i] = extension.HookConfig{
			Event:   h.Event,
			Command: h.Command,
			Tools:   h.Tools,
			Timeout: h.Timeout,
		}
	}
	return hooks
}

// gitCmdTimeout bounds every git subprocess call spawned during init so a
// stalled repo (blocked hook, lock contention, unreachable network mount)
// can never hang the init pipeline indefinitely.
const gitCmdTimeout = 5 * time.Second

// memoryDrainTimeout bounds the memory worker's drain at session end.
//
// It is larger than the 5s this used to be because draining is now the only
// step that can be slow: with the default model-free compressor each queued
// observation is a local insert, but a host configured for subagent compression
// pays a child process per observation, and a budget shorter than one of those
// abandons the tail of the session.
//
// A timeout here is not fatal but it does suppress the summary, so it is also
// the point at which the session stops being described rather than just
// recorded.
const memoryDrainTimeout = 60 * time.Second

// oneShotMemoryDrainTimeout bounds the memory worker's drain in one-shot runs
// (print/json): the process must not linger once its answer is delivered.
//
// Derived as half the summary budget. The default model-free compressor drains
// queued observations in milliseconds — local inserts, measured sub-ms — so the
// cap only bites a host configured for subagent compression, which pays a child
// process per observation (~5.6s each, see internal/memory/summarize.go). A
// one-shot run accepts losing the tail of its own memory rather than holding
// the caller for it. Interactive keeps the full [memoryDrainTimeout]: there the
// session tail is the product, and exit latency is not billed to anyone.
const oneShotMemoryDrainTimeout = defaultSessionSummaryTimeout / 2

// hasObservationsProbeTimeout bounds the empty-session probe on the shutdown
// path. A one-row indexed lookup, so a small fixed ceiling: the probe must
// never become the thing the exit waits on.
const hasObservationsProbeTimeout = 2 * time.Second

// memoryDrainBudget returns the memory worker's drain budget for a run mode.
// One-shot modes get the reduced ceiling; long-running modes (rpc/socket
// servers) and interactive keep the full [memoryDrainTimeout].
func memoryDrainBudget(mode string) time.Duration {
	switch mode {
	case "print", "json":
		return oneShotMemoryDrainTimeout
	default:
		return memoryDrainTimeout
	}
}

// defaultSessionSummaryTimeout bounds the end-of-session summary: one model call.
//
// It gets its own budget rather than sharing the drain's, because a summary is
// work that has not started when the drain finishes. It is a hard bound and not
// an open wait — this runs on the exit path, so a provider that never answers
// must not hold the process open. A summary that overruns is logged and
// abandoned; the observations it would have described are already stored.
const defaultSessionSummaryTimeout = 30 * time.Second

// sessionSummaryBudget returns the per-session-summary timeout: the env var
// PI_SUMMARY_TIMEOUT_MS when valid and positive, otherwise the default.
//
// An env var is used rather than a config field because summary and drain share
// one lifecycle phase (the shutdown closer) and there is no config object in
// scope at that point. The PI_ prefix matches the pattern of other execution
// knobs (PI_SUBAGENT_TIMEOUT_MS, PI_SUBAGENT_CONCURRENCY).
//
// A syntactically invalid or non-positive value logs a single warning through
// slog.Debug and falls back to the default.
func sessionSummaryBudget() time.Duration {
	if envMs := os.Getenv("PI_SUMMARY_TIMEOUT_MS"); envMs != "" {
		ms, err := strconv.Atoi(envMs)
		if err == nil && ms > 0 {
			return time.Duration(ms) * time.Millisecond
		}
		slog.Debug("memory: invalid PI_SUMMARY_TIMEOUT_MS, using default",
			"value", envMs, "default", defaultSessionSummaryTimeout)
	}
	return defaultSessionSummaryTimeout
}

// detectGitRoot returns the git repository root for the given directory,
// or empty string if not inside a git repo.
//
// Inside a linked worktree this resolves the *main* checkout, not the worktree
// — the value becomes PI_SANDBOX_ROOT for spawned subagents, and rooting that
// at a worktree makes every file-tool access to the rest of the repo fail.
// See internal/gitroot.
func detectGitRoot(ctx context.Context, dir string) string {
	return gitroot.Detect(ctx, dir)
}

// LoadDotEnv loads environment variables from ~/.pirate/.env and the nearest
// project .pirate/.env. Project values override global values and both override
// the inherited shell environment.
func LoadDotEnv() {
	loadDotEnv()
}

// loadDotEnv loads environment variables from ~/.pirate/.env and project
// .pirate/.env. These files are written by login/config flows and take
// precedence over the inherited shell environment — a user who ran `/login`
// expects the saved credential to be used even if their shell still exports a
// different API key from earlier. Lines in the files override the process env;
// missing keys fall through to whatever the shell set.
func loadDotEnv() {
	loadDotEnvFile(filepath.Join(config.PirateHome(), ".env"))
	if cwd, err := os.Getwd(); err == nil {
		if projectEnv := findNearestDotEnv(cwd); projectEnv != "" {
			loadDotEnvFile(projectEnv)
		}
	}
}

func loadDotEnvFile(path string) {
	data, err := os.ReadFile(path)
	if err != nil {
		return
	}
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, val, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		key = strings.TrimSpace(key)
		val = strings.TrimSpace(val)
		if val == "" {
			continue
		}
		_ = os.Setenv(key, val)
	}
}

func findNearestDotEnv(start string) string {
	dir, err := filepath.Abs(start)
	if err != nil {
		return ""
	}
	for {
		candidate := filepath.Join(dir, config.ProjectDirName, ".env")
		if st, err := os.Stat(candidate); err == nil && !st.IsDir() {
			return candidate
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return ""
		}
		dir = parent
	}
}

// palaceConfigFromCLI resolves the palace config section into the full
// palace.PalaceConfig every palace.New call site consumes: paths with their
// defaults plus the embedder decision — an external API endpoint first, then
// Ollama, then the in-process model. Single source of truth: search, status,
// wake-up, kg and the TUI sidebar must open the same embedder mining uses, or
// the dim filter (RankBySimilarity) silently discards api-mined vectors and
// status reports the wrong backend.
//
// ${VAR} in the api settings expands here, where the embedder is built. The
// stored config keeps the literal placeholder so a later Save never writes an
// expanded key — see config.ResolveEnvValue.
func palaceConfigFromCLI(cfg *config.Config) palace.PalaceConfig {
	palaceCfg := palace.DefaultConfig()
	if p := cfg.Palace; p != nil {
		if p.DBPath != "" {
			palaceCfg.DBPath = p.DBPath
		}
		if p.ModelPath != "" {
			palaceCfg.ModelPath = p.ModelPath
		}
		if p.OllamaURL != "" {
			palaceCfg.OllamaURL = p.OllamaURL
		}
		if p.OllamaModel != "" {
			palaceCfg.OllamaModel = p.OllamaModel
		}
		if p.LocalEmbedder {
			palaceCfg.UseOllama = false
		}
		if p.EmbeddingsURL != "" {
			palaceCfg.APIEmbedderURL = config.ResolveEnvValue(p.EmbeddingsURL)
			palaceCfg.APIEmbedderModel = config.ResolveEnvValue(p.EmbeddingsModel)
			palaceCfg.APIEmbedderKey = config.ResolveEnvValue(p.EmbeddingsAPIKey)
		}
	}
	if palaceCfg.DBPath == "" {
		palaceCfg.DBPath = filepath.Join(config.PirateHome(), "palace.db")
	}
	if palaceCfg.ModelPath == "" {
		palaceCfg.ModelPath = filepath.Join(config.PirateHome(), "models", "KnightsAnalytics_all-MiniLM-L6-v2")
	}
	return palaceCfg
}

// Execute runs the root command.
func Execute() error {
	// Defensive repeat of the migration done at the very top of main(): direct
	// callers of Execute (tests, future entry points) must not skip it. It is
	// idempotent — a home that already carries migrated content is a no-op —
	// so this costs a few Lstats when main already ran.
	config.MigrateLegacyHome()
	return newRootCmd().Execute()
}
