// Package testenv holds small helpers that keep tests portable across
// operating systems. It is only imported from _test.go files.
package testenv

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
)

// dirtyEnvVars is the registry of environment variables pirate reads from the
// process environment whose ambient values make tests machine-dependent: a
// machine where pirate itself is configured (spawn-tree variables from a
// running session, provider keys, PIRATE_HOME) leaks that configuration into
// tests and produces failures that do not reproduce on CI.
//
// The list mirrors the os.Getenv/LookupEnv call sites under internal/, grouped
// by reader. Variables matching the PI_ prefix or the _API_KEY / _BASE_URL
// suffixes do not need to be listed here -- dirtyNames sweeps them from the
// live environment anyway -- but every variable outside those patterns must
// appear, or sanitization misses it.
var dirtyEnvVars = []string{
	// Spawned-agent place in its run tree (internal/session/agentenv.go,
	// internal/permission).
	"PI_AGENT_ID",
	"PI_AGENT_TYPE",
	"PI_RUN_ID",
	"PI_SPEC_NAME",
	"PI_RUN_SLICE",
	"PI_RUN_CYCLE",
	"PI_PARENT_SESSION",
	"PI_AGENT_BRANCH",
	"PI_WORKTREE_ROOT",
	"PI_AGENT_PERMISSION",

	// Home and data directories (internal/config/home.go, cli, acp/server).
	"PIRATE_HOME",
	"PI_GO_HOME",
	"PI_SANDBOX_ROOT",
	"PI_SESSIONS_DIR",

	// Runtime knobs (internal/config, internal/subagent, internal/cli).
	"PI_SUBAGENT_CONCURRENCY",
	"PI_SUBAGENT_TIMEOUT_MS",
	"PI_SUBAGENT_INACTIVITY_MS",
	"PI_STREAM_IDLE_TIMEOUT_MS",
	"PI_MIDTURN_ATTEMPTS",
	"PI_SKILLS_DIRS",
	"PI_DISABLE_LEGACY_SKILLS",
	"PI_SUMMARY_TIMEOUT_MS",
	"PI_QUESTION_TIMEOUT_MS",
	"PI_PLAN_MAX_FIX_CYCLES",
	"PI_OLLAMA_NUM_PREDICT",

	// Process-wide feature toggles (internal/agent, internal/tools,
	// internal/provider, internal/palace, internal/cli).
	"PI_NO_GROUNDING",
	"PI_WEB_SEARCH",
	"PI_XAI_TOOLS",
	"PI_NO_XAI_TOOLS",
	"PI_NO_COREML",
	"PI_ONNXRUNTIME_LIB",
	"PI_GO_UPDATE_CHECK",

	// External command overrides (internal/codex, internal/acp/client/*).
	"PI_CODEX_CMD",
	"PI_ACP_AGY_CMD",
	"PI_ACP_CLAUDE_CMD",
	"PI_ACP_COPILOT_CMD",
	"PI_ACP_CURSOR_CMD",
	"PI_ACP_GEMINI_CMD",

	// Headless entry points (internal/cli a2a/serve) and kagent.
	"PI_MODEL",
	"PI_SYSTEM",
	"PI_BASE_URL",
	"GEMINI_LIVE_MODEL",
	"KAGENT_AGENT_CARD_JSON",

	// Provider API keys (config.APIKeys, internal/provider, internal/tools).
	"ANTHROPIC_API_KEY",
	"ANTHROPIC_AUTH_TOKEN",
	"OPENAI_API_KEY",
	"AZUREOPENAI_API_KEY",
	"AZURE_OPENAI_API_KEY",
	"AZURE_API_KEY",
	"AZURE_OPENAI_ENDPOINT",
	"GEMINI_API_KEY",
	"GOOGLE_API_KEY",
	"MISTRAL_API_KEY",
	"XAI_API_KEY",
	"OPENROUTER_API_KEY",
	"OLLAMA_API_KEY",
	"OPENCODE_API_KEY",
	"AGENTGATEWAY_API_KEY",

	// Provider base URLs (config.BaseURLs, internal/provider).
	"ANTHROPIC_BASE_URL",
	"OPENAI_BASE_URL",
	"GEMINI_BASE_URL",
	"MISTRAL_BASE_URL",
	"XAI_BASE_URL",
	"OPENROUTER_BASE_URL",
	"OPENCODE_BASE_URL",
	"AGENTGATEWAY_BASE_URL",
	"OLLAMA_HOST",

	// Ambient Anthropic authentication sources consulted by the SDK behind
	// the provider (surfaced by TestNewPromptHandler_NoAPIKey).
	"ANTHROPIC_FEDERATION_RULE_ID",
	"ANTHROPIC_ORGANIZATION_ID",
	"ANTHROPIC_IDENTITY_TOKEN_FILE",
}

// dirtyName reports whether name is polluted by pattern: pirate-owned PI_*
// variables and provider credentials/endpoints (*_API_KEY, *_BASE_URL). This
// catches variables added after this list was written -- a new provider
// reading NEWPROVIDER_BASE_URL must not reopen the ambient-env bug.
func dirtyName(name string) bool {
	return strings.HasPrefix(name, "PI_") ||
		strings.HasSuffix(name, "_API_KEY") ||
		strings.HasSuffix(name, "_BASE_URL")
}

// dirtyNames returns the variables to sanitize: the explicit registry plus
// every currently-set variable matching dirtyName.
func dirtyNames() []string {
	names := slices.Clone(dirtyEnvVars)
	for _, kv := range os.Environ() {
		name, _, _ := strings.Cut(kv, "=")
		if !slices.Contains(names, name) && dirtyName(name) {
			names = append(names, name)
		}
	}
	return names
}

// UnsetEnv removes every dirty variable from the process environment, with no
// way back. Call it from TestMain, before m.Run(): the whole package then runs
// against a clean environment. Tests that need one of these variables set it
// explicitly afterwards (t.Setenv), which restores it at cleanup.
//
// os.Unsetenv is used instead of setting variables to the empty string on
// purpose: os.LookupEnv must report the variable as absent, not merely empty,
// and empty is a meaningful value to readers like PI_AGENT_PERMISSION (an
// empty value fails rules parsing rather than reading as unset).
func UnsetEnv() {
	for _, name := range dirtyNames() {
		os.Unsetenv(name)
	}
}

// SanitizeEnv removes every dirty variable for the duration of the test and
// restores the previous state (present or absent) at cleanup. It is the
// per-test counterpart of UnsetEnv, for packages or individual tests that
// cannot sanitize in TestMain.
//
// Like t.Setenv, it panics if the test (or its parent) has called Parallel.
func SanitizeEnv(tb testing.TB) {
	tb.Helper()
	for _, name := range dirtyNames() {
		// t.Setenv registers restoration of the previous state, including
		// absence; the Unsetenv below makes the variable truly absent for the
		// test body (the same belt-and-braces as UnsetHome).
		tb.Setenv(name, "")
		os.Unsetenv(name)
	}
}

// TempPirateHome points PIRATE_HOME and its legacy alias PI_GO_HOME at a fresh
// temporary directory for the duration of the test, so config resolution
// cannot reach the developer's real ~/.pirate. PirateHome consults
// PIRATE_HOME first, so setting both closes the fallback. Returns the
// directory, which exists.
func TempPirateHome(tb testing.TB) string {
	tb.Helper()
	dir := tb.TempDir()
	tb.Setenv("PIRATE_HOME", dir)
	tb.Setenv("PI_GO_HOME", dir)
	return dir
}

// SetHome points the user's home directory at dir for the duration of the test.
//
// os.UserHomeDir reads $HOME on Unix but %USERPROFILE% on Windows, so a test
// that sets only HOME still resolves to the real profile directory there --
// which makes the test read and write the developer's (or CI runner's) actual
// ~/.pirate instead of its sandbox.
func SetHome(t *testing.T, dir string) {
	t.Helper()
	t.Setenv("HOME", dir)
	if runtime.GOOS == "windows" {
		t.Setenv("USERPROFILE", dir)
	}
}

// SetUnwritableHome points the home directory at a regular file for the
// duration of the test, so anything the code under test tries to create below
// ~ fails. It returns that path.
//
// Tests used to spell this as HOME=/nonexistent/..., which does not travel:
// on Windows a leading slash is drive-relative, so os.MkdirAll cheerfully
// creates the directory and the expected failure never happens.
func SetUnwritableHome(t *testing.T) string {
	t.Helper()
	home := filepath.Join(t.TempDir(), "home-is-a-file")
	if err := os.WriteFile(home, nil, 0o600); err != nil {
		t.Fatalf("creating the file that stands in for HOME: %v", err)
	}
	SetHome(t, home)
	return home
}

// RequireShell skips the test unless a POSIX shell is on PATH.
//
// The GitHub Windows runners ship Git for Windows, so "bash" and "sh" resolve
// there; a bare Windows box has neither. Tests that drive shell commands call
// this instead of hardcoding /bin/sh, which never exists on Windows.
func RequireShell(t *testing.T) string {
	t.Helper()
	sh, err := exec.LookPath("bash")
	if err != nil {
		sh, err = exec.LookPath("sh")
	}
	if err != nil {
		t.Skipf("no POSIX shell on PATH: %v", err)
	}
	return sh
}

// UnsetHome removes the home-directory environment variables for the duration
// of the test, so os.UserHomeDir has nothing to resolve. Unsetting HOME alone
// leaves %USERPROFILE% in place, which is the only variable Windows consults.
func UnsetHome(t *testing.T) {
	t.Helper()
	t.Setenv("HOME", "")
	if runtime.GOOS == "windows" {
		t.Setenv("USERPROFILE", "")
	}
	os.Unsetenv("HOME")
	if runtime.GOOS == "windows" {
		os.Unsetenv("USERPROFILE")
	}
}

// FakeBinary writes a do-nothing executable named name into dir and returns
// its path.
//
// On Windows the file gets a .bat suffix, because exec.LookPath there resolves
// only the suffixes listed in %PATHEXT% -- an extensionless file is invisible
// to it, however executable its mode bits claim to be.
func FakeBinary(t *testing.T, dir, name string) string {
	t.Helper()
	body := "#!/bin/sh\nexit 0\n"
	if runtime.GOOS == "windows" {
		name += ".bat"
		body = "@exit /b 0\r\n"
	}
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(body), 0o755); err != nil {
		t.Fatalf("write fake binary %s: %v", path, err)
	}
	return path
}
