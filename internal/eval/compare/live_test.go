package compare

import (
	"context"
	"errors"
	"iter"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	llmmodel "google.golang.org/adk/v2/model"

	"github.com/spa-skyson/pi-rate/internal/agent"
	"github.com/spa-skyson/pi-rate/internal/testenv"
)

// inventoryFixture renders a session-log body with the given inventory lines,
// the way logger entries actually carry them (one JSON record per line).
func inventoryFixture(lines ...string) string {
	var b strings.Builder
	for _, l := range lines {
		b.WriteString(`{"level":"info","content":` + quoteJSON(l) + `}` + "\n")
	}
	return b.String()
}

// quoteJSON renders s as a JSON string literal.
func quoteJSON(s string) string {
	var b strings.Builder
	b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"':
			b.WriteString(`\"`)
		case '\\':
			b.WriteString(`\\`)
		case '\n':
			b.WriteString(`\n`)
		default:
			b.WriteRune(r)
		}
	}
	b.WriteByte('"')
	return b.String()
}

// TestInventoryParse pins the driver's inventory reader: the LAST
// "context inventory:" line of a session log wins, non-inventory lines and
// non-JSON lines are skipped, and a log without an inventory line is an
// error — the arm must not silently report zero prompt bytes.
func TestInventoryParse(t *testing.T) {
	t.Parallel()

	t.Run("last line wins", func(t *testing.T) {
		t.Parallel()
		total, err := parseInventoryTotal(inventoryFixture(
			"context inventory: base=100 rules=20 skills=0 menu=0 total=120 tools=900",
			"unrelated log line",
			"context inventory: base=200 rules=30 skills=0 menu=0 total=230 tools=900",
		))
		if err != nil {
			t.Fatalf("parseInventoryTotal: %v", err)
		}
		if total != 230 {
			t.Errorf("total = %d, want 230 (the last inventory line)", total)
		}
	})

	t.Run("no inventory line is an error", func(t *testing.T) {
		t.Parallel()
		if _, err := parseInventoryTotal("just some lines\nnothing here\n"); err == nil {
			t.Error("parseInventoryTotal must fail when the log carries no inventory line")
		}
	})

	t.Run("empty log is an error", func(t *testing.T) {
		t.Parallel()
		if _, err := parseInventoryTotal(""); err == nil {
			t.Error("parseInventoryTotal must fail on an empty log")
		}
	})

	t.Run("InventoryTotalFromLog empty path is zero", func(t *testing.T) {
		t.Parallel()
		total, err := InventoryTotalFromLog("")
		if err != nil {
			t.Fatalf("InventoryTotalFromLog(\"\"): %v", err)
		}
		if total != 0 {
			t.Errorf("total = %d, want 0", total)
		}
	})

	t.Run("InventoryTotalFromLog reads the file", func(t *testing.T) {
		t.Parallel()
		path := filepath.Join(t.TempDir(), "session-12-00-00.log")
		body := inventoryFixture("context inventory: base=10 rules=2 skills=1 menu=1 total=13 tools=42")
		if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
		total, err := InventoryTotalFromLog(path)
		if err != nil {
			t.Fatalf("InventoryTotalFromLog: %v", err)
		}
		if total != 13 {
			t.Errorf("total = %d, want 13", total)
		}
	})
}

// TestLiveRunner_ErrNoProvider proves the provider gate: a Runner over an
// environment with no usable key fails RunArm with ErrNoProvider — before any
// workspace is seeded or any agent built. The test clears only the candidate
// variables (not the whole environment) and skips when a real key is set, so
// a developer's machine with credentials never takes the negative path.
// Not parallel: t.Setenv (through testenv.SetHome) forbids it.
func TestLiveRunner_ErrNoProvider(t *testing.T) {
	for _, v := range []string{"ANTHROPIC_API_KEY", "OPENAI_API_KEY"} {
		if os.Getenv(v) != "" {
			t.Skipf("%s is set — the negative provider path cannot be exercised", v)
		}
	}
	t.Setenv("ANTHROPIC_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "")

	testenv.SetHome(t, t.TempDir())
	lr := &LiveRunner{Root: t.TempDir()}
	_, err := lr.RunArm(context.Background(), Variant{Name: "control"}, 1)
	if err == nil {
		t.Fatal("RunArm with no provider candidate must fail")
	}
	if !errors.Is(err, ErrNoProvider) {
		t.Fatalf("error = %v, want ErrNoProvider", err)
	}
}

// TestLiveRunner_BuildConfig pins the config assembly, the half of the driver
// that needs no network: control (nil Parts) gets the file-driven assembly
// from the arm's workspace, variant gets exactly the supplied parts, and both
// point their SessionService at their own sessions dir.
// Not parallel: t.Setenv (through testenv.SetHome) forbids it.
func TestLiveRunner_BuildConfig(t *testing.T) {
	testenv.SetHome(t, t.TempDir())

	lr := &LiveRunner{
		Root:  t.TempDir(),
		Model: &fakeLLM{},
	}
	tmpl, err := lr.workspaceTemplate()
	if err != nil {
		t.Fatalf("workspaceTemplate: %v", err)
	}

	variantParts := &agent.InstructionParts{
		Base:  agent.SystemInstruction,
		Rules: "\n\n# Project Rules\n\nNever use bash; use read and ls only\n",
	}

	for _, tc := range []struct {
		name    string
		parts   *agent.InstructionParts
		wantSub string // a substring only this arm's prompt assembly carries
	}{
		{name: "control-file-assembly", parts: nil, wantSub: "Answer in at most three sentences."},
		{name: "variant-injected", parts: variantParts, wantSub: "Never use bash; use read and ls only"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ws := filepath.Join(lr.Root, tc.name+"-ws")
			if err := copyDir(ws, tmpl); err != nil {
				t.Fatalf("copyDir: %v", err)
			}
			sessionsDir := filepath.Join(lr.Root, tc.name+"-sessions")
			cfg, log, err := lr.buildConfig(Variant{Name: tc.name, Parts: tc.parts}, ws, sessionsDir)
			if err != nil {
				t.Fatalf("buildConfig: %v", err)
			}
			defer func() { _ = log.Close() }()

			if cfg.Model != lr.Model {
				t.Error("config must carry the runner's model")
			}
			if cfg.WorkingDir != ws {
				t.Errorf("WorkingDir = %s, want %s", cfg.WorkingDir, ws)
			}
			if len(cfg.Tools) == 0 {
				t.Error("config must carry the core tool set")
			}
			if cfg.SessionService == nil {
				t.Error("config must carry a session service")
			}
			if cfg.InstructionParts == nil {
				t.Fatal("InstructionParts must be set — the driver always supplies resolved parts")
			}
			if !strings.Contains(cfg.InstructionParts.String(), tc.wantSub) {
				t.Errorf("prompt assembly must carry %q, got:\n%.400s", tc.wantSub, cfg.InstructionParts.String())
			}
			if _, err := os.Stat(sessionsDir); err != nil {
				t.Errorf("sessions dir %s must exist after buildConfig: %v", sessionsDir, err)
			}
		})
	}
}

// fakeLLM is a model.LLM stand-in for config assembly; it is never called.
type fakeLLM struct{}

func (fakeLLM) Name() string { return "fake" }

func (fakeLLM) GenerateContent(context.Context, *llmmodel.LLMRequest, bool) iter.Seq2[*llmmodel.LLMResponse, error] {
	return func(yield func(*llmmodel.LLMResponse, error) bool) {
		yield(nil, errors.New("fakeLLM must not be called"))
	}
}

// TestLiveCompare is the live driver over the real tool loop: control on the
// file-driven assembly from the seeded workspace, variant on injected parts
// whose rules ban bash, both aggregated by Compare and rendered by Render.
//
// Opt-in like the eval harness: it skips unless PI_HARNESS_AB_LIVE=1, because
// it makes real model calls, and skips with an explicit reason when no live
// provider answers (ErrNoProvider) — the flag alone must not require keys.
// Live runs are weather: the assertions cover the report's shape, not the
// arms' behavior.
func TestLiveCompare(t *testing.T) {
	if os.Getenv("PI_HARNESS_AB_LIVE") != "1" {
		t.Skipf("set PI_HARNESS_AB_LIVE=1 to run the live comparison (makes real model calls)")
	}
	if testing.Short() {
		t.Skip("live comparison skipped in -short mode")
	}
	// A HOME per test: the per-run loggers write under it, so the inventory
	// lines this test reads are its own.
	testenv.SetHome(t, t.TempDir())
	// No skills menu: an empty skills dir keeps both arms' prompt assembly
	// deterministic apart from the injected rules.
	t.Setenv("PI_SKILLS_DIRS", filepath.Join(t.TempDir(), "no-skills"))

	runs := 1
	if v := os.Getenv("PI_HARNESS_AB_RUNS"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n <= 0 {
			t.Fatalf("PI_HARNESS_AB_RUNS=%q is not a positive integer", v)
		}
		runs = n
	}

	ctx := context.Background()
	// Resolve once up front so ErrNoProvider reads as a clean skip instead of
	// an arm error buried in Compare.
	mdl, modelName, err := NewLiveModel(ctx)
	if errors.Is(err, ErrNoProvider) {
		t.Skipf("live comparison skipped: %v", err)
	}
	if err != nil {
		t.Fatalf("NewLiveModel: %v", err)
	}

	lr := &LiveRunner{
		Root:      t.TempDir(),
		Model:     mdl,
		ModelName: modelName,
	}
	t.Cleanup(func() { _ = os.RemoveAll(lr.Root) })

	control := Variant{
		Name: "control",
		// Parts nil: the file-driven assembly from the arm's workspace —
		// the production path.
		Label: "Count the files in the current directory and tell me the number.",
	}
	variant := Variant{
		Name: "no-bash",
		Parts: &agent.InstructionParts{
			Base:  agent.SystemInstruction,
			Rules: "\n\n# Project Rules\n\nNever use bash; use read and ls only\n",
		},
		Label: "Count the files in the current directory and tell me the number.",
	}

	res, err := Compare(ctx, control, variant, lr, runs)
	if err != nil {
		t.Fatalf("Compare: %v", err)
	}

	report := Render(res)
	t.Logf("live model: %s, runs=%d per arm", modelName, runs)
	t.Logf("comparison report:\n%s", report)

	if strings.TrimSpace(report) == "" {
		t.Fatal("Render must produce a non-empty report")
	}
	if !strings.Contains(report, "## Control") || !strings.Contains(report, "## Variant") {
		t.Error("report must contain both arm sections")
	}
	for i, fp := range res.Fingerprint {
		if fp == "" {
			t.Errorf("fingerprint[%d] is empty", i)
		}
	}
}
