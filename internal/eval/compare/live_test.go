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
			cfg, closer, err := lr.buildConfig(Variant{Name: tc.name, Parts: tc.parts}, ws, sessionsDir)
			if err != nil {
				t.Fatalf("buildConfig: %v", err)
			}
			// The config holds the sandbox's open root over ws and the
			// logger's open session log; both must be released before the
			// outer t.TempDir cleanup removes lr.Root (Windows is strict).
			t.Cleanup(closer)

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

	// Run-store reuse: a comparison of these exact variants that already
	// finished is loaded from .harness-compare/ instead of re-run, and a
	// fresh one is saved for the next invocation.
	storeDir, err := StoreDir()
	if err != nil {
		t.Fatalf("StoreDir: %v", err)
	}
	res, reused, err := compareWithStore(ctx, control, variant, lr, runs, storeDir)
	if err != nil {
		t.Fatalf("Compare: %v", err)
	}
	t.Logf("run store: %s (reused=%v)", storeDir, reused)

	report := Render(res)
	t.Logf("live model: %s, runs=%d per arm", modelName, runs)
	t.Logf("comparison report:\n%s", report)

	// Blind mode (PI_HARNESS_AB_BLIND=1): log the anonymized report next to
	// the named one — the blind copy is what a judge would see.
	if os.Getenv("PI_HARNESS_AB_BLIND") == "1" {
		t.Logf("blind report:\n%s", blindReport(res))
	}

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

// TestCompareWithStore_Reuse pins the wiring between the comparison core and
// the run Store: the first call runs the Runner and saves, the second call
// for the same variants must be served entirely from the store — the Runner
// is never invoked and the loaded result equals the saved one. A different
// fingerprint pair must not reuse anything. Uses a throwaway store dir, so
// the repository-root cache is untouched.
func TestCompareWithStore_Reuse(t *testing.T) {
	dir := t.TempDir()
	runner := &fakeRunner{arms: map[string]ArmResult{
		"control": buildArm(armSpec{name: "control", steps: 10, calls: map[string]int{"read": 4}}),
		"variant": buildArm(armSpec{name: "variant", steps: 8, calls: map[string]int{"read": 6}}),
	}}

	runs := []RunResult{{Scenario: "grep-basics", Status: "pass"}}
	res1, reused1, err := compareWithStore(context.Background(), controlVariant(), variantVariant(), withArmRuns(runner, runs), 1, dir)
	if err != nil {
		t.Fatalf("first compareWithStore: %v", err)
	}
	if reused1 {
		t.Error("first run must not report reuse")
	}
	if got := len(runner.calls); got != 2 {
		t.Fatalf("first run: runner called %d time(s), want 2", got)
	}

	// The store now holds the comparison; a second call for the same run
	// count must not reach the runner.
	res2, reused2, err := compareWithStore(context.Background(), controlVariant(), variantVariant(), withArmRuns(runner, runs), 1, dir)
	if err != nil {
		t.Fatalf("second compareWithStore: %v", err)
	}
	if !reused2 {
		t.Error("second run must report reuse")
	}
	if got := len(runner.calls); got != 2 {
		t.Errorf("second run: runner called %d time(s), want 2 (no re-execution)", got)
	}
	if res2.Control.Variant != res1.Control.Variant || res2.Variant.Variant != res1.Variant.Variant {
		t.Errorf("reused arms drifted: (%q,%q) -> (%q,%q)",
			res1.Control.Variant, res1.Variant.Variant, res2.Control.Variant, res2.Variant.Variant)
	}
	if res2.Fingerprint != res1.Fingerprint {
		t.Error("reused fingerprint drifted")
	}
	if len(res2.Contrasts) != len(res1.Contrasts) {
		t.Errorf("reused contrast count drifted: %d -> %d", len(res1.Contrasts), len(res2.Contrasts))
	}

	// A different pair must not be served from the first pair's file.
	other := Variant{Name: "other", Parts: &agent.InstructionParts{Base: "other base"}}
	runner2 := withArmRuns(&fakeRunner{arms: map[string]ArmResult{
		"control": buildArm(armSpec{name: "control", steps: 1}),
		"other":   buildArm(armSpec{name: "other", steps: 2}),
	}}, runs)
	_, reusedOther, err := compareWithStore(context.Background(), controlVariant(), other, runner2, 1, dir)
	if err != nil {
		t.Fatalf("other-pair compareWithStore: %v", err)
	}
	if reusedOther {
		t.Error("a different fingerprint pair must not reuse another pair's stored run")
	}
}

// withArmRuns gives every scripted arm the given run list, so a stored
// result carries the run count the reuse gate (issue 80) checks against the
// request.
func withArmRuns(f *fakeRunner, runs []RunResult) *fakeRunner {
	for name, arm := range f.arms {
		arm.Runs = runs
		arm.Delivered = countDelivered(runs)
		f.arms[name] = arm
	}
	return f
}

// TestCompareWithStore_ReuseHonorsRuns pins the run-count half of the reuse
// gate (issue 80): the store file pins the variants' configuration, not how
// many runs it holds, so a stored comparison with fewer runs than requested
// must be re-executed — with the reason printed — and only an adequately
// stocked result is reused.
func TestCompareWithStore_ReuseHonorsRuns(t *testing.T) {
	dir := t.TempDir()

	oneRun := []RunResult{{Scenario: "live", Status: "pass"}}
	arm := func(name string, runs []RunResult) ArmResult {
		a := buildArm(armSpec{name: name, calls: map[string]int{"read": 1}})
		a.Runs = runs
		a.Delivered = countDelivered(runs)
		return a
	}

	// A stored 1-run-per-arm result must not satisfy a 2-run request: the
	// runner is called again and the fresh 2-run result is saved over it.
	runner := &fakeRunner{arms: map[string]ArmResult{
		"control": arm("control", oneRun),
		"variant": arm("variant", oneRun),
	}}
	if _, reused, err := compareWithStore(context.Background(), controlVariant(), variantVariant(), runner, 1, dir); err != nil {
		t.Fatalf("seeding run: %v", err)
	} else if reused {
		t.Fatal("seeding run must not report reuse")
	}

	runner = &fakeRunner{arms: map[string]ArmResult{
		"control": arm("control", []RunResult{oneRun[0], oneRun[0]}),
		"variant": arm("variant", []RunResult{oneRun[0], oneRun[0]}),
	}}
	res, reused, err := compareWithStore(context.Background(), controlVariant(), variantVariant(), runner, 2, dir)
	if err != nil {
		t.Fatalf("short-store compareWithStore: %v", err)
	}
	if reused {
		t.Error("a 1-run stored result must not satisfy a 2-run request")
	}
	if got := len(runner.calls); got != 2 {
		t.Errorf("short store: runner called %d time(s), want 2 (fresh execution)", got)
	}
	if len(res.Control.Runs) != 2 || len(res.Variant.Runs) != 2 {
		t.Errorf("fresh result carries %d/%d runs, want 2/2", len(res.Control.Runs), len(res.Variant.Runs))
	}
	// The re-run overwrote the short store file, so the same request now reuses.
	again, reused, err := compareWithStore(context.Background(), controlVariant(), variantVariant(), runner, 2, dir)
	if err != nil {
		t.Fatalf("re-request compareWithStore: %v", err)
	}
	if !reused {
		t.Error("a 2-run stored result must be reused for a 2-run request")
	}
	if len(again.Control.Runs) != 2 || len(again.Variant.Runs) != 2 {
		t.Errorf("reused result carries %d/%d runs, want 2/2", len(again.Control.Runs), len(again.Variant.Runs))
	}

	// More runs stored than requested is still reuse.
	_, reused, err = compareWithStore(context.Background(), controlVariant(), variantVariant(), runner, 1, dir)
	if err != nil {
		t.Fatalf("fewer-runs compareWithStore: %v", err)
	}
	if !reused {
		t.Error("a 2-run stored result must satisfy a 1-run request")
	}
}

// TestAnonymize_DoesNotMutateContrasts pins the contrast deep copy: the
// shallow struct copy shares the Contrasts backing array with the input, and
// the direction rewrite must not leak into it.
func TestAnonymize_DoesNotMutateContrasts(t *testing.T) {
	runner := &fakeRunner{arms: map[string]ArmResult{
		"control": buildArm(armSpec{name: "control", steps: 10, calls: map[string]int{"read": 4}}),
		"variant": buildArm(armSpec{name: "variant", steps: 8, calls: map[string]int{"read": 6}}),
	}}
	res := mustCompare(t, controlVariant(), variantVariant(), runner, 5)
	before := make([]Contrast, len(res.Contrasts))
	copy(before, res.Contrasts)

	_ = Anonymize(res, 3)

	for i, c := range res.Contrasts {
		if c != before[i] {
			t.Errorf("Anonymize mutated input contrast %d: %+v -> %+v", i, before[i], c)
		}
	}
}

// TestBlindReport pins the blind-mode rendering: output stays a valid
// rendered report while the real arm names disappear from it, and the seed
// environment variable decides the A/B assignment deterministically.
// Not parallel: t.Setenv forbids it.
func TestBlindReport(t *testing.T) {
	res := mustCompare(t, controlVariant(), variantVariant(),
		&fakeRunner{arms: map[string]ArmResult{
			"control": buildArm(armSpec{name: "control", steps: 10, calls: map[string]int{"read": 4}}),
			"variant": buildArm(armSpec{name: "variant", steps: 8, calls: map[string]int{"read": 6}}),
		}}, 5)

	blind := blindReport(res)
	if !strings.Contains(blind, "# Harness comparison") {
		t.Error("blind report must still be a rendered report")
	}
	if strings.Contains(blind, "control") || strings.Contains(blind, "variant") {
		t.Errorf("blind report must not carry the real arm names:\n%s", blind)
	}

	// Deterministic in the seed: same env, same assignment.
	t.Setenv(blindSeedEnv, "42")
	first := blindReport(res)
	second := blindReport(res)
	if first != second {
		t.Error("same seed must render byte-identically")
	}
	if !strings.Contains(first, " A ") && !strings.Contains(first, "A (fingerprint") {
		t.Errorf("blind report must carry anonymized slot names:\n%s", first)
	}

	// A non-numeric seed falls back to the default with a note, not an error.
	t.Setenv(blindSeedEnv, "abc")
	if fallback := blindReport(res); fallback == "" {
		t.Error("non-numeric seed must fall back to the default, not break rendering")
	}
}
