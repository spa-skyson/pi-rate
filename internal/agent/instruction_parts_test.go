package agent

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/spa-skyson/pi-rate/internal/logger"
	"github.com/spa-skyson/pi-rate/internal/testenv"
)

// TestInstructionPartsOverrideRulesFile is the injection contract's core
// promise: the same working directory, two agents with different supplied
// InstructionParts, two different prompts — and the file-backed Rules the
// directory carries never reach the prompt. Everything happens without a
// single write to the context files on disk.
func TestInstructionPartsOverrideRulesFile(t *testing.T) {
	testenv.SetHome(t, t.TempDir())

	// A working directory with an AGENTS.md whose marker must NOT appear in
	// any injected prompt.
	cwd := t.TempDir()
	const fileMarker = "FILE-BACKED-RULES-MARKER"
	if err := os.WriteFile(filepath.Join(cwd, "AGENTS.md"), []byte("# Project Rules\n\n"+fileMarker+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}

	variants := []struct {
		name         string
		parts        InstructionParts
		wantInPrompt string
	}{
		{
			name: "variant A",
			parts: InstructionParts{
				Base:   "BASE A",
				Rules:  rulesSectionMarker + "variant-a rules",
				Skills: skillsSectionMarker + "- /alpha-a: A skill.\n",
			},
			wantInPrompt: "variant-a rules",
		},
		{
			name: "variant B",
			parts: InstructionParts{
				Base:   "BASE B",
				Rules:  rulesSectionMarker + "variant-b rules",
				Skills: skillsSectionMarker + "- /alpha-b: B skill.\n",
			},
			wantInPrompt: "variant-b rules",
		},
	}

	prompts := make([]string, len(variants))
	for i, v := range variants {
		llm := &capturingLLM{name: "m", response: "ok"}
		a, err := New(Config{
			Model:            llm,
			Instruction:      v.parts.Base,
			InstructionParts: &v.parts,
			WorkingDir:       cwd,
		})
		if err != nil {
			t.Fatalf("%s: New() error: %v", v.name, err)
		}
		if err := runForPrompt(t, a, llm); err != nil {
			t.Fatalf("%s: %v", v.name, err)
		}
		prompts[i] = llm.capturedSystemInstruction()

		if !strings.Contains(prompts[i], v.wantInPrompt) {
			t.Errorf("%s: prompt must carry the supplied Rules, got %q", v.name, prompts[i])
		}
		if strings.Contains(prompts[i], fileMarker) {
			t.Errorf("%s: prompt must not carry the file-backed Rules, got %q", v.name, prompts[i])
		}
	}

	if prompts[0] == prompts[1] {
		t.Errorf("two supplied parts for one cwd must yield two different prompts, both are %q", prompts[0])
	}
	// The ordering invariant survives injection: the marker of the earlier
	// section sits before the marker of the later one.
	if i := strings.Index(prompts[0], rulesSectionMarker); i >= 0 {
		if j := strings.Index(prompts[0], skillsSectionMarker); j >= 0 && j < i {
			t.Errorf("Skills section precedes Rules in the assembled prompt: %q", prompts[0])
		}
	}
}

// TestInstructionPartsEnvAlwaysRebuilt pins the Env rule: a caller-supplied
// Env is dropped, the "# Runtime Environment" block is recomposed from the
// live process environment and stays the last block of the prompt.
func TestInstructionPartsEnvAlwaysRebuilt(t *testing.T) {
	testenv.SetHome(t, t.TempDir())
	cwd := t.TempDir()

	const callerEnv = "CALLER-SUPPLIED-ENV-MARKER"
	parts := InstructionParts{
		Base:   "BASE",
		Rules:  rulesSectionMarker + "rules",
		Skills: skillsSectionMarker + "- /skill: A skill.\n",
		Env:    "\n\n# Runtime Environment\n\n" + callerEnv + "\n",
	}

	llm := &capturingLLM{name: "m", response: "ok"}
	a, err := New(Config{
		Model:            llm,
		Instruction:      parts.Base,
		InstructionParts: &parts,
		WorkingDir:       cwd,
	})
	if err != nil {
		t.Fatalf("New() error: %v", err)
	}
	if err := runForPrompt(t, a, llm); err != nil {
		t.Fatal(err)
	}
	got := llm.capturedSystemInstruction()

	if strings.Contains(got, callerEnv) {
		t.Errorf("supplied Env must never reach the prompt, got %q", got)
	}
	if !strings.Contains(got, envSectionMarker) {
		t.Errorf("recomposed Runtime Environment block missing from prompt: %q", got)
	}
	// Env is last: the environment block's content outlives every other
	// section in the assembled string.
	envStart := strings.Index(got, envSectionMarker)
	if tail := got[envStart+len(envSectionMarker):]; strings.Contains(tail, skillsSectionMarker) || strings.Contains(tail, rulesSectionMarker) {
		t.Errorf("Runtime Environment block is not the last section: %q", got)
	}
	if !strings.Contains(got, "- HOME=") {
		t.Errorf("recomposed Env block lacks the environment entries: %q", got)
	}
}

// TestInstructionPartsGateRejectsSuppliedOverrun runs the 32K total gate over
// injected parts: the build fails naming the section that is itself over the
// budget, and nothing lands on the agent when it does.
func TestInstructionPartsGateRejectsSuppliedOverrun(t *testing.T) {
	testenv.SetHome(t, t.TempDir())

	huge := strings.Repeat("r", maxInstructionTotal+1)
	parts := InstructionParts{
		Base:   "BASE",
		Rules:  rulesSectionMarker + huge,
		Skills: skillsSectionMarker + "- /skill: A skill.\n",
	}

	a, err := New(Config{
		Model:            &mockLLM{name: "m", response: "ok"},
		Instruction:      parts.Base,
		InstructionParts: &parts,
		WorkingDir:       t.TempDir(),
	})
	if err == nil {
		t.Fatal("New() with an over-budget injected prompt must fail")
	}
	if a != nil {
		t.Errorf("a failed build must not return an agent, got %v", a)
	}
	if !strings.Contains(err.Error(), "Rules=") {
		t.Errorf("gate error must name the oversized Rules section: %v", err)
	}

	// The same gate guards the rebuild path.
	a2, err := New(Config{
		Model:       &mockLLM{name: "m", response: "ok"},
		Instruction: "small",
	})
	if err != nil {
		t.Fatalf("New() error: %v", err)
	}
	if err := a2.RebuildWithParts(parts); err == nil {
		t.Fatal("RebuildWithParts() with an over-budget assembly must fail")
	} else if !strings.Contains(err.Error(), "Rules=") {
		t.Errorf("gate error must name the oversized Rules section: %v", err)
	}
}

// TestNilInstructionPartsKeepsFileAssembly pins the nil contract: without the
// field the agent assembles the prompt from disk exactly as before — the
// file-backed Rules and the skills menu are present, and the string matches
// LoadInstructionPartsFor for the same directory (modulo the cwd line New
// appends to its base).
func TestNilInstructionPartsKeepsFileAssembly(t *testing.T) {
	testenv.SetHome(t, t.TempDir())

	cwd := t.TempDir()
	const fileMarker = "NIL-PATH-FILE-RULES"
	if err := os.WriteFile(filepath.Join(cwd, "AGENTS.md"), []byte("# Project Rules\n\n"+fileMarker+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}

	llm := &capturingLLM{name: "m", response: "ok"}
	a, err := New(Config{
		Model:       llm,
		Instruction: "BASE",
		WorkingDir:  cwd,
	})
	if err != nil {
		t.Fatalf("New() error: %v", err)
	}
	if err := runForPrompt(t, a, llm); err != nil {
		t.Fatal(err)
	}
	got := llm.capturedSystemInstruction()

	// The nil-field contract, byte-pinned: the prompt is the base with the
	// cwd line appended — exactly what New built before the field existed.
	// Context-file reads and the env block belong to the callers of New
	// (the CLI and piagent assemble via LoadInstruction*), so a directory
	// with an AGENTS.md must NOT leak its rules into this prompt, and the
	// runtime adendum the instruction processor appends is all that follows.
	want := "BASE\nCurrent working directory: " + cwd + "\n"
	if !strings.HasPrefix(got, want) {
		t.Errorf("nil-field prompt drifted from the pre-injection assembly:\n got %q\nwant prefix %q", got, want)
	}
	if strings.Contains(got, fileMarker) {
		t.Errorf("nil-field path must not pull file-backed Rules in (that is the callers' job), got %q", got)
	}
	if !strings.Contains(got, "You are an agent") {
		t.Errorf("nil-field path must keep the runtime instruction adendum, got %q", got)
	}
}

// TestRebuildWithPartsReplacesPrompt drives the rebuild path end to end: two
// RebuildWithParts calls on one agent produce the two supplied prompts at the
// runner, and the config the agent carries follows the last one.
func TestRebuildWithPartsReplacesPrompt(t *testing.T) {
	testenv.SetHome(t, t.TempDir())
	cwd := t.TempDir()

	llm := &capturingLLM{name: "m", response: "ok"}
	a, err := New(Config{
		Model:       llm,
		Instruction: "BASE",
		WorkingDir:  cwd,
	})
	if err != nil {
		t.Fatalf("New() error: %v", err)
	}

	first := InstructionParts{
		Base:   "INJECTED BASE ONE",
		Rules:  rulesSectionMarker + "first injected rules",
		Skills: skillsSectionMarker + "- /one: First skill.\n",
	}
	if err := a.RebuildWithParts(first); err != nil {
		t.Fatalf("RebuildWithParts() error: %v", err)
	}
	if err := runForPrompt(t, a, llm); err != nil {
		t.Fatal(err)
	}
	got := llm.capturedSystemInstruction()
	if !strings.Contains(got, "first injected rules") {
		t.Errorf("rebuild must switch the prompt to the supplied parts, got %q", got)
	}
	if strings.Contains(got, envSectionMarker+"first injected rules") {
		t.Errorf("supplied Env leaked into the rebuilt prompt: %q", got)
	}

	second := InstructionParts{
		Base:   "INJECTED BASE TWO",
		Rules:  rulesSectionMarker + "second injected rules",
		Skills: skillsSectionMarker + "- /two: Second skill.\n",
	}
	if err := a.RebuildWithParts(second); err != nil {
		t.Fatalf("RebuildWithParts() error: %v", err)
	}
	if err := runForPrompt(t, a, llm); err != nil {
		t.Fatal(err)
	}
	got = llm.capturedSystemInstruction()
	if !strings.Contains(got, "second injected rules") || strings.Contains(got, "first injected rules") {
		t.Errorf("second rebuild must fully replace the first assembly, got %q", got)
	}
	if a.config.InstructionParts == nil || a.config.InstructionParts.Base != "INJECTED BASE TWO" {
		t.Errorf("RebuildWithParts must store the last supplied parts on the config: %+v", a.config.InstructionParts)
	}
}

// TestRebuildWithPartsRunsInventoryAndHygiene mirrors the
// RebuildWithInstruction guarantees on the parts path: the context inventory
// line lands in the session log describing the new prompt, and the skill-menu
// hygiene warns for an on-disk skill without an activation marker — deduped
// on repeat.
func TestRebuildWithPartsRunsInventoryAndHygiene(t *testing.T) {
	testenv.SetHome(t, t.TempDir())

	// One skill whose description carries no activation marker, isolated via
	// PI_SKILLS_DIRS so no repo skill leaks into the notice count.
	skillDir := t.TempDir()
	skillRoot := filepath.Join(skillDir, "mute-skill")
	if err := os.MkdirAll(skillRoot, 0o755); err != nil {
		t.Fatal(err)
	}
	body := "---\nname: mute-skill\ndescription: Does things quietly and well.\n---\nbody\n"
	if err := os.WriteFile(filepath.Join(skillRoot, "SKILL.md"), []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PI_SKILLS_DIRS", skillDir)

	log, err := logger.New()
	if err != nil {
		t.Fatalf("logger.New() error: %v", err)
	}
	t.Cleanup(func() { _ = log.Close() })

	notices := collectNotices(t)

	a, err := New(Config{
		Model:       &mockLLM{name: "m", response: "ok"},
		Instruction: "base",
		WorkingDir:  t.TempDir(),
		Logger:      log,
	})
	if err != nil {
		t.Fatalf("New() error: %v", err)
	}
	if _, _, err := a.CreateSession(t.Context()); err != nil {
		t.Fatalf("CreateSession() error: %v", err)
	}
	// New ran the hygiene pass already: the mute skill warned exactly once…
	if hits := countMsgs(notices(), "skill mute-skill:"); hits != 1 {
		t.Fatalf("after New: got %d warnings, want exactly 1: %q", hits, notices())
	}

	injected := InstructionParts{
		Base:   "injected base",
		Rules:  rulesSectionMarker + "injected rules",
		Skills: skillsSectionMarker + "- /one: Injected skill.\n",
		Env:    "supplied env that must be ignored",
	}
	if err := a.RebuildWithParts(injected); err != nil {
		t.Fatalf("RebuildWithParts() error: %v", err)
	}

	// …and the rebuild over the same on-disk set stays silent (dedup), while
	// the inventory line describes the injected prompt.
	if hits := countMsgs(notices(), "skill mute-skill:"); hits != 1 {
		t.Errorf("unchanged rebuild must stay deduplicated, got %d warnings: %q", hits, notices())
	}

	data, err := os.ReadFile(log.Path())
	if err != nil {
		t.Fatalf("reading session log: %v", err)
	}
	entries := inventoryEntries(string(data))
	if len(entries) < 2 {
		t.Fatalf("RebuildWithParts must write a context inventory line, got %d entries in log:\n%s", len(entries), data)
	}
	last := entries[len(entries)-1]
	if last.base < len("injected base") {
		t.Errorf("rebuild inventory base=%d, want >= %d (the injected base)", last.base, len("injected base"))
	}

	// A NEW misauthored skill appearing on disk after New is warned by the
	// rebuild's hygiene pass — exactly as RebuildWithInstruction does.
	secondRoot := filepath.Join(skillDir, "mute-two")
	if err := os.MkdirAll(secondRoot, 0o755); err != nil {
		t.Fatal(err)
	}
	body2 := "---\nname: mute-two\ndescription: Also has no trigger phrase.\n---\nbody\n"
	if err := os.WriteFile(filepath.Join(secondRoot, "SKILL.md"), []byte(body2), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := a.RebuildWithParts(InstructionParts{Base: "second injected", Rules: rulesSectionMarker + "r", Skills: skillsSectionMarker + "- /two: S.\n"}); err != nil {
		t.Fatalf("second RebuildWithParts() error: %v", err)
	}
	if hits := countMsgs(notices(), "skill mute-two:"); hits != 1 {
		t.Errorf("rebuild must warn for the new skill exactly once, got %d: %q", hits, notices())
	}
}

// runForPrompt runs one turn against the agent so the capturing LLM records
// the system instruction the runner actually sent.
func runForPrompt(t *testing.T, a *Agent, llm *capturingLLM) error {
	t.Helper()
	ctx := context.Background()
	sessionID, _, err := a.CreateSession(ctx)
	if err != nil {
		return err
	}
	_ = llm
	for _, err := range a.Run(ctx, sessionID, "hello") {
		if err != nil {
			return err
		}
	}
	return nil
}

// inventoryEntries parses every context inventory line out of a session log,
// in order — the shared shape of the inventory assertions here.
func inventoryEntries(logText string) []inventoryEntry {
	var entries []inventoryEntry
	for _, l := range strings.Split(logText, "\n") {
		if !strings.Contains(l, "context inventory:") {
			continue
		}
		if e, ok := findInventoryEntry(l); ok {
			entries = append(entries, e)
		}
	}
	return entries
}
