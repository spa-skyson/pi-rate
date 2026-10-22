package agent

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/spa-skyson/pi-rate/internal/extension"
	"github.com/spa-skyson/pi-rate/internal/logger"
	"github.com/spa-skyson/pi-rate/internal/notice"
	"github.com/spa-skyson/pi-rate/internal/testenv"
)

// TestInstructionBudgetOverFailsWithAllSections builds a prompt whose total
// exceeds maxInstructionTotal through a single oversized Rules block, and a
// second variant that spreads the overage across Base and Rules. Both must
// fail, and the error must name every section that is itself over the budget.
func TestInstructionBudgetOverFailsWithAllSections(t *testing.T) {
	hugeRules := strings.Repeat("r", maxInstructionTotal+1)

	t.Run("single oversized section is named", func(t *testing.T) {
		parts := InstructionParts{Base: SystemInstruction, Rules: hugeRules}
		err := checkInstructionTotal(parts.String(), &parts)
		if err == nil {
			t.Fatal("expected the total-budget gate to fail, got nil")
		}
		if !strings.Contains(err.Error(), "Rules=") {
			t.Errorf("error must name the oversized Rules section: %v", err)
		}
		if strings.Contains(err.Error(), "Base=") {
			t.Errorf("Base is under the budget and must not be named: %v", err)
		}
	})

	t.Run("every oversized section is named", func(t *testing.T) {
		parts := InstructionParts{
			Base:   strings.Repeat("b", maxInstructionTotal+1),
			Rules:  strings.Repeat("r", maxInstructionTotal+2),
			Skills: strings.Repeat("s", maxInstructionTotal+3),
		}
		err := checkInstructionTotal(parts.String(), &parts)
		if err == nil {
			t.Fatal("expected the total-budget gate to fail, got nil")
		}
		for _, want := range []string{"Base=", "Rules=", "Skills="} {
			if !strings.Contains(err.Error(), want) {
				t.Errorf("error must name %s: %v", want, err)
			}
		}
	})

	t.Run("over the total but no single section reports the spread", func(t *testing.T) {
		parts := InstructionParts{
			Base:  strings.Repeat("b", maxInstructionTotal-1),
			Rules: strings.Repeat("r", maxInstructionTotal-1),
			Env:   environmentContext("/home/t", "t", "/w", "", "/w"),
		}
		err := checkInstructionTotal(parts.String(), &parts)
		if err == nil {
			t.Fatal("expected the total-budget gate to fail, got nil")
		}
		if strings.Contains(err.Error(), "Base=") || strings.Contains(err.Error(), "Rules=") {
			t.Errorf("no section is individually over; none may be named: %v", err)
		}
		if !strings.Contains(err.Error(), "spread") {
			t.Errorf("the spread case must say so: %v", err)
		}
	})
}

// TestInstructionBudgetUnderPasses keeps the gate open for a prompt at and
// just below the budget, and recovers the section sizes from the assembled
// string the way a caller without InstructionParts would.
func TestInstructionBudgetUnderPasses(t *testing.T) {
	parts := InstructionParts{
		Base:   strings.Repeat("b", maxInstructionTotal-56),
		Rules:  rulesSectionMarker + "small",
		Skills: skillsSectionMarker + "small",
	}
	if len(parts.String()) != maxInstructionTotal-5 {
		t.Fatalf("test prompt is %d chars, want %d", len(parts.String()), maxInstructionTotal-5)
	}
	if err := checkInstructionTotal(parts.String(), &parts); err != nil {
		t.Fatalf("prompt at %d chars must pass: %v", len(parts.String()), err)
	}

	recovered := instructionOnlyParts(parts.String())
	if recovered.Base != parts.Base {
		t.Errorf("recovered Base length = %d, want %d", len(recovered.Base), len(parts.Base))
	}
	if recovered.Rules != parts.Rules {
		t.Errorf("recovered Rules = %q, want %q", recovered.Rules, parts.Rules)
	}
}

// TestInstructionBudgetGateRejectsAgentBuild runs the gate end to end: an
// agent assembled from a prompt over the budget fails to build, a normal one
// still builds.
func TestInstructionBudgetGateRejectsAgentBuild(t *testing.T) {
	over := strings.Repeat("x", maxInstructionTotal+1)
	if _, err := New(Config{Instruction: over}); err == nil {
		t.Fatal("New() with an over-budget instruction must fail")
	}
	if _, err := New(Config{Instruction: "small enough"}); err != nil {
		t.Fatalf("New() with a normal instruction must pass: %v", err)
	}
}

// TestDefaultPromptUnderBudget measures the prompt a production run in this
// repository assembles and asserts the default block passes the always-on
// gate — the same measurement TestSystemPromptSizeReport prints, held against
// maxInstructionTotal.
func TestDefaultPromptUnderBudget(t *testing.T) {
	root := repoRoot(t)
	parts := loadInstructionPartsFrom(SystemInstruction, root, t.TempDir())
	prompt := parts.String()
	if len(prompt) > maxInstructionTotal {
		t.Fatalf("default system prompt is %d chars, over the %d gate; shrink it before raising the limit", len(prompt), maxInstructionTotal)
	}
	if err := checkInstructionTotal(prompt, &parts); err != nil {
		t.Fatalf("default prompt must pass the gate: %v", err)
	}
}

// collectNotices swaps the notice sink for a recorder and restores it.
func collectNotices(t *testing.T) func() []string {
	t.Helper()
	var mu sync.Mutex
	var raised []string
	prev := notice.SetSink(func(msg string) {
		mu.Lock()
		raised = append(raised, msg)
		mu.Unlock()
	})
	t.Cleanup(func() { notice.SetSink(prev) })
	return func() []string {
		mu.Lock()
		defer mu.Unlock()
		return append([]string(nil), raised...)
	}
}

// resetWarnedSkills clears the once-per-skill dedup between tests. Safe to
// install as a t.Cleanup as well as call directly.
func resetWarnedSkills(t *testing.T) {
	t.Helper()
	clear := func() {
		warnedSkills.Range(func(key, _ any) bool {
			warnedSkills.Delete(key)
			return true
		})
	}
	clear()
	t.Cleanup(clear)
}

// TestSkillTriggerHygiene raises exactly one warning for a skill whose
// description carries no activation marker, none for one that does, and
// leaves the menu byte-identical either way.
func TestSkillTriggerHygiene(t *testing.T) {
	resetWarnedSkills(t)
	notices := collectNotices(t)

	noTrigger := extension.Skill{Name: "mute-skill", Description: "Does things quickly and well."}
	withTrigger := extension.Skill{Name: "loud-skill", Description: "Use when the user asks for noise. Also more detail."}

	menu := appendSkillsMenu([]extension.Skill{noTrigger, withTrigger})

	raised := notices()
	for _, msg := range raised {
		if strings.Contains(msg, "skill loud-skill:") {
			t.Errorf("skill with an activation marker must not be warned: %s", msg)
		}
	}
	if hits := countMsgs(raised, "skill mute-skill:"); hits != 1 {
		t.Errorf("got %d warnings for mute-skill, want exactly 1: %q", hits, raised)
	}

	// A second assembly of the same menu is the every-turn case: still one
	// warning per skill per session, not one per rebuild.
	appendSkillsMenu([]extension.Skill{noTrigger, withTrigger})
	if hits := countMsgs(notices(), "skill mute-skill:"); hits != 1 {
		t.Errorf("got %d warnings after a rebuild, want exactly 1", hits)
	}

	// Hygiene observes only: the menu content and order are unchanged.
	want := "\n\n# Available Skills\n\n- /mute-skill: Does things quickly and well.\n- /loud-skill: Use when the user asks for noise.\n"
	if menu != want {
		t.Errorf("menu changed:\n got %q\nwant %q", menu, want)
	}
}

// TestSanitizedSkillNameInNotice feeds a skill whose name carries an ESC/OSC
// escape sequence and is longer than 64 runes, and asserts the notice carries
// a clean, length-capped name while a valid name passes through unchanged.
func TestSanitizedSkillNameInNotice(t *testing.T) {
	resetWarnedSkills(t)
	notices := collectNotices(t)

	// ESC survives the audit scanner's ASCII fast path; the ']' OSC payload
	// is ASCII and by design stays in the menu name, so the notice test
	// keys on the sanitized remainder of the name.
	dirty := "\x1b]2;pwned\x07evil-\x1b[31mred\n\r\tname"
	appendSkillsMenu([]extension.Skill{{Name: dirty, Description: "No trigger here."}})

	raised := notices()
	if len(raised) != 1 {
		t.Fatalf("got %d notices, want 1: %q", len(raised), raised)
	}
	got := raised[0]
	if strings.ContainsAny(got, "\x1b\x07\n\r\t") {
		t.Errorf("notice carries control characters: %q", got)
	}
	for _, r := range got {
		if r <= 0x1F || (r >= 0x7F && r <= 0x9F) {
			t.Errorf("notice carries control rune %q: %q", r, got)
		}
	}
	if !strings.Contains(got, "skill ]2;pwnedevil-[31mredname:") {
		t.Errorf("sanitized name missing from notice: %q", got)
	}

	// A name with nothing to strip passes through byte-for-byte.
	// A name with nothing to strip passes through byte-for-byte. The dirty
	// notice stays in the recorder (dedup is by raw name), so read the last
	// one — the second appendSkillsMenu raised exactly one more.
	resetWarnedSkills(t)
	const plain = "perfectly-fine-skill"
	appendSkillsMenu([]extension.Skill{{Name: plain, Description: "No trigger here either."}})
	raised = notices()
	if n := len(raised); n != 2 || !strings.Contains(raised[n-1], "skill "+plain+":") {
		t.Errorf("valid name mangled: %q", raised)
	}
}

// TestLongSkillNameTruncatedInNotice caps a long name at maxSkillNameRunes
// runes with an ellipsis, so one giant frontmatter name cannot flood the
// notice area.
func TestLongSkillNameTruncatedInNotice(t *testing.T) {
	resetWarnedSkills(t)
	notices := collectNotices(t)

	appendSkillsMenu([]extension.Skill{{
		Name:        strings.Repeat("ы", maxSkillNameRunes+20),
		Description: "No trigger here.",
	}})

	raised := notices()
	if len(raised) != 1 {
		t.Fatalf("got %d notices, want 1: %q", len(raised), raised)
	}
	prefix := "skill "
	suffix := ": в description нет условия активации"
	i := strings.Index(raised[0], prefix) + len(prefix)
	j := strings.Index(raised[0], suffix)
	if j < 0 {
		t.Fatalf("unexpected notice shape: %q", raised[0])
	}
	name := raised[0][i:j]
	runes := []rune(name)
	if len(runes) != maxSkillNameRunes+1 || !strings.HasSuffix(name, "…") {
		t.Errorf("name not capped at %d runes + ellipsis: %d runes, %q", maxSkillNameRunes, len(runes), name)
	}
}

// TestRussianTriggerMarkers accepts the Russian activation markers the same
// way: a description that only says "используй когда" or "если" is fine.
func TestRussianTriggerMarkers(t *testing.T) {
	resetWarnedSkills(t)
	notices := collectNotices(t)

	appendSkillsMenu([]extension.Skill{
		{Name: "ru-when", Description: "Используй когда нужно планировать."},
		{Name: "ru-if", Description: "Помогает, если задача объёмная."},
	})

	for _, msg := range notices() {
		if strings.Contains(msg, "skill ru-when:") || strings.Contains(msg, "skill ru-if:") {
			t.Errorf("Russian marker missed: %s", msg)
		}
	}
}

func countMsgs(ss []string, sub string) int {
	n := 0
	for _, s := range ss {
		if strings.Contains(s, sub) {
			n++
		}
	}
	return n
}

// TestContextInventoryFormat pins the line format, so a consumer can grep it
// without parsing JSON. menu= measures the skills menu line itself — the
// skills span up to the env block — while skills= stays the raw span.
func TestContextInventoryFormat(t *testing.T) {
	parts := InstructionParts{
		Base:   "12345",
		Rules:  "678",
		Skills: "90",
		Env:    environmentContext("/home/test", "test", "/w", "", "/w"),
	}
	prompt := parts.String()
	got := contextInventory(parts, prompt, 42)
	menu := len(parts.Skills)
	want := fmt.Sprintf(
		"context inventory: base=5 rules=3 skills=2 menu=%d total=%d tools=42",
		menu, len(prompt))
	if got != want {
		t.Errorf("contextInventory = %q, want %q", got, want)
	}
	if !strings.Contains(got, " menu=") {
		t.Errorf("inventory lost the menu field: %q", got)
	}
}

// TestContextInventoryInSessionLog builds an agent with a session logger and
// asserts the inventory line lands in it; the numbers must match the prompt
// the agent was built on.
func TestContextInventoryInSessionLog(t *testing.T) {
	testenv.SetHome(t, t.TempDir())

	log, err := logger.New()
	if err != nil {
		t.Fatalf("logger.New() error: %v", err)
	}
	t.Cleanup(func() { _ = log.Close() })

	const base = "base prompt for inventory test"
	a, err := New(Config{
		Model:       &mockLLM{name: "test-model", response: "ok"},
		Instruction: base,
		WorkingDir:  t.TempDir(),
		Logger:      log,
	})
	if err != nil {
		t.Fatalf("New() error: %v", err)
	}
	if _, _, err := a.CreateSession(t.Context()); err != nil {
		t.Fatalf("CreateSession() error: %v", err)
	}

	data, err := os.ReadFile(log.Path())
	if err != nil {
		t.Fatalf("reading session log: %v", err)
	}

	entry, ok := findInventoryEntry(string(data))
	if !ok {
		t.Fatalf("context inventory line missing from session log:\n%s", data)
	}

	// The prompt New() assembled is base + cwd line + env block, so the total
	// must exceed the caller's base, and base= must carry at least it.
	if entry.total <= len(base) {
		t.Errorf("inventory total=%d, want > %d (the assembled prompt)", entry.total, len(base))
	}
	if entry.base < len(base) {
		t.Errorf("inventory base=%d, want >= %d (the base section)", entry.base, len(base))
	}
}

// inventoryEntry is the parsed inventory line.
type inventoryEntry struct {
	raw    string
	base   int
	rules  int
	skills int
	menu   int
	total  int
	tools  int
}

// findInventoryEntry scans a session log's JSON entries for the inventory
// line and parses its numbers back out.
func findInventoryEntry(logText string) (inventoryEntry, bool) {
	for _, l := range strings.Split(logText, "\n") {
		if !strings.Contains(l, "context inventory:") {
			continue
		}
		var rec struct {
			Content string `json:"content"`
		}
		if err := json.Unmarshal([]byte(l), &rec); err != nil {
			continue
		}
		var ie inventoryEntry
		ie.raw = rec.Content
		fields := map[string]*int{
			"base":   &ie.base,
			"rules":  &ie.rules,
			"skills": &ie.skills,
			"menu":   &ie.menu,
			"total":  &ie.total,
			"tools":  &ie.tools,
		}
		ok := true
		for name, dst := range fields {
			v, err := fieldAfter(rec.Content, name+"=")
			if err != nil {
				ok = false
				break
			}
			*dst = v
		}
		if ok {
			return ie, true
		}
	}
	return inventoryEntry{}, false
}

// fieldAfter parses the integer that follows prefix in s.
func fieldAfter(s, prefix string) (int, error) {
	i := strings.Index(s, prefix)
	if i < 0 {
		return 0, strconv.ErrSyntax
	}
	rest := s[i+len(prefix):]
	end := strings.IndexFunc(rest, func(r rune) bool { return r < '0' || r > '9' })
	if end < 0 {
		end = len(rest)
	}
	return strconv.Atoi(rest[:end])
}

// TestInventoryPartsSumMatchesPrompt asserts the recovered sections of a
// fully marker-structured prompt reconcile to the full prompt length, so the
// numbers in the log line add up against what the model receives.
func TestInventoryPartsSumMatchesPrompt(t *testing.T) {
	parts := InstructionParts{
		Base:   "BASE",
		Rules:  rulesSectionMarker + "RULES",
		Skills: skillsSectionMarker + "SKILLS",
		Env:    environmentContext("/home/test", "test", "/w", "", "/w"),
	}
	prompt := parts.String()
	got := instructionOnlyParts(prompt)
	// Skills carries everything from its marker on, Env included, so the
	// recovered sections always sum to the whole prompt.
	if sum := len(got.Base) + len(got.Rules) + len(got.Skills); sum != len(prompt) {
		t.Errorf("recovered sections sum to %d, want %d (full prompt)", sum, len(prompt))
	}
}
