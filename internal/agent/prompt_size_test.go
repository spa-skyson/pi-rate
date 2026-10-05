package agent

import (
	"fmt"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/spa-skyson/pi-rate/internal/config"
	"github.com/spa-skyson/pi-rate/internal/extension"
)

// repoRoot returns the repository root relative to this test file, so the
// measurement below reads the real AGENTS.md a production run in this repo
// would pick up, without depending on the process working directory.
func repoRoot(t *testing.T) string {
	t.Helper()
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	return filepath.Join(filepath.Dir(thisFile), "..", "..")
}

// fullMenu is the pre-compaction menu format: every skill carries its full
// frontmatter description. Kept only so the size report can print an honest
// before/after for the same skill list.
func fullMenu(skills []extension.Skill) string {
	var b strings.Builder
	b.WriteString("\n\n# Available Skills\n\n")
	for _, s := range skills {
		fmt.Fprintf(&b, "- /%s: %s\n", s.Name, s.Description)
	}
	return b.String()
}

// measure is a section size snapshot used by the prompt size report.
func measure(name, s string) (string, int, int, int) {
	lines := 0
	if trimmed := strings.TrimSpace(s); trimmed != "" {
		lines = strings.Count(trimmed, "\n") + 1
	}
	return name, len(s), lines, len(s) / 4
}

// TestSystemPromptSizeReport measures the system prompt a production run in
// this repository assembles, section by section, and prints an honest
// before/after for the skills menu. Log-only: it pins no budget (the menu
// budget lives in TestSkillsMenuBudget) but makes a size regression visible
// in test output.
//
// Excluded from the numbers (machine-dependent, small): the Runtime
// Environment preamble (~200 chars) and any global ~/.pirate/AGENTS.md.
func TestSystemPromptSizeReport(t *testing.T) {
	root := repoRoot(t)

	parts := loadInstructionPartsFrom(SystemInstruction, root, t.TempDir())
	skills, err := extension.LoadSkills()
	if err != nil {
		t.Fatalf("LoadSkills() error: %v", err)
	}
	compact := appendSkillsMenu(skills)
	legacy := fullMenu(skills)

	for _, m := range []struct {
		s string
		v string
	}{
		{"Base (SystemInstruction)", parts.Base},
		{"Rules (repo AGENTS.md)", parts.Rules},
		{"Skills menu (before)", legacy},
		{"Skills menu (after)", compact},
	} {
		name, chars, lines, tokens := measure(m.s, m.v)
		t.Logf("%-28s %7d chars %5d lines ~%5d tokens", name, chars, lines, tokens)
	}
	name, chars, lines, tokens := measure("TOTAL-before", parts.Base+parts.Rules+legacy)
	t.Logf("%-28s %7d chars %5d lines ~%5d tokens", name, chars, lines, tokens)
	name, chars, lines, tokens = measure("TOTAL-after", parts.Base+parts.Rules+compact)
	t.Logf("%-28s %7d chars %5d lines ~%5d tokens", name, chars, lines, tokens)
	t.Logf("skills menu: %d -> %d chars (%.0f%% smaller), %d skills",
		len(legacy), len(compact), 100*(1-float64(len(compact))/float64(len(legacy))), len(skills))

	if len(skills) == 0 {
		t.Fatal("expected bundled skills to be discovered")
	}
	if !strings.Contains(compact, "# Available Skills") {
		t.Fatal("compact menu lost its header")
	}

	// Machine-realistic variant: the full production skill list (bundled +
	// user + project dirs resolved from the repo root). Log-only because the
	// exact set depends on the machine's ~/.pirate and ~/.claude contents.
	if all, err := extension.LoadSkills(extension.DefaultSkillDirsIn(root, config.Config{})...); err == nil && len(all) > len(skills) {
		name, chars, lines, tokens := measure("Skills ALL dirs (before)", fullMenu(all))
		t.Logf("%-28s %7d chars %5d lines ~%5d tokens (%d skills)", name, chars, lines, tokens, len(all))
		name, chars, lines, tokens = measure("Skills ALL dirs (after)", appendSkillsMenu(all))
		t.Logf("%-28s %7d chars %5d lines ~%5d tokens (%d skills)", name, chars, lines, tokens, len(all))
	}
}

// menuEntry renders one menu line the way appendSkillsMenu does, so the
// budget below tracks the production renderer instead of duplicating it.
func menuEntry(name, desc string) string {
	return fmt.Sprintf("- /%s: %s\n", name, desc)
}

// TestSkillsMenuBudget is the anti-bloat guard: the menu exists for skill
// discovery, so every skill must render as exactly one bounded line. Before
// the compaction a multi-sentence SKILL.md frontmatter description inflated
// the system prompt by hundreds of tokens per skill with no discovery
// benefit, silently. This fails loudly instead.
func TestSkillsMenuBudget(t *testing.T) {
	skills, err := extension.LoadSkills()
	if err != nil {
		t.Fatalf("LoadSkills() error: %v", err)
	}
	if len(skills) == 0 {
		t.Fatal("LoadSkills() returned no bundled skills")
	}

	menu := appendSkillsMenu(skills)
	const header = "\n\n# Available Skills\n\n"
	if !strings.HasPrefix(menu, header) {
		t.Fatalf("menu must start with %q, got %q", header, menu[:20])
	}
	body := menu[len(header):]

	// Every skill contributes exactly one line: no embedded newlines, so a
	// multi-paragraph frontmatter description cannot smuggle extra lines in.
	if gotLines := strings.Count(body, "\n"); gotLines != len(skills) {
		t.Errorf("skills menu has %d lines for %d skills; each skill must render as exactly one line (full descriptions belong in /skills and the Active Skill body)", gotLines, len(skills))
	}

	// Each rendered entry stays under the description cap plus room for the
	// "- /name: " prefix, so even a long skill name cannot blow the cap.
	for _, s := range skills {
		entry := menuEntry(s.Name, skillMenuDescription(s.Description))
		if max := maxSkillMenuDesc + len(s.Name) + 12; len(entry) > max {
			t.Errorf("menu entry for /%s is %d chars, want <= %d: %q", s.Name, len(entry), max, entry)
		}
	}

	// Whole-menu budget for the bundled set, derived from the per-skill cap
	// rather than a hardcoded constant, so adding a skill scales the budget
	// and only a genuinely verbose entry can fail it.
	budget := len(header) + len(skills)*(maxSkillMenuDesc+16)
	if len(menu) > budget {
		t.Errorf("skills menu is %d chars, want <= %d (%d skills × %d-char description cap + header)", len(menu), budget, len(skills), maxSkillMenuDesc)
	}
}

// TestSkillMenuDescriptionTruncates pins the description reduction rules the
// budget above relies on: first line, first sentence, cap with ellipsis.
func TestSkillMenuDescriptionTruncates(t *testing.T) {
	tests := []struct {
		name string
		desc string
		want string
	}{
		{"single short sentence kept whole", "Run lint checks before finishing.", "Run lint checks before finishing."},
		{"first sentence only", "First sentence. Second sentence. Third.", "First sentence."},
		{"first line only", "First line\nSecond line. More.", "First line"},
		{
			"long first sentence capped at word boundary",
			strings.Repeat("word ", 30) + ". Trailing.",
			strings.Repeat("word ", 14) + "word…", // 75 runes, no trailing space before the ellipsis
		},
		{"empty stays empty", "", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := skillMenuDescription(tt.desc); got != tt.want {
				t.Errorf("skillMenuDescription(%q) = %q, want %q", tt.desc, got, tt.want)
			}
		})
	}
}
