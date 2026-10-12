package agent

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// largeAgentsFixture builds a markdown file over the inline threshold
// (maxRulesFileBudget): 400 fenced-delimited sections, >40KB, with bash
// fences whose `#` lines must not be mistaken for headings.
func largeAgentsFixture() string {
	var b strings.Builder
	for i := 0; i < 400; i++ {
		fmt.Fprintf(&b, "## Section %d\n\nRule line %d with enough words to look like real guidance.\n\n```bash\n# shell comment inside a fence, not a heading\necho %d\n```\n\n", i, i, i)
	}
	return b.String()
}

// TestLargeContextFileCompactsToOutline is the size guard from issue #67: a
// directory with a large AGENTS.md must assemble a system prompt that stays
// under 12KB, carries the section outline (headings + line numbers) instead
// of the full text, and tells the model how to read a section on demand.
func TestLargeContextFileCompactsToOutline(t *testing.T) {
	dir := t.TempDir()
	content := largeAgentsFixture()
	if len(content) <= 40*1024 {
		t.Fatalf("fixture is only %d bytes, want >40KB", len(content))
	}
	if err := os.WriteFile(filepath.Join(dir, "AGENTS.md"), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	parts := loadInstructionPartsFrom("Base instruction.", dir, "")
	got := parts.String()

	if len(got) > 12*1024 {
		t.Fatalf("system prompt with a >40KB AGENTS.md is %d chars, want <= 12KB", len(got))
	}
	if limit := maxRulesBlock + len("\n\n# Project Rules\n\n"); len(parts.Rules) > limit {
		t.Errorf("Rules block is %d chars, want <= %d", len(parts.Rules), limit)
	}
	if !strings.Contains(parts.Rules, "- L1: ## Section 0 — ") {
		t.Errorf("outline must carry headings with line numbers and a description, got %q", parts.Rules[:200])
	}
	if strings.Contains(got, "shell comment inside a fence") {
		t.Error("a `#` line inside a code fence must not become an outline heading")
	}
	if strings.Contains(got, "Rule line 399") {
		t.Error("full section text leaked into the prompt; only the outline belongs there")
	}
	if !strings.Contains(parts.Rules, "more sections") {
		t.Error("outline must say how many sections were dropped, not cap silently")
	}
	if !strings.Contains(parts.Rules, "read tool: file_path=<path>, offset=<line number>") {
		t.Error("outline must tell the model how to read a full section on demand")
	}
}

// TestCompactContextFilesBudgetStubs pins the whole-block budget: when the
// per-file outlines no longer fit maxRulesBlock together, a later file is
// replaced by an explicit pointer instead of being dropped or inlined.
func TestCompactContextFilesBudgetStubs(t *testing.T) {
	files := []contextFile{
		{path: "/repo/AGENTS.md", content: largeAgentsFixture()},
		{path: "/repo/inner/AGENTS.md", content: largeAgentsFixture()},
		{path: "/repo/inner/CLAUDE.md", content: "# Tiny\n- rule"},
	}

	got := compactContextFiles(files)

	if len(got) > maxRulesBlock {
		t.Fatalf("compacted rules are %d chars, want <= %d", len(got), maxRulesBlock)
	}
	if !strings.Contains(got, "Skipped: the project-rules prompt budget is exhausted") {
		t.Errorf("a file that does not fit must leave an explicit pointer, got tail %q", got[maxRulesBlock-400:])
	}
	if !strings.Contains(got, "# Tiny") {
		t.Error("a small file after an exhausted budget must still be inlined")
	}
	if !strings.HasSuffix(got, rulesReadHint) {
		t.Error("an outlined rules body must end with the read hint")
	}
}

// TestSmallContextFileInlinesWhole keeps the small-file behavior: a file at
// or under maxRulesFileBudget goes in verbatim — no outline header, no read
// hint, nothing lost.
func TestSmallContextFileInlinesWhole(t *testing.T) {
	content := "# Custom Rules\n- Rule 1\n"
	files := []contextFile{{path: "/repo/.pirate/AGENTS.md", content: content}}

	got := compactContextFiles(files)

	if got != content {
		t.Fatalf("small file must inline verbatim, got %q", got)
	}
	if strings.Contains(got, "###") {
		t.Error("an inlined file must not grow an outline header")
	}
}

// TestSectionOutlineSkipsFences pins heading extraction: `#` lines inside
// fenced code blocks are shell comments, not sections.
func TestSectionOutlineSkipsFences(t *testing.T) {
	content := "# Real\n\n```bash\n# not a heading\necho hi\n```\n\n## After fence\n"
	got := sectionOutline(content)

	if len(got) != 2 {
		t.Fatalf("sectionOutline found %d sections, want 2: %+v", len(got), got)
	}
	if got[0].line != 1 || got[0].head != "# Real" {
		t.Errorf("first section = %+v, want line 1 head %q", got[0], "# Real")
	}
	if got[1].line != 8 || got[1].head != "## After fence" {
		t.Errorf("second section = %+v, want line 8 head %q", got[1], "## After fence")
	}
}

// TestOutlineWithoutHeadings pins the degenerate input: a large file with no
// markdown headings compacts to an explicit pointer at its head instead of
// silently vanishing.
func TestOutlineWithoutHeadings(t *testing.T) {
	f := contextFile{
		path:    "/repo/WALL.md",
		content: strings.Repeat("plain text wall\n", 1000),
	}

	got := outlineContextFile(f)

	if !strings.Contains(got, "(no markdown headings)") {
		t.Errorf("outline of a heading-less file must say so, got %q", got)
	}
	if !strings.Contains(got, "plain text wall") {
		t.Errorf("outline must keep the file's first line as the description, got %q", got)
	}
}

// TestNoContextFilesKeepsRulesEmpty pins the no-AGENTS.md repository: no
// "# Project Rules" block is synthesized, and the volatile environment block
// is still appended last.
func TestNoContextFilesKeepsRulesEmpty(t *testing.T) {
	dir := t.TempDir()

	parts := loadInstructionPartsFrom("Base instruction.", dir, "")

	if parts.Rules != "" {
		t.Errorf("Rules = %q, want empty without context files", parts.Rules)
	}
	got := parts.String()
	if !strings.Contains(got, "# Runtime Environment") {
		t.Fatalf("environment block must still be present, got %q", got)
	}
	if strings.Index(got, "Base instruction.") > strings.Index(got, "# Runtime Environment") {
		t.Errorf("environment block must come after the static base, got %q", got)
	}
}
