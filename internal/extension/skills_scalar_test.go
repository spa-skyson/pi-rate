package extension

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// parseDesc parses the given SKILL.md content and returns the description
// the parser produced for it.
func parseDesc(t *testing.T, content string) string {
	t.Helper()
	skill, _, err := parseSkillContent(content, "fallback")
	if err != nil {
		t.Fatalf("parseSkillContent: %v", err)
	}
	return skill.Description
}

// TestParseSkillContent_FoldedScalar: description: > collects the indented
// continuation lines and folds them into one value instead of leaving the
// bare ">" (issue 86).
func TestParseSkillContent_FoldedScalar(t *testing.T) {
	content := "---\n" +
		"name: glab\n" +
		"description: >\n" +
		"  GitLab CLI (glab) for working with GitLab from the command line.\n" +
		"  Use this skill whenever you run any `glab` command or GitLab API\n" +
		"  call. Read this skill before running any `glab` command.\n" +
		"---\n" +
		"body\n"
	got := parseDesc(t, content)
	want := "GitLab CLI (glab) for working with GitLab from the command line. " +
		"Use this skill whenever you run any `glab` command or GitLab API call. " +
		"Read this skill before running any `glab` command.\n"
	if got != want {
		t.Errorf("Description:\n got %q\nwant %q", got, want)
	}
}

// TestParseSkillContent_LiteralScalar: description: | keeps the line breaks.
func TestParseSkillContent_LiteralScalar(t *testing.T) {
	got := parseDesc(t, "---\ndescription: |\n  first line\n  second line\n---\nbody\n")
	want := "first line\nsecond line\n"
	if got != want {
		t.Errorf("Description = %q, want %q", got, want)
	}
}

// TestParseSkillContent_FoldedStripModifier: description: >- drops the
// trailing newline a clip chiral keeps.
func TestParseSkillContent_FoldedStripModifier(t *testing.T) {
	got := parseDesc(t, "---\ndescription: >-\n  folded strip\n---\nbody\n")
	if got != "folded strip" {
		t.Errorf("Description = %q, want %q (no trailing newline)", got, "folded strip")
	}
	if strings.HasSuffix(got, "\n") {
		t.Errorf("strip modifier must not leave a trailing newline: %q", got)
	}
}

// TestParseSkillContent_SingleLineUnchanged: a plain one-line description
// parses exactly as before the scalar support.
func TestParseSkillContent_SingleLineUnchanged(t *testing.T) {
	got := parseDesc(t, "---\ndescription: Review code for quality issues\n---\nbody\n")
	if got != "Review code for quality issues" {
		t.Errorf("Description = %q, want unchanged one-line value", got)
	}
}

// TestParseSkillContent_ContinuationKeepsKeyValueLikeLines: a continuation
// line that itself looks like "key: value" (a bullet item, an example) stays
// inside the scalar — it does not open a new frontmatter key.
func TestParseSkillContent_ContinuationKeepsKeyValueLikeLines(t *testing.T) {
	got := parseDesc(t, "---\ndescription: >\n  Skills loader.\n  - lesson: fake\n  - note: also fake\n---\nbody\n")
	want := "Skills loader. - lesson: fake - note: also fake\n"
	if got != want {
		t.Errorf("Description = %q, want %q", got, want)
	}
}

// TestParseSkillContent_DelimiterEndsScalar: the closing "---" ends the
// scalar; the body is not swallowed into the description.
func TestParseSkillContent_DelimiterEndsScalar(t *testing.T) {
	content := "---\ndescription: >\n  folded text\n---\nreal body line\n"
	skill, body, err := parseSkillContent(content, "fallback")
	if err != nil {
		t.Fatalf("parseSkillContent: %v", err)
	}
	if !strings.Contains(skill.Description, "folded text") || strings.Contains(skill.Description, "real body") {
		t.Errorf("Description = %q, want the folded value only", skill.Description)
	}
	if body != "real body line" {
		t.Errorf("body = %q, want %q", body, "real body line")
	}
}

// TestParseSkillContent_FoldedBlankLineBecomesNewline: an empty line inside
// a folded scalar folds to a newline, not to a space.
func TestParseSkillContent_FoldedBlankLineBecomesNewline(t *testing.T) {
	got := parseDesc(t, "---\ndescription: >\n  first paragraph line\n\n  second paragraph line\n---\nbody\n")
	want := "first paragraph line\nsecond paragraph line\n"
	if got != want {
		t.Errorf("Description = %q, want %q", got, want)
	}
}

// TestParseSkillContent_BareKeyContinuationIsFolded: "description:" with
// indented continuation lines and no ">" marker is treated as folded —
// leniency toward skill authors.
func TestParseSkillContent_BareKeyContinuationIsFolded(t *testing.T) {
	got := parseDesc(t, "---\ndescription:\n  written without the marker\n  but multi-line\n---\nbody\n")
	want := "written without the marker but multi-line\n"
	if got != want {
		t.Errorf("Description = %q, want %q", got, want)
	}
}

// TestParseSkillContent_FoldedNameAndTools: block scalars work for every
// recognized key, not just description.
func TestParseSkillContent_FoldedNameAndTools(t *testing.T) {
	skill, _, err := parseSkillContent("---\nname: multi-skill\ntools: >\n  read, write,\n  bash\n---\nbody\n", "fallback")
	if err != nil {
		t.Fatalf("parseSkillContent: %v", err)
	}
	if skill.Name != "multi-skill" {
		t.Errorf("Name = %q", skill.Name)
	}
	if len(skill.Tools) != 3 || skill.Tools[0] != "read" || skill.Tools[2] != "bash" {
		t.Errorf("Tools = %v, want [read write bash]", skill.Tools)
	}
}

// TestParseSkillContent_UnknownKeysDoNotOpenScalars: an unknown key with a
// ">" value must not capture the following lines — only name, description
// and tools open blocks.
func TestParseSkillContent_UnknownKeysDoNotOpenScalars(t *testing.T) {
	content := "---\nmetadata: >\n  not a skill key\nlicense: MIT\n---\nbody\n"
	skill, body, err := parseSkillContent(content, "fallback")
	if err != nil {
		t.Fatalf("parseSkillContent: %v", err)
	}
	if len(skill.Tools) != 0 || skill.Name != "fallback" {
		t.Errorf("unexpected skill fields from unknown keys: %+v", skill)
	}
	if body != "body" {
		t.Errorf("body = %q, want %q", body, "body")
	}
}

// TestParseSkillContent_FoldedScalarEndsAtNextKey: a non-indented
// "key: value" line ends the scalar; both keys keep their own values.
func TestParseSkillContent_FoldedScalarEndsAtNextKey(t *testing.T) {
	content := "---\nname: glab\ndescription: >\n  first part\nname2: ignored\n---\nbody\n"
	skill, _, err := parseSkillContent(content, "fallback")
	if err != nil {
		t.Fatalf("parseSkillContent: %v", err)
	}
	if skill.Description != "first part\n" {
		t.Errorf("Description = %q, want %q", skill.Description, "first part")
	}
}

// TestLoadSkillFileWithFoldedDescription is the integration case from issue
// 86: the real glab SKILL.md frontmatter (folded description with URLs and
// backticks) must produce the full description text, and the skill hygiene
// marker "Use this skill whenever" must be found in it.
func TestLoadSkillFileWithFoldedDescription(t *testing.T) {
	// Fixture copied from ~/.pirate/skills/glab/SKILL.md (GitLab CLI's own
	// bundled skill, folded scalar, unchanged).
	const glabFrontmatter = `---
name: glab
description: >
  GitLab CLI (glab) for working with GitLab from the command line. Use this skill whenever you run any
  ` + "`glab`" + ` command or GitLab API call. Read this
  skill before running any ` + "`glab`" + ` or GitLab API command — it applies to every
  GitLab operation, whether reading or writing (for example merge requests,
  issues, work items, discussions and threaded replies, comments, CI/CD
  pipelines, releases, packages, members, and project settings). Whenever a
  task touches GitLab in any way, consult this skill first so you use the
  correct, safe command on the first try. Prefer glab over raw API calls for
  all GitLab operations.
metadata:
  source-project: gitlab-org/cli
  source-path: internal/commands/skills/bundled/assets/glab
---

# GitLab CLI (glab)
`
	dir := t.TempDir()
	skillDir := filepath.Join(dir, "glab")
	if err := os.MkdirAll(skillDir, 0o755); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(skillDir, "SKILL.md")
	if err := os.WriteFile(path, []byte(glabFrontmatter), 0o644); err != nil {
		t.Fatal(err)
	}

	skill, err := parseSkillFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if skill.Name != "glab" {
		t.Errorf("Name = %q, want glab", skill.Name)
	}
	for _, marker := range []string{
		"Use this skill whenever",
		"merge requests",
		"project settings",
		"Prefer glab over raw API calls for all GitLab operations.",
	} {
		if !strings.Contains(skill.Description, marker) {
			t.Errorf("Description missing %q:\n%s", marker, skill.Description)
		}
	}
	if strings.Contains(skill.Description, ">") && !strings.Contains(skill.Description, "`glab` command") {
		// The bare ">" artifact must be gone.
		t.Errorf("Description still carries the bare > artifact:\n%s", skill.Description)
	}
	if skill.Description == ">" {
		t.Fatalf("issue 86 regression: Description = %q", skill.Description)
	}
}
