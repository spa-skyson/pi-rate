package tools

import (
	"fmt"
	"strings"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool"
)

// GitHunkInput defines the parameters for the git-hunk tool.
type GitHunkInput struct {
	// File path to inspect hunks for (relative to repo root).
	File string `json:"file"`
	// If true, show hunks from staged diff. Default: false.
	Staged bool `json:"staged,omitempty"`
}

// Hunk represents a single diff hunk with metadata.
type Hunk struct {
	// Hunk header (e.g. "@@ -1,3 +1,5 @@").
	Header string `json:"header"`
	// Raw hunk content (context + added + removed lines).
	Content string `json:"content"`
	// Number of lines added in this hunk.
	Added int `json:"added"`
	// Number of lines removed in this hunk.
	Removed int `json:"removed"`
}

// GitHunkOutput contains parsed hunks for a file.
type GitHunkOutput struct {
	// File path.
	File string `json:"file"`
	// Parsed hunks.
	Hunks []Hunk `json:"hunks"`
	// Total number of hunks.
	TotalHunks int `json:"total_hunks"`
}

func newGitHunkTool(sb *Sandbox) (tool.Tool, error) {
	return newTool("git-hunk", "Parsed diff hunks for one file: header, content, line stats per hunk.", func(ctx agent.Context, input GitHunkInput) (GitHunkOutput, error) {
		return gitHunkHandler(sb, ctx, input)
	})
}

func gitHunkHandler(sb *Sandbox, ctx agent.Context, input GitHunkInput) (GitHunkOutput, error) {
	if input.File == "" {
		return GitHunkOutput{}, fmt.Errorf("file is required")
	}

	dir := sb.Dir()

	// Check if directory is a git repo
	if _, err := runGit(ctx, dir, "rev-parse", "--git-dir"); err != nil {
		return GitHunkOutput{}, fmt.Errorf("not a git repository")
	}

	args := []string{"diff", "-U3"}
	if input.Staged {
		args = append(args, "--cached")
	}
	args = append(args, "--", input.File)

	diff, err := runGit(ctx, dir, args...)
	if err != nil {
		return GitHunkOutput{}, fmt.Errorf("git diff failed: %w", err)
	}

	hunks := parseHunks(diff)

	return GitHunkOutput{
		File:       input.File,
		Hunks:      hunks,
		TotalHunks: len(hunks),
	}, nil
}

// parseHunks splits a unified diff into individual Hunk structs. A thin
// adapter over ParseUnifiedDiff (gitdiff.go) — the one parser both the tool
// handlers and the TUI viewer read — so the Content stays the raw hunk body
// the tool has always returned.
func parseHunks(diff string) []Hunk {
	parsed := ParseUnifiedDiff(diff)
	if len(parsed) == 0 {
		return nil
	}

	hunks := make([]Hunk, 0, len(parsed))
	for _, h := range parsed {
		contentLines := make([]string, 0, len(h.Lines))
		for _, l := range h.Lines {
			contentLines = append(contentLines, l.Text)
		}
		hunks = append(hunks, Hunk{
			Header:  h.Header,
			Content: strings.Join(contentLines, "\n"),
			Added:   h.Added,
			Removed: h.Removed,
		})
	}
	return hunks
}
