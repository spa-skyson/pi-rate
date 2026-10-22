package tools

import (
	"fmt"
	"strings"
	"testing"
)

// The shared truncation marker: every cut that reaches the model carries
// `… (+N bytes truncated)` with N = the number of cut bytes, so a
// rendered artifact can never be read as the whole answer.

func TestTruncationMarkerFormat(t *testing.T) {
	if got, want := truncationMarker(42), "\n… (+42 bytes truncated)"; got != want {
		t.Errorf("truncationMarker(42) = %q, want %q", got, want)
	}
}

// TestTruncateOutputCarriesCutSize pins truncateOutput's marker to the exact
// byte count removed at the 256KB ceiling.
func TestTruncateOutputCarriesCutSize(t *testing.T) {
	s := strings.Repeat("x", maxOutputBytes+1000)
	got := truncateOutput(s)
	want := truncationMarker(len(s) - maxOutputBytes)
	if !strings.HasSuffix(got, want) {
		t.Errorf("truncateOutput marker = %q, want suffix %q", got[max(0, len(got)-60):], want)
	}
}

// TestTruncateLineCarriesCutSize pins the per-line clamp the grep/read tools use.
func TestTruncateLineCarriesCutSize(t *testing.T) {
	s := strings.Repeat("y", maxLineLength+100)
	got := truncateLine(s)
	want := truncationMarker(len(s) - maxLineLength)
	if !strings.HasSuffix(got, want) {
		t.Errorf("truncateLine marker missing: %q", got[max(0, len(got)-60):])
	}
}

// TestHardTruncateCarriesCutSize covers the bash/read/tree compactor's char
// ceiling: the marker names exactly maxChars cut.
func TestHardTruncateCarriesCutSize(t *testing.T) {
	s := strings.Repeat("y", 100)
	got, applied := hardTruncate(s, 30)
	if !applied {
		t.Fatal("hardTruncate did not apply")
	}
	if !strings.HasSuffix(got, truncationMarker(70)) {
		t.Errorf("hardTruncate marker missing: %q", got)
	}
}

// TestHardTruncateLinesCarriesCutSize covers the line ceiling: the marker
// counts everything from the cut to the end of the input.
func TestHardTruncateLinesCarriesCutSize(t *testing.T) {
	s := strings.Repeat("line\n", 200) // 1000 bytes
	got, applied := hardTruncateLines(s, 10)
	if !applied {
		t.Fatal("hardTruncateLines did not apply")
	}
	// hardTruncateLines cuts at the boundary newline, so the marker counts
	// from the first byte after the kept 10 lines — the newline included.
	cutAt := 10*len("line\n") - 1
	// The marker for the ~950-char tail is longer than the kept head, so the
	// failure report shows sizes rather than a tail slice of the output.
	if !strings.HasSuffix(got, truncationMarker(len(s)-cutAt)) {
		t.Errorf("hardTruncateLines output is %d bytes, want it to end with the marker for %d cut chars", len(got), len(s)-cutAt)
	}
}

// TestSmartTruncateCarriesMarker runs the priority-selection path over an
// oversized input and requires the shared marker on the output.
func TestSmartTruncateCarriesMarker(t *testing.T) {
	cfg := DefaultCompactorConfig()
	cfg.MaxLines = 5
	var lines []string
	for range 100 {
		lines = append(lines, strings.Repeat("z", 40))
	}
	s := strings.Join(lines, "\n")
	got, applied := smartTruncate(s, cfg)
	if !applied {
		t.Fatal("smartTruncate did not apply")
	}
	if !strings.Contains(got, "bytes truncated)") {
		t.Errorf("smartTruncate output lacks the shared marker: %q", got[len(got)-60:])
	}
}

// TestCompactGitDiffTextCarriesMarker covers the git diff text compactor.
func TestCompactGitDiffTextCarriesMarker(t *testing.T) {
	cfg := DefaultCompactorConfig()
	cfg.MaxDiffLines = 3
	cfg.MaxDiffHunkLines = 2
	var lines []string
	lines = append(lines, "diff --git a/big.go b/big.go", "index 111..222 100644", "--- a/big.go", "+++ b/big.go", "@@ -1,9 +1,9 @@")
	for range 60 {
		lines = append(lines, "+added line number "+strings.Repeat("n", 20))
	}
	s := strings.Join(lines, "\n")
	got, applied := compactGitDiffText(s, cfg)
	if !applied {
		t.Fatal("compactGitDiffText did not apply")
	}
	if !strings.Contains(got, "bytes truncated)") {
		t.Errorf("git diff compaction lacks the shared marker: %q", got[len(got)-80:])
	}
}

// TestCompactGitLogTextCarriesMarker covers the git log entry cap.
func TestCompactGitLogTextCarriesMarker(t *testing.T) {
	cfg := DefaultCompactorConfig()
	var lines []string
	for i := range 50 {
		lines = append(lines,
			fmt.Sprintf("commit %040d", i),
			"Author: someone",
			"    some message that is reasonably long for the log entry body")
	}
	s := strings.Join(lines, "\n")
	got, applied := compactGitLogText(s, cfg)
	if !applied {
		t.Fatal("compactGitLogText did not apply")
	}
	if !strings.Contains(got, "bytes truncated)") {
		t.Errorf("git log compaction lacks the shared marker: %q", got[len(got)-80:])
	}
}

// TestCompactGitStatusTextCarriesMarker covers the git status file cap.
func TestCompactGitStatusTextCarriesMarker(t *testing.T) {
	cfg := DefaultCompactorConfig()
	var lines []string
	lines = append(lines, "On branch main", "Changes not staged for commit:", "")
	for i := range 40 {
		lines = append(lines, fmt.Sprintf("M  some/path/file_%02d.go", i))
	}
	s := strings.Join(lines, "\n")
	got, applied := compactGitStatusText(s, cfg)
	if !applied {
		t.Fatal("compactGitStatusText did not apply")
	}
	if !strings.Contains(got, "bytes truncated)") {
		t.Errorf("git status compaction lacks the shared marker: %q", got[len(got)-80:])
	}
}

// TestGroupSearchOutputCarriesMarker covers the search grouping cap.
func TestGroupSearchOutputCarriesMarker(t *testing.T) {
	cfg := DefaultCompactorConfig()
	var lines []string
	for i := range 200 {
		lines = append(lines, fmt.Sprintf("big.go:%d:some match content here that is long enough", i))
	}
	s := strings.Join(lines, "\n")
	got, applied := groupSearchOutput(s, cfg)
	if !applied {
		t.Fatal("groupSearchOutput did not apply")
	}
	if !strings.Contains(got, "bytes truncated)") {
		t.Errorf("grouped search output lacks the shared marker: %q", got[len(got)-80:])
	}
}

// TestTreeHandlerCarriesMarker runs the tree tool over a directory big enough
// to cross the entry cap and requires the marker on the listing.
func TestTreeHandlerCarriesMarker(t *testing.T) {
	dir := t.TempDir()
	sb := testSandbox(t, dir)
	for i := range maxTreeEntries + 50 {
		name := fmt.Sprintf("file_%c%c.txt", 'a'+i%26, 'a'+(i/26)%26)
		if err := sb.WriteFile(name, []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	out, err := treeHandler(sb, TreeInput{})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.Tree, "bytes truncated)") {
		t.Errorf("tree listing lacks the shared marker: %q", out.Tree[max(0, len(out.Tree)-80):])
	}
}

// TestFilterBuildOutputCarriesMarker covers the build filter's cap path: an
// input with more errors than the cap must end in the shared marker.
func TestFilterBuildOutputCarriesMarker(t *testing.T) {
	cfg := DefaultCompactorConfig()
	cfg.MaxBuildErrors = 2
	var lines []string
	for i := range 30 {
		lines = append(lines, fmt.Sprintf("file.go:%d:5: error: something failed badly", i))
	}
	s := strings.Join(lines, "\n")
	got, applied := filterBuildOutput(s, cfg)
	if !applied {
		t.Fatal("filterBuildOutput did not apply")
	}
	if !strings.Contains(got, "bytes truncated)") {
		t.Errorf("build filter output lacks the shared marker: %q", got[len(got)-80:])
	}
}

// TestAggregateTestOutputCarriesMarker covers the test aggregation path.
func TestAggregateTestOutputCarriesMarker(t *testing.T) {
	cfg := DefaultCompactorConfig()
	var lines []string
	for i := range 30 {
		lines = append(lines, fmt.Sprintf("=== RUN   TestThing%d", i))
	}
	lines = append(lines, "ok  \tpkg/one\t0.10s", "ok  \tpkg/two\t0.20s")
	s := strings.Join(lines, "\n")
	got, applied := aggregateTestOutput(s, cfg)
	if !applied {
		t.Fatal("aggregateTestOutput did not apply")
	}
	if !strings.Contains(got, "bytes truncated)") {
		t.Errorf("test summary lacks the shared marker: %q", got[len(got)-80:])
	}
}

// TestAggregateLinterOutputCarriesMarker covers the linter aggregation path.
func TestAggregateLinterOutputCarriesMarker(t *testing.T) {
	cfg := DefaultCompactorConfig()
	var lines []string
	for f := range 30 {
		for i := range 5 {
			lines = append(lines, fmt.Sprintf("file%02d.go:%d:%d: some lint message worth showing", f, i, i+1))
		}
	}
	s := strings.Join(lines, "\n")
	got, applied := aggregateLinterOutput(s, cfg)
	if !applied {
		t.Fatal("aggregateLinterOutput did not apply")
	}
	if !strings.Contains(got, "bytes truncated)") {
		t.Errorf("linter summary lacks the shared marker: %q", got[len(got)-80:])
	}
}
