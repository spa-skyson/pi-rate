package tools

// Tests for issue #69: the read tools accept absolute paths outside the
// sandbox root in a single call, while mutating paths stay strictly inside
// the sandbox and device files stay unreadable.

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestReadHandler_AbsolutePathOutsideRoot is the acceptance case: a file in
// a directory outside the sandbox root is read in one handler call, with
// line numbers — no second model turn through bash.
func TestReadHandler_AbsolutePathOutsideRoot(t *testing.T) {
	sb := testSandbox(t, t.TempDir())

	// A second temp dir is never inside the sandbox root.
	outsideDir := t.TempDir()
	path := filepath.Join(outsideDir, "note.txt")
	if err := os.WriteFile(path, []byte("hello outside\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	out, err := readHandler(sb, ReadInput{FilePath: path})
	if err != nil {
		t.Fatalf("readHandler(%s): %v", path, err)
	}
	if out.TotalLines != 1 {
		t.Errorf("TotalLines = %d, want 1", out.TotalLines)
	}
	if !strings.Contains(out.Content, "hello outside") {
		t.Errorf("Content = %q, want the file's text", out.Content)
	}
	if !strings.Contains(out.Content, "1\t") {
		t.Errorf("Content = %q, want line numbers", out.Content)
	}
}

// TestReadHandler_ProcFdStillBlocked pins the device block-list: a
// /proc/<pid>/fd/* read is refused before any I/O, sandbox or not.
func TestReadHandler_ProcFdStillBlocked(t *testing.T) {
	sb := testSandbox(t, t.TempDir())

	_, err := readHandler(sb, ReadInput{FilePath: "/proc/1/fd/1"})
	if err == nil {
		t.Fatal("expected /proc/1/fd/1 read to be refused")
	}
	if !strings.Contains(err.Error(), "device file") {
		t.Errorf("err = %v, want the device-file refusal", err)
	}
}

// TestReadHandler_RelativeEscapeStillBlocked pins that the fallback is for
// absolute paths only: a relative ".."-escape keeps the strict sandbox
// error, at the handler and at the accessor.
func TestReadHandler_RelativeEscapeStillBlocked(t *testing.T) {
	root := t.TempDir()
	sub := filepath.Join(root, "sub")
	if err := os.MkdirAll(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	sb := testSandbox(t, sub)
	if err := os.WriteFile(filepath.Join(root, "peek.txt"), []byte("secret"), 0o644); err != nil {
		t.Fatal(err)
	}

	if _, err := readHandler(sb, ReadInput{FilePath: "../peek.txt"}); err == nil {
		t.Fatal("expected ../peek.txt read to be refused")
	}
	_, err := sb.StatAnywhere("../peek.txt")
	if err == nil {
		t.Fatal("expected StatAnywhere to refuse a relative escape")
	}
	if !strings.Contains(err.Error(), "escapes sandbox root") {
		t.Errorf("err = %v, want the unchanged escapes-sandbox error", err)
	}
}

// TestLsHandler_AbsolutePathOutsideRoot pins ls's new reach: an absolute
// directory outside the root lists in one call.
func TestLsHandler_AbsolutePathOutsideRoot(t *testing.T) {
	sb := testSandbox(t, t.TempDir())

	outsideDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(outsideDir, "a.txt"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}

	out, err := lsHandler(sb, LsInput{Path: outsideDir})
	if err != nil {
		t.Fatalf("lsHandler(%s): %v", outsideDir, err)
	}
	if out.TotalEntries != 1 {
		t.Errorf("TotalEntries = %d, want 1", out.TotalEntries)
	}
	if len(out.Entries) != 1 || !strings.HasSuffix(out.Entries[0].Name, "a.txt") {
		t.Errorf("Entries = %+v, want a.txt", out.Entries)
	}
}

// TestSandbox_WriteFile_OutsideStillBlocked pins the mutating side: the
// strict WriteFile refuses an outside path and creates nothing.
func TestSandbox_WriteFile_OutsideStillBlocked(t *testing.T) {
	sb := testSandbox(t, t.TempDir())

	outsidePath := filepath.Join(t.TempDir(), "new.txt")
	if err := sb.WriteFile(outsidePath, []byte("nope"), 0o644); err == nil {
		t.Fatal("expected WriteFile outside the sandbox to fail")
	}
	if _, err := os.Stat(outsidePath); !os.IsNotExist(err) {
		t.Errorf("outside file created anyway: %v", err)
	}
}

// TestEditHandler_OutsideStillBlocked pins edit's unchanged failure: the
// read that feeds the edit goes through the strict ReadFile, so the error
// is the sandbox-escape one and the file is left untouched on disk.
func TestEditHandler_OutsideStillBlocked(t *testing.T) {
	sb := testSandbox(t, t.TempDir())

	outsidePath := filepath.Join(t.TempDir(), "target.txt")
	if err := os.WriteFile(outsidePath, []byte("untouched\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	_, err := editHandler(sb, EditInput{
		FilePath:  outsidePath,
		OldString: "untouched",
		NewString: "edited",
	})
	if err == nil {
		t.Fatal("expected edit outside the sandbox to fail")
	}
	if !strings.Contains(err.Error(), "escapes sandbox root") {
		t.Errorf("err = %v, want the unchanged escapes-sandbox error", err)
	}
	data, readErr := os.ReadFile(outsidePath)
	if readErr != nil {
		t.Fatal(readErr)
	}
	if string(data) != "untouched\n" {
		t.Errorf("file changed on disk: %q", data)
	}
}
