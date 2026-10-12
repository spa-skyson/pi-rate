package tools

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	gopath "path"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/spa-skyson/pi-rate/internal/config"
)

const (
	maxReadRetries = 3
	readRetryDelay = 50 * time.Millisecond
)

// isTransientReadError returns true for errors that might succeed on retry.
func isTransientReadError(err error) bool {
	if err == nil {
		return false
	}
	msg := err.Error()
	transient := []string{
		"text file busy",
		"resource temporarily unavailable",
		"input/output error",
	}
	for _, t := range transient {
		if strings.Contains(msg, t) {
			return true
		}
	}
	return errors.Is(err, syscall.ETIMEDOUT)
}

// Sandbox provides a secure file system abstraction that restricts
// mutating file operations to a specific directory tree.
//
// SECURITY MODEL:
//   - Mutating operations (WriteFile, MkdirAll) and the reads that feed
//     them (Stat, Open, ReadFile, ReadDir) are restricted to the sandbox
//     root via os.Root (Go 1.24+): paths resolve relative to the root,
//     symlinks pointing outside are blocked, absolute paths are converted
//     to relative.
//   - The read-only tools (read, read_image, ls) go through the *Anywhere
//     accessors instead, which additionally allow absolute paths anywhere
//     on the filesystem for reading; device files stay blocked (issue #69).
type Sandbox struct {
	root        *os.Root
	dir         string // absolute path of the root directory
	worktreeDir string // absolute path of worktree (if subagent context)
	extraRoots  []*os.Root
	extraDirs   []string // absolute paths of extra allowed directories

	// Parsed .gitignore patterns, memoized across tool calls. A Sandbox lives
	// for the whole session, and collecting these costs a full tree walk, so
	// re-deriving them on every grep/find was the dominant cost of both tools.
	gitignoreMu   sync.Mutex
	gitignorePats []GitignorePattern
	gitignoreAt   time.Time
}

// gitignoreTTL bounds how stale the memoized patterns may be. Long enough that
// a burst of tool calls within one agent turn walks the tree once; short enough
// that editing a .gitignore is picked up promptly.
const gitignoreTTL = 5 * time.Second

// NewSandbox opens an os.Root anchored at dir.
// Optionally pass worktreeDir if this is a subagent sandbox that needs to
// resolve relative paths from a different working directory.
func NewSandbox(dir string, worktreeDir ...string) (*Sandbox, error) {
	abs, err := filepath.Abs(dir)
	if err != nil {
		return nil, fmt.Errorf("resolving sandbox dir: %w", err)
	}
	root, err := os.OpenRoot(abs)
	if err != nil {
		return nil, fmt.Errorf("opening sandbox root %s: %w", abs, err)
	}
	wt := ""
	if len(worktreeDir) > 0 && worktreeDir[0] != "" {
		wt, err = filepath.Abs(worktreeDir[0])
		if err != nil {
			return nil, fmt.Errorf("resolving worktree dir: %w", err)
		}
	}
	return &Sandbox{root: root, dir: abs, worktreeDir: wt}, nil
}

// AddExtraDir registers an additional directory that the sandbox can access.
// Paths under this directory are resolved using a separate os.Root.
func (s *Sandbox) AddExtraDir(dir string) error {
	abs, err := filepath.Abs(dir)
	if err != nil {
		return fmt.Errorf("resolving extra dir: %w", err)
	}
	// Create the directory if it doesn't exist.
	if err := os.MkdirAll(abs, 0o700); err != nil {
		return fmt.Errorf("creating extra dir %s: %w", abs, err)
	}
	root, err := os.OpenRoot(abs)
	if err != nil {
		return fmt.Errorf("opening extra root %s: %w", abs, err)
	}
	s.extraRoots = append(s.extraRoots, root)
	s.extraDirs = append(s.extraDirs, abs)
	return nil
}

// matchExtraRoot returns the extra os.Root and the relative path if the
// absolute path falls under one of the extra allowed directories.
// Returns nil, "" if no match.
func (s *Sandbox) matchExtraRoot(absPath string) (*os.Root, string) {
	for i, dir := range s.extraDirs {
		if absPath == dir || strings.HasPrefix(absPath, dir+string(filepath.Separator)) {
			rel, err := filepath.Rel(dir, absPath)
			if err != nil {
				continue
			}
			return s.extraRoots[i], rel
		}
	}
	return nil, ""
}

// Close releases the underlying os.Root file descriptor.
func (s *Sandbox) Close() error {
	for _, r := range s.extraRoots {
		_ = r.Close()
	}
	return s.root.Close()
}

// LoadGitignorePatterns loads .gitignore patterns from the sandbox root.
// Returns nil if no patterns found (non-fatal).
//
// The result is memoized for gitignoreTTL: collecting it costs a walk of the
// whole tree, and grep/find call this on every invocation.
func (s *Sandbox) LoadGitignorePatterns() ([]GitignorePattern, error) {
	s.gitignoreMu.Lock()
	defer s.gitignoreMu.Unlock()

	if !s.gitignoreAt.IsZero() && time.Since(s.gitignoreAt) < gitignoreTTL {
		return s.gitignorePats, nil
	}

	patterns, err := s.loadGitignore()
	if err != nil {
		return nil, err
	}
	if len(patterns) == 0 {
		patterns = nil
	}
	s.gitignorePats = patterns
	s.gitignoreAt = time.Now()
	return patterns, nil
}

// FS returns an fs.FS scoped to the sandbox root directory.
func (s *Sandbox) FS() fs.FS {
	return s.root.FS()
}

// Dir returns the absolute path of the sandbox root.
func (s *Sandbox) Dir() string {
	return s.dir
}

// SetWorktreeDir sets the worktree directory for path normalization.
// This is used by subagent sandboxes that need to resolve relative paths
// from a different working directory than the sandbox root.
func (s *Sandbox) SetWorktreeDir(dir string) error {
	if dir == "" {
		s.worktreeDir = ""
		return nil
	}
	abs, err := filepath.Abs(dir)
	if err != nil {
		return fmt.Errorf("resolving worktree dir: %w", err)
	}
	s.worktreeDir = abs
	return nil
}

// Resolve converts an absolute or relative path to a relative path
// under the sandbox root. Returns an error with the sandbox root path
// if the resolved path would escape the directory tree.
//
// For subagent contexts with worktree directories, paths like "../../go.mod"
// are resolved relative to the worktree, then made relative to the sandbox root.
//
// SECURITY: This is intentional. The sandbox restricts file system
// access to prevent the agent from reading/writing files outside
// the working directory.
func (s *Sandbox) Resolve(name string) (string, error) {
	// Handle worktree-relative paths (../../ patterns from subagent worktree)
	if s.worktreeDir != "" && strings.HasPrefix(name, "../") {
		return s.resolveWorktreePath(name)
	}

	var rel string
	if filepath.IsAbs(name) {
		var err error
		rel, err = filepath.Rel(s.dir, name)
		if err != nil {
			return "", fmt.Errorf("path %s is outside sandbox root %s — use absolute paths under this directory", name, s.dir)
		}
	} else {
		rel = name
	}

	// Check if the cleaned relative path escapes the sandbox root.
	cleaned := filepath.Clean(rel)
	if cleaned == ".." || strings.HasPrefix(cleaned, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("path %q escapes sandbox root %s — use absolute paths starting with %s/", name, s.dir, s.dir)
	}

	return rel, nil
}

// resolveWorktreePath handles paths like "../../go.mod" from subagent worktrees.
// It converts the worktree-relative path to an absolute path, then makes it
// relative to the sandbox root while ensuring it doesn't escape.
func (s *Sandbox) resolveWorktreePath(name string) (string, error) {
	// Convert worktree-relative path to absolute
	abs, err := filepath.Abs(filepath.Join(s.worktreeDir, name))
	if err != nil {
		return "", fmt.Errorf("resolving worktree-relative path %q: %w", name, err)
	}

	// Make it relative to sandbox root
	rel, err := filepath.Rel(s.dir, abs)
	if err != nil {
		return "", fmt.Errorf("path %q resolves to %s which is outside sandbox root %s", name, abs, s.dir)
	}

	// Verify the relative path doesn't escape the sandbox
	cleaned := filepath.Clean(rel)
	if cleaned == ".." || strings.HasPrefix(cleaned, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("path %q escapes sandbox root %s", name, s.dir)
	}

	return rel, nil
}

// resolveToRoot returns the os.Root and relative path for the given name.
// It first checks extra roots (for absolute paths under allowed dirs),
// then falls back to the primary sandbox root.
func (s *Sandbox) resolveToRoot(name string) (*os.Root, string, error) {
	// Check extra roots for absolute paths.
	if filepath.IsAbs(name) {
		if root, rel := s.matchExtraRoot(name); root != nil {
			return root, rel, nil
		}
	}
	// Fall back to the primary sandbox root.
	rel, err := s.Resolve(name)
	if err != nil {
		return nil, "", err
	}
	return s.root, rel, nil
}

// ReadFile reads the named file within the sandbox.
// Transient errors (e.g. "text file busy") are retried up to 3 times
// with increasing delay. Non-transient errors are returned immediately.
func (s *Sandbox) ReadFile(name string) ([]byte, error) {
	root, rel, err := s.resolveToRoot(name)
	if err != nil {
		return nil, err
	}
	return readFileRoot(root, rel)
}

// readFileRoot reads rel from root, retrying transient errors. Shared by the
// strict ReadFile and ReadFileAnywhere so both behave identically for
// anything that resolves inside the sandbox.
func readFileRoot(root *os.Root, rel string) ([]byte, error) {
	var lastErr error
	for attempt := 0; attempt < maxReadRetries; attempt++ {
		data, err := root.ReadFile(rel)
		if err == nil {
			return data, nil
		}

		if !isTransientReadError(err) {
			return nil, err
		}
		lastErr = err

		if attempt < maxReadRetries-1 {
			time.Sleep(readRetryDelay * time.Duration(attempt+1))
		}
	}
	return nil, lastErr
}

// ─── Read-only access outside the sandbox root (issue #69) ──────────────
//
// The read tools (read, read_image, ls) accept absolute paths anywhere on
// the filesystem: read is non-destructive, and refusing it turned one cheap
// tool call into a second model turn through bash — +15-20s and a wasted
// model call per attempt (measured 2026-10-12). The *Anywhere accessors are
// the whole mechanism: identical to the strict accessors for anything that
// resolves inside the sandbox — os.Root still enforces symlink containment
// there — plus plain os calls for absolute paths outside it.
//
// DECISION, against the alternatives:
//   - A second Sandbox rooted at the filesystem root would resolve relative
//     paths against that root and break every relative read.
//   - A read-only flag on Resolve would put the permissive path one flag
//     away from write.go and edit.go, which call the same accessors and
//     must keep today's exact errors. Distinct methods keep the strict path
//     as the default and permissiveness opt-in per call site.
//
// The fallback fires for absolute paths only: relative paths, including
// ".."-escapes, surface the strict Resolve error unchanged. Device files
// stay blocked via isBlockedPath. grep/find/tree are not extended: they walk
// the sandbox FS, and searching an arbitrary root is a different mechanism.

// outsideAbs returns the absolute path the read-only fallback may touch.
// An empty path with a nil error means the fallback does not apply (name is
// relative) and the caller must surface the original sandbox error.
func outsideAbs(name string) (abs string, err error) {
	if !filepath.IsAbs(name) {
		return "", nil
	}
	if isBlockedPath(name) {
		return "", errBlockedDevice(name)
	}
	return filepath.Clean(name), nil
}

// readAnywhere applies the read-only fallback: run fn on the cleaned
// absolute path when name is absolute and not a blocked device, otherwise
// surface resolveErr — the sandbox error — unchanged.
func readAnywhere[T any](name string, resolveErr error, fn func(abs string) (T, error)) (T, error) {
	var zero T
	abs, blocked := outsideAbs(name)
	switch {
	case blocked != nil:
		return zero, blocked
	case abs == "":
		return zero, resolveErr
	}
	return fn(abs)
}

// StatAnywhere is Stat for read-only callers: outside absolute paths stat
// against the real filesystem.
func (s *Sandbox) StatAnywhere(name string) (os.FileInfo, error) {
	root, rel, err := s.resolveToRoot(name)
	if err == nil {
		return root.Lstat(rel)
	}
	return readAnywhere(name, err, os.Stat)
}

// OpenAnywhere is Open for read-only callers: outside absolute paths open
// against the real filesystem.
func (s *Sandbox) OpenAnywhere(name string) (*os.File, error) {
	root, rel, err := s.resolveToRoot(name)
	if err == nil {
		return root.Open(rel)
	}
	return readAnywhere(name, err, os.Open)
}

// ReadFileAnywhere is ReadFile for read-only callers: outside absolute
// paths are read from the real filesystem. Inside the sandbox both share
// readFileRoot, transient-error retries included.
func (s *Sandbox) ReadFileAnywhere(name string) ([]byte, error) {
	root, rel, err := s.resolveToRoot(name)
	if err == nil {
		return readFileRoot(root, rel)
	}
	return readAnywhere(name, err, os.ReadFile)
}

// ReadDirAnywhere is ReadDir for read-only callers: outside absolute
// directories are listed from the real filesystem.
func (s *Sandbox) ReadDirAnywhere(name string) ([]os.DirEntry, error) {
	root, rel, err := s.resolveToRoot(name)
	if err == nil {
		f, openErr := root.Open(rel)
		if openErr != nil {
			return nil, openErr
		}
		defer f.Close()
		return f.ReadDir(-1)
	}
	return readAnywhere(name, err, func(abs string) ([]os.DirEntry, error) {
		f, openErr := os.Open(abs)
		if openErr != nil {
			return nil, openErr
		}
		defer f.Close()
		return f.ReadDir(-1)
	})
}

// WriteFile writes data to the named file within the sandbox, creating it if
// necessary (parent directories are created automatically).
func (s *Sandbox) WriteFile(name string, data []byte, perm os.FileMode) error {
	root, rel, err := s.resolveToRoot(name)
	if err != nil {
		return err
	}
	dir := filepath.Dir(rel)
	if dir != "." {
		if err := root.MkdirAll(dir, 0o755); err != nil {
			return fmt.Errorf("creating directories: %w", err)
		}
	}
	f, err := root.OpenFile(rel, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, perm)
	if err != nil {
		return err
	}
	_, writeErr := f.Write(data)
	closeErr := f.Close()
	if writeErr != nil {
		return writeErr
	}
	return closeErr
}

// Open opens a file for reading within the sandbox.
func (s *Sandbox) Open(name string) (*os.File, error) {
	root, rel, err := s.resolveToRoot(name)
	if err != nil {
		return nil, err
	}
	return root.Open(rel)
}

// Stat returns FileInfo for a path within the sandbox.
func (s *Sandbox) Stat(name string) (os.FileInfo, error) {
	root, rel, err := s.resolveToRoot(name)
	if err != nil {
		return nil, err
	}
	return root.Lstat(rel)
}

// ReadDir lists entries in a directory within the sandbox.
func (s *Sandbox) ReadDir(name string) ([]os.DirEntry, error) {
	root, rel, err := s.resolveToRoot(name)
	if err != nil {
		return nil, err
	}
	f, err := root.Open(rel)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return f.ReadDir(-1)
}

// MkdirAll creates a directory path within the sandbox.
func (s *Sandbox) MkdirAll(name string, perm os.FileMode) error {
	root, rel, err := s.resolveToRoot(name)
	if err != nil {
		return err
	}
	return root.MkdirAll(rel, perm)
}

// GitignorePattern represents a parsed .gitignore pattern.
type GitignorePattern struct {
	dir     string // directory the pattern applies to ("" = root)
	pattern string // the pattern text
	isNeg   bool   // true if negated (!pattern)
	isDir   bool   // true if pattern ends with /
}

// ParseGitignoreLine parses a single .gitignore line into a pattern.
// Handles negation (!prefix), directory-only (/suffix), and recursive (**) patterns.
func parseGitignoreLine(line string, dir string) (GitignorePattern, bool) {
	line = strings.TrimRight(line, "\r")

	// Skip empty lines and comments
	if line == "" || strings.HasPrefix(line, "#") {
		return GitignorePattern{}, false
	}

	isNeg := strings.HasPrefix(line, "!")
	if isNeg {
		line = line[1:]
	}

	isDir := strings.HasSuffix(line, "/")
	if isDir {
		line = strings.TrimSuffix(line, "/")
	}

	// Convert .gitignore pattern to filepath.Match pattern
	// ** anywhere becomes * with recursive handling
	pattern := line
	pattern = strings.ReplaceAll(pattern, "**/", "**/")
	pattern = strings.ReplaceAll(pattern, "**", "*")

	return GitignorePattern{
		dir:     dir,
		pattern: pattern,
		isNeg:   isNeg,
		isDir:   isDir,
	}, true
}

// matchesGitignore reports whether path matches the given gitignore pattern.
// The path is relative to the sandbox root.
func (p GitignorePattern) matches(path string) bool {
	// Get the directory portion of the path
	dir := filepath.Dir(path)
	name := filepath.Base(path)

	// Pattern applies to its own directory or subdirectories
	if p.dir != "" && dir != p.dir && !strings.HasPrefix(dir, p.dir+string(filepath.Separator)) {
		return false
	}

	// path.Match, not filepath.Match: .gitignore syntax escapes with a
	// backslash ("\#literal" names a file starting with '#'), and so does
	// path.Match on every OS. filepath.Match on Windows treats '\' as an
	// ordinary character instead, so such a pattern never matched there. The
	// operand is a bare name with no separator, so the two agree otherwise.
	matched, _ := gopath.Match(p.pattern, name)
	if matched && p.isDir {
		// Directory pattern: only match if path is a directory
		// (handled by caller checking d.IsDir())
		return true
	}
	return matched
}

// loadGitignore reads and parses .gitignore files from the sandbox,
// building a list of patterns applicable to each directory.
func (s *Sandbox) loadGitignore() ([]GitignorePattern, error) {
	var patterns []GitignorePattern
	dirPatterns := make(map[string][]GitignorePattern)

	fsys := s.root.FS()

	// Walk the directory tree collecting .gitignore files.
	//
	// Prune with the same shouldSkipDir the search walks use (grep.go, find.go).
	// Skipping only .git here meant descending into node_modules, vendor and
	// every dotdir — directories the search then skips anyway — so collecting
	// the patterns cost more than applying them.
	err := fs.WalkDir(fsys, ".", func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() && path != "." && shouldSkipDir(d.Name()) {
			return filepath.SkipDir
		}
		if d.Name() != ".gitignore" {
			return nil
		}
		collectGitignoreFile(fsys, path, dirPatterns)
		return nil
	})
	if err != nil {
		return nil, err
	}

	// Merge all patterns into one list
	for _, patList := range dirPatterns {
		patterns = append(patterns, patList...)
	}

	return patterns, nil
}

// collectGitignoreFile parses the .gitignore at path and appends its patterns
// to the bucket for the directory that contains it. An unreadable file is
// skipped: a .gitignore that cannot be read contributes no patterns.
func collectGitignoreFile(fsys fs.FS, path string, dirPatterns map[string][]GitignorePattern) {
	gitignoreData, readErr := readFileFromFS(fsys, path)
	if readErr != nil {
		return
	}

	gitignoreDir := filepath.Dir(path)
	if gitignoreDir == "." {
		gitignoreDir = ""
	}

	scanner := bufio.NewScanner(strings.NewReader(string(gitignoreData)))
	for scanner.Scan() {
		if pat, ok := parseGitignoreLine(scanner.Text(), gitignoreDir); ok {
			dirPatterns[gitignoreDir] = append(dirPatterns[gitignoreDir], pat)
		}
	}
}

// readFileFromFS reads a file from an fs.FS.
func readFileFromFS(fsys fs.FS, path string) ([]byte, error) {
	f, err := fsys.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return io.ReadAll(f)
}

// shouldSkipPath reports whether the given path (relative to sandbox root)
// should be skipped based on .gitignore patterns and hardcoded rules.
func shouldSkipPath(relPath string, d fs.DirEntry, patterns []GitignorePattern) bool {
	base := d.Name()

	// Hardcoded directory skips
	// Explicitly do NOT skip .pirate, .cursor, .claude - these contain agent/skill files
	agentDirs := map[string]bool{config.ProjectDirName: true, ".cursor": true, ".claude": true}
	if (strings.HasPrefix(base, ".") && base != "." && !agentDirs[base]) || base == "node_modules" || base == "vendor" || base == "__pycache__" {
		return true
	}

	// Check binary directories
	if base == "target" {
		return true
	}

	// Apply .gitignore patterns
	for _, pat := range patterns {
		if pat.matches(relPath) {
			if pat.isNeg {
				// Negation: don't skip
				return false
			}
			// Match: skip unless it's a directory-only pattern and we're looking at a file
			if pat.isDir && !d.IsDir() {
				continue
			}
			return true
		}
	}

	return false
}
