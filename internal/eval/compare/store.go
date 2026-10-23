package compare

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Store persists finished comparisons on disk so a later invocation can
// reuse an arm instead of re-running it (harness2 rule P11). Files are
// keyed by the fingerprint pair — `<fp0>-<fp1>.json` — so a reused result
// is by construction the same comparison of the same variants.
type Store struct {
	dir string
}

// NewStore returns a Store rooted at dir. The directory is created lazily,
// on the first Save.
func NewStore(dir string) *Store {
	return &Store{dir: dir}
}

// Dir returns the store's root directory.
func (s *Store) Dir() string { return s.dir }

// stored is the on-disk envelope: the result plus the fingerprint pair it
// was saved under, so Load can reject a file whose identity drifted (a
// renamed or edited variant, a hand-copied file).
type stored struct {
	Fingerprint   [2]string     `json:"fingerprint"`
	CompareResult CompareResult `json:"result"`
}

// Has reports whether the store holds a comparison for this fingerprint
// pair.
func (s *Store) Has(fp [2]string) bool {
	_, err := os.Stat(s.path(fp))
	return err == nil
}

// Save writes the comparison under its fingerprint pair, creating the store
// directory when needed. The saved file round-trips: Load returns a result
// equal to the one saved.
func (s *Store) Save(res CompareResult) error {
	if err := os.MkdirAll(s.dir, 0o755); err != nil {
		return fmt.Errorf("compare: store mkdir: %w", err)
	}
	data, err := json.MarshalIndent(stored{Fingerprint: res.Fingerprint, CompareResult: res}, "", "  ")
	if err != nil {
		return fmt.Errorf("compare: store encode: %w", err)
	}
	path := s.path(res.Fingerprint)
	// Write-then-rename: a crash or a concurrent reader must never see a
	// half-written store file, which would fail to parse on the next Load.
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return fmt.Errorf("compare: store write %s: %w", tmp, err)
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("compare: store rename %s -> %s: %w", tmp, path, err)
	}
	return nil
}

// ErrFingerprintMismatch is returned by Load when the stored file's
// fingerprint pair does not match the requested one — the file exists but
// is not this comparison.
var ErrFingerprintMismatch = errors.New("compare: stored fingerprint does not match")

// Load reads the comparison for the fingerprint pair. ok is false when no
// file exists, the file is unreadable, or the stored fingerprint does not
// match fp (a mismatch also returns the error, so the caller can print the
// reason before re-running the arm).
func (s *Store) Load(fp [2]string) (CompareResult, bool, error) {
	path := s.path(fp)
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return CompareResult{}, false, nil
	}
	if err != nil {
		return CompareResult{}, false, fmt.Errorf("compare: store read %s: %w", path, err)
	}
	var st stored
	if err := json.Unmarshal(data, &st); err != nil {
		return CompareResult{}, false, fmt.Errorf("compare: store parse %s: %w", path, err)
	}
	if st.Fingerprint != fp {
		return CompareResult{}, false, fmt.Errorf("%w: want %v, have %v", ErrFingerprintMismatch, fp, st.Fingerprint)
	}
	return st.CompareResult, true, nil
}

// path renders the file name for a fingerprint pair. The full hex digests
// are kept — shortening them would make collisions possible across
// different comparisons sharing a prefix.
func (s *Store) path(fp [2]string) string {
	name := strings.Join([]string{fp[0], fp[1]}, "-") + ".json"
	return filepath.Join(s.dir, name)
}
