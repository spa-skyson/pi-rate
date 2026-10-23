package compare

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestStore_RoundtripAndGuard(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir)

	runner := &fakeRunner{arms: map[string]ArmResult{
		"control": buildArm(armSpec{name: "control", steps: 10, calls: map[string]int{"read": 4}}),
		"variant": buildArm(armSpec{name: "variant", steps: 8, calls: map[string]int{"read": 6}}),
	}}
	res := mustCompare(t, controlVariant(), variantVariant(), runner, 5)

	if s.Has(res.Fingerprint) {
		t.Fatal("store reports a result before anything was saved")
	}

	if err := s.Save(res); err != nil {
		t.Fatalf("Save: %v", err)
	}
	if !s.Has(res.Fingerprint) {
		t.Error("Has=false after Save")
	}

	got, ok, err := s.Load(res.Fingerprint)
	if err != nil || !ok {
		t.Fatalf("Load: ok=%v err=%v", ok, err)
	}
	if got.Control.Variant != res.Control.Variant || got.Variant.Variant != res.Variant.Variant {
		t.Errorf("round-trip changed the arms: (%q,%q) -> (%q,%q)",
			res.Control.Variant, res.Variant.Variant, got.Control.Variant, got.Variant.Variant)
	}
	if len(got.Contrasts) != len(res.Contrasts) {
		t.Errorf("round-trip changed the contrast count: %d -> %d", len(res.Contrasts), len(got.Contrasts))
	}
	if got.Fingerprint != res.Fingerprint {
		t.Error("round-trip changed the fingerprint")
	}

	// File name is <fp0>-<fp1>.json.
	wantPath := filepath.Join(dir, res.Fingerprint[0]+"-"+res.Fingerprint[1]+".json")
	if _, err := os.Stat(wantPath); err != nil {
		t.Errorf("Save did not write %s: %v", wantPath, err)
	}

	// A file that exists but stores a different fingerprint must not load.
	// Tamper with the envelope: same file name, different stored pair.
	path := filepath.Join(dir, res.Fingerprint[0]+"-"+res.Fingerprint[1]+".json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	tampered := strings.Replace(string(data), res.Fingerprint[0], strings.Repeat("a", 64), 1)
	if tampered == string(data) {
		t.Fatal("tamper was a no-op")
	}
	if err := os.WriteFile(path, []byte(tampered), 0o644); err != nil {
		t.Fatal(err)
	}
	_, ok, err = s.Load(res.Fingerprint)
	if ok {
		t.Error("Load returned ok for a file with a mismatched stored fingerprint")
	}
	if !errors.Is(err, ErrFingerprintMismatch) {
		t.Errorf("Load error = %v, want ErrFingerprintMismatch", err)
	}

	// A missing file is a plain not-found, not an error.
	missing := [2]string{"no", "such"}
	if _, ok, err := s.Load(missing); ok || err != nil {
		t.Errorf("Load of missing fp: ok=%v err=%v, want ok=false err=nil", ok, err)
	}
}

func TestStore_SaveCreatesDir(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "nested", "store")
	s := NewStore(dir)
	runner := &fakeRunner{arms: map[string]ArmResult{
		"control": buildArm(armSpec{name: "control"}),
		"variant": buildArm(armSpec{name: "variant"}),
	}}
	res := mustCompare(t, controlVariant(), variantVariant(), runner, 5)
	if err := s.Save(res); err != nil {
		t.Fatalf("Save into a missing dir: %v", err)
	}
	if !s.Has(res.Fingerprint) {
		t.Error("Has=false after Save into a fresh dir")
	}
}

func TestToolCallsFromDir_CountsPerTool(t *testing.T) {
	dir := t.TempDir()
	writeTraj := func(session, read, bash string) {
		t.Helper()
		sd := filepath.Join(dir, session)
		if err := os.MkdirAll(sd, 0o755); err != nil {
			t.Fatal(err)
		}
		var b strings.Builder
		b.WriteString(`{"schema_version":"ATIF-v1.6","session_id":"` + session + `","agent":{"name":"pirate"},"steps":[`)
		steps := make([]string, 0, 2)
		for _, tc := range []struct{ id, name string }{{"c1", read}, {"c2", bash}} {
			if tc.name == "" {
				continue
			}
			steps = append(steps, `{"step_id":1,"source":"assistant","tool_calls":[{"tool_call_id":"`+tc.id+`","function_name":"`+tc.name+`","arguments":{}}],"observation":{"results":[{"source_call_id":"`+tc.id+`","content":"ok"}]}}`)
		}
		b.WriteString(strings.Join(steps, ","))
		b.WriteString(`]}`)
		if err := os.WriteFile(filepath.Join(sd, "trajectory.atif.json"), []byte(b.String()), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	writeTraj("sess-a", "read", "bash")
	writeTraj("sess-b", "read", "") // a subagent session: another read

	got, err := ToolCallsFromDir(dir)
	if err != nil {
		t.Fatalf("ToolCallsFromDir: %v", err)
	}
	want := map[string]int{"read": 2, "bash": 1}
	for name, n := range want {
		if got[name] != n {
			t.Errorf("%s = %d, want %d (full map: %v)", name, got[name], n, got)
		}
	}
	if ToolCallsTotal(got) != 3 {
		t.Errorf("total = %d, want 3", ToolCallsTotal(got))
	}

	// Deterministic name listing.
	if names := ToolCallsNames(got); strings.Join(names, ",") != "bash,read" {
		t.Errorf("names = %v, want [bash read]", names)
	}

	// An empty directory yields an empty map, not an error.
	empty, err := ToolCallsFromDir(t.TempDir())
	if err != nil {
		t.Errorf("empty dir: %v", err)
	}
	if len(empty) != 0 || empty == nil {
		t.Errorf("empty dir map = %v, want empty non-nil", empty)
	}
}

func TestMergeToolCalls(t *testing.T) {
	dst := map[string]int{"read": 1}
	got := MergeToolCalls(dst, map[string]int{"read": 2, "bash": 1})
	if got["read"] != 3 || got["bash"] != 1 {
		t.Errorf("merge = %v, want read=3 bash=1", got)
	}
	if &got == &dst {
		t.Error("MergeToolCalls must return the folded dst")
	}
	nilMerged := MergeToolCalls(nil, map[string]int{"read": 2})
	if nilMerged["read"] != 2 {
		t.Errorf("merge into nil = %v, want read=2", nilMerged)
	}
}
