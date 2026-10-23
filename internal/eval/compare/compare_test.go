package compare

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/spa-skyson/pi-rate/internal/agent"
	"github.com/spa-skyson/pi-rate/internal/eval"
)

// fakeRunner returns the scripted ArmResult for each variant name, recording
// the calls it received so tests can assert on execution order and run
// counts.
type fakeRunner struct {
	arms  map[string]ArmResult
	err   error
	calls []string
}

func (f *fakeRunner) RunArm(_ context.Context, v Variant, runs int) (ArmResult, error) {
	f.calls = append(f.calls, fmt.Sprintf("%s/%d", v.Name, runs))
	if f.err != nil {
		return ArmResult{}, f.err
	}
	arm, ok := f.arms[v.Name]
	if !ok {
		return ArmResult{}, fmt.Errorf("fakeRunner: no scripted arm for variant %q", v.Name)
	}
	return arm, nil
}

// staticJudge returns a fixed verdict and evidence.
type staticJudge struct {
	verdict  string
	evidence string
	err      error
}

func (s staticJudge) Judge(_ context.Context, _, _ ArmResult) (string, string, error) {
	return s.verdict, s.evidence, s.err
}

// controlVariant is the minimal control variant every comparison in these
// tests starts from.
func controlVariant() Variant {
	return Variant{Name: "control", Parts: &agent.InstructionParts{Base: "base prompt"}}
}

// variantVariant differs from the control by one rules byte, so the two
// fingerprints are distinct.
func variantVariant() Variant {
	return Variant{Name: "variant", Parts: &agent.InstructionParts{Base: "base prompt", Rules: "extra rules"}}
}

// arm builds an ArmResult with the given knobs; the zero-value fields make
// individual tests' intent obvious at the call site.
type armSpec struct {
	name      string
	steps     int
	toolCalls int
	calls     map[string]int
	bytes     int
	runs      []RunResult
}

func buildArm(s armSpec) ArmResult {
	arm := ArmResult{
		Variant:          s.name,
		Runs:             s.runs,
		ToolCalls:        s.calls,
		PromptTotalBytes: s.bytes,
	}
	arm.Metrics.TotalSteps = s.steps
	arm.Metrics.TotalToolCalls = s.toolCalls
	return arm
}

// mustCompare runs Compare and fails t on error.
func mustCompare(t *testing.T, ctrl, varnt Variant, r Runner, runs int, opts ...Option) CompareResult {
	t.Helper()
	res, err := Compare(context.Background(), ctrl, varnt, r, runs, opts...)
	if err != nil {
		t.Fatalf("Compare: unexpected error: %v", err)
	}
	return res
}

func TestCompare_BuildsContrastsWithEvidence(t *testing.T) {
	runner := &fakeRunner{arms: map[string]ArmResult{
		"control": buildArm(armSpec{
			name:      "control",
			steps:     10,
			toolCalls: 12,
			calls:     map[string]int{"read": 4, "bash": 8},
			bytes:     1000,
			runs: []RunResult{
				{Scenario: "grep-basics", Status: eval.StatusPass, Deliverables: []Deliverable{{Path: "out.txt", Found: true, Bytes: 100}}},
			},
		}),
		"variant": buildArm(armSpec{
			name:      "variant",
			steps:     8,
			toolCalls: 9,
			calls:     map[string]int{"read": 6, "grep": 3},
			bytes:     1300,
			runs: []RunResult{
				{Scenario: "grep-basics", Status: eval.StatusPass, Deliverables: []Deliverable{{Path: "out.txt", Found: true, Bytes: 150}}},
			},
		}),
	}}

	res := mustCompare(t, controlVariant(), variantVariant(), runner, 5)

	kinds := map[string]bool{}
	for _, c := range res.Contrasts {
		if strings.TrimSpace(c.Evidence) == "" {
			t.Fatalf("contrast %s has empty evidence", c.Kind)
		}
		kinds[c.Kind] = true
	}
	for _, want := range []string{ContrastToolset, ContrastSteps, ContrastDeliverable, ContrastInventory} {
		if !kinds[want] {
			t.Errorf("missing contrast kind %q; got %v", want, kinds)
		}
	}
	if kinds[ContrastVerdict] {
		t.Error("verdict contrast present without a judge")
	}
	if kinds[ContrastNoiseFloor] {
		t.Error("noise-floor contrast present at runs=5")
	}
	// Arms run control-first.
	wantCalls := []string{"control/5", "variant/5"}
	if strings.Join(runner.calls, ",") != strings.Join(wantCalls, ",") {
		t.Errorf("runner call order = %v, want %v", runner.calls, wantCalls)
	}
	// Delivered is derived from run statuses.
	if res.Control.Delivered != 1 || res.Variant.Delivered != 1 {
		t.Errorf("delivered = (%d, %d), want (1, 1)", res.Control.Delivered, res.Variant.Delivered)
	}
	// Fingerprint identifies the pair.
	if res.Fingerprint[0] == res.Fingerprint[1] {
		t.Error("identical fingerprints for distinct variants")
	}
	if res.Fingerprint[0] != Fingerprint(controlVariant()) || res.Fingerprint[1] != Fingerprint(variantVariant()) {
		t.Error("fingerprint does not match per-variant Fingerprint()")
	}
}

func TestCompare_NoiseFloorBelowThreshold(t *testing.T) {
	for _, runs := range []int{1, 4} {
		runner := &fakeRunner{arms: map[string]ArmResult{
			"control": buildArm(armSpec{name: "control", calls: map[string]int{}}),
			"variant": buildArm(armSpec{name: "variant", calls: map[string]int{}}),
		}}
		res := mustCompare(t, controlVariant(), variantVariant(), runner, runs)
		var nf *Contrast
		for i := range res.Contrasts {
			if res.Contrasts[i].Kind == ContrastNoiseFloor {
				nf = &res.Contrasts[i]
			}
		}
		if nf == nil {
			t.Fatalf("runs=%d: no noise-floor contrast", runs)
		}
		if nf.Direction != DirectionNone {
			t.Errorf("runs=%d: direction = %q, want none", runs, nf.Direction)
		}
		if !strings.Contains(nf.Evidence, fmt.Sprintf("runs=%d<%d", runs, MinSignificantRuns)) {
			t.Errorf("runs=%d: evidence = %q, want it to name the threshold", runs, nf.Evidence)
		}
	}
}

func TestCompare_AtThresholdNoNoiseFloor(t *testing.T) {
	runner := &fakeRunner{arms: map[string]ArmResult{
		"control": buildArm(armSpec{name: "control"}),
		"variant": buildArm(armSpec{name: "variant"}),
	}}
	res := mustCompare(t, controlVariant(), variantVariant(), runner, MinSignificantRuns)
	for _, c := range res.Contrasts {
		if c.Kind == ContrastNoiseFloor {
			t.Fatalf("noise-floor contrast present at runs=%d", MinSignificantRuns)
		}
	}
}

func TestCompare_ToolsetDelta(t *testing.T) {
	runner := &fakeRunner{arms: map[string]ArmResult{
		"control": buildArm(armSpec{name: "control", calls: map[string]int{"read": 4, "bash": 8, "edit": 1}}),
		"variant": buildArm(armSpec{name: "variant", calls: map[string]int{"read": 6, "bash": 8, "grep": 2}}),
	}}
	res := mustCompare(t, controlVariant(), variantVariant(), runner, 5)

	var got []string
	for _, c := range res.Contrasts {
		if c.Kind != ContrastToolset {
			continue
		}
		got = append(got, c.Evidence)
		switch {
		case strings.HasPrefix(c.Evidence, "read:"):
			if c.Direction != DirectionVariant {
				t.Errorf("read: direction = %q, want variant", c.Direction)
			}
			if !strings.Contains(c.Evidence, "delta +2") {
				t.Errorf("read: evidence = %q, want delta +2", c.Evidence)
			}
		case strings.HasPrefix(c.Evidence, "edit:"):
			if c.Direction != DirectionControl {
				t.Errorf("edit: direction = %q, want control", c.Direction)
			}
			if !strings.Contains(c.Evidence, "delta -1") {
				t.Errorf("edit: evidence = %q, want delta -1", c.Evidence)
			}
		case strings.HasPrefix(c.Evidence, "grep:"):
			if c.Direction != DirectionVariant {
				t.Errorf("grep: direction = %q, want variant", c.Direction)
			}
			if !strings.Contains(c.Evidence, "delta +2") {
				t.Errorf("grep: evidence = %q, want delta +2", c.Evidence)
			}
		default:
			t.Errorf("unexpected toolset contrast %q", c.Evidence)
		}
	}
	if len(got) != 3 {
		t.Fatalf("got %d toolset contrasts (%v), want 3 — bash is equal and must not appear", len(got), got)
	}
}

func TestCompare_JudgeOptional(t *testing.T) {
	arms := func() map[string]ArmResult {
		return map[string]ArmResult{
			"control": buildArm(armSpec{name: "control"}),
			"variant": buildArm(armSpec{name: "variant"}),
		}
	}

	t.Run("nil judge no verdict", func(t *testing.T) {
		res := mustCompare(t, controlVariant(), variantVariant(), &fakeRunner{arms: arms()}, 5)
		for _, c := range res.Contrasts {
			if c.Kind == ContrastVerdict {
				t.Fatalf("verdict contrast without judge: %+v", c)
			}
		}
	})

	t.Run("judge present adds verdict", func(t *testing.T) {
		j := staticJudge{verdict: "variant", evidence: "variant ran 3 fewer steps"}
		res := mustCompare(t, controlVariant(), variantVariant(), &fakeRunner{arms: arms()}, 5, WithJudge(j))
		var found *Contrast
		for i := range res.Contrasts {
			if res.Contrasts[i].Kind == ContrastVerdict {
				found = &res.Contrasts[i]
			}
		}
		if found == nil {
			t.Fatal("no verdict contrast with judge set")
		}
		if !strings.Contains(found.Evidence, "variant ran 3 fewer steps") || !strings.Contains(found.Evidence, "verdict: variant") {
			t.Errorf("verdict evidence = %q, want verdict and evidence folded in", found.Evidence)
		}
	})

	t.Run("judge error degrades", func(t *testing.T) {
		j := staticJudge{err: errors.New("model unavailable")}
		res := mustCompare(t, controlVariant(), variantVariant(), &fakeRunner{arms: arms()}, 5, WithJudge(j))
		var found *Contrast
		for i := range res.Contrasts {
			if res.Contrasts[i].Kind == ContrastVerdict {
				found = &res.Contrasts[i]
			}
		}
		if found == nil {
			t.Fatal("no verdict contrast despite judge error — the judge is advisory and must degrade, not fail")
		}
		if !strings.Contains(found.Evidence, "model unavailable") {
			t.Errorf("verdict evidence = %q, want the judge error carried", found.Evidence)
		}
	})

	t.Run("runner error surfaces", func(t *testing.T) {
		_, err := Compare(context.Background(), controlVariant(), variantVariant(), &fakeRunner{err: errors.New("boom")}, 5)
		if err == nil || !strings.Contains(err.Error(), "boom") {
			t.Fatalf("err = %v, want the runner error wrapped", err)
		}
	})

	t.Run("invalid inputs", func(t *testing.T) {
		runner := &fakeRunner{arms: arms()}
		if _, err := Compare(context.Background(), controlVariant(), variantVariant(), nil, 5); err == nil {
			t.Error("nil runner accepted")
		}
		if _, err := Compare(context.Background(), controlVariant(), variantVariant(), runner, 0); err == nil {
			t.Error("zero runs accepted")
		}
		if _, err := Compare(context.Background(), controlVariant(), variantVariant(), runner, -3); err == nil {
			t.Error("negative runs accepted")
		}
	})
}

func TestCompare_EqualArmsNoContrasts(t *testing.T) {
	same := buildArm(armSpec{name: "x", steps: 3, toolCalls: 5, calls: map[string]int{"read": 5}, bytes: 500})
	runner := &fakeRunner{arms: map[string]ArmResult{"control": same, "variant": same}}
	res := mustCompare(t, controlVariant(), variantVariant(), runner, 5)
	if len(res.Contrasts) != 0 {
		t.Errorf("equal arms produced contrasts: %+v", res.Contrasts)
	}
}

func TestFingerprint_StableAndDistinct(t *testing.T) {
	a := Variant{Name: "v", Label: "L", Parts: &agent.InstructionParts{Base: "b", Rules: "r", Skills: "s", Env: "e"}}
	b := Variant{Name: "v", Label: "L", Parts: &agent.InstructionParts{Base: "b", Rules: "r", Skills: "s", Env: "e"}}
	if Fingerprint(a) != Fingerprint(b) {
		t.Fatal("identical variants must share a fingerprint")
	}
	// Every part participates.
	for i, mutate := range []func(*Variant){
		func(v *Variant) { v.Name = "other" },
		func(v *Variant) { v.Label = "other" },
		func(v *Variant) { v.Parts.Base = "other" },
		func(v *Variant) { v.Parts.Rules = "other" },
		func(v *Variant) { v.Parts.Skills = "other" },
		func(v *Variant) { v.Parts.Env = "other" },
	} {
		m := *a.Parts
		v := Variant{Name: a.Name, Label: a.Label, Parts: &m}
		mutate(&v)
		if Fingerprint(v) == Fingerprint(a) {
			t.Errorf("mutation %d did not change the fingerprint", i)
		}
	}
	// Nil parts fingerprint like empty parts, and stay stable across calls.
	nilParts := Variant{Name: "v"}
	emptyParts := Variant{Name: "v", Parts: &agent.InstructionParts{}}
	if Fingerprint(nilParts) != Fingerprint(emptyParts) {
		t.Error("nil parts and empty parts must fingerprint the same")
	}
	if Fingerprint(nilParts) != Fingerprint(Variant{Name: "v"}) {
		t.Error("fingerprint is not stable across calls")
	}
}
