package compare

import (
	"strings"
	"testing"
)

func TestAnonymize_DeterministicMapping(t *testing.T) {
	build := func() CompareResult {
		runner := &fakeRunner{arms: map[string]ArmResult{
			"control": buildArm(armSpec{name: "control", steps: 10, toolCalls: 12, calls: map[string]int{"read": 4}, bytes: 1000}),
			"variant": buildArm(armSpec{name: "variant", steps: 8, toolCalls: 9, calls: map[string]int{"read": 6}, bytes: 1300}),
		}}
		return mustCompare(t, controlVariant(), variantVariant(), runner, 5)
	}

	first := Anonymize(build(), 42)
	second := Anonymize(build(), 42)

	if first.Control.Variant != second.Control.Variant || first.Variant.Variant != second.Variant.Variant {
		t.Fatalf("same seed produced different names: (%q,%q) vs (%q,%q)",
			first.Control.Variant, first.Variant.Variant, second.Control.Variant, second.Variant.Variant)
	}
	for _, name := range []string{first.Control.Variant, first.Variant.Variant} {
		if name != "A" && name != "B" {
			t.Errorf("anonymized name = %q, want A or B", name)
		}
	}
	if first.Control.Variant == first.Variant.Variant {
		t.Fatal("both arms got the same anonymized name")
	}

	// Directions are remapped consistently with the names.
	for _, c := range first.Contrasts {
		switch c.Direction {
		case DirectionControl, DirectionVariant:
			t.Errorf("contrast %s kept a non-anonymous direction %q", c.Kind, c.Direction)
		case "A", "B", DirectionNone:
		default:
			t.Errorf("contrast %s has unknown direction %q", c.Kind, c.Direction)
		}
	}

	// Different seeds may swap the assignment; verify over a spread of seeds
	// that both mappings occur, without asserting which seed gives which.
	seen := map[string]bool{}
	for seed := int64(0); seed < 64; seed++ {
		anon := Anonymize(build(), seed)
		seen[anon.Control.Variant] = true
	}
	if !seen["A"] || !seen["B"] {
		t.Errorf("seed sweep produced control names %v — the mapping does not vary with the seed", seen)
	}
}

func TestAnonymize_EvidenceScrubbed(t *testing.T) {
	runner := &fakeRunner{arms: map[string]ArmResult{
		"control": buildArm(armSpec{name: "control", steps: 10, toolCalls: 12, calls: map[string]int{"read": 4}}),
		"variant": buildArm(armSpec{name: "variant", steps: 8, toolCalls: 9, calls: map[string]int{"read": 6}}),
	}}
	res := mustCompare(t, controlVariant(), variantVariant(), runner, 5)
	anon := Anonymize(res, 7)

	for _, c := range anon.Contrasts {
		if strings.Contains(c.Evidence, "control") || strings.Contains(c.Evidence, "variant") {
			t.Errorf("contrast %s evidence leaks an arm name: %q", c.Kind, c.Evidence)
		}
	}
}

func TestAnonymize_DoesNotMutateOriginal(t *testing.T) {
	runner := &fakeRunner{arms: map[string]ArmResult{
		"control": buildArm(armSpec{name: "control", steps: 10, calls: map[string]int{"read": 4}}),
		"variant": buildArm(armSpec{name: "variant", steps: 8, calls: map[string]int{"read": 6}}),
	}}
	res := mustCompare(t, controlVariant(), variantVariant(), runner, 5)
	_ = Anonymize(res, 3)
	if res.Control.Variant != "control" || res.Variant.Variant != "variant" {
		t.Fatal("Anonymize mutated its input")
	}
}

func TestRender_SectionsAndDeterminism(t *testing.T) {
	runner := &fakeRunner{arms: map[string]ArmResult{
		"control": buildArm(armSpec{
			name:      "control",
			steps:     10,
			toolCalls: 12,
			calls:     map[string]int{"read": 4, "bash": 8},
			bytes:     1000,
			runs:      []RunResult{{Scenario: "grep-basics", Status: "pass"}},
		}),
		"variant": buildArm(armSpec{
			name:      "variant",
			steps:     8,
			toolCalls: 9,
			calls:     map[string]int{"read": 6},
			bytes:     1300,
			runs:      []RunResult{{Scenario: "grep-basics", Status: "pass"}},
		}),
	}}
	res := mustCompare(t, controlVariant(), variantVariant(), runner, 2)

	out := Render(res)
	for _, want := range []string{
		"# Harness comparison",
		"## Control: control",
		"## Variant: variant",
		"## Contrasts",
		"noise-floor",
		"WARNING",
		"fingerprint " + res.Fingerprint[0],
		"fingerprint " + res.Fingerprint[1],
		"read: control 4 call(s)",
		"prompt bytes: 1000",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("render missing %q", want)
		}
	}
	if out != Render(res) {
		t.Error("Render is not deterministic")
	}
	if strings.Contains(out, "Mermaid") {
		t.Error("no mermaid in the plain report")
	}
}

func TestRender_NoContrasts(t *testing.T) {
	same := buildArm(armSpec{name: "x", calls: map[string]int{"read": 1}})
	runner := &fakeRunner{arms: map[string]ArmResult{"control": same, "variant": same}}
	res := mustCompare(t, controlVariant(), variantVariant(), runner, 5)
	out := Render(res)
	if !strings.Contains(out, "mechanically indistinguishable") {
		t.Errorf("empty-contrast report missing the explicit none line:\n%s", out)
	}
}
