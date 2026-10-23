package compare

import (
	"fmt"
	"math/rand/v2"
	"strings"
)

// Anonymize prepares a CompareResult for blind judging (harness2 rule P9):
// it returns a copy whose arm names are replaced with "A"/"B", so a judge
// cannot tell which arm is the control from the report alone.
//
// The mapping is deterministic in the seed: the arms are ordered by their
// fingerprint strings, a seeded shuffle of the two-slot name list picks
// which slot gets "A", and the assignment is applied consistently to both
// arms and to every contrast direction. Same seed, same assignment —
// different seeds can swap it, which is the point of seeding: rejudge under
// a different mapping to confirm the verdict does not track the name.
func Anonymize(res CompareResult, seed int64) CompareResult {
	out := res
	// Contrasts is a slice: the shallow copy above shares the backing array
	// with res, and the Direction rewrite below would leak into the caller's
	// result. Copy it before mutating.
	out.Contrasts = append([]Contrast(nil), res.Contrasts...)
	// Order arms by fingerprint so the mapping depends on the comparison's
	// identity, not on struct field order.
	ctrl, varnt := res.Control.Variant, res.Variant.Variant
	firstIsControl := res.Fingerprint[0] <= res.Fingerprint[1]
	rng := rand.New(rand.NewPCG(uint64(seed), 0))

	// names[0] is the anonymized name of the fingerprint-first arm.
	names := [2]string{"A", "B"}
	rng.Shuffle(len(names), func(i, j int) { names[i], names[j] = names[j], names[i] })

	nameOf := map[string]string{
		ctrl:  names[b2i(!firstIsControl)],
		varnt: names[b2i(firstIsControl)],
	}

	out.Control.Variant = nameOf[ctrl]
	out.Variant.Variant = nameOf[varnt]
	for i := range out.Contrasts {
		switch out.Contrasts[i].Direction {
		case DirectionControl:
			out.Contrasts[i].Direction = nameOf[ctrl]
		case DirectionVariant:
			out.Contrasts[i].Direction = nameOf[varnt]
		}
	}
	// Evidence strings quote the arm names where the drivers wrote them;
	// anonymize those mentions too, longest name first so no partial
	// replacement slips through.
	for i := range out.Contrasts {
		for _, name := range longestFirst([]string{ctrl, varnt}) {
			if name != "" && name != "A" && name != "B" {
				out.Contrasts[i].Evidence = strings.ReplaceAll(out.Contrasts[i].Evidence, name, nameOf[name])
			}
		}
	}
	return out
}

// longestFirst orders names by descending length so replacing the longer of
// two names that share a prefix cannot leave a partial match behind.
func longestFirst(names []string) []string {
	out := append([]string(nil), names...)
	for i := 1; i < len(out); i++ {
		for j := i; j > 0 && len(out[j]) > len(out[j-1]); j-- {
			out[j], out[j-1] = out[j-1], out[j]
		}
	}
	return out
}

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}

// Render produces the human-readable report: identity, both arm summaries,
// the contrast table, and the noise-floor warning when present. Identical
// inputs render byte-identically — no timestamps, no map order.
func Render(res CompareResult) string {
	var b strings.Builder
	fp := func(i int) string {
		if i < len(res.Fingerprint) {
			return res.Fingerprint[i]
		}
		return ""
	}
	fmt.Fprintf(&b, "# Harness comparison\n\n")
	fmt.Fprintf(&b, "Control: %s (fingerprint %s)\n", res.Control.Variant, fp(0))
	fmt.Fprintf(&b, "Variant: %s (fingerprint %s)\n\n", res.Variant.Variant, fp(1))

	writeArm(&b, "Control", res.Control)
	writeArm(&b, "Variant", res.Variant)

	fmt.Fprintf(&b, "## Contrasts\n\n")
	if len(res.Contrasts) == 0 {
		b.WriteString("none — the arms are mechanically indistinguishable\n\n")
	} else {
		fmt.Fprintf(&b, "%-12s %-10s %s\n", "kind", "direction", "evidence")
		for _, c := range res.Contrasts {
			fmt.Fprintf(&b, "%-12s %-10s %s\n", c.Kind, c.Direction, c.Evidence)
		}
		b.WriteString("\n")
	}
	if hasNoiseFloor(res.Contrasts) {
		b.WriteString("WARNING: run count below the significance threshold — treat every contrast above as noise, not signal.\n\n")
	}
	return b.String()
}

// writeArm renders one arm's summary section.
func writeArm(b *strings.Builder, title string, arm ArmResult) {
	fmt.Fprintf(b, "## %s: %s\n\n", title, arm.Variant)
	fmt.Fprintf(b, "- runs: %d, delivered: %d\n", len(arm.Runs), arm.Delivered)
	fmt.Fprintf(b, "- steps: %d, tool calls: %d\n", arm.Metrics.TotalSteps, arm.Metrics.TotalToolCalls)
	fmt.Fprintf(b, "- prompt bytes: %d\n", arm.PromptTotalBytes)
	if len(arm.ToolCalls) > 0 {
		b.WriteString("- tool calls by tool:")
		for _, name := range ToolCallsNames(arm.ToolCalls) {
			fmt.Fprintf(b, " %s=%d", name, arm.ToolCalls[name])
		}
		b.WriteString("\n")
	}
	if len(arm.Runs) > 0 {
		b.WriteString("- runs by scenario:")
		for _, r := range arm.Runs {
			status := r.Status
			if status == "" {
				status = "n/a"
			}
			fmt.Fprintf(b, " %s=%s", r.Scenario, status)
		}
		b.WriteString("\n")
	}
	b.WriteString("\n")
}

// hasNoiseFloor reports whether the contrast list carries a noise-floor
// marker.
func hasNoiseFloor(cs []Contrast) bool {
	for _, c := range cs {
		if c.Kind == ContrastNoiseFloor {
			return true
		}
	}
	return false
}
