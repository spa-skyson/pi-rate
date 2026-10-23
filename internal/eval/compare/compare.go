// Package compare implements the harness A/B comparison core behind
// make eval-harness-compare (issue #80): it runs two harness variants
// against the same scenarios, contrasts their trajectories and artifacts
// mechanically, renders a deterministic plain-text report, supports blind
// anonymization for judging, and persists results keyed by a variant
// fingerprint so finished arms can be reused.
//
// The package owns no live execution: arms are produced through the Runner
// interface, so unit tests drive it with scripted results and the wiring
// increment can connect the real driver without touching this core.
package compare

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"

	"github.com/spa-skyson/pi-rate/internal/agent"
	"github.com/spa-skyson/pi-rate/internal/eval"
)

// Variant is one harness configuration under comparison: a name for reports,
// the instruction parts injected through the harness injection point
// (issue #79), and an optional free-form label.
type Variant struct {
	Name  string
	Parts *agent.InstructionParts
	Label string
}

// Deliverable is one artifact a run was expected to produce, as recorded by
// the driver that executed the run.
type Deliverable struct {
	Path  string `json:"path"`
	Found bool   `json:"found"`
	Bytes int    `json:"bytes"`
}

// RunResult is the outcome of one scenario run inside an arm.
type RunResult struct {
	// Scenario is the eval scenario name the run executed.
	Scenario string `json:"scenario"`
	// Status is the eval scenario status: eval.StatusPass, StatusFail,
	// StatusSkip or StatusError.
	Status string `json:"status,omitempty"`
	// Deliverables lists the artifacts this run produced (or was expected to
	// produce), for the deliverable contrast.
	Deliverables []Deliverable `json:"deliverables,omitempty"`
	// TrajDir is the sessions directory holding the run's
	// trajectory.atif.json files. Empty for scripted runs.
	TrajDir string `json:"traj_dir,omitempty"`
}

// ArmResult is everything the comparison knows about one arm: all runs of
// the arm, the aggregated trajectory metrics, the per-tool call counts, how
// many runs delivered, and the total prompt inventory size in bytes.
type ArmResult struct {
	Variant          string                 `json:"variant"`
	Runs             []RunResult            `json:"runs,omitempty"`
	Metrics          eval.TrajectoryMetrics `json:"metrics"`
	ToolCalls        map[string]int         `json:"tool_calls,omitempty"`
	Delivered        int                    `json:"delivered"`
	PromptTotalBytes int                    `json:"prompt_total_bytes"`
}

// Contrast kinds. The first four are built mechanically from the arm data;
// verdict appears only when a Judge is supplied; noise-floor marks
// comparisons too small to be significant.
const (
	// ContrastToolset: per-tool call-count deltas between the arms.
	ContrastToolset = "toolset"
	// ContrastSteps: trajectory step and tool-call totals.
	ContrastSteps = "steps"
	// ContrastDeliverable: per-scenario artifact presence and size.
	ContrastDeliverable = "deliverable"
	// ContrastInventory: prompt inventory size difference.
	ContrastInventory = "inventory"
	// ContrastVerdict: the optional judge's comparison verdict.
	ContrastVerdict = "verdict"
	// ContrastNoiseFloor: the run count is below MinSignificantRuns.
	ContrastNoiseFloor = "noise-floor"
)

// Contrast directions. Direction marks the arm in which the measured
// quantity is higher (calls, steps, bytes) or that gained a deliverable;
// DirectionNone when the sides are equal or the contrast is not directional.
const (
	DirectionControl = "control"
	DirectionVariant = "variant"
	DirectionNone    = "none"
)

// MinSignificantRuns is the per-arm run count below which differences are
// reported as noise rather than signal (harness2 research, rule P8).
const MinSignificantRuns = 5

// Contrast is one mechanically established difference between the arms.
// Every contrast carries concrete evidence — numbers or tool names; a
// contrast without evidence is a programming error and panics at
// aggregation time.
type Contrast struct {
	Kind      string `json:"kind"`
	Direction string `json:"direction"`
	Evidence  string `json:"evidence"`
}

// CompareResult is the full outcome of one control-vs-variant comparison.
// Fingerprint holds the sha256 identities of (control, variant).
type CompareResult struct {
	Control     ArmResult  `json:"control"`
	Variant     ArmResult  `json:"variant"`
	Contrasts   []Contrast `json:"contrasts"`
	Fingerprint [2]string  `json:"fingerprint"`
}

// Runner produces one arm of the comparison: it executes (or replays) a
// variant against `runs` runs and returns the aggregated result. The core
// is agnostic of how arms come to be; the live driver implements this
// interface, tests implement it with scripted data.
type Runner interface {
	RunArm(ctx context.Context, v Variant, runs int) (ArmResult, error)
}

// Judge is the optional comparison judge. When supplied to Compare it adds
// one verdict contrast; nil means the arms are compared mechanically only.
type Judge interface {
	Judge(ctx context.Context, control, variant ArmResult) (verdict string, evidence string, err error)
}

// Option configures Compare.
type Option func(*options)

type options struct {
	judge Judge
}

// WithJudge adds an optional judge: Compare then appends one verdict
// contrast built from the judge's verdict and evidence. A judge error does
// not fail the comparison — it degrades into an error-carrying verdict
// contrast, mirroring how the eval judge degrades its report.
func WithJudge(j Judge) Option {
	return func(o *options) { o.judge = j }
}

// Compare runs both arms through the Runner, aggregates the arm results and
// builds the contrast list. Control runs first, then variant, so the whole
// comparison is deterministic for a deterministic Runner.
//
// Compare derives Delivered per arm from the run statuses; Metrics,
// ToolCalls and PromptTotalBytes come from the Runner (see ToolCallsFromDir
// for the per-tool counting helper). With runs < MinSignificantRuns the
// result carries a noise-floor contrast: differences are not significant at
// that sample size.
func Compare(ctx context.Context, control, variant Variant, r Runner, runs int, opts ...Option) (CompareResult, error) {
	if r == nil {
		return CompareResult{}, errors.New("compare: nil Runner")
	}
	if runs <= 0 {
		return CompareResult{}, fmt.Errorf("compare: runs must be positive, got %d", runs)
	}
	var o options
	for _, opt := range opts {
		if opt != nil {
			opt(&o)
		}
	}

	ctrl, err := r.RunArm(ctx, control, runs)
	if err != nil {
		return CompareResult{}, fmt.Errorf("compare: control arm: %w", err)
	}
	varArm, err := r.RunArm(ctx, variant, runs)
	if err != nil {
		return CompareResult{}, fmt.Errorf("compare: variant arm: %w", err)
	}

	ctrl.Variant, varArm.Variant = control.Name, variant.Name
	ctrl.Delivered = countDelivered(ctrl.Runs)
	varArm.Delivered = countDelivered(varArm.Runs)

	res := CompareResult{
		Control:     ctrl,
		Variant:     varArm,
		Fingerprint: [2]string{Fingerprint(control), Fingerprint(variant)},
	}
	res.Contrasts = buildContrasts(ctx, res, runs, o.judge)
	return res, nil
}

// countDelivered counts the runs that reached a passing outcome.
func countDelivered(runs []RunResult) int {
	n := 0
	for _, r := range runs {
		if r.Status == eval.StatusPass {
			n++
		}
	}
	return n
}

// buildContrasts assembles every contrast in fixed, sorted order: toolset
// deltas by tool name, steps, deliverable diffs by scenario, inventory,
// verdict (only with a judge), noise-floor (only under the threshold). It
// ends with the evidence invariant check — a contrast without a kind or
// evidence is a programming error, not a reportable state.
func buildContrasts(ctx context.Context, res CompareResult, runs int, j Judge) []Contrast {
	var cs []Contrast
	cs = append(cs, toolsetContrasts(res.Control.ToolCalls, res.Variant.ToolCalls)...)
	if c, ok := stepsContrast(res.Control.Metrics, res.Variant.Metrics); ok {
		cs = append(cs, c)
	}
	cs = append(cs, deliverableContrasts(res)...)
	if c, ok := inventoryContrast(res.Control.PromptTotalBytes, res.Variant.PromptTotalBytes); ok {
		cs = append(cs, c)
	}
	if j != nil {
		cs = append(cs, verdictContrast(ctx, j, res.Control, res.Variant))
	}
	if runs < MinSignificantRuns {
		cs = append(cs, noiseFloorContrast(runs))
	}
	if err := validateContrasts(cs); err != nil {
		panic(err)
	}
	return cs
}

// toolsetContrasts emits one contrast per tool whose call count differs
// between the arms, sorted by tool name.
func toolsetContrasts(control, variant map[string]int) []Contrast {
	seen := make(map[string]bool, len(control)+len(variant))
	for t := range control {
		seen[t] = true
	}
	for t := range variant {
		seen[t] = true
	}
	var out []Contrast
	for _, t := range sortedKeys(seen) {
		c, v := control[t], variant[t]
		if c == v {
			continue
		}
		direction := DirectionVariant
		if v < c {
			direction = DirectionControl
		}
		out = append(out, Contrast{
			Kind:      ContrastToolset,
			Direction: direction,
			Evidence:  fmt.Sprintf("%s: control %d call(s), variant %d call(s) (delta %+d)", t, c, v, v-c),
		})
	}
	return out
}

// stepsContrast contrasts the trajectory totals. Emitted when either total
// differs; direction points at the heavier arm.
func stepsContrast(control, variant eval.TrajectoryMetrics) (Contrast, bool) {
	if control.TotalSteps == variant.TotalSteps && control.TotalToolCalls == variant.TotalToolCalls {
		return Contrast{}, false
	}
	c := Contrast{
		Kind:      ContrastSteps,
		Direction: DirectionNone,
		Evidence: fmt.Sprintf("steps: control %d, variant %d; tool calls: control %d, variant %d",
			control.TotalSteps, variant.TotalSteps, control.TotalToolCalls, variant.TotalToolCalls),
	}
	switch {
	case control.TotalSteps != variant.TotalSteps:
		c.Direction = heavierDirection(control.TotalSteps, variant.TotalSteps)
	default:
		c.Direction = heavierDirection(control.TotalToolCalls, variant.TotalToolCalls)
	}
	return c, true
}

// heavierDirection names the arm carrying the larger value, none when equal.
func heavierDirection(control, variant int) string {
	switch {
	case control > variant:
		return DirectionControl
	case variant > control:
		return DirectionVariant
	default:
		return DirectionNone
	}
}

// artifact is the aggregated per-path deliverable state across an arm's runs
// of one scenario: present if any run produced it, bytes = the largest size
// seen.
type artifact struct {
	present bool
	bytes   int
}

// deliverableContrasts contrasts per-scenario artifacts. One contrast per
// scenario with at least one presence or size difference, sorted by
// scenario; direction points at the arm with more or larger deliverables,
// none when the diffs pull both ways.
func deliverableContrasts(res CompareResult) []Contrast {
	scenarios := make(map[string]bool)
	for _, r := range res.Control.Runs {
		if len(r.Deliverables) > 0 {
			scenarios[r.Scenario] = true
		}
	}
	for _, r := range res.Variant.Runs {
		if len(r.Deliverables) > 0 {
			scenarios[r.Scenario] = true
		}
	}

	var out []Contrast
	for _, sc := range sortedKeys(scenarios) {
		ctrl := deliverablesByPath(res.Control.Runs, sc)
		varnt := deliverablesByPath(res.Variant.Runs, sc)
		paths := make(map[string]bool, len(ctrl)+len(varnt))
		for p := range ctrl {
			paths[p] = true
		}
		for p := range varnt {
			paths[p] = true
		}

		var parts []string
		ctrlPresent, varntPresent := 0, 0
		ctrlBytes, varntBytes := 0, 0
		for _, p := range sortedKeys(paths) {
			c, v := ctrl[p], varnt[p]
			if c == v {
				continue
			}
			parts = append(parts, describeDeliverableDiff(p, c, v))
			if c.present {
				ctrlPresent++
				ctrlBytes += c.bytes
			}
			if v.present {
				varntPresent++
				varntBytes += v.bytes
			}
		}
		if len(parts) == 0 {
			continue
		}
		out = append(out, Contrast{
			Kind:      ContrastDeliverable,
			Direction: deliverableDirection(ctrlPresent, varntPresent, ctrlBytes, varntBytes),
			Evidence:  fmt.Sprintf("%s: %s", sc, strings.Join(parts, "; ")),
		})
	}
	return out
}

// deliverablesByPath folds an arm's runs of one scenario into per-path
// artifact states.
func deliverablesByPath(runs []RunResult, scenario string) map[string]artifact {
	m := make(map[string]artifact)
	for _, r := range runs {
		if r.Scenario != scenario {
			continue
		}
		for _, d := range r.Deliverables {
			a := m[d.Path]
			if d.Found {
				a.present = true
			}
			if d.Bytes > a.bytes {
				a.bytes = d.Bytes
			}
			m[d.Path] = a
		}
	}
	return m
}

// deliverableDirection picks the arm whose deliverables gained: by presence
// count first, then by total bytes; none when tied or mixed.
func deliverableDirection(ctrlPresent, varntPresent, ctrlBytes, varntBytes int) string {
	switch {
	case ctrlPresent == varntPresent && ctrlBytes == varntBytes:
		return DirectionNone
	case ctrlPresent >= varntPresent && ctrlBytes >= varntBytes:
		return DirectionControl
	case varntPresent >= ctrlPresent && varntBytes >= ctrlBytes:
		return DirectionVariant
	default:
		return DirectionNone
	}
}

func describeDeliverableDiff(path string, c, v artifact) string {
	switch {
	case c.present && !v.present:
		return fmt.Sprintf("%s: present in control (%d B), absent in variant", path, c.bytes)
	case !c.present && v.present:
		return fmt.Sprintf("%s: absent in control, present in variant (%d B)", path, v.bytes)
	default:
		return fmt.Sprintf("%s: control %d B, variant %d B", path, c.bytes, v.bytes)
	}
}

// inventoryContrast contrasts the prompt inventory sizes. Emitted when they
// differ; direction points at the larger inventory.
func inventoryContrast(controlBytes, variantBytes int) (Contrast, bool) {
	if controlBytes == variantBytes {
		return Contrast{}, false
	}
	return Contrast{
		Kind:      ContrastInventory,
		Direction: heavierDirection(controlBytes, variantBytes),
		Evidence: fmt.Sprintf("prompt bytes: control %d, variant %d (delta %+d)",
			controlBytes, variantBytes, variantBytes-controlBytes),
	}, true
}

// verdictContrast folds the judge's answer into one contrast. A judge error
// degrades into an error-carrying contrast instead of failing the
// comparison — the judge is advisory.
func verdictContrast(ctx context.Context, j Judge, control, variant ArmResult) Contrast {
	verdict, evidence, err := j.Judge(ctx, control, variant)
	if err != nil {
		return Contrast{
			Kind:      ContrastVerdict,
			Direction: DirectionNone,
			Evidence:  fmt.Sprintf("judge error: %v", err),
		}
	}
	var b strings.Builder
	if verdict != "" {
		b.WriteString("verdict: " + verdict)
	}
	if evidence != "" {
		if b.Len() > 0 {
			b.WriteString("; ")
		}
		b.WriteString(evidence)
	}
	ev := b.String()
	if ev == "" {
		ev = "judge returned neither verdict nor evidence"
	}
	return Contrast{Kind: ContrastVerdict, Direction: DirectionNone, Evidence: ev}
}

// noiseFloorContrast marks the comparison as below the significance
// threshold.
func noiseFloorContrast(runs int) Contrast {
	return Contrast{
		Kind:      ContrastNoiseFloor,
		Direction: DirectionNone,
		Evidence:  fmt.Sprintf("runs=%d<%d; differences are not significant", runs, MinSignificantRuns),
	}
}

// validateContrasts enforces the contrast invariant: every contrast has a
// kind, concrete evidence and a known direction.
func validateContrasts(cs []Contrast) error {
	for i, c := range cs {
		if c.Kind == "" {
			return fmt.Errorf("compare: contrast %d: empty kind", i)
		}
		if strings.TrimSpace(c.Evidence) == "" {
			return fmt.Errorf("compare: contrast %d (%s): empty evidence", i, c.Kind)
		}
		switch c.Direction {
		case DirectionControl, DirectionVariant, DirectionNone:
		default:
			return fmt.Errorf("compare: contrast %d (%s): unknown direction %q", i, c.Kind, c.Direction)
		}
	}
	return nil
}

// sortedKeys returns the map keys in ascending order, so every listing in
// this package is deterministic.
func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
