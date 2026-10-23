package compare

import (
	"sort"

	"github.com/spa-skyson/pi-rate/internal/atif"
	"github.com/spa-skyson/pi-rate/internal/eval"
)

// ToolCallsFromDir counts per-tool calls over every ATIF trajectory under a
// sessions directory — the same shape of counting the eval coverage matrix
// uses (eval's callStatsByTool over LoadTrajectories), exported here so the
// live Runner can fill ArmResult.ToolCalls from what the runs actually
// wrote. Subagent trajectories are included: every sessions subdirectory
// with a trajectory.atif.json counts.
//
// Malformed or incomplete trajectory files are skipped, mirroring
// eval.LoadTrajectories. A directory with no readable trajectories yields
// an empty (non-nil) map and no error.
func ToolCallsFromDir(sessionsDir string) (map[string]int, error) {
	out := make(map[string]int)
	loaded, err := eval.LoadTrajectories(sessionsDir)
	if err != nil {
		return nil, err
	}
	for _, lt := range loaded {
		if lt == nil || lt.Traj == nil {
			continue
		}
		for i := range lt.Traj.Steps {
			for _, tc := range lt.Traj.Steps[i].ToolCalls {
				if tc.FunctionName == "" {
					continue
				}
				out[tc.FunctionName]++
			}
		}
	}
	return out, nil
}

// ToolCallsFromTraj counts per-tool calls in one parsed trajectory. Used by
// ToolCallsFromDir and available to drivers that already hold parsed
// trajectories.
func ToolCallsFromTraj(t *atif.Trajectory) map[string]int {
	out := make(map[string]int)
	if t == nil {
		return out
	}
	for i := range t.Steps {
		for _, tc := range t.Steps[i].ToolCalls {
			if tc.FunctionName == "" {
				continue
			}
			out[tc.FunctionName]++
		}
	}
	return out
}

// MergeToolCalls folds src into dst, summing counts.
func MergeToolCalls(dst, src map[string]int) map[string]int {
	if dst == nil {
		dst = make(map[string]int, len(src))
	}
	for k, n := range src {
		dst[k] += n
	}
	return dst
}

// ToolCallsTotal returns the sum of all per-tool counts.
func ToolCallsTotal(calls map[string]int) int {
	n := 0
	for _, c := range calls {
		n += c
	}
	return n
}

// ToolCallsNames returns an arm's tool names in sorted order, so drivers and
// tests can assert on the map deterministically.
func ToolCallsNames(calls map[string]int) []string {
	names := make([]string, 0, len(calls))
	for name := range calls {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}
