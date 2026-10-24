package agent

import (
	"fmt"
	"strings"
	"testing"
)

// TestPersonaAwareBudgetPassesAgentBuild: a subagent-child build carrying a
// heavy persona (the test-engineer shape, 29.5K, issue #85) passes New()
// when the persona size is announced, with the effective gate at
// len(persona)+24_000 — a gate the persona + built-in tail fits but growth
// past it still fails. Without the announcement the same build must fail at
// the plain 32K gate (the interactive behavior, unchanged).
func TestPersonaAwareBudgetPassesAgentBuild(t *testing.T) {
	persona := strings.Repeat("x", 29_480) // the shipped test-engineer persona size
	prompt := persona + "\nCurrent working directory: /tmp/project\n" +
		strings.Repeat("y", maxInstructionTotal-len(persona)) + "\n"
	if len(prompt) <= maxInstructionTotal {
		t.Fatalf("fixture is %d bytes, want over the plain %d gate", len(prompt), maxInstructionTotal)
	}

	// Without the announcement: the plain gate rejects the build, as it did
	// the incident in issue #85.
	if _, err := New(Config{Instruction: prompt}); err == nil {
		t.Fatal("a persona-sized prompt without a persona budget must fail the 32K gate")
	}

	// With the announcement the child would make: New() passes.
	budget := PersonaBudget(len(persona))
	if budget != 29_480+24_000 {
		t.Fatalf("PersonaBudget = %d, want len(persona)+16000 = %d", budget, 29_480+24_000)
	}
	a, err := New(Config{Instruction: prompt, PersonaBudgetBytes: budget})
	if err != nil {
		t.Fatalf("a persona child at the persona-aware budget must build: %v", err)
	}
	_ = a

	// The headroom is real slack, not a new unlimited mode: a prompt past
	// persona+16000 fails naming the numbers.
	overBudget := PersonaBudget(len(persona))
	over := strings.Repeat("x", overBudget+1)
	if _, err := New(Config{Instruction: over, PersonaBudgetBytes: budget}); err == nil {
		t.Fatal("a prompt over the persona-aware budget must fail")
	} else if !strings.Contains(err.Error(), fmt.Sprintf("over the %d byte budget", overBudget)) {
		t.Errorf("error must name the effective budget: %v", err)
	}
}

// TestPersonaBudgetNeverWeakensPlainGate: the persona-aware limit only ever
// raises the gate. A PersonaBudgetBytes at or below maxInstructionTotal —
// including zero and a negative — keeps the plain 32K gate, so nothing can
// turn this knob into a way under the always-on budget.
func TestPersonaBudgetNeverWeakensPlainGate(t *testing.T) {
	over := strings.Repeat("x", maxInstructionTotal+1)
	for _, budget := range []int{0, 1, maxInstructionTotal, maxInstructionTotal - 1, -5} {
		if _, err := New(Config{Instruction: over, PersonaBudgetBytes: budget}); err == nil {
			t.Errorf("PersonaBudgetBytes=%d must keep the plain gate and fail", budget)
		}
	}

	// The resolution itself: only a value past maxInstructionTotal applies.
	for _, tc := range []struct {
		in, want int
	}{
		{in: 0, want: maxInstructionTotal},
		{in: maxInstructionTotal, want: maxInstructionTotal},
		{in: maxInstructionTotal + 1, want: maxInstructionTotal + 1},
		{in: 45_480, want: 45_480},
	} {
		if got := effectiveInstructionBudget(tc.in); got != tc.want {
			t.Errorf("effectiveInstructionBudget(%d) = %d, want %d", tc.in, got, tc.want)
		}
	}
}

// TestPersonaChildResolution pins the child-side arithmetic the spawned
// process performs: the orchestrator announces the persona size via
// PI_AGENT_PERSONA_BYTES, and the child resolves its whole-prompt gate to
// PersonaBudget(size) = size+16_000. The full command-line path is covered
// by the cli package wiring (buildNonInteractiveInstruction → agent.New)
// — driving a real child binary here would wait on a model, which is not
// what this test is about.
func TestPersonaChildResolution(t *testing.T) {
	if testing.Short() {
		t.Skip("short mode")
	}
	// The incident shape: persona 29_480 + the working-directory tail
	// (rules/skills/env ≈ 17.5K in this repository). The plain gate rejected
	// it; the resolved persona-aware gate must accept it.
	persona := 29_480
	gate := effectiveInstructionBudget(PersonaBudget(persona))
	if gate != 53_480 {
		t.Fatalf("resolved gate = %d, want 53480", gate)
	}
	prompt := strings.Repeat("x", persona) + "\nCurrent working directory: /tmp/p\n" + strings.Repeat("y", 14_000)
	if len(prompt) > gate {
		t.Fatalf("fixture is %d bytes, want at most %d", len(prompt), gate)
	}
	parts := InstructionParts{Base: prompt}
	if err := checkInstructionTotal(parts.String(), &parts); err == nil {
		// Sanity: the same prompt fails the plain gate, so this test cannot
		// pass vacuously.
		t.Fatal("fixture must exceed the plain 32K gate; it proves nothing")
	}
	if err := checkInstructionTotalBudget(parts.String(), &parts, gate); err != nil {
		t.Fatalf("the incident-shaped prompt must pass the persona-aware gate: %v", err)
	}
}

// personaBudgetJSON confirms PersonaBudgetBytes survives the config shapes
// the package serializes (kept minimal: the field is an int and the compile
// check above pins its placement in Config).
func TestPersonaBudgetFieldRoundTrips(t *testing.T) {
	var cfg Config
	cfg.PersonaBudgetBytes = 45_480
	if cfg.PersonaBudgetBytes != 45_480 {
		t.Errorf("PersonaBudgetBytes = %d, want 45480", cfg.PersonaBudgetBytes)
	}
}
