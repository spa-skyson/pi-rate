package cli

import (
	"testing"

	"github.com/spa-skyson/pi-rate/internal/agent"
	"github.com/spa-skyson/pi-rate/internal/config"
)

// TestPersonaBudgetFromEnv pins the child wiring (issue #85 c): a spawned
// persona child resolves its whole-prompt gate from PI_AGENT_PERSONA_BYTES
// the way runNonInteractive feeds agent.Config, and a run without the env
// var — every plain one-shot and interactive session — resolves zero and
// keeps the plain 32K gate (d).
func TestPersonaBudgetFromEnv(t *testing.T) {
	t.Setenv(config.EnvPersonaBytes, "29480")
	personaBytes, ok := config.ResolvePersonaBytes()
	if !ok {
		t.Fatal("the env var must resolve when set")
	}
	if got := agent.PersonaBudget(personaBytes); got != 53_480 {
		t.Errorf("PersonaBudget(29480) = %d, want 53480 (len(persona)+24000)", got)
	}
	cfg := agent.Config{PersonaBudgetBytes: agent.PersonaBudget(personaBytes)}
	if cfg.PersonaBudgetBytes != 53_480 {
		t.Errorf("Config.PersonaBudgetBytes = %d, want 53480", cfg.PersonaBudgetBytes)
	}

	// The plain run: no announcement, zero budget, plain gate.
	t.Setenv(config.EnvPersonaBytes, "")
	if _, ok := config.ResolvePersonaBytes(); ok {
		t.Error("an unset var must resolve as no-persona; the child keeps the plain 32K gate")
	}
}
