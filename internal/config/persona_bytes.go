package config

import (
	"os"
	"strconv"
	"strings"
)

// ResolvePersonaBytes returns the persona size announced to this process by
// its spawning subagent parent (PI_AGENT_PERSONA_BYTES), and whether one was
// announced at all. The child uses it to raise its whole-prompt budget gate
// to len(persona)+16_000 (agent.Config.PersonaBytes, issue #85): the persona
// rides --system whole, and the interactive 32K gate is not the budget a
// single-task persona answers to.
//
// An unset variable means "no persona" (ok=false). An unparsable or negative
// value is treated the same way — a typo must not fail a spawn; the child
// falls back to the plain 32K gate and the build either passes (small
// persona) or fails with the gate's own message, which names the numbers.
func ResolvePersonaBytes() (n int, ok bool) {
	v := strings.TrimSpace(os.Getenv(EnvPersonaBytes))
	if v == "" {
		return 0, false
	}
	parsed, err := strconv.Atoi(v)
	if err != nil || parsed <= 0 {
		return 0, false
	}
	return parsed, true
}
