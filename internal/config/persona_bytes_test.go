package config

import (
	"testing"
)

// TestResolvePersonaBytes pins the child-side read of the persona size the
// spawning parent announced (issue #85): set → (size, true); unset, empty,
// unparsable or non-positive → (0, false), so a typo falls back to the plain
// gate instead of failing the spawn.
func TestResolvePersonaBytes(t *testing.T) {
	t.Setenv(EnvPersonaBytes, "")

	if n, ok := ResolvePersonaBytes(); ok {
		t.Errorf("unset var must resolve as no-persona, got (%d, %v)", n, ok)
	}

	t.Setenv(EnvPersonaBytes, "29480")
	if n, ok := ResolvePersonaBytes(); !ok || n != 29480 {
		t.Errorf("set var must resolve, got (%d, %v), want (29480, true)", n, ok)
	}

	for _, bad := range []string{"", "  ", "abc", "-1", "0"} {
		t.Setenv(EnvPersonaBytes, bad)
		if n, ok := ResolvePersonaBytes(); ok {
			t.Errorf("value %q must resolve as no-persona, got (%d, %v)", bad, n, ok)
		}
	}
}
