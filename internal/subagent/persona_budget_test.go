package subagent

import (
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/spa-skyson/pi-rate/internal/notice"
)

// personaFileOf returns a valid agent persona whose total file size is n
// bytes: a frontmatter header plus a body padded with comment lines. The
// padding lands in the markdown body so the file parses normally once the
// budget gate lets it through.
func personaFileOf(t *testing.T, dir, name string, n int) string {
	t.Helper()
	header := "---\nname: " + name + "\nrole: smol\n---\n"
	if n < len(header) {
		t.Fatalf("persona size %d smaller than frontmatter header", n)
	}
	pad := n - len(header)
	body := strings.Repeat("x", pad)
	path := filepath.Join(dir, name+".md")
	if err := os.WriteFile(path, []byte(header+body), 0o600); err != nil {
		t.Fatal(err)
	}
	if got := fileSize(t, path); got != n {
		t.Fatalf("fixture %s is %d bytes, want %d", path, got, n)
	}
	return path
}

func fileSize(t *testing.T, path string) int {
	t.Helper()
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	return int(info.Size())
}

// collectNotices swaps the notice sink for a recorder and restores it.
func collectNotices(t *testing.T) func() []string {
	t.Helper()
	var mu sync.Mutex
	var raised []string
	prev := notice.SetSink(func(msg string) {
		mu.Lock()
		raised = append(raised, msg)
		mu.Unlock()
	})
	t.Cleanup(func() { notice.SetSink(prev) })
	return func() []string {
		mu.Lock()
		defer mu.Unlock()
		return append([]string(nil), raised...)
	}
}

// TestPersonaOverBudgetFailsLoad: a persona file over maxAgentPersonaBytes
// fails to load, and the error names the file, its size and the limit
// (issue #85 a). The failure must come from the plain parse path — the same
// one agent discovery uses — so an oversized persona is loud at startup,
// not at first spawn.
func TestPersonaOverBudgetFailsLoad(t *testing.T) {
	dir := t.TempDir()
	path := personaFileOf(t, dir, "bloated", maxAgentPersonaBytes+1)

	if _, err := ParseAgentFile(path); err == nil {
		t.Fatal("a persona over the budget must fail to load")
	} else {
		if !strings.Contains(err.Error(), filepath.Base(path)) {
			t.Errorf("error must name the file: %v", err)
		}
		if !strings.Contains(err.Error(), "48001 bytes") {
			t.Errorf("error must state the file size: %v", err)
		}
		if !strings.Contains(err.Error(), "48000") {
			t.Errorf("error must state the limit: %v", err)
		}
	}

	// The same file discovered through the directory loader fails the whole
	// directory load: discovery is where startup surfaces it.
	if _, err := LoadAgentsFromDir(dir); err == nil {
		t.Fatal("LoadAgentsFromDir must fail on an over-budget persona")
	}
}

// TestPersonaNearBudgetWarnsOnce: a persona within
// maxAgentPersonaWarnGap of the cap loads fine and raises exactly one
// notice (issue #85 b). Just below the warn band the same persona loads
// silently.
func TestPersonaNearBudgetWarnsOnce(t *testing.T) {
	dir := t.TempDir()
	// 46_000 bytes: inside the (limit-4096, limit] warn band.
	path := personaFileOf(t, dir, "chubby", maxAgentPersonaBytes-maxAgentPersonaWarnGap+2048)

	notices := collectNotices(t)
	agent, err := ParseAgentFile(path)
	if err != nil {
		t.Fatalf("a persona just under the cap must load: %v", err)
	}
	if agent.Name != "chubby" {
		t.Errorf("expected the persona to parse, got name %q", agent.Name)
	}

	raised := notices()
	if len(raised) != 1 {
		t.Fatalf("got %d notices, want exactly 1: %q", len(raised), raised)
	}
	if !strings.HasPrefix(raised[0], "persona chubby.md is ") || !strings.HasSuffix(raised[0], " bytes, close to the 48000 budget") {
		t.Errorf("notice = %q, want the persona/size/limit shape", raised[0])
	}

	// The same file discovered through the directory loader loads and warns
	// once per near-budget file: each parse re-checks, the load path is not
	// the deduped hygiene pass.
	personaFileOf(t, dir, "chubby-two", maxAgentPersonaBytes-1024)
	if _, err := LoadAgentsFromDir(dir); err != nil {
		t.Fatalf("LoadAgentsFromDir with near-budget personas: %v", err)
	}
	if got := notices(); len(got) != 3 {
		t.Errorf("got %d notices after the first pass and a dir load, want 3 (1 initial + 2 per-file): %q", len(got), got)
	}

	// A persona comfortably under the warn band stays silent.
	quiet := t.TempDir()
	personaFileOf(t, quiet, "slim", maxAgentPersonaBytes-maxAgentPersonaWarnGap-1)
	notices = collectNotices(t)
	if _, err := ParseAgentFile(filepath.Join(quiet, "slim.md")); err != nil {
		t.Fatalf("a persona under the warn band must load: %v", err)
	}
	if got := notices(); len(got) != 0 {
		t.Errorf("got %d notices for a persona below the warn band, want 0: %q", len(got), got)
	}
}

// TestPersonaEnvOnlyWhenPersonaCarried pins the child-budget handshake the
// orchestrator performs: a spawn of an agent with a persona announces its
// size via PI_AGENT_PERSONA_BYTES, an agent without an instruction announces
// nothing and the child keeps the plain 32K gate (issue #85 c/d, spawn side).
func TestPersonaEnvOnlyWhenPersonaCarried(t *testing.T) {
	withPersona := AgentConfig{Name: "heavy", Instruction: strings.Repeat("x", 29_480)}
	env := personaEnv(withPersona)
	if len(env) != 1 {
		t.Fatalf("persona agent must set the env var, got %v", env)
	}
	if env[0] != "PI_AGENT_PERSONA_BYTES=29480" {
		t.Errorf("env = %q, want the persona byte count", env[0])
	}

	without := AgentConfig{Name: "bare"}
	if env := personaEnv(without); env != nil {
		t.Errorf("an agent without a persona must set no env var, got %v", env)
	}
}
