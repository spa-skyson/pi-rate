package cli

import (
	"context"
	"testing"

	"github.com/spa-skyson/pi-rate/internal/testenv"

	adktool "google.golang.org/adk/v2/tool"
)

// todoToolNames reports whether the todo tools are among ts.
func todoToolNames(ts []adktool.Tool) (write, read bool) {
	for _, x := range ts {
		switch x.Name() {
		case "todo_write":
			write = true
		case "todo_read":
			read = true
		}
	}
	return write, read
}

// TestDeferredInitCoreTools_TodoWiring pins the interactive wiring: a session
// ID registers the todo tools (nil todoCh included — the notifier is optional),
// and no session ID registers none.
func TestDeferredInitCoreTools_TodoWiring(t *testing.T) {
	t.Run("registered with a session ID", func(t *testing.T) {
		root := t.TempDir()
		res := &initResources{}
		t.Cleanup(res.cleanup)

		core, err := deferredInitCoreTools(root, root, "sess-cli", "sess-cli", nil, nil, res)
		if err != nil {
			t.Fatalf("deferredInitCoreTools: %v", err)
		}
		write, read := todoToolNames(core)
		if !write || !read {
			t.Errorf("todo tools missing (write=%v read=%v) with a session ID", write, read)
		}
		if res.sandbox == nil || res.bashSup == nil {
			t.Error("sandbox/supervisor not recorded on res for cleanup")
		}
	})

	t.Run("absent without a session ID", func(t *testing.T) {
		root := t.TempDir()
		res := &initResources{}
		t.Cleanup(res.cleanup)

		core, err := deferredInitCoreTools(root, root, "", "", nil, nil, res)
		if err != nil {
			t.Fatalf("deferredInitCoreTools: %v", err)
		}
		if write, read := todoToolNames(core); write || read {
			t.Errorf("todo tools registered without a session ID (write=%v read=%v)", write, read)
		}
	})
}

// TestBuildRootRuntime_HeaderSessionID_AlwaysSet pins the print-mode wiring:
// buildRootRuntime pre-generates the session ID even when no ${SESSION_ID}
// headers need one. The todo tools register through that ID in every mode
// (coreToolOptions → initNonInteractiveRuntime), and resolveSessionID then
// creates the session under it — so todos.json lands in the session the run
// actually used. Before the fix the ID stayed empty without such headers and
// print/json runs silently lost todo_write/todo_read.
//
// The ollama model keeps this offline: buildRootRuntime health-checks a local
// daemon only without a key, and the key is exported (same pattern as
// ollama_routing_test.go).
func TestBuildRootRuntime_HeaderSessionID_AlwaysSet(t *testing.T) {
	resetGlobalFlags(t)
	testenv.SetHome(t, t.TempDir())
	t.Setenv("OLLAMA_API_KEY", "sk-ollama-test")
	t.Setenv("OLLAMA_HOST", "")

	flagModel = "ollama/qwen3.8:27b-mlx" // default config carries no ${SESSION_ID} headers
	flagMode = "print"
	flagSession = ""
	flagContinue = false

	rt, err := buildRootRuntime(context.Background(), []string{"hi"})
	if err != nil {
		t.Fatalf("buildRootRuntime: %v", err)
	}
	if rt.headerSessionID == "" {
		t.Fatal("headerSessionID empty without ${SESSION_ID} headers: print-mode todo tools would not register")
	}

	// The print path builds its tools from exactly this ID.
	nrt, err := initNonInteractiveRuntime(context.Background(), &rt.cfg, rt.cwd, rt.sandboxRoot, rt.worktreeDir, rt.headerSessionID)
	if err != nil {
		t.Fatalf("initNonInteractiveRuntime: %v", err)
	}
	nrt.close() // releases the sandbox: an open root breaks t.TempDir cleanup on Windows

	if write, read := todoToolNames(nrt.coreTools); !write || !read {
		t.Errorf("todo tools missing in print mode (write=%v read=%v)", write, read)
	}
}
