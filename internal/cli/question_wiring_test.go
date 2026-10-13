package cli

import (
	"context"
	"testing"
	"time"

	"google.golang.org/adk/v2/agent"
	adktool "google.golang.org/adk/v2/tool"

	"github.com/spa-skyson/pi-rate/internal/config"
	"github.com/spa-skyson/pi-rate/internal/testenv"
	"github.com/spa-skyson/pi-rate/internal/tools"
	"github.com/spa-skyson/pi-rate/internal/tui"
)

// TestNonInteractiveRuntime_QuestionHeadless is the #32 guard for the
// subagent path: a spawned `pi --mode json` child builds its tools through
// initNonInteractiveRuntime, which must keep the question tool headless (nil
// notifier) — the call answers canceled immediately and never blocks, so a
// background agent asking a question can never park its run.
func TestNonInteractiveRuntime_QuestionHeadless(t *testing.T) {
	resetGlobalFlags(t)
	testenv.SetHome(t, t.TempDir())

	nrt, err := initNonInteractiveRuntime(context.Background(), &config.Config{}, t.TempDir(), t.TempDir(), "", "sess-subagent-q")
	if err != nil {
		t.Fatalf("initNonInteractiveRuntime: %v", err)
	}
	nrt.close() // releases the sandbox: an open root breaks t.TempDir cleanup on Windows

	var qt adktool.Tool
	for _, x := range nrt.coreTools {
		if x.Name() == "question" {
			qt = x
		}
	}
	if qt == nil {
		t.Fatal("question tool not registered in the non-interactive runtime")
	}

	done := make(chan map[string]any, 1)
	go func() {
		r := qt.(interface {
			Run(agent.Context, any) (map[string]any, error)
		})
		out, err := r.Run(&cliToolCtx{Context: context.Background()}, map[string]any{
			"question": "Anyone there?",
			"options":  []any{map[string]any{"label": "no"}},
		})
		if err != nil {
			t.Errorf("Run error: %v", err)
		}
		done <- out
	}()
	select {
	case out := <-done:
		if out["selected"] != "canceled" {
			t.Errorf("output = %#v, want immediate canceled in the subagent runtime", out)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("question call blocked in the non-interactive runtime — the subagent would hang")
	}
}

// TestDeferredInitCoreTools_QuestionWiring pins the interactive wiring: a
// question channel registers the question tool with a live notifier (the
// request reaches the channel, the dialog's answer reaches the tool), and a
// nil channel keeps the headless variant — the tool still registers, it just
// answers canceled immediately.
func TestDeferredInitCoreTools_QuestionWiring(t *testing.T) {
	t.Run("bridged with a channel", func(t *testing.T) {
		root := t.TempDir()
		res := &initResources{}
		t.Cleanup(res.cleanup)

		questionCh := make(chan tui.SessionQuestion, 1)
		core, err := deferredInitCoreTools(root, root, "sess-q", "sess-q", nil, questionCh, res)
		if err != nil {
			t.Fatalf("deferredInitCoreTools: %v", err)
		}

		var qt adktool.Tool
		for _, x := range core {
			if x.Name() == "question" {
				qt = x
			}
		}
		if qt == nil {
			t.Fatal("question tool not registered with a question channel")
		}

		// Invoke the tool; the request must land on the channel and the
		// dialog's answer must come back as the tool result.
		done := make(chan map[string]any, 1)
		go func() {
			r := qt.(interface {
				Run(agent.Context, any) (map[string]any, error)
			})
			out, err := r.Run(&cliToolCtx{Context: context.Background()}, map[string]any{
				"question": "Merge now?",
				"options":  []any{map[string]any{"label": "yes"}},
			})
			if err != nil {
				t.Errorf("Run error: %v", err)
			}
			done <- out
		}()

		select {
		case tagged := <-questionCh:
			tagged.Req.Reply <- tools.QuestionAnswer{Selected: "option", Label: "yes", Index: 0}
		case <-time.After(2 * time.Second):
			t.Fatal("no request reached the question channel")
		}
		out := <-done
		if out["selected"] != "option" || out["label"] != "yes" {
			t.Errorf("output = %#v, want selected=option label=yes", out)
		}
	})

	t.Run("full slot displaces the stale request", func(t *testing.T) {
		// #32 guard: the notifier must never block on its send. A stale
		// request sitting unread in the buffer is displaced — answered
		// canceled — and the new request takes the slot.
		root := t.TempDir()
		res := &initResources{}
		t.Cleanup(res.cleanup)

		questionCh := make(chan tui.SessionQuestion, 1)
		stale := tools.QuestionRequest{Reply: make(chan tools.QuestionAnswer, 1)}
		questionCh <- tui.SessionQuestion{Req: stale, SessionID: "sess-q"}

		core, err := deferredInitCoreTools(root, root, "sess-q", "sess-q", nil, questionCh, res)
		if err != nil {
			t.Fatalf("deferredInitCoreTools: %v", err)
		}
		var qt adktool.Tool
		for _, x := range core {
			if x.Name() == "question" {
				qt = x
			}
		}
		if qt == nil {
			t.Fatal("question tool not registered")
		}

		done := make(chan map[string]any, 1)
		go func() {
			r := qt.(interface {
				Run(agent.Context, any) (map[string]any, error)
			})
			out, err := r.Run(&cliToolCtx{Context: context.Background()}, map[string]any{
				"question": "The fresh one",
				"options":  []any{map[string]any{"label": "yes"}},
			})
			if err != nil {
				t.Errorf("Run error: %v", err)
			}
			done <- out
		}()

		// The stale request is answered canceled so its tool call stops
		// waiting on a Reply nobody owns.
		select {
		case ans := <-stale.Reply:
			if ans.Selected != "canceled" {
				t.Errorf("stale answer = %q, want canceled", ans.Selected)
			}
		case <-time.After(2 * time.Second):
			t.Fatal("stale request was not displaced")
		}

		// The fresh request took the slot; the dialog's answer completes it.
		select {
		case tagged := <-questionCh:
			if tagged.Req.Question != "The fresh one" {
				t.Errorf("channel request = %q, want the fresh one", tagged.Req.Question)
			}
			if tagged.SessionID != "sess-q" {
				t.Errorf("session tag = %q, want sess-q", tagged.SessionID)
			}
			tagged.Req.Reply <- tools.QuestionAnswer{Selected: "option", Label: "yes", Index: 0}
		case <-time.After(2 * time.Second):
			t.Fatal("fresh request did not reach the channel")
		}
		out := <-done
		if out["selected"] != "option" {
			t.Errorf("output = %#v, want selected=option", out)
		}
	})

	t.Run("headless without a channel", func(t *testing.T) {
		root := t.TempDir()
		res := &initResources{}
		t.Cleanup(res.cleanup)

		core, err := deferredInitCoreTools(root, root, "sess-q", "sess-q", nil, nil, res)
		if err != nil {
			t.Fatalf("deferredInitCoreTools: %v", err)
		}

		var qt adktool.Tool
		for _, x := range core {
			if x.Name() == "question" {
				qt = x
			}
		}
		if qt == nil {
			t.Fatal("question tool must register even without a question channel")
		}
		r := qt.(interface {
			Run(agent.Context, any) (map[string]any, error)
		})
		out, err := r.Run(&cliToolCtx{Context: context.Background()}, map[string]any{
			"question": "Anyone there?",
			"options":  []any{map[string]any{"label": "no"}},
		})
		if err != nil {
			t.Fatalf("Run error: %v", err)
		}
		if out["selected"] != "canceled" {
			t.Errorf("output = %#v, want immediate canceled", out)
		}
	})
}
