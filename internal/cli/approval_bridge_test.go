package cli

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/spa-skyson/pi-rate/internal/config"
	"github.com/spa-skyson/pi-rate/internal/permission"
	"github.com/spa-skyson/pi-rate/internal/tools"
	"github.com/spa-skyson/pi-rate/internal/tui"
)

func TestApprovalAsker(t *testing.T) {
	t.Run("delivers the request and returns the answer", func(t *testing.T) {
		ch := make(chan tui.SessionApproval)
		ask := approvalAsker(ch, "sess_test")

		done := make(chan permission.ApprovalResult, 1)
		go func() {
			done <- ask(context.Background(), permission.ApprovalRequest{
				Tool:  "bash",
				Rule:  "git *",
				Reply: make(chan permission.ApprovalResult, 1),
			})
		}()

		var req permission.ApprovalRequest
		select {
		case tagged := <-ch:
			req = tagged.Req
			if tagged.SessionID != "sess_test" {
				t.Errorf("session tag = %q, want sess_test — the TUI routes by it", tagged.SessionID)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("request never reached the dialog channel")
		}
		if req.Tool != "bash" || req.Rule != "git *" {
			t.Errorf("request = %+v", req)
		}
		req.Reply <- permission.ApprovalResult{Allowed: true}

		select {
		case got := <-done:
			if !got.Allowed {
				t.Errorf("answer = %+v, want allowed", got)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("asker did not return")
		}
	})

	t.Run("canceled turn denies without reaching the channel", func(t *testing.T) {
		ch := make(chan tui.SessionApproval, 1)
		ctx, cancel := context.WithCancel(context.Background())
		cancel()

		got := approvalAsker(ch, "sess_test")(ctx, permission.ApprovalRequest{
			Reply: make(chan permission.ApprovalResult, 1),
		})
		if got.Allowed {
			t.Error("canceled wait must deny")
		}
		select {
		case <-ch:
			t.Error("canceled wait must not hand the request to the dialog")
		default:
		}
	})

	t.Run("cancellation while waiting for the answer denies", func(t *testing.T) {
		ch := make(chan tui.SessionApproval)
		ctx, cancel := context.WithCancel(context.Background())

		done := make(chan permission.ApprovalResult, 1)
		reqCh := make(chan permission.ApprovalRequest, 1)
		go func() {
			done <- approvalAsker(ch, "sess_test")(ctx, permission.ApprovalRequest{
				Tool:  "edit",
				Reply: make(chan permission.ApprovalResult, 1), // buffered: a late answer never blocks
			})
		}()

		select {
		case tagged := <-ch:
			reqCh <- tagged.Req
			cancel() // the turn dies while the dialog is still up, unanswered
		case <-time.After(5 * time.Second):
			t.Fatal("request never reached the channel")
		}

		// The reply has not been sent, so the asker can only observe Done:
		// this return is deterministic.
		select {
		case got := <-done:
			if got.Allowed {
				t.Error("canceled wait must deny")
			}
		case <-time.After(5 * time.Second):
			t.Fatal("asker stayed blocked after cancellation")
		}

		// Only now does the user answer — into the buffered Reply of a wait
		// that has already been abandoned. The send must not block.
		req := <-reqCh
		select {
		case req.Reply <- permission.ApprovalResult{Allowed: true}:
		case <-time.After(5 * time.Second):
			t.Fatal("late answer blocked on Reply")
		}
	})
}

// TestDeferredCallbacks_PermissionBridge is the wiring half of the approval
// flow: with ask rules configured, the deferred callback chain must install
// the asking gate when a bridge channel is provided and the hard-denying gate
// when it is not — the same rules, two behaviors, selected by one argument.
func TestDeferredCallbacks_PermissionBridge(t *testing.T) {
	resetGlobalFlags(t)
	sandbox, err := tools.NewSandbox(t.TempDir())
	if err != nil {
		t.Fatalf("NewSandbox: %v", err)
	}
	t.Cleanup(func() { _ = sandbox.Close() })

	ask := permission.Ask
	cfg := config.Config{Permission: &permission.Rules{
		Tools: map[string]permission.Directive{"edit": ask},
	}}

	tool := &namedToolStub{name: "edit"}
	callback := func(t *testing.T, bridge chan tui.SessionApproval) func() error {
		t.Helper()
		cbs := buildDeferredCallbacks(cfg, "anthropic", sandbox, nil, nil, bridge, "sess-cbs")
		if len(cbs.beforeTool) == 0 {
			t.Fatal("no before-tool callbacks installed")
		}
		return func() error {
			_, err := cbs.beforeTool[0](nil, tool, nil)
			return err
		}
	}

	t.Run("without a bridge ask denies", func(t *testing.T) {
		err := callback(t, nil)()
		if err == nil || !strings.Contains(err.Error(), "requires interactive approval") {
			t.Errorf("headless gate: want ask-denial, got %v", err)
		}
	})

	t.Run("with a bridge ask consults the dialog channel", func(t *testing.T) {
		ch := make(chan tui.SessionApproval, 1)
		errCh := make(chan error, 1)
		go func() {
			errCh <- callback(t, ch)()
		}()

		select {
		case tagged := <-ch:
			if tagged.Req.Tool != "edit" {
				t.Errorf("dialog got tool %q, want edit", tagged.Req.Tool)
			}
			if tagged.SessionID == "" {
				t.Error("the bridge did not tag the request with a session — the TUI would deny it")
			}
			tagged.Req.Reply <- permission.ApprovalResult{Allowed: true}
		case <-time.After(5 * time.Second):
			t.Fatal("ask request never reached the bridge")
		}
		select {
		case err := <-errCh:
			if err != nil {
				t.Errorf("approved ask returned an error: %v", err)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("gate stayed blocked after approval")
		}
	})
}
