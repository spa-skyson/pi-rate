package tools

import (
	"context"
	"fmt"
	"strings"
	"time"
	"unicode"

	"go.opentelemetry.io/otel/attribute"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool"

	"github.com/spa-skyson/pi-rate/internal/otel"
)

const (
	// defaultBashTimeout caps how long a command holds the foreground before it
	// is handed to the supervisor. It is a budget for the *turn*, not for the
	// command: nothing is killed at the threshold, the command keeps running,
	// and the caller gets a handle to watch it with.
	//
	// One minute is chosen so the common case stays whole — the vast majority
	// of commands in this repo (git, go vet, a package test run, a warm build)
	// finish well inside it — while anything genuinely long says so a minute in
	// rather than running the turn out. A caller that knows its command is long
	// should raise `timeout` rather than discover this limit; the idle check
	// then does the real work, firing well before the raised hard limit.
	defaultBashTimeout = time.Minute

	// maxBashTimeout caps a caller-supplied limit. Past ten minutes the caller
	// should be backgrounding deliberately and polling with bash_wait rather
	// than holding the foreground.
	maxBashTimeout = 10 * time.Minute
)

// BashInput defines the parameters for the bash tool.
type BashInput struct {
	// The shell command to execute.
	Command string `json:"command"`
	// Optional timeout in SECONDS. Default 60, max 600. On expiry the command
	// is moved to the background, not killed. Raise it for work you already
	// know is long so it finishes in the foreground instead of costing a round
	// trip.
	Timeout int `json:"timeout,omitempty"`
	// Optional idle timeout in SECONDS. A command that produces no output at
	// all for this long is moved to the background. Default 90.
	IdleTimeout int `json:"idle_timeout,omitempty"`
}

// BashOutput contains the result of executing a shell command.
type BashOutput struct {
	// Standard output from the command.
	Stdout string `json:"stdout"`
	// Standard error from the command.
	Stderr string `json:"stderr"`
	// Exit code of the command. -1 when the command has not exited.
	ExitCode int `json:"exit_code"`
	// Running is true when the command outlived its timeout and is still
	// executing in the background.
	Running bool `json:"running,omitempty"`
	// Handle identifies a still-running command for bash_wait and bash_kill.
	Handle string `json:"handle,omitempty"`
	// Elapsed is how long the command has been running in total.
	Elapsed string `json:"elapsed,omitempty"`
	// Idle is how long it has been since the command last produced output.
	Idle string `json:"idle,omitempty"`
	// Timeout and IdleTimeout are the limits this command actually ran under,
	// after defaults and clamping. They are reported rather than left implicit
	// because a handoff is otherwise unexplainable from the result alone: the
	// caller sees "moved to the background" with no way to tell whether the
	// limit it hit was the 90s default or a 1s value it passed itself.
	Timeout     string `json:"timeout,omitempty"`
	IdleTimeout string `json:"idle_timeout,omitempty"`
	// Note explains a non-obvious outcome in terms the model can act on.
	Note string `json:"note,omitempty"`
}

// BashStatus is the result of inspecting or stopping a backgrounded command.
type BashStatus struct {
	Handle  string `json:"handle"`
	Command string `json:"command"`
	// Running reports whether the command is still executing.
	Running bool `json:"running"`
	// ExitCode is meaningful only once Running is false.
	ExitCode int `json:"exit_code,omitempty"`
	// Stdout and Stderr carry only what was produced since the previous read.
	Stdout string `json:"stdout"`
	Stderr string `json:"stderr"`
	// Elapsed is how long the command has been running in total.
	Elapsed string `json:"elapsed,omitempty"`
	// Idle is how long it has been since the command last produced output.
	Idle string `json:"idle,omitempty"`
	// Timeout and IdleTimeout are the limits the command was started under, so
	// a poll can say why it was handed off without the caller having to
	// remember what it asked for several turns ago.
	Timeout     string `json:"timeout,omitempty"`
	IdleTimeout string `json:"idle_timeout,omitempty"`
	Note        string `json:"note,omitempty"`
}

// BashWaitInput asks for output accumulated by a backgrounded command.
type BashWaitInput struct {
	// Handle returned by a previous bash call.
	Handle string `json:"handle"`
	// WaitSec blocks up to this long, in SECONDS, for new output or for the
	// command to exit. Defaults to 60, capped at 600 — one generous wait is
	// the way to follow a long run without a polling loop. Prefer one long
	// wait over polling in a loop.
	WaitSec int `json:"wait_sec,omitempty"`
}

// BashKillInput identifies a backgrounded command to stop.
type BashKillInput struct {
	// Handle returned by a previous bash call.
	Handle string `json:"handle"`
}

const (
	// defaultBashWait is how long bash_wait blocks when wait_sec is omitted.
	// It stays at one minute: a caller that says nothing about how long to wait
	// used to get exactly this, and a quiet command silently stretching a
	// single call to ten minutes would hold the turn far longer than the
	// caller asked for.
	defaultBashWait = 60 * time.Second

	// maxBashWait caps a caller-supplied wait. Raised from 60s so one call can
	// follow a multi-minute test suite or build — the polling loop it replaces
	// burns a round trip per minute, which is what cost issue #45's acceptance
	// agent whole minutes of sleeps.
	maxBashWait = 600 * time.Second
)

// The bash tool's description is a lead plus a shared tail. Only the lead
// differs by platform: on Windows without bash the command runs through
// PowerShell, which rejects bash syntax like `&&`, so the model has to be told
// which shell it is writing for. The backgrounding contract and the units are
// the same either way, so they are stated once.
const bashLead = `Run a shell command and return its output (system ops, tests, builds, git).`

const powershellLead = `Run a shell command and return its output. Commands go through powershell.exe -NoProfile -Command (no bash): use ; or a newline instead of &&, Test-Path instead of test -f, Get-ChildItem instead of ls. Native tools (git, go, curl.exe) work as usual.`

const bashLimits = `

Past its timeout (60s) or 90s silent, a command is backgrounded, not killed: result has running=true and a handle — bash_wait reads more output, bash_kill stops it; an empty handle result means the command is too broad. For known-long work raise timeout; for very long runs let the handoff happen and follow with one generous bash_wait (wait_sec up to 600) instead of re-running or polling.

timeout, idle_timeout, wait_sec: SECONDS, capped at 600 — write 300, not 300000.`

func executeDescription() string {
	if CurrentShellKind() == "powershell" {
		return powershellLead + bashLimits
	}
	return bashLead + bashLimits
}

func newBashTool(sb *Sandbox, sup *BashSupervisor) (tool.Tool, error) {
	// The tool keeps the well-known "bash" name across platforms so sessions,
	// configs and muscle memory referencing it stay valid; only the
	// description tells the model which syntax to write.
	return newTool("bash", executeDescription(), func(ctx agent.Context, input BashInput) (BashOutput, error) {
		return bashHandler(sb, sup, ctx, input)
	})
}

func bashHandler(sb *Sandbox, sup *BashSupervisor, ctx agent.Context, input BashInput) (BashOutput, error) {
	if input.Command == "" {
		return BashOutput{}, fmt.Errorf("command is required")
	}
	if name := controlToolName(input.Command); name != "" {
		return BashOutput{}, fmt.Errorf(
			"%q is a tool name, not a shell command — call the %s tool directly instead of typing it into the bash tool",
			name, name)
	}

	// Use background context if agent.Context is nil (e.g. in unit tests)
	var parentCtx context.Context
	if ctx != nil {
		parentCtx = ctx
	} else {
		parentCtx = context.Background()
	}

	parentCtx, span := otel.Tracer("pi-go").Start(parentCtx, "tool.bash")
	span.SetAttributes(
		attribute.String("bash.command", input.Command),
		attribute.String("bash.cwd", sb.Dir()),
	)
	defer span.End()

	out, err := sup.Run(parentCtx, runRequest{
		dir:         sb.Dir(),
		command:     input.Command,
		timeout:     clampDuration(input.Timeout, defaultBashTimeout, 0, maxBashTimeout),
		idleTimeout: clampDuration(input.IdleTimeout, 0, 0, maxBashTimeout),
	})
	if err != nil {
		span.RecordError(err)
		return BashOutput{}, fmt.Errorf("executing command: %w", err)
	}

	span.SetAttributes(
		attribute.Int("bash.exit_code", out.ExitCode),
		attribute.Int("bash.stdout_len", len(out.Stdout)),
		attribute.Int("bash.stderr_len", len(out.Stderr)),
		attribute.Bool("bash.backgrounded", out.Running),
	)

	return out, nil
}

// clampDuration converts a seconds input to a duration, substituting fallback
// when unset and holding the result within [minDur, maxDur].
//
// A floor applies only to a value the caller actually supplied: an unset input
// takes fallback unchanged, so a caller that passes nothing gets the default
// chosen for it rather than one bent by a floor. Pass minDur = 0 where no
// floor applies, which is every current caller — the units are seconds now, so
// there is no unit mistake left to guard against.
func clampDuration(sec int, fallback, minDur, maxDur time.Duration) time.Duration {
	if sec <= 0 {
		return fallback
	}
	d := time.Duration(sec) * time.Second
	if d < minDur {
		return minDur
	}
	if d > maxDur {
		return maxDur
	}
	return d
}

// bashWaitDur resolves a bash_wait wait_sec input: an omitted value takes the
// default wait, and anything above the cap lands on the cap. It is its own
// function rather than an inline clampDuration call so the advertised limits
// — default 60, cap 600 — have one testable seam.
func bashWaitDur(sec int) time.Duration {
	return clampDuration(sec, defaultBashWait, 0, maxBashWait)
}

// controlToolName reports which bash control tool (bash_wait, bash_kill) a
// command string is trying to invoke, or "" when the command does not look
// like one. It exists because backgrounding results embed the control tools'
// names, and models occasionally paste the suggested call — e.g.
// bash_wait(handle="bg_1") — into the bash tool as shell text, where it fails
// with an unhelpful bash syntax error. Matching only at the start of the
// command keeps ordinary shell lines that merely mention these names working.
func controlToolName(command string) string {
	for _, name := range []string{"bash_wait", "bash_kill"} {
		if rest, ok := strings.CutPrefix(command, name); ok {
			trimmed := strings.TrimLeftFunc(rest, unicode.IsSpace)
			if !strings.HasPrefix(trimmed, "(") {
				continue
			}
			// `name(...)` is also valid shell function-definition syntax
			// (`bash_wait() { ... }`); a pasted tool call has no space there.
			if strings.HasPrefix(trimmed, "()") {
				continue
			}
			return name
		}
	}
	return ""
}

// BashControlTools returns the tools for inspecting and stopping commands that
// the bash tool moved to the background.
//
// They are separate from CoreTools because they are only worth advertising to
// the model when something can actually background a command — a caller that
// builds its own supervisor and never streams need not carry the extra schema.
func BashControlTools(sup *BashSupervisor) ([]tool.Tool, error) {
	waitTool, err := newTool("bash_wait",
		"Wait on a backgrounded command; returns output since the last wait. Blocks up to 60s (wait_sec, capped at 600), returning early on new output or exit. Prefer one generous wait_sec (300-600) over looping — each call is a round trip, and an empty result means nothing new, not a stuck command. When done: running=false plus the exit code; the handle is spent.",
		func(_ agent.Context, input BashWaitInput) (BashStatus, error) {
			if input.Handle == "" {
				return BashStatus{}, fmt.Errorf("handle is required (running: %v)", sup.Handles())
			}
			return sup.readOutput(input.Handle, bashWaitDur(input.WaitSec))
		})
	if err != nil {
		return nil, err
	}

	killTool, err := newTool("bash_kill",
		"Stop a backgrounded command and its whole process tree; returns output since the last read.",
		func(_ agent.Context, input BashKillInput) (BashStatus, error) {
			if input.Handle == "" {
				return BashStatus{}, fmt.Errorf("handle is required (running: %v)", sup.Handles())
			}
			return sup.killHandle(input.Handle)
		})
	if err != nil {
		return nil, err
	}

	return []tool.Tool{waitTool, killTool}, nil
}
