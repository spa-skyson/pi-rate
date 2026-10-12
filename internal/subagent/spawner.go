package subagent

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"
)

// ErrSubagentTimeout marks a subagent that was stopped by one of its own time
// limits rather than by failing.
//
// It exists because the kill is indistinguishable from a crash at the point the
// caller reads it. os/exec's Wait prefers the *ExitError from Process.Wait over
// the context error — watch.err is only consulted when err == nil — and a
// SIGKILLed process always exits unsuccessfully. So context.DeadlineExceeded is
// discarded and every timeout surfaced as the bare text "signal: killed", which
// tells the reader nothing about which limit was hit or how to raise it.
var ErrSubagentTimeout = errors.New("subagent timeout")

// ErrProcessDone marks a Steer on a subagent process that has already exited.
var ErrProcessDone = errors.New("subagent process already done")

// maxStderrCapture bounds the stderr retained for error reporting. The whole
// point of draining stderr concurrently is to unblock a chatty child, so the
// buffer must not become the new place the memory goes.
const maxStderrCapture = 64 * 1024

// The two limits a subagent can die of, documented together because a user
// hit by one usually needs to know the other exists:
//
//   - absolute wall-clock cap — PI_SUBAGENT_TIMEOUT_MS env var, or a
//   `timeout:` (milliseconds) key in the agent's frontmatter (timeoutHint);
//   - inactivity cap — PI_SUBAGENT_INACTIVITY_MS env var (inactivityHint).
//   There is no frontmatter key for it.
//
// Raising the absolute cap does not raise the inactivity one, and
// ResolveTimeout clamps inactivity to never exceed absolute — so following
// the wrong hint cannot fix a kill, only move the other limit.
//
// Each kill names the hint for the limit that actually fired.

// timeoutHint names the knobs for the absolute time limit, because a subagent
// killed by a limit is exactly the moment the reader wants to know the limit
// is adjustable.
const timeoutHint = "raise it with PI_SUBAGENT_TIMEOUT_MS or a `timeout:` (milliseconds) key in the agent's frontmatter"

// inactivityHint names the knob that governs the inactivity limit — the one
// that fires when a child produces no output, distinct from the absolute cap
// above. (The stream-idle watch and its keep-alive heartbeat, issue #37,
// exist so healthy children never get to test this limit.)
const inactivityHint = "raise it with PI_SUBAGENT_INACTIVITY_MS (milliseconds)"

// SpawnOpts holds options for spawning a subagent process.
type SpawnOpts struct {
	AgentID     string   // Unique ID for this agent
	Model       string   // Model name to use
	WorkDir     string   // Working directory for the process
	Prompt      string   // Task prompt to send
	Instruction string   // System instruction for the subagent
	Timeout     int      // Timeout in milliseconds (0 = use default)
	Env         []string // Additional environment variables (merged with filtered process env)
	BaseURL     string   // LLM API base URL (passed as --url flag)
	Insecure    bool     // Skip TLS verification (passed as --insecure flag)
	Headers     []string // Extra HTTP headers (passed as --header flags)
	// LSP selects the child's language-server tool surface (passed as --lsp).
	// Empty leaves the flag off, so the child uses its own default. This is the
	// seam that lets a navigation-heavy agent run with the full LSP set while
	// the parent session pays for none of it.
	LSP string
	// Tools restricts the child to the named tools (passed as --tools). Empty
	// leaves the flag off, so the child keeps its full tool set. This carries
	// the agent's frontmatter `tools:` list to the process that owns the tools.
	Tools []string
	// Temperature is the child's sampling temperature (passed as
	// --temperature). Zero leaves the flag off, so the child inherits its
	// caller's default; see AgentConfig.Temperature for the 0-means-unset tie.
	Temperature float64
	// ThinkingLevel is the child's reasoning effort (passed as --thinking),
	// already normalized by NormalizeReasoningEffort. Empty leaves the flag
	// off and the child on its configured default.
	ThinkingLevel string
	// Steps caps the child's tool-call iterations (passed as --steps). Zero
	// leaves the flag off — no limit.
	Steps int
}

// Spawner creates and manages subagent pi processes.
type Spawner struct {
	// PiBinary is the path to the pi binary. Defaults to os.Executable() when
	// created via NewSpawner(""), ensuring subagents match the parent version.
	PiBinary string
}

// NewSpawner creates a new Spawner. If piBinary is empty, uses the currently
// running executable path (via os.Executable) to ensure subagents run the same
// version as the parent process.
func NewSpawner(piBinary string) *Spawner {
	if piBinary == "" {
		if exe, err := os.Executable(); err == nil {
			piBinary = exe
		} else {
			piBinary = "pi"
		}
	}
	return &Spawner{PiBinary: piBinary}
}

// Process represents a running subagent pi process.
type Process struct {
	cmd    *exec.Cmd
	events chan Event
	done   chan struct{}
	cancel context.CancelFunc
	// stdin is the steer channel to a json-mode child: Process.Steer writes
	// one line per follow-up message. Set only by spawner.Spawn — ACP and
	// codex adapters leave it nil, and steering those is not supported.
	stdin  io.WriteCloser
	result string
	err    error
	mu     sync.Mutex
}

// Events returns a channel that receives streaming events from the subagent.
// The channel is closed when the process exits.
func (p *Process) Events() <-chan Event {
	return p.events
}

// Steer sends text to the child as a follow-up message. The child (pi --mode
// json) queues it and runs it as a full extra turn in the same session once
// its current turn finishes; see runJSON in internal/cli. Fails with
// ErrProcessDone once the child has exited, and for agents whose runner has
// no stdin channel (ACP, codex).
func (p *Process) Steer(text string) error {
	select {
	case <-p.done:
		return fmt.Errorf("steer subagent: %w", ErrProcessDone)
	default:
	}
	p.mu.Lock()
	w := p.stdin
	p.mu.Unlock()
	if w == nil {
		return fmt.Errorf("steer subagent: no stdin channel (agent runner does not support steering)")
	}
	if _, err := io.WriteString(w, text+"\n"); err != nil {
		return fmt.Errorf("steer subagent: %w", err)
	}
	return nil
}

// Wait blocks until the process exits and returns the accumulated result or an error.
func (p *Process) Wait() (string, error) {
	<-p.done
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.result, p.err
}

// Cancel kills the subagent process.
func (p *Process) Cancel() {
	p.cancel()
}

// spawnArgs builds the child pi command line. Split out from Spawn so the flag
// wiring can be tested without starting a process — the --lsp value in
// particular decides how much context the child spends on tool declarations.
func spawnArgs(opts SpawnOpts) []string {
	args := []string{"--mode", "json"}
	if opts.Model != "" {
		args = append(args, "--model", opts.Model)
	}
	if opts.BaseURL != "" {
		args = append(args, "--url", opts.BaseURL)
	}
	if opts.Insecure {
		args = append(args, "--insecure")
	}
	for _, h := range opts.Headers {
		args = append(args, "--header", h)
	}
	if opts.Instruction != "" {
		args = append(args, "--system", opts.Instruction)
	}
	if opts.LSP != "" {
		args = append(args, "--lsp", opts.LSP)
	}
	if len(opts.Tools) > 0 {
		args = append(args, "--tools", strings.Join(opts.Tools, ","))
	}
	if opts.Temperature != 0 {
		args = append(args, "--temperature", strconv.FormatFloat(opts.Temperature, 'g', -1, 64))
	}
	if opts.ThinkingLevel != "" {
		args = append(args, "--thinking", opts.ThinkingLevel)
	}
	if opts.Steps > 0 {
		args = append(args, "--steps", strconv.Itoa(opts.Steps))
	}
	// The prompt is positional and must stay last. The "--" terminator keeps
	// cobra from reading it as a subcommand or as flags: a prompt whose first
	// word matches a root subcommand ("ping", "model", "memory", ...) made the
	// child run "pi ping", which has no --mode flag, and die on arg parsing.
	return append(args, "--", opts.Prompt)
}

// buildCommand assembles the child pi command: arguments, environment, kill
// semantics and working directory.
func (s *Spawner) buildCommand(procCtx context.Context, opts SpawnOpts) *exec.Cmd {
	cmd := exec.CommandContext(procCtx, s.PiBinary, spawnArgs(opts)...)

	// Set up environment: filtered process env + additional env vars. The
	// child's concurrency budget is this process's share, not a copy of it —
	// see ChildEnv.
	baseEnv := ChildEnv(ConcurrencyFromEnv())
	if len(opts.Env) > 0 {
		cmd.Env = append(overrideEnv(baseEnv, opts.Env), opts.Env...)
	} else {
		cmd.Env = baseEnv
	}

	// Ensure the process and its children are killed on cancel.
	setPlatformAttrs(cmd)
	cmd.WaitDelay = 3 * time.Second
	if opts.WorkDir != "" {
		cmd.Dir = opts.WorkDir
	}
	return cmd
}

// overrideEnv drops from base the entries whose name also appears in
// overrides, so a spawn env var replaces the inherited one instead of
// racing it as a duplicate: the base carries every allowlisted PI_* variable
// the parent had, and the child's getenv resolves duplicates by order — the
// inherited copy can win. This is how an agent's frontmatter `mcp:` (or a
// permission blob, or a stream-idle budget) beats the parent's own value of
// the same variable.
func overrideEnv(base, overrides []string) []string {
	names := make(map[string]bool, len(overrides))
	for _, e := range overrides {
		if k, _, ok := strings.Cut(e, "="); ok {
			names[k] = true
		}
	}
	out := make([]string, 0, len(base))
	for _, e := range base {
		if k, _, ok := strings.Cut(e, "="); ok && names[k] {
			continue
		}
		out = append(out, e)
	}
	return out
}

// startChildProcess opens all three pipes and starts cmd. The pipes must be
// created before Start, so a failure at any of the four steps aborts the spawn.
// The stdin write end is the steer channel — the child reads it as follow-up
// messages, and exec's Wait closes it after the process exits.
func startChildProcess(cmd *exec.Cmd) (stdin io.WriteCloser, stdout, stderr io.ReadCloser, err error) {
	stdin, err = cmd.StdinPipe()
	if err != nil {
		return nil, nil, nil, fmt.Errorf("creating stdin pipe: %w", err)
	}

	stdout, err = cmd.StdoutPipe()
	if err != nil {
		return nil, nil, nil, fmt.Errorf("creating stdout pipe: %w", err)
	}

	stderr, err = cmd.StderrPipe()
	if err != nil {
		return nil, nil, nil, fmt.Errorf("creating stderr pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		return nil, nil, nil, fmt.Errorf("starting pi process: %w", err)
	}
	return stdin, stdout, stderr, nil
}

// stderrCollector drains a child's stderr into a bounded buffer, concurrently
// with stdout.
//
// These are two independent pipes with their own kernel buffers. Reading them
// in sequence — stdout to EOF, then stderr — deadlocks the moment the child
// writes more than a pipe buffer (~64KB) to stderr: the child blocks in
// write(2), so it never closes stdout, so the parent never finishes the stdout
// scan and never reaches the stderr read. The child then sits there until a
// timeout kills it, and because stderr was never drained the error arrives with
// no diagnostic text attached at all.
type stderrCollector struct {
	mu   sync.Mutex
	buf  strings.Builder
	done chan struct{}
}

// startStderrCollector begins draining r in the background. done is closed at
// EOF, which is the signal that the pipe is safe to Wait on.
func startStderrCollector(r io.Reader) *stderrCollector {
	c := &stderrCollector{done: make(chan struct{})}
	go func() {
		defer close(c.done)
		sc := bufio.NewScanner(r)
		for sc.Scan() {
			c.mu.Lock()
			if c.buf.Len() < maxStderrCapture {
				c.buf.WriteString(sc.Text())
				c.buf.WriteByte('\n')
			}
			c.mu.Unlock()
		}
	}()
	return c
}

// Text returns the captured stderr, trimmed.
func (c *stderrCollector) Text() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return strings.TrimSpace(c.buf.String())
}

// startLineScanner feeds stdout lines to the reader goroutine rather than
// having it scan inline, so the reader can wait on "a line arrived" and
// "nothing has arrived in a while" at the same time. A bare
// `for scanner.Scan()` can only block.
func startLineScanner(r io.Reader) <-chan string {
	lines := make(chan string, 64)
	go func() {
		defer close(lines)
		sc := bufio.NewScanner(r)
		sc.Buffer(make([]byte, 0, 256*1024), 1024*1024) // up to 1MB lines
		for sc.Scan() {
			lines <- sc.Text()
		}
	}()
	return lines
}

// Spawn starts a pi subprocess in JSON mode and returns a Process handle for streaming events.
func (s *Spawner) Spawn(ctx context.Context, opts SpawnOpts) (*Process, error) {
	if opts.Prompt == "" {
		return nil, fmt.Errorf("prompt is required")
	}

	// Resolve timeout configuration (applies defaults if not set).
	timeoutCfg := ResolveTimeout(opts.Timeout)
	procCtx, cancel := context.WithTimeout(ctx, timeoutCfg.Absolute)

	cmd := s.buildCommand(procCtx, opts)

	stdin, stdout, stderr, err := startChildProcess(cmd)
	if err != nil {
		cancel()
		return nil, err
	}

	proc := &Process{
		cmd:    cmd,
		events: make(chan Event, 64),
		done:   make(chan struct{}),
		cancel: cancel,
		stdin:  stdin,
	}

	stderrC := startStderrCollector(stderr)
	lines := startLineScanner(stdout)

	// Reader goroutine: parse JSONL from stdout, send events.
	go proc.pumpChildOutput(procCtx, timeoutCfg, lines, stderrC, cancel)

	return proc, nil
}

// pumpChildOutput is the reader goroutine: it consumes stdout lines until the
// child is done or goes silent, waits for the process, and records the result.
func (p *Process) pumpChildOutput(procCtx context.Context, timeoutCfg TimeoutConfig, lines <-chan string, stderrC *stderrCollector, cancel context.CancelFunc) {
	defer close(p.done)
	defer close(p.events)

	// The inactivity timer is what separates "slow" from "wedged". Without
	// it the absolute cap is the only limit, so a long but productive agent
	// is killed on the same rule as one that has hung — which is precisely
	// the failure this fixes.
	idle := NewInactivityTimer(timeoutCfg.Inactivity)
	defer idle.Stop()

	result, timedOutIdle := p.readChildLines(lines, idle, cancel)

	// Both pipes must be at EOF before Wait, and stderr is wanted for the
	// error message below.
	<-stderrC.done

	// Wait for process exit.
	waitErr := p.cmd.Wait()

	p.mu.Lock()
	p.result = result
	p.err = childExitError(timeoutCfg, timedOutIdle, procCtx.Err(), waitErr, stderrC.Text())
	if p.err != nil {
		p.sendEvent(Event{Type: "error", Error: p.err.Error()})
	}
	p.mu.Unlock()
}

// readChildLines consumes stdout lines, emitting an Event for each and
// accumulating the assistant text. It returns the accumulated result and
// whether the inactivity timer fired.
func (p *Process) readChildLines(lines <-chan string, idle *InactivityTimer, cancel context.CancelFunc) (string, bool) {
	var resultBuilder strings.Builder

	for {
		select {
		case line, ok := <-lines:
			if !ok {
				return resultBuilder.String(), false
			}
			idle.Reset()
			if line == "" {
				continue
			}
			p.emitChildLine(line, &resultBuilder)

		case <-idle.C():
			cancel()          // kills the process group; stdout closes, so lines drains
			for range lines { //nolint:revive // drain so the scanner goroutine can exit
			}
			return resultBuilder.String(), true
		}
	}
}

// emitChildLine parses one JSONL line from the child and sends the matching
// Event, appending assistant text to result.
func (p *Process) emitChildLine(line string, result *strings.Builder) {
	var ev jsonEvent
	if err := json.Unmarshal([]byte(line), &ev); err != nil {
		// Non-JSON output; emit as text.
		p.sendEvent(Event{Type: "text_delta", Content: line})
		return
	}

	switch ev.Type {
	case "error":
		// Child emits provider failures as {"type":"error","error":...}
		// (internal/cli/cli.go); keep the text in Event.Error so the parent can
		// classify it instead of dropping it into Content.
		p.sendEvent(Event{Type: "error", Error: ev.Error})
	case "keepalive":
		// The child's LLM-wait pulse (issue #37). readChildLines already
		// reset the inactivity watchdog before this call — the line arriving
		// is the whole point — so swallow it here: it carries no content and
		// must not surface in the event stream.
		return
	case "text_delta":
		result.WriteString(ev.Delta)
		p.sendEvent(Event{Type: "text_delta", Content: ev.Delta})
	case "tool_call":
		p.sendEvent(Event{Type: "tool_call", Content: ev.ToolName, ToolArgs: ev.ToolInput})
	case "tool_result":
		p.sendEvent(Event{Type: "tool_result", Content: ev.Content})
	case "message_start":
		p.sendEvent(Event{Type: "message_start", SessionID: ev.SessionID})
	case "message_end":
		p.sendEvent(Event{Type: "message_end"})
	default:
		p.sendEvent(Event{Type: ev.Type, Content: ev.Delta + ev.Content})
	}
}

// childExitError classifies how a child pi process ended, returning nil for a
// clean exit. The two timeout cases come first because a limit kill is also a
// signal kill, and would otherwise be reported as a bare process failure.
// Each branch names the knob for the limit that fired — the absolute branch
// the TIMEOUT_MS/frontmatter knobs, the inactivity branch INACTIVITY_MS —
// so following the message moves the limit that actually killed.
func childExitError(timeoutCfg TimeoutConfig, timedOutIdle bool, ctxErr, waitErr error, stderrStr string) error {
	switch {
	case timedOutIdle:
		return fmt.Errorf("pi subagent produced no output for %s: %w (%s)",
			timeoutCfg.Inactivity, ErrSubagentTimeout, inactivityHint)
	case errors.Is(ctxErr, context.DeadlineExceeded):
		return fmt.Errorf("pi subagent exceeded its %s time limit: %w (%s)",
			timeoutCfg.Absolute, ErrSubagentTimeout, timeoutHint)
	case waitErr != nil && stderrStr != "":
		return fmt.Errorf("pi process failed: %w: %s", waitErr, stderrStr)
	case waitErr != nil:
		return fmt.Errorf("pi process failed: %w", waitErr)
	}
	return nil
}

// sendEvent sends an event to the channel without blocking.
func (p *Process) sendEvent(ev Event) {
	select {
	case p.events <- ev:
	default:
		// Channel full; drop event to avoid blocking.
	}
}

// jsonEvent mirrors the JSONL event format from pi --mode json (see internal/cli/cli.go).
type jsonEvent struct {
	Type      string `json:"type"`
	Agent     string `json:"agent,omitempty"`
	Role      string `json:"role,omitempty"`
	Delta     string `json:"delta,omitempty"`
	Content   string `json:"content,omitempty"`
	Error     string `json:"error,omitempty"`
	ToolName  string `json:"tool_name,omitempty"`
	ToolInput any    `json:"tool_input,omitempty"`
	SessionID string `json:"session_id,omitempty"`
}
