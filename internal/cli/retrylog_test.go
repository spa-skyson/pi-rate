package cli

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"iter"
	"os"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"

	"github.com/spa-skyson/pi-rate/internal/agent"
	"github.com/spa-skyson/pi-rate/internal/logger"
	"github.com/spa-skyson/pi-rate/internal/testenv"
)

// cliFlakyLLM fails its first `failures` GenerateContent calls with a
// transient error, then answers with a fixed text response.
type cliFlakyLLM struct {
	mu       sync.Mutex
	failures int
	err      error
	response string
}

func (m *cliFlakyLLM) Name() string { return "flaky" }

func (m *cliFlakyLLM) GenerateContent(_ context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	m.mu.Lock()
	fail := m.failures > 0
	if fail {
		m.failures--
	}
	m.mu.Unlock()

	return func(yield func(*model.LLMResponse, error) bool) {
		if fail {
			yield(nil, m.err)
			return
		}
		yield(&model.LLMResponse{
			Content: genai.NewContentFromText(m.response, genai.RoleModel),
		}, nil)
	}
}

// readLogEntries closes the session log (flushing any pending stream) and
// decodes every JSON line in it.
func readLogEntries(t *testing.T, log *logger.Logger) []logger.Entry {
	t.Helper()
	if err := log.Close(); err != nil {
		t.Fatalf("closing log: %v", err)
	}
	data, err := os.ReadFile(log.Path())
	if err != nil {
		t.Fatalf("reading log: %v", err)
	}
	var entries []logger.Entry
	dec := json.NewDecoder(bytes.NewReader(data))
	for dec.More() {
		var e logger.Entry
		if err := dec.Decode(&e); err != nil {
			t.Fatalf("decoding log entry: %v", err)
		}
		entries = append(entries, e)
	}
	return entries
}

// entriesOfType returns the log entries carrying the given type.
func entriesOfType(entries []logger.Entry, typ string) []logger.Entry {
	var found []logger.Entry
	for _, e := range entries {
		if e.Type == typ {
			found = append(found, e)
		}
	}
	return found
}

// Two transient failures followed by a clean attempt must leave two retry
// entries in the session log — numbered against the budget, with the delay
// and the cause — and no error entries. Three stream aborts and three
// retries used to leave zero records there (issue #64).
func TestRunPrintLogsRetryAttempts(t *testing.T) {
	tmpDir := t.TempDir()
	testenv.SetHome(t, tmpDir)

	lg, err := logger.New()
	if err != nil {
		t.Fatalf("logger.New: %v", err)
	}

	const cause = "llm stream idle: no stream data for 1m30s"
	llm := &cliFlakyLLM{failures: 2, err: errors.New(cause), response: "recovered"}
	ag, sessionID := newTestAgent(t, llm)

	stdout := captureStdout(t, func() {
		stderr := captureStderr(t, func() {
			if err := runPrint(context.Background(), ag, sessionID, "hello", lg, ""); err != nil {
				t.Errorf("runPrint error: %v", err)
			}
		})
		if n := strings.Count(stderr, "Retrying in"); n != 2 {
			t.Errorf("stderr announced %d retries, want 2", n)
		}
	})
	if !strings.Contains(stdout, "recovered") {
		t.Errorf("stdout missing the reply, got %q", stdout)
	}

	entries := readLogEntries(t, lg)
	retries := entriesOfType(entries, "info")
	if len(retries) != 2 {
		t.Fatalf("got %d info entries, want 2 retry entries: %+v", len(retries), retries)
	}

	max := agent.DefaultRetryConfig().MaxRetries
	wantAttempts := []string{"(1/" + strconv.Itoa(max) + ")", "(2/" + strconv.Itoa(max) + ")"}
	wantDelays := []string{"in 1s ", "in 2s "}
	for i, e := range retries {
		if !strings.Contains(e.Content, wantAttempts[i]) {
			t.Errorf("retry entry %d missing attempt %q: %q", i, wantAttempts[i], e.Content)
		}
		if !strings.Contains(e.Content, wantDelays[i]) {
			t.Errorf("retry entry %d missing delay %q: %q", i, wantDelays[i], e.Content)
		}
		if !strings.Contains(e.Content, cause) {
			t.Errorf("retry entry %d missing cause %q: %q", i, cause, e.Content)
		}
	}
	if errs := entriesOfType(entries, "error"); len(errs) != 0 {
		t.Errorf("unexpected error entries on a recovered turn: %+v", errs)
	}
}

// A turn whose every attempt fails transiently must end with the
// exhausted-budget wrapper in an "error" log entry, next to the per-retry
// "info" entries.
func TestRunJSONLogsExhaustedRetries(t *testing.T) {
	tmpDir := t.TempDir()
	testenv.SetHome(t, tmpDir)

	lg, err := logger.New()
	if err != nil {
		t.Fatalf("logger.New: %v", err)
	}

	const cause = "llm stream idle: no stream data for 1m30s"
	llm := &cliFlakyLLM{failures: 99, err: errors.New(cause), response: "never"}
	ag, sessionID := newTestAgent(t, llm)

	cfg := agent.RetryConfig{MaxRetries: 2, InitialDelay: time.Millisecond, MaxDelay: time.Millisecond}
	var buf bytes.Buffer
	em := newJSONEmitter(json.NewEncoder(&buf), lg, false)

	err = runJSONTurn(context.Background(), ag, sessionID, "hello", em, cfg, lg)
	if err == nil || !strings.Contains(err.Error(), "transient error after 2 retries") {
		t.Fatalf("runJSONTurn error = %v, want the exhausted-budget wrapper", err)
	}

	entries := readLogEntries(t, lg)
	retries := entriesOfType(entries, "info")
	if len(retries) != 2 {
		t.Fatalf("got %d info entries, want 2 retry entries: %+v", len(retries), retries)
	}
	for i, want := range []string{"(1/2)", "(2/2)"} {
		if !strings.Contains(retries[i].Content, want) {
			t.Errorf("retry entry %d missing attempt %q: %q", i, want, retries[i].Content)
		}
		if !strings.Contains(retries[i].Content, cause) {
			t.Errorf("retry entry %d missing cause: %q", i, retries[i].Content)
		}
	}
	var exhausted []logger.Entry
	for _, e := range entriesOfType(entries, "error") {
		if strings.Contains(e.Content, "transient error after 2 retries") {
			exhausted = append(exhausted, e)
		}
	}
	if len(exhausted) != 1 {
		t.Fatalf("got %d exhausted-budget error entries, want 1: %+v", len(exhausted), entriesOfType(entries, "error"))
	}
	if !strings.Contains(exhausted[0].Content, cause) {
		t.Errorf("error entry missing cause: %q", exhausted[0].Content)
	}
}
