package memory

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"
)

// mockCompressor implements Compressor for testing.
type mockCompressor struct {
	mu      sync.Mutex
	calls   []RawObservation
	failAt  int // fail on the Nth call (-1 = never fail)
	latency time.Duration
}

func newMockCompressor() *mockCompressor {
	return &mockCompressor{failAt: -1}
}

func (m *mockCompressor) CompressObservation(_ context.Context, raw RawObservation) (*Observation, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.calls = append(m.calls, raw)
	callNum := len(m.calls)

	if m.latency > 0 {
		time.Sleep(m.latency)
	}

	if m.failAt > 0 && callNum == m.failAt {
		return nil, fmt.Errorf("mock compression failure on call %d", callNum)
	}

	return &Observation{
		SessionID:   raw.SessionID,
		Project:     raw.Project,
		Title:       "Compressed: " + raw.ToolName,
		Type:        TypeDiscovery,
		Text:        "compressed observation text",
		SourceFiles: []string{},
		ToolName:    raw.ToolName,
		CreatedAt:   raw.Timestamp,
	}, nil
}

func (m *mockCompressor) getCalls() []RawObservation {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]RawObservation, len(m.calls))
	copy(out, m.calls)
	return out
}

// mockStore implements Store for worker tests.
type mockStore struct {
	mu             sync.Mutex
	observations   []*Observation
	insertObsError error
}

func newMockStore() *mockStore {
	return &mockStore{}
}

func (s *mockStore) InsertObservation(_ context.Context, obs *Observation) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.insertObsError != nil {
		return s.insertObsError
	}
	obs.ID = int64(len(s.observations) + 1)
	s.observations = append(s.observations, obs)
	return nil
}

func (s *mockStore) getObservations() []*Observation {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]*Observation, len(s.observations))
	copy(out, s.observations)
	return out
}

// Unused Store methods — satisfy interface.
func (s *mockStore) CreateSession(context.Context, *Session) error { return nil }
func (s *mockStore) CompleteSession(context.Context, string) error { return nil }
func (s *mockStore) GetObservations(context.Context, []int64) ([]*Observation, error) {
	return nil, nil
}
func (s *mockStore) RecentObservations(context.Context, string, int) ([]*Observation, error) {
	return nil, nil
}
func (s *mockStore) SessionObservations(context.Context, string) ([]*Observation, error) {
	return s.getObservations(), nil
}
func (s *mockStore) HasObservations(context.Context, string) (bool, error) {
	return len(s.getObservations()) > 0, nil
}
func (s *mockStore) UpsertSummary(context.Context, *SessionSummary) error { return nil }
func (s *mockStore) RecentSummaries(context.Context, string, int) ([]*SessionSummary, error) {
	return nil, nil
}
func (s *mockStore) Search(context.Context, SearchQuery) (*SearchResult, error) {
	return &SearchResult{}, nil
}
func (s *mockStore) Timeline(context.Context, int64, int, int) ([]*Observation, error) {
	return nil, nil
}
func (s *mockStore) Close() error { return nil }

// fakeToolForCallback is a minimal implementation of toolpkg.Tool for testing.
type fakeToolForCallback struct{ name string }

func (f *fakeToolForCallback) Name() string        { return f.name }
func (f *fakeToolForCallback) Description() string { return "" }
func (f *fakeToolForCallback) IsLongRunning() bool { return false }
func (f *fakeToolForCallback) Declaration() (interface{}, error) {
	return nil, nil
}

func makeRaw(toolName string) RawObservation {
	return RawObservation{
		SessionID:  "test-session",
		Project:    "/test/project",
		ToolName:   toolName,
		ToolInput:  map[string]any{"file_path": "/test/file.go"},
		ToolOutput: map[string]any{"content": "hello"},
		Timestamp:  time.Now(),
	}
}

func TestWorkerEnqueueAndProcess(t *testing.T) {
	store := newMockStore()
	comp := newMockCompressor()
	w := NewWorker(store, comp, 10)

	ctx := context.Background()
	w.Start(ctx)

	for i := 0; i < 5; i++ {
		w.Enqueue(makeRaw(fmt.Sprintf("tool-%d", i)))
	}

	shutdownCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	if err := w.Shutdown(shutdownCtx); err != nil {
		t.Fatalf("shutdown: %v", err)
	}

	calls := comp.getCalls()
	if len(calls) != 5 {
		t.Errorf("compressor calls = %d, want 5", len(calls))
	}

	obs := store.getObservations()
	if len(obs) != 5 {
		t.Errorf("stored observations = %d, want 5", len(obs))
	}

	for i, o := range obs {
		expected := fmt.Sprintf("Compressed: tool-%d", i)
		if o.Title != expected {
			t.Errorf("obs[%d].Title = %q, want %q", i, o.Title, expected)
		}
	}
}

func TestWorkerChannelFullDrop(t *testing.T) {
	store := newMockStore()
	comp := newMockCompressor()
	// Buffer of 1 — don't start worker so channel stays full
	w := NewWorker(store, comp, 1)

	// First enqueue fills the buffer
	w.Enqueue(makeRaw("tool-1"))
	// Second should be dropped (non-blocking)
	w.Enqueue(makeRaw("tool-2"))
	// Third should also be dropped
	w.Enqueue(makeRaw("tool-3"))

	// Now start and drain
	ctx := context.Background()
	w.Start(ctx)

	shutdownCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	if err := w.Shutdown(shutdownCtx); err != nil {
		t.Fatalf("shutdown: %v", err)
	}

	obs := store.getObservations()
	if len(obs) != 1 {
		t.Errorf("stored observations = %d, want 1 (others dropped)", len(obs))
	}
}

func TestWorkerShutdownDrain(t *testing.T) {
	store := newMockStore()
	comp := newMockCompressor()
	w := NewWorker(store, comp, 10)

	ctx := context.Background()
	w.Start(ctx)

	// Enqueue observations
	for i := 0; i < 3; i++ {
		w.Enqueue(makeRaw(fmt.Sprintf("tool-%d", i)))
	}

	// Shutdown should drain all pending
	shutdownCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	if err := w.Shutdown(shutdownCtx); err != nil {
		t.Fatalf("shutdown: %v", err)
	}

	obs := store.getObservations()
	if len(obs) != 3 {
		t.Errorf("stored observations = %d, want 3 (all drained)", len(obs))
	}
}

func TestWorkerCompressionFailureFallback(t *testing.T) {
	store := newMockStore()
	comp := newMockCompressor()
	comp.failAt = 2 // Fail on second call

	w := NewWorker(store, comp, 10)

	ctx := context.Background()
	w.Start(ctx)

	w.Enqueue(makeRaw("tool-ok"))
	w.Enqueue(makeRaw("tool-fail"))
	w.Enqueue(makeRaw("tool-ok2"))

	shutdownCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	if err := w.Shutdown(shutdownCtx); err != nil {
		t.Fatalf("shutdown: %v", err)
	}

	obs := store.getObservations()
	if len(obs) != 3 {
		t.Fatalf("stored observations = %d, want 3", len(obs))
	}

	// First: compressed normally
	if obs[0].Title != "Compressed: tool-ok" {
		t.Errorf("obs[0].Title = %q, want compressed", obs[0].Title)
	}

	// Second: fallback (compression failed)
	if obs[1].Title != "tool-fail (uncompressed)" {
		t.Errorf("obs[1].Title = %q, want fallback", obs[1].Title)
	}
	if obs[1].Type != TypeChange {
		t.Errorf("obs[1].Type = %q, want %q", obs[1].Type, TypeChange)
	}

	// Third: compressed normally
	if obs[2].Title != "Compressed: tool-ok2" {
		t.Errorf("obs[2].Title = %q, want compressed", obs[2].Title)
	}
}

func TestWorkerPrivacyFiltering(t *testing.T) {
	store := newMockStore()
	comp := newMockCompressor()
	w := NewWorker(store, comp, 10)

	ctx := context.Background()
	w.Start(ctx)

	raw := RawObservation{
		SessionID: "test-session",
		Project:   "/test",
		ToolName:  "read",
		ToolInput: map[string]any{
			"file_path": "/secret.go",
			"note":      "contains <private>secret-key</private> data",
		},
		ToolOutput: map[string]any{
			"content": "API_KEY=<private>abc123</private>",
		},
		Timestamp: time.Now(),
	}
	w.Enqueue(raw)

	shutdownCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	if err := w.Shutdown(shutdownCtx); err != nil {
		t.Fatalf("shutdown: %v", err)
	}

	calls := comp.getCalls()
	if len(calls) != 1 {
		t.Fatalf("compressor calls = %d, want 1", len(calls))
	}

	// Verify privacy tags were stripped before compression
	call := calls[0]
	if note, ok := call.ToolInput["note"].(string); ok {
		if note != "contains [PRIVATE] data" {
			t.Errorf("input note = %q, want privacy stripped", note)
		}
	} else {
		t.Error("input note not found or not string")
	}

	if content, ok := call.ToolOutput["content"].(string); ok {
		if content != "API_KEY=[PRIVATE]" {
			t.Errorf("output content = %q, want privacy stripped", content)
		}
	} else {
		t.Error("output content not found or not string")
	}
}

func TestExtractSourceFiles(t *testing.T) {
	tests := []struct {
		name  string
		input map[string]any
		want  int
	}{
		{"file_path key", map[string]any{"file_path": "/a.go"}, 1},
		{"path key", map[string]any{"path": "/b.go"}, 1},
		{"no file keys", map[string]any{"query": "search"}, 0},
		{"empty map", map[string]any{}, 0},
		{"multiple keys", map[string]any{"file_path": "/a.go", "path": "/b.go"}, 2},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			files := extractSourceFiles(tt.input)
			if len(files) != tt.want {
				t.Errorf("extractSourceFiles() = %d files, want %d", len(files), tt.want)
			}
		})
	}
}

func TestBuildMemoryCallback(t *testing.T) {
	store := newMockStore()
	comp := newMockCompressor()
	w := NewWorker(store, comp, 10)

	ctx := context.Background()
	w.Start(ctx)

	cb := BuildMemoryCallback(w, "sess-1", "/my/project")
	cb("read", map[string]any{"file_path": "/test.go"}, map[string]any{"content": "hello"})

	shutdownCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	if err := w.Shutdown(shutdownCtx); err != nil {
		t.Fatalf("shutdown: %v", err)
	}

	obs := store.getObservations()
	if len(obs) != 1 {
		t.Fatalf("stored observations = %d, want 1", len(obs))
	}
	if obs[0].SessionID != "sess-1" {
		t.Errorf("SessionID = %q, want %q", obs[0].SessionID, "sess-1")
	}
	if obs[0].Project != "/my/project" {
		t.Errorf("Project = %q, want %q", obs[0].Project, "/my/project")
	}
}

func TestNewWorkerDefaultBufSize(t *testing.T) {
	// bufSize <= 0 should default to 100.
	store := newMockStore()
	comp := newMockCompressor()
	w := NewWorker(store, comp, 0)

	ctx := context.Background()
	w.Start(ctx)

	// Enqueue 10 items — well under the default 100 buffer.
	for i := 0; i < 10; i++ {
		w.Enqueue(makeRaw(fmt.Sprintf("t%d", i)))
	}

	shutdownCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	if err := w.Shutdown(shutdownCtx); err != nil {
		t.Fatalf("shutdown: %v", err)
	}

	if n := len(store.getObservations()); n != 10 {
		t.Errorf("observations = %d, want 10", n)
	}
}

func TestWorkerShutdownTimeout(t *testing.T) {
	store := newMockStore()
	// Use a compressor with a long latency so it doesn't finish before the timeout.
	comp := &mockCompressor{failAt: -1, latency: 200 * time.Millisecond}
	w := NewWorker(store, comp, 10)

	ctx := context.Background()
	w.Start(ctx)

	// Fill the queue.
	for i := 0; i < 5; i++ {
		w.Enqueue(makeRaw(fmt.Sprintf("slow-%d", i)))
	}

	// Shutdown with a very short timeout — should time out.
	shortCtx, cancel := context.WithTimeout(ctx, 1*time.Millisecond)
	defer cancel()
	err := w.Shutdown(shortCtx)
	if err == nil {
		t.Error("expected timeout error from Shutdown, got nil")
	}
}

func TestBuildAfterToolCallback(t *testing.T) {
	store := newMockStore()
	comp := newMockCompressor()
	w := NewWorker(store, comp, 10)

	ctx := context.Background()
	w.Start(ctx)

	excluded := map[string]bool{"skip-me": true}
	cb := BuildAfterToolCallback(w, "sess-cb", "/project", excluded)

	// A nil tool error means it should be enqueued.
	mockTool := &fakeToolForCallback{name: "real-tool"}
	cb(nil, mockTool, map[string]any{"key": "val"}, map[string]any{"out": "ok"}, nil)

	// An excluded tool should not be enqueued.
	skippedTool := &fakeToolForCallback{name: "skip-me"}
	cb(nil, skippedTool, map[string]any{}, map[string]any{}, nil)

	// A tool error should not be enqueued.
	cb(nil, mockTool, map[string]any{}, map[string]any{}, fmt.Errorf("tool failed"))

	shutdownCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	if err := w.Shutdown(shutdownCtx); err != nil {
		t.Fatalf("shutdown: %v", err)
	}

	obs := store.getObservations()
	if len(obs) != 1 {
		t.Errorf("stored observations = %d, want 1 (excluded + errored should be skipped)", len(obs))
	}
	if obs[0].ToolName != "real-tool" {
		t.Errorf("ToolName = %q, want real-tool", obs[0].ToolName)
	}
}

func TestWorkerAfterStoreHook(t *testing.T) {
	store := newMockStore()
	comp := newMockCompressor()
	w := NewWorker(store, comp, 10)
	var hooked []*Observation
	w.OnAfterStore(func(_ context.Context, obs *Observation) {
		hooked = append(hooked, obs)
	})

	ctx := context.Background()
	w.Start(ctx)
	w.Enqueue(makeRaw("hook-tool"))
	shutdownCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	if err := w.Shutdown(shutdownCtx); err != nil {
		t.Fatalf("shutdown: %v", err)
	}
	if len(hooked) != 1 || hooked[0].ToolName != "hook-tool" {
		t.Fatalf("hooked = %#v", hooked)
	}
}

func TestWorkerStoreFailureSkipsHook(t *testing.T) {
	store := newMockStore()
	store.insertObsError = fmt.Errorf("store down")
	comp := newMockCompressor()
	w := NewWorker(store, comp, 10)
	hookCalled := false
	w.OnAfterStore(func(context.Context, *Observation) { hookCalled = true })

	ctx := context.Background()
	w.Start(ctx)
	w.Enqueue(makeRaw("store-fail"))
	shutdownCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	if err := w.Shutdown(shutdownCtx); err != nil {
		t.Fatalf("shutdown: %v", err)
	}
	if len(store.getObservations()) != 0 {
		t.Fatal("expected no stored observations")
	}
	if hookCalled {
		t.Fatal("hook should not run when storing fails")
	}
}

func TestTruncateFallbackText(t *testing.T) {
	raw := RawObservation{
		ToolName:   "test",
		ToolInput:  map[string]any{"key": "value"},
		ToolOutput: map[string]any{"result": "ok"},
	}
	text := truncateFallbackText(raw)
	if text == "" {
		t.Error("truncateFallbackText returned empty string")
	}

	// Test truncation of large output
	largeVal := make([]byte, 8192)
	for i := range largeVal {
		largeVal[i] = 'x'
	}
	raw.ToolOutput = map[string]any{"big": string(largeVal)}
	text = truncateFallbackText(raw)
	if len(text) > 4200 { // 4096 + "...(truncated)" + some JSON overhead
		t.Errorf("truncated text too long: %d", len(text))
	}
}

func TestTruncateFallbackTextMarshalError(t *testing.T) {
	raw := RawObservation{
		ToolName:   "bad-json",
		ToolInput:  map[string]any{"bad": func() {}},
		ToolOutput: map[string]any{"result": "ok"},
	}
	text := truncateFallbackText(raw)
	if text != "tool=bad-json (marshal error)" {
		t.Fatalf("text = %q", text)
	}
}

// TestWorkerEnqueueAfterShutdownClosedChannel pins the production panic from
// the after-tool callback: Enqueue must drop the observation instead of
// sending on the closed obsChan.
func TestWorkerEnqueueAfterShutdownClosedChannel(t *testing.T) {
	store := newMockStore()
	comp := newMockCompressor()
	w := NewWorker(store, comp, 10)

	ctx := context.Background()
	w.Start(ctx)
	w.Enqueue(makeRaw("tool-before-shutdown"))

	shutdownCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	if err := w.Shutdown(shutdownCtx); err != nil {
		t.Fatalf("shutdown: %v", err)
	}

	// A late after-tool callback fires after the channel is closed.
	w.Enqueue(makeRaw("tool-after-shutdown"))

	obs := store.getObservations()
	if len(obs) != 1 {
		t.Fatalf("stored observations = %d, want 1 (late enqueue must be dropped)", len(obs))
	}
	if obs[0].ToolName != "tool-before-shutdown" {
		t.Errorf("obs[0].ToolName = %q, want %q", obs[0].ToolName, "tool-before-shutdown")
	}
}

// TestWorkerShutdownIdempotent pins the "close of closed channel" panic on a
// second Shutdown. The case the fix targets is a retry after a drain timeout:
// the first Shutdown returns an error while the consumer is still working, so
// the caller retries — the channel is already closed, and the retried call
// must wait out the in-flight drain instead of panicking or returning early.
func TestWorkerShutdownIdempotent(t *testing.T) {
	t.Run("repeat after successful drain", func(t *testing.T) {
		store := newMockStore()
		comp := newMockCompressor()
		w := NewWorker(store, comp, 10)

		ctx := context.Background()
		w.Start(ctx)
		w.Enqueue(makeRaw("tool-1"))

		shutdownCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
		defer cancel()
		if err := w.Shutdown(shutdownCtx); err != nil {
			t.Fatalf("first shutdown: %v", err)
		}
		if err := w.Shutdown(shutdownCtx); err != nil {
			t.Fatalf("second shutdown: %v", err)
		}

		if obs := store.getObservations(); len(obs) != 1 {
			t.Errorf("stored observations = %d, want 1", len(obs))
		}
	})

	t.Run("retry after drain timeout", func(t *testing.T) {
		store := newMockStore()
		// Slow compressor so the first Shutdown times out mid-drain while the
		// consumer is still working through the queue.
		comp := &mockCompressor{failAt: -1, latency: 100 * time.Millisecond}
		w := NewWorker(store, comp, 10)

		ctx := context.Background()
		w.Start(ctx)

		const want = 5
		for i := 0; i < want; i++ {
			w.Enqueue(makeRaw(fmt.Sprintf("slow-%d", i)))
		}

		// The drain needs 5 * 100ms; a 10ms context fires well before it
		// finishes. The channel is now closed and the consumer is still busy.
		shortCtx, shortCancel := context.WithTimeout(ctx, 10*time.Millisecond)
		if err := w.Shutdown(shortCtx); err == nil {
			t.Error("first shutdown: expected timeout error, got nil")
		}
		shortCancel()

		// The retry must not panic on the closed channel and must wait for
		// the drain left running by the timed-out call.
		longCtx, longCancel := context.WithTimeout(ctx, 10*time.Second)
		defer longCancel()
		if err := w.Shutdown(longCtx); err != nil {
			t.Fatalf("second shutdown: %v", err)
		}

		if obs := store.getObservations(); len(obs) != want {
			t.Errorf("stored observations = %d, want %d (queue must drain fully)",
				len(obs), want)
		}
	})
}

// TestWorkerEnqueueShutdownRace hammers Enqueue and Shutdown from many
// goroutines: no send on a closed channel and no race between the close and
// an in-flight enqueue. Under the old code this panics or trips the race
// detector.
//
// A start gate keeps the enqueuers from racing the Shutdown goroutine to the
// scheduler: every sender is up and spinning before Shutdown can close the
// channel, so the close lands in the middle of the send storm instead of
// before it — the enqueue-vs-close window is the one the production panic
// needed. The final send after Shutdown returns pins the late-callback drop
// path under the same concurrency.
func TestWorkerEnqueueShutdownRace(t *testing.T) {
	store := newMockStore()
	comp := newMockCompressor()
	w := NewWorker(store, comp, 64)

	ctx := context.Background()
	w.Start(ctx)

	const (
		enqueuers = 8
		perWorker = 100
	)
	start := make(chan struct{})
	shutdownReturned := make(chan struct{})

	var wg sync.WaitGroup
	for i := 0; i < enqueuers; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start
			for j := 0; j < perWorker; j++ {
				w.Enqueue(makeRaw(fmt.Sprintf("race-tool-%d-%d", i, j)))
			}
			// Shutdown has returned and the channel is closed; this send
			// must be dropped, not panic.
			<-shutdownReturned
			w.Enqueue(makeRaw(fmt.Sprintf("race-tool-late-%d", i)))
		}(i)
	}
	wg.Add(1)
	go func() {
		defer wg.Done()
		<-start
		if err := w.Shutdown(context.Background()); err != nil {
			t.Errorf("shutdown: %v", err)
		}
		close(shutdownReturned)
	}()

	// Release the gate: senders and Shutdown start at the same moment, so
	// sends are in flight while the channel closes and drains.
	close(start)
	wg.Wait()
}
