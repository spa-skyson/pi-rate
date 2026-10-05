package piagent

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"sync"
	"testing"

	adkagent "google.golang.org/adk/v2/agent"
	adksession "google.golang.org/adk/v2/session"
	adktool "google.golang.org/adk/v2/tool"
	adktoolconfirmation "google.golang.org/adk/v2/tool/toolconfirmation"

	"github.com/spa-skyson/pi-rate/internal/config"
	"github.com/spa-skyson/pi-rate/internal/palace"
)

// The palace is opt-in for a library, so nothing else in this package exercises
// it. That left the whole opt-in path untested — which is the wrong way round:
// a feature nobody switches on by default is precisely the one that rots.

// newPalaceAt builds a palace with one drawer, so setupPalace's content gate
// opens. Uses the in-process embedder rather than Ollama so the test needs no
// daemon.
func newPalaceAt(t *testing.T, dbPath string) {
	t.Helper()
	p, err := palace.New(palace.WithDBPath(dbPath), palace.WithLocalEmbedder())
	if err != nil {
		t.Fatalf("palace.New: %v", err)
	}
	defer func() { _ = p.Close() }()

	if _, err := p.AddDrawer(context.Background(), palace.DrawerInput{
		Wing:       "pi-go",
		Room:       "piagent",
		Content:    "the palace gate reads drawer count, not file existence",
		Importance: 7,
	}); err != nil {
		t.Fatalf("AddDrawer: %v", err)
	}
}

func palaceCfg(dbPath string) config.Config {
	return config.Config{Palace: &config.PalaceConfig{DBPath: dbPath}}
}

func TestSetupPalaceEmptyPalaceRegistersNoTools(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "empty.db")
	p, err := palace.New(palace.WithDBPath(dbPath), palace.WithLocalEmbedder())
	if err != nil {
		t.Fatalf("palace.New: %v", err)
	}
	_ = p.Close()

	tools, _, closeFn := setupPalace(options{palaceEnabled: true}, palaceCfg(dbPath), nil)
	t.Cleanup(closeFn)

	if tools != nil {
		t.Fatalf("an empty palace registered %d tools; the gate reads drawer count", len(tools))
	}
}

// TestSetupPalaceWithContentRegistersTools is the path an embedder opts in for.
// It was entirely untested: palace.New, the content gate, tool building and the
// wake-up context all ran for the first time here.
func TestSetupPalaceWithContentRegistersTools(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "palace.db")
	newPalaceAt(t, dbPath)

	tools, wakeUp, closeFn := setupPalace(options{palaceEnabled: true}, palaceCfg(dbPath), nil)
	t.Cleanup(closeFn)

	if len(tools) == 0 {
		t.Fatal("a palace with drawers registered no tools")
	}
	// WakeUp over all wings should surface the drawer we just filed.
	if wakeUp == "" {
		t.Error("a palace with drawers produced no wake-up context")
	}
	if closeFn == nil {
		t.Fatal("closer is nil; the caller defers it unconditionally")
	}
}

// TestResolvePalaceConfigChecksEmbedderMapping: the config file's palace
// section must reach palace.PalaceConfig, because openEmbedder picks the
// backend from these fields. Before this was wired, setupPalace called
// palace.New with paths only, so palace.DefaultConfig's UseOllama=true and
// empty APIEmbedderURL meant a configured embeddings endpoint was ignored.
func TestResolvePalaceConfigChecksEmbedderMapping(t *testing.T) {
	isolate(t)

	t.Run("empty config keeps the default backend choice", func(t *testing.T) {
		got := resolvePalaceConfig(config.Config{})
		want := palace.DefaultConfig()
		if got.UseOllama != want.UseOllama || got.OllamaURL != want.OllamaURL ||
			got.OllamaModel != want.OllamaModel || got.APIEmbedderURL != "" {
			t.Errorf("resolvePalaceConfig(empty) = %+v, want the palace defaults", got)
		}
	})

	t.Run("embeddings url selects the api backend", func(t *testing.T) {
		cfg := config.Config{Palace: &config.PalaceConfig{
			EmbeddingsURL:    "http://llm.internal:8000/v1",
			EmbeddingsModel:  "bge-m3",
			EmbeddingsAPIKey: "sk-test",
			OllamaURL:        "http://elsewhere:11434",
			OllamaModel:      "nomic-embed-text",
		}}
		got := resolvePalaceConfig(cfg)
		if got.APIEmbedderURL != "http://llm.internal:8000/v1" {
			t.Errorf("APIEmbedderURL = %q, want the configured endpoint", got.APIEmbedderURL)
		}
		if got.APIEmbedderModel != "bge-m3" {
			t.Errorf("APIEmbedderModel = %q", got.APIEmbedderModel)
		}
		if got.APIEmbedderKey != "sk-test" {
			t.Errorf("APIEmbedderKey = %q", got.APIEmbedderKey)
		}
		if got.OllamaURL != "http://elsewhere:11434" || got.OllamaModel != "nomic-embed-text" {
			t.Errorf("ollama overrides lost: %+v", got)
		}
		// UseOllama stays true: openEmbedder checks the api URL first, so this
		// is the fallback, not the primary.
		if !got.UseOllama {
			t.Error("UseOllama was turned off; it must remain the fallback behind the api backend")
		}
	})

	t.Run("local_embedder forces the in-process model", func(t *testing.T) {
		cfg := config.Config{Palace: &config.PalaceConfig{LocalEmbedder: true}}
		if got := resolvePalaceConfig(cfg); got.UseOllama {
			t.Error("LocalEmbedder did not disable Ollama")
		}
	})
}

// embeddingsSpy is an OpenAI-compatible /v1/embeddings double that records the
// requests it served, so a test can prove the palace reached this endpoint
// rather than Ollama or the in-process model.
type embeddingsSpy struct {
	*httptest.Server

	mu     sync.Mutex
	bodies []map[string]any
}

func newEmbeddingsSpy(t *testing.T) *embeddingsSpy {
	t.Helper()
	spy := &embeddingsSpy{}
	spy.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		spy.mu.Lock()
		spy.bodies = append(spy.bodies, body)
		spy.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":[{"index":0,"embedding":[0.1,0.2,0.3]}]}`))
	}))
	t.Cleanup(spy.Close)
	return spy
}

func (s *embeddingsSpy) requestCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.bodies)
}

// toolCtx wraps StrictContextMock with the handful of methods
// functiontool.Run touches before the handler, so a palace tool can be invoked
// without the full ADK runtime. Everything else keeps the mock's loud panics.
type toolCtx struct {
	*adkagent.StrictContextMock
}

func (toolCtx) ToolConfirmation() *adktoolconfirmation.ToolConfirmation { return nil }
func (toolCtx) RequestConfirmation(string, any) error                   { return nil }
func (toolCtx) Actions() *adksession.EventActions                       { return &adksession.EventActions{} }

var _ adkagent.Context = toolCtx{&adkagent.StrictContextMock{Ctx: context.Background()}}

// TestSetupPalaceUsesConfiguredEmbedder is the end-to-end pin for the bug: with
// palace.embeddings_url pointing at a live endpoint, the palace setupPalace
// builds must embed through it. The search tool runs the query embedding — the
// read path that silently degrades when the backend differs from the one
// mining used — against whatever embedder setupPalace actually opened, so a
// request to the spy is proof the configured one was honored.
func TestSetupPalaceUsesConfiguredEmbedder(t *testing.T) {
	isolate(t)

	spy := newEmbeddingsSpy(t)
	dbPath := filepath.Join(t.TempDir(), "palace.db")
	newPalaceAt(t, dbPath)

	cfg := config.Config{Palace: &config.PalaceConfig{
		DBPath:           dbPath,
		EmbeddingsURL:    spy.URL,
		EmbeddingsModel:  "bge-m3",
		EmbeddingsAPIKey: "sk-test",
	}}
	palaceTools, _, closeFn := setupPalace(options{palaceEnabled: true}, cfg, nil)
	t.Cleanup(closeFn)

	var search adktool.Tool
	for _, tl := range palaceTools {
		if tl.Name() == "palace-search" {
			search = tl
			break
		}
	}
	if search == nil {
		t.Fatal("palace-search tool not registered; the drawer-count gate should have opened")
	}

	runner, ok := search.(interface {
		Run(ctx adkagent.Context, args any) (map[string]any, error)
	})
	if !ok {
		t.Fatalf("palace-search (%T) does not implement Run", search)
	}
	ctx := toolCtx{&adkagent.StrictContextMock{Ctx: context.Background()}}
	if _, err := runner.Run(ctx, map[string]any{"query": "palace gate"}); err != nil {
		t.Fatalf("palace-search Run: %v", err)
	}

	if spy.requestCount() == 0 {
		t.Fatal("the configured embeddings endpoint served no requests — setupPalace opened a different embedder backend")
	}
}
