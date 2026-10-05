package cli

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/spa-skyson/pi-rate/internal/config"
	"github.com/spa-skyson/pi-rate/internal/palace"
	"github.com/spa-skyson/pi-rate/internal/testenv"
)

// newEmbeddingEndpoint is a minimal OpenAI-compatible /embeddings double: it
// counts requests and answers one 4-dim vector per call, which is all a
// single-query embed needs. The count is what the wiring assertions read — a
// palace built from defaults never sends a request at all.
func newEmbeddingEndpoint(t *testing.T) (url string, hits func() int) {
	t.Helper()
	var mu sync.Mutex
	n := 0
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		mu.Lock()
		n++
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":[{"index":0,"embedding":[0.1,0.2,0.3,0.4]}]}`))
	}))
	t.Cleanup(ts.Close)
	return ts.URL, func() int {
		mu.Lock()
		defer mu.Unlock()
		return n
	}
}

// apiEmbedderProject points the process at a temp project whose .pirate/config.json
// selects the given embeddings endpoint: PIRATE_HOME moves to an empty temp dir
// and cwd to the project, so config.Load resolves exactly this section.
func apiEmbedderProject(t *testing.T, embeddingsURL string) {
	t.Helper()
	t.Setenv("PIRATE_HOME", t.TempDir())
	project := t.TempDir()
	t.Chdir(project)
	pirateDir := filepath.Join(project, config.ProjectDirName)
	if err := os.MkdirAll(pirateDir, 0o755); err != nil {
		t.Fatal(err)
	}
	cfg := map[string]any{
		"palace": map[string]any{
			"embeddings_url":   embeddingsURL,
			"embeddings_model": "bge-m3",
		},
	}
	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(pirateDir, "config.json"), data, 0o600); err != nil {
		t.Fatal(err)
	}
}

// TestPalaceConfigFromCLI_ExpandsEmbedderEnv checks that ${VAR} in all three
// embeddings_* settings resolves at the single source of truth, so every
// palace built from it — mine, search, status, the session — carries the
// expanded values while config.json keeps the placeholder.
func TestPalaceConfigFromCLI_ExpandsEmbedderEnv(t *testing.T) {
	home := t.TempDir()
	t.Setenv("PIRATE_HOME", home)
	project := t.TempDir()
	t.Chdir(project)
	pirateDir := filepath.Join(project, config.ProjectDirName)
	if err := os.MkdirAll(pirateDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(pirateDir, ".env"), []byte(
		"PALACE_URL=http://llm.internal:8000/v1\nPALACE_MODEL=bge-m3\nPALACE_KEY=sk-from-env\n",
	), 0o600); err != nil {
		t.Fatal(err)
	}

	cfg := config.Config{Palace: &config.PalaceConfig{
		EmbeddingsURL:    "${PALACE_URL}",
		EmbeddingsModel:  "${PALACE_MODEL}",
		EmbeddingsAPIKey: "${PALACE_KEY}",
	}}
	got := palaceConfigFromCLI(&cfg)
	if got.APIEmbedderURL != "http://llm.internal:8000/v1" {
		t.Errorf("url = %q, want expanded", got.APIEmbedderURL)
	}
	if got.APIEmbedderModel != "bge-m3" {
		t.Errorf("model = %q, want expanded", got.APIEmbedderModel)
	}
	if got.APIEmbedderKey != "sk-from-env" {
		t.Errorf("key = %q, want expanded", got.APIEmbedderKey)
	}

	// The stored config is untouched: the placeholder survives for Save.
	if cfg.Palace.EmbeddingsAPIKey != "${PALACE_KEY}" {
		t.Errorf("config key mutated to %q", cfg.Palace.EmbeddingsAPIKey)
	}
}

// TestPalaceWiring_APIEmbedderReachesEverySite is the wiring guard: a config
// with embeddings_url must reach every place that builds a Palace, not just
// mining. Revert any subtest's call site to plain defaults and that subtest
// goes red — either Status stops naming the api backend or the endpoint stops
// seeing requests.
func TestPalaceWiring_APIEmbedderReachesEverySite(t *testing.T) {
	endpoint, hits := newEmbeddingEndpoint(t)
	apiEmbedderProject(t, endpoint)

	t.Run("openPalaceDB drives search, status, wake-up and kg", func(t *testing.T) {
		p, err := openPalaceDB("")
		if err != nil {
			t.Fatalf("openPalaceDB: %v", err)
		}
		defer p.Close()
		status, err := p.Status(t.Context())
		if err != nil {
			t.Fatalf("Status: %v", err)
		}
		if status.Embedder != "api/bge-m3" {
			t.Errorf("status embedder = %q, want api/bge-m3 (the configured backend)", status.Embedder)
		}
	})

	t.Run("runMemorySearch embeds the query on the endpoint", func(t *testing.T) {
		captureStdout(t, func() {
			if err := runMemorySearch("anything", "", "", "", 5); err != nil {
				t.Errorf("runMemorySearch: %v", err)
			}
		})
		if hits() == 0 {
			t.Error("embeddings endpoint saw no requests — search built a palace from defaults")
		}
	})

	t.Run("mining opens the api palace", func(t *testing.T) {
		palaceCfg := resolvePalaceConfig(filepath.Join(t.TempDir(), "palace.db"), filepath.Join(t.TempDir(), "model"))
		p, err := palace.New(minePalaceOptions(palaceCfg)...)
		if err != nil {
			t.Fatalf("palace.New: %v", err)
		}
		defer p.Close()
		status, err := p.Status(t.Context())
		if err != nil {
			t.Fatalf("Status: %v", err)
		}
		if status.Embedder != "api/bge-m3" {
			t.Errorf("mining embedder = %q, want api/bge-m3", status.Embedder)
		}
	})

	t.Run("runMemoryInit opens the api palace without ollama warnings", func(t *testing.T) {
		// init runs before any model is mined, so there is nothing to embed
		// yet; the observable is the embedder the palace was built with. A
		// regressed runMemoryInit builds from defaults (UseOllama=true), whose
		// probe fails against the test's endpoint and logs the fallback; the
		// configured api backend constructs without any network at all.
		warnings := captureEmbedderWarnings(t)
		project := t.TempDir()
		captureStdout(t, func() {
			if err := runMemoryInit(project, ""); err != nil {
				t.Fatalf("runMemoryInit: %v", err)
			}
		})
		if msgs := warnings(); len(msgs) != 0 {
			t.Errorf("embedder fallbacks logged: %q — init ignored the configured embeddings_url", msgs)
		}
	})
}

// embedderWarnRecorder captures messages logged through the default slog
// handler. openEmbedder logs its fallbacks synchronously from the calling
// goroutine, so no locking is needed under -race.
type embedderWarnRecorder struct {
	msgs []string
}

func (w *embedderWarnRecorder) Enabled(context.Context, slog.Level) bool { return true }

func (w *embedderWarnRecorder) Handle(_ context.Context, r slog.Record) error {
	if strings.Contains(r.Message, "embedder") {
		w.msgs = append(w.msgs, r.Message)
	}
	return nil
}

func (w *embedderWarnRecorder) WithAttrs([]slog.Attr) slog.Handler { return w }

func (w *embedderWarnRecorder) WithGroup(string) slog.Handler { return w }

// captureEmbedderWarnings installs the recorder for the test and returns a
// reader of the embedder fallback messages logged so far.
func captureEmbedderWarnings(t *testing.T) func() []string {
	t.Helper()
	w := &embedderWarnRecorder{}
	prev := slog.Default()
	slog.SetDefault(slog.New(w))
	t.Cleanup(func() { slog.SetDefault(prev) })
	return func() []string { return w.msgs }
}

// TestEnsureMineModel_SkipsDownloadForNetworkBackends pins the gate: with an
// api endpoint or Ollama selected, the local MiniLM must not be fetched — the
// vectors come from the network backend, and EmbedderAvailability has already
// approved it by the time this runs. A regressed gate either errors on the
// real download or lands the weights, both visible below.
func TestEnsureMineModel_SkipsDownloadForNetworkBackends(t *testing.T) {
	testenv.SetHome(t, t.TempDir())
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "palace.db")
	modelPath := filepath.Join(dir, "models", "mini") // absent on purpose

	for name, palaceCfg := range map[string]palace.PalaceConfig{
		"api":    {APIEmbedderURL: "http://llm.internal:8000/v1"},
		"ollama": {UseOllama: true, OllamaURL: "http://d:1", OllamaModel: "embed-m"},
	} {
		t.Run(name, func(t *testing.T) {
			if err := ensureMineModel(dbPath, modelPath, palaceCfg); err != nil {
				t.Fatalf("ensureMineModel: %v", err)
			}
			if palace.ModelReady(modelPath) {
				t.Error("local model was downloaded although a network backend serves this run")
			}
		})
	}
}
