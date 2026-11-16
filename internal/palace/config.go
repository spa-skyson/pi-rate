package palace

import (
	"path/filepath"

	"github.com/spa-skyson/pi-rate/internal/config"
)

const (
	defaultDeduplicationThreshold = float32(0.9)
	defaultDBPath                 = "palace.db"

	// DefaultModelDirName is the on-disk directory name of the in-process
	// embedding model: the subdir `memory model download` creates under its
	// destination (hugot names it after the sentence-transformers/all-MiniLM-L6-v2
	// repo id), and the leaf DefaultModelPath appends. Every literal that
	// names this directory must derive from this constant — split copies of
	// the name made `memory model download` write a cache the embedder never
	// looked at.
	DefaultModelDirName = "sentence-transformers_all-MiniLM-L6-v2"
)

// DefaultModelPath is the single source of truth for where the embedding
// model lives by default: the path `memory model download` writes to (under
// its default destination) and every embedder opens — cli's
// palaceConfigFromCLI and piagent's palacePaths both fall back to it. If
// these paths drift apart, the download lands where no embedder looks.
func DefaultModelPath() string {
	return filepath.Join(config.PirateHome(), "models", DefaultModelDirName)
}

// PalaceConfig holds configuration for the Palace system.
type PalaceConfig struct {
	DBPath                 string
	ModelPath              string
	IdentityFile           string
	DeduplicationThreshold float32
	L1TopK                 int
	L1MaxChars             int
	L2MaxDrawers           int
	L2MaxCharsPerDrawer    int

	// UseOllama selects the Ollama daemon for embedding instead of the
	// in-process model. Default on: it is an order of magnitude faster and
	// retrieves better. See DefaultOllamaEmbedModel for the measurements.
	UseOllama bool
	// OllamaURL is the daemon address; empty means DefaultOllamaURL.
	OllamaURL string
	// OllamaModel is the embedding model; empty means DefaultOllamaEmbedModel.
	OllamaModel string

	// APIEmbedderURL points at an external OpenAI-compatible embeddings
	// endpoint (root up to /embeddings, e.g. "http://llm.internal:8000/v1").
	// Non-empty selects the api backend ahead of Ollama and the in-process
	// model — see openEmbedder. Changing it (or switching away) swaps vector
	// spaces, so stored vectors must be re-mined, exactly like OllamaModel.
	APIEmbedderURL string
	// APIEmbedderModel is the model name sent to the endpoint; empty means
	// DefaultAPIEmbedModel. Changing it invalidates every stored vector.
	APIEmbedderModel string
	// APIEmbedderKey is the Bearer token for the endpoint; empty sends no
	// Authorization header. It is a secret: never log it.
	APIEmbedderKey string
}

// DefaultConfig returns a PalaceConfig with sensible defaults.
func DefaultConfig() PalaceConfig {
	return PalaceConfig{
		DBPath:                 defaultDBPath,
		DeduplicationThreshold: defaultDeduplicationThreshold,
		L1TopK:                 15,
		L1MaxChars:             3200,
		L2MaxDrawers:           10,
		L2MaxCharsPerDrawer:    300,
		UseOllama:              true,
		OllamaURL:              DefaultOllamaURL,
		OllamaModel:            DefaultOllamaEmbedModel,
	}
}

// Option is a functional option for configuring a Palace.
type Option func(*PalaceConfig)

// WithDBPath sets the database file path.
func WithDBPath(path string) Option {
	return func(c *PalaceConfig) { c.DBPath = path }
}

// WithModelPath sets the path to the embedding model directory.
func WithModelPath(path string) Option {
	return func(c *PalaceConfig) { c.ModelPath = path }
}

// WithIdentityFile sets the path to the L0 identity file.
func WithIdentityFile(path string) Option {
	return func(c *PalaceConfig) { c.IdentityFile = path }
}

// WithDeduplicationThreshold sets the cosine similarity threshold for duplicate detection.
func WithDeduplicationThreshold(t float32) Option {
	return func(c *PalaceConfig) { c.DeduplicationThreshold = t }
}

// WithOllamaEmbedder points the palace at an Ollama daemon for embedding.
// Empty arguments fall back to DefaultOllamaURL and DefaultOllamaEmbedModel.
func WithOllamaEmbedder(baseURL, model string) Option {
	return func(c *PalaceConfig) {
		c.UseOllama = true
		if baseURL != "" {
			c.OllamaURL = baseURL
		}
		if model != "" {
			c.OllamaModel = model
		}
	}
}

// WithAPIEmbedder points the palace at an external OpenAI-compatible
// embeddings endpoint. An empty url selects nothing; empty model and key fall
// back to DefaultAPIEmbedModel and no Authorization header.
func WithAPIEmbedder(url, model, key string) Option {
	return func(c *PalaceConfig) {
		if url != "" {
			c.APIEmbedderURL = url
		}
		if model != "" {
			c.APIEmbedderModel = model
		}
		c.APIEmbedderKey = key
	}
}

// WithLocalEmbedder forces the in-process model, bypassing Ollama. Used when no
// daemon is available and the caller would rather be slow than fail.
func WithLocalEmbedder() Option {
	return func(c *PalaceConfig) { c.UseOllama = false }
}

// WithConfig adopts a fully resolved config — what the CLI's
// palaceConfigFromCLI builds: defaults overlaid with the user's palace
// section, embedder decision (api / ollama / local) included. Every command
// that opens a palace goes through it so the backend configured once is the
// backend everywhere; a site that needs different paths can pass
// WithDBPath/WithModelPath after it.
func WithConfig(resolved PalaceConfig) Option {
	return func(c *PalaceConfig) { *c = resolved }
}
