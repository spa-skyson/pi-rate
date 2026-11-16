package cli

import (
	"github.com/spf13/cobra"

	"github.com/spa-skyson/pi-rate/internal/config"
	"github.com/spa-skyson/pi-rate/internal/palace"
)

// defaultPalaceModelPath returns the default embedding model path. The
// palace package owns the value (palace.DefaultModelPath) so the download
// target and every embedder's open path cannot drift apart — split literals
// here and in palaceConfigFromCLI meant the download wrote a cache the
// embedder never looked at.
func defaultPalaceModelPath() string {
	return palace.DefaultModelPath()
}

// resolvePalaceConfig resolves the user's palace section onto the defaults,
// with dbPath and modelPath — when non-empty — overriding the config values,
// since each command names its own store. A broken config file is not fatal
// here: the defaults are what the command ran with before the config existed.
// Together with palaceConfigFromCLI (which does the resolution) this is the
// entrance every memory command takes, so the configured embedder — api,
// ollama or local — is the same one mining opened.
func resolvePalaceConfig(dbPath, modelPath string) palace.PalaceConfig {
	userCfg, err := config.Load()
	if err != nil {
		userCfg = config.Config{}
	}
	palaceCfg := palaceConfigFromCLI(&userCfg)
	if dbPath != "" {
		palaceCfg.DBPath = dbPath
	}
	if modelPath != "" {
		palaceCfg.ModelPath = modelPath
	}
	return palaceCfg
}

func newMemoryCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "memory",
		Short: "Manage the MemPalace memory system",
		Long: `Commands for managing the MemPalace memory system: download embedding models,
initialize palace databases, and view palace status.`,
	}

	cmd.AddCommand(newMemoryModelCmd())
	cmd.AddCommand(newMemoryInitCmd())
	cmd.AddCommand(newMemoryStatusCmd())
	cmd.AddCommand(newMemoryMineCmd())
	cmd.AddCommand(newMemorySearchCmd())
	cmd.AddCommand(newMemoryKGCmd())
	cmd.AddCommand(newMemoryWakeUpCmd())
	cmd.AddCommand(newMemoryRecentCmd())
	cmd.AddCommand(newMemoryClearCmd())

	return cmd
}
