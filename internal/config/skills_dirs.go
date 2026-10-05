package config

import (
	"log/slog"
	"os"
	"strings"
)

// ResolveSkillDirs returns the effective skill directory override. Empty means
// the default discovery applies (see extension.DefaultSkillDirsIn).
//
// Precedence: PI_SKILLS_DIRS, then config.json `skillsDirs:`. The env var is
// colon-separated like PATH; empty segments are dropped so a trailing colon or
// an unset entry cannot resolve to the working directory.
func (c Config) ResolveSkillDirs() []string {
	if v := strings.TrimSpace(os.Getenv(EnvSkillsDirs)); v != "" {
		dirs := splitDirs(v)
		if len(dirs) == 0 {
			slog.Warn("config: unusable PI_SKILLS_DIRS ignored; using default discovery",
				"value", v)
			return nil
		}
		return dirs
	}
	return splitDirsList(c.SkillsDirs)
}

// ResolveDisableLegacySkillDirs reports whether the .claude/skills and
// .cursor/skills directories are excluded from skill discovery.
//
// Precedence: PI_DISABLE_LEGACY_SKILLS ("1"/"true", case-insensitive — the
// same truthy tokens as PI_NO_GROUNDING), then config.json
// `disableLegacySkillDirs:`.
func (c Config) ResolveDisableLegacySkillDirs() bool {
	if v := strings.TrimSpace(os.Getenv(EnvDisableLegacySkillDirs)); v != "" {
		switch strings.ToLower(v) {
		case "1", "true", "yes", "on":
			return true
		}
		return false
	}
	return c.DisableLegacySkillDirs
}

// splitDirs splits a colon-separated directory list, dropping empty segments.
func splitDirs(v string) []string {
	return splitDirsList(strings.Split(v, ":"))
}

func splitDirsList(entries []string) []string {
	dirs := make([]string, 0, len(entries))
	for _, e := range entries {
		if e = strings.TrimSpace(e); e != "" {
			dirs = append(dirs, e)
		}
	}
	return dirs
}
