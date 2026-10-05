package config

import (
	"log/slog"
	"os"
	"strings"
)

// ResolveSkillDirs returns the effective skill directory override. Empty means
// the default discovery applies (see extension.DefaultSkillDirsIn).
//
// Precedence: PI_SKILLS_DIRS, then config.json `skillsDirs:`. The value is a
// separated directory list like PATH: ";" on Windows, ":" otherwise — both
// separators are accepted regardless of the host OS, with drive-letter
// re-joining so a list authored for the other platform still parses (a "C"
// fragment followed by "\…" or "/…" is glued back into "C:\…" / "C:/…").
// Empty segments are dropped so a trailing separator or an unset entry cannot
// resolve to the working directory.
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

// splitDirs splits a separated directory list, dropping empty segments. Both
// ":" and ";" separate entries, so a value parses no matter which platform's
// convention its author used. Splitting on ":" also cuts a Windows drive path
// ("C:\a" → "C" + "\a"), so a one-letter fragment followed by an entry
// starting with "\" or "/" is glued back with the colon restored
// (issue #61). ponytail: the glue misfires on a Unix one-letter directory
// immediately followed by an absolute path ("a:/b") — pathological config,
// upgrade to a per-platform separator only if that ever shows up.
func splitDirs(v string) []string {
	parts := strings.FieldsFunc(v, func(r rune) bool { return r == ':' || r == ';' })
	dirs := make([]string, 0, len(parts))
	for i := 0; i < len(parts); i++ {
		p := strings.TrimSpace(parts[i])
		if p == "" {
			continue
		}
		if len(p) == 1 && isDriveLetter(p[0]) && i+1 < len(parts) {
			if next := strings.TrimSpace(parts[i+1]); strings.HasPrefix(next, `\`) || strings.HasPrefix(next, `/`) {
				p += ":" + next
				i++
			}
		}
		dirs = append(dirs, p)
	}
	return dirs
}

func isDriveLetter(c byte) bool {
	return c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z'
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
