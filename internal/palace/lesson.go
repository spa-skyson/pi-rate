// Structured lesson format for palace drawers.
//
// Issue 81: palace entries are free text, so a "project fact" and a
// "transferable lesson" are indistinguishable, and nothing enforces
// evidence. A Lesson is a markdown section embedded in a drawer's Content
// with fixed field headers, so FTS5 keyword search and the embedding
// pipeline index it like any other drawer content while ParseLesson
// recovers the fields verbatim.
package palace

import (
	"context"
	"fmt"
	"strings"
	"time"
)

// Lesson is the structured harness2-style lesson carried inside a palace
// drawer's content. All four fields are required on write — a lesson without
// the contrast that produced it poisons the library for every later task
// (issue 81).
type Lesson struct {
	// Cue states when the lesson applies.
	Cue string
	// Lesson states what to do.
	Lesson string
	// Evidence states what the lesson is based on.
	Evidence string
	// Confidence annotates how well-established the lesson is.
	Confidence string
}

// LessonSectionHeader is the fixed heading that marks a lesson section inside
// a drawer's content.
const LessonSectionHeader = "## Lesson"

// lessonField is one lesson field: its serialized label, an accessor, and a
// setter. The slice below fixes both the on-disk field order and the order
// fields are named in validation errors.
type lessonField struct {
	label string
	get   func(Lesson) string
	set   func(*Lesson, string)
}

var lessonFields = []lessonField{
	{"cue", func(l Lesson) string { return l.Cue }, func(l *Lesson, v string) { l.Cue = v }},
	{"lesson", func(l Lesson) string { return l.Lesson }, func(l *Lesson, v string) { l.Lesson = v }},
	{"evidence", func(l Lesson) string { return l.Evidence }, func(l *Lesson, v string) { l.Evidence = v }},
	{"confidence", func(l Lesson) string { return l.Confidence }, func(l *Lesson, v string) { l.Confidence = v }},
}

// FormatLesson renders the lesson as a fixed-layout markdown section:
//
//	## Lesson
//	- cue: …
//	- lesson: …
//	- evidence: …
//	- confidence: …
//
// Values are trimmed; multi-line values keep their inner line breaks indented
// by two spaces, so a "- <label>:" marker at column zero is always a field
// start and ParseLesson undoes the indentation exactly.
func FormatLesson(l Lesson) string {
	var sb strings.Builder
	sb.WriteString(LessonSectionHeader + "\n")
	for _, f := range lessonFields {
		sb.WriteString("- " + f.label + ": " + indentContinuation(strings.TrimSpace(f.get(l))) + "\n")
	}
	return sb.String()
}

// indentContinuation indents every line after the first so only the first
// line of a field value starts at column zero.
func indentContinuation(s string) string {
	return strings.ReplaceAll(s, "\n", "\n  ")
}

// ParseLesson extracts the lesson section from drawer content. ok is false
// when the content carries no lesson section or the section has no fields.
// Trailing free text after the section is ignored.
func ParseLesson(text string) (Lesson, bool) {
	section, ok := cutLessonSection(text)
	if !ok {
		return Lesson{}, false
	}

	var l Lesson
	found := false
	for _, f := range lessonFields {
		val, ok := lessonFieldValue(section, f.label)
		if !ok {
			continue
		}
		f.set(&l, strings.TrimSpace(val))
		found = true
	}
	if !found {
		return Lesson{}, false
	}
	return l, true
}

// cutLessonSection returns the body of the first "## Lesson" section in text.
// Headings are recognized at column zero only: FormatLesson indents value
// continuation lines, so an indented "## …" line belongs to a value, not to
// the section layout.
func cutLessonSection(text string) (string, bool) {
	lines := strings.Split(text, "\n")
	start := -1
	for i, line := range lines {
		if line == LessonSectionHeader {
			start = i
			break
		}
	}
	if start < 0 {
		return "", false
	}
	end := len(lines)
	for i := start + 1; i < len(lines); i++ {
		if strings.HasPrefix(lines[i], "## ") {
			end = i
			break
		}
	}
	return strings.Join(lines[start+1:end], "\n"), true
}

// lessonFieldValue extracts one "- <label>:" field from a section body,
// including its indented continuation lines.
func lessonFieldValue(section, label string) (string, bool) {
	prefix := "- " + label + ":"
	lines := strings.Split(section, "\n")
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if !strings.HasPrefix(trimmed, prefix) {
			continue
		}
		value := strings.TrimSpace(strings.TrimPrefix(trimmed, prefix))
		for j := i + 1; j < len(lines); j++ {
			next := lines[j]
			// Continuation lines are indented by FormatLesson; anything at
			// column zero (the next field, the next section) ends the value.
			if !strings.HasPrefix(next, "  ") {
				break
			}
			value += "\n" + strings.TrimPrefix(next, "  ")
		}
		return value, true
	}
	return "", false
}

// ValidateLesson reports every missing lesson field in one error, so the
// writer fixes the whole note at once instead of discovering the gaps one
// rejection at a time (issue 81). Whitespace-only counts as absent.
func ValidateLesson(l Lesson) error {
	var missing []string
	for _, f := range lessonFields {
		if strings.TrimSpace(f.get(l)) == "" {
			missing = append(missing, f.label)
		}
	}
	if len(missing) == 0 {
		return nil
	}
	return fmt.Errorf("palace: lesson missing required fields: %s", strings.Join(missing, ", "))
}

// AddLesson is the gated write path for lessons: it validates the lesson,
// embeds it into the drawer content, and stores the drawer through AddDrawer.
// The gate sits here, in front of the single store path, so every consumer —
// the palace-add-lesson tool and any CLI code writing lessons — shares one
// validation. A lesson with a missing field is rejected before embedding or
// dedup work is spent on it. input.Wing/Room/Hall/SourceFile/Importance are
// honored; input.Content is overwritten with the formatted lesson.
func (p *Palace) AddLesson(ctx context.Context, input DrawerInput, l Lesson) (*Drawer, error) {
	if err := ValidateLesson(l); err != nil {
		return nil, err
	}
	input.Content = FormatLesson(l)
	return p.AddDrawer(ctx, input)
}

// LessonsBefore implements the prefix-only rule: a session or task sees only
// the lessons accumulated strictly before its start, so nothing leaks in from
// parallel or later runs (issue 81). cutoff is the session/task start time; a
// lesson's creation time is its drawer's CreatedAt, so a lesson written
// during the session is invisible to its own context injection.
func (p *Palace) LessonsBefore(ctx context.Context, cutoff time.Time) ([]Lesson, error) {
	drawers, err := p.store.ListDrawers(ctx, DrawerFilter{})
	if err != nil {
		return nil, fmt.Errorf("palace: lessons before cutoff: %w", err)
	}
	var lessons []Lesson
	for _, d := range drawers {
		if !d.CreatedAt.Before(cutoff) {
			continue
		}
		if l, ok := ParseLesson(d.Content); ok {
			lessons = append(lessons, l)
		}
	}
	return lessons, nil
}

// TODO(issue 81): prefix-only is enforced at this API level; the consumer
// mixing lessons into a live session's context must call LessonsBefore with
// the session start time. The wiring points — internal/cli.setupPalace
// (WakeUp context injection) and the palace-search tool path — sit outside
// internal/palace and are deliberately untouched by this change.
