package palace

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"
)

// fullLesson is a lesson with every field set, reused across the tests.
func fullLesson() Lesson {
	return Lesson{
		Cue:        "before changing the merge layout",
		Lesson:     "run go test ./internal/... before pushing",
		Evidence:   "v0.5.7 tag push failed the Lint job and had to be re-pushed",
		Confidence: "high — reproduced twice",
	}
}

func TestFormatLesson_FixedLayout(t *testing.T) {
	got := FormatLesson(fullLesson())
	want := "## Lesson\n" +
		"- cue: before changing the merge layout\n" +
		"- lesson: run go test ./internal/... before pushing\n" +
		"- evidence: v0.5.7 tag push failed the Lint job and had to be re-pushed\n" +
		"- confidence: high — reproduced twice\n"
	if got != want {
		t.Fatalf("FormatLesson() =\n%q\nwant\n%q", got, want)
	}
}

func TestLessonFormatParseRoundtrip(t *testing.T) {
	tests := []struct {
		name   string
		lesson Lesson
	}{
		{
			name:   "ascii single-line fields",
			lesson: fullLesson(),
		},
		{
			name: "unicode and cyrillic",
			lesson: Lesson{
				Cue:        "перед сменой схемы миграций",
				Lesson:     "прогоняй миграции на копии прод-данных",
				Evidence:   "инцидент 2026-09-14: миграция уронила индекс — «Планировщик» потерял расписание",
				Confidence: "высокая — подтверждено на стенде",
			},
		},
		{
			name: "multi-line evidence",
			lesson: Lesson{
				Cue: "when a fixture test fails only on Windows",
				Lesson: "check the fixture path separator before debugging the code:\n" +
					"1. reproduce with the exact OS path\n" +
					"2. compare against the tracked fixture",
				Evidence:   "the ripgrep-named fixture test passed on macOS and failed\non Windows CI, where the tool is not on PATH",
				Confidence: "medium",
			},
		},
		{
			name: "value containing a heading-like line",
			lesson: Lesson{
				Cue:        "before hand-editing markdown fixtures",
				Lesson:     "remember an indented heading is value text:\n  ## Lesson\nis data, not a new section",
				Evidence:   "ParseLesson must not split a lesson at an indented heading inside a value",
				Confidence: "high",
			},
		},
		{
			// Format trims field values, so the roundtrip promise is
			// "parsed == formatted", not byte-identity with the input.
			name: "whitespace is trimmed on format",
			lesson: Lesson{
				Cue:        strings.TrimSpace("  padded cue  "),
				Lesson:     strings.TrimSpace("\tleading and trailing whitespace\n"),
				Evidence:   strings.TrimSpace(" evidence with padding "),
				Confidence: strings.TrimSpace("  high  "),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			formatted := FormatLesson(tt.lesson)
			got, ok := ParseLesson(formatted)
			if !ok {
				t.Fatalf("ParseLesson(FormatLesson(...)) = ok=false for\n%s", formatted)
			}
			if got != tt.lesson {
				t.Fatalf("roundtrip mismatch:\ngot  %+q\nwant %+q", got, tt.lesson)
			}
		})
	}
}

func TestParseLesson(t *testing.T) {
	tests := []struct {
		name   string
		text   string
		want   Lesson
		wantOK bool
	}{
		{
			name:   "no lesson section",
			text:   "just a project fact about the auth module",
			wantOK: false,
		},
		{
			name:   "empty section header only",
			text:   "## Lesson\n",
			wantOK: false,
		},
		{
			name:   "section with no fields",
			text:   "## Lesson\nfree prose without field markers\n",
			wantOK: false,
		},
		{
			name: "section inside larger drawer content",
			text: strings.Join([]string{
				"# project notes",
				"",
				"the scheduler uses a ram index today.",
				"",
				"## Lesson",
				"- cue: after changing the drawer schema",
				"- lesson: run the migrations before embedding",
				"- evidence: stale embeddings dropped every search hit once",
				"- confidence: high",
				"",
				"trailing free text stays outside the section.",
			}, "\n"),
			want: Lesson{
				Cue:        "after changing the drawer schema",
				Lesson:     "run the migrations before embedding",
				Evidence:   "stale embeddings dropped every search hit once",
				Confidence: "high",
			},
			wantOK: true,
		},
		{
			name: "partial fields parse, ok stays true",
			text: strings.Join([]string{
				"## Lesson",
				"- cue: any change to palace",
				"- lesson: keep the store interface small",
			}, "\n"),
			want: Lesson{
				Cue:    "any change to palace",
				Lesson: "keep the store interface small",
			},
			wantOK: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := ParseLesson(tt.text)
			if ok != tt.wantOK {
				t.Fatalf("ParseLesson() ok = %v, want %v", ok, tt.wantOK)
			}
			if got != tt.want {
				t.Fatalf("ParseLesson() = %+q, want %+q", got, tt.want)
			}
		})
	}
}

func TestValidateLesson_MissingFieldsInOneError(t *testing.T) {
	tests := []struct {
		name        string
		lesson      Lesson
		wantMissing []string
	}{
		{
			name:        "empty lesson lists all four fields",
			lesson:      Lesson{},
			wantMissing: []string{"cue", "lesson", "evidence", "confidence"},
		},
		{
			name: "lesson with only cue lists the other three",
			lesson: Lesson{
				Cue: "when touching the embedder",
			},
			wantMissing: []string{"lesson", "evidence", "confidence"},
		},
		{
			name: "whitespace-only field counts as missing",
			lesson: Lesson{
				Cue:        "cue",
				Lesson:     "lesson",
				Evidence:   "   ",
				Confidence: "high",
			},
			wantMissing: []string{"evidence"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateLesson(tt.lesson)
			if err == nil {
				t.Fatal("ValidateLesson() = nil, want an error")
			}
			for _, field := range tt.wantMissing {
				if !strings.Contains(err.Error(), field) {
					t.Errorf("error %q does not name missing field %q", err, field)
				}
			}
			// One message, not one error per field: a single line listing all
			// the gaps together.
			if n := strings.Count(err.Error(), "\n"); n != 0 {
				t.Errorf("error spans %d lines, want a single-line message: %q", n+1, err)
			}
		})
	}
}

func TestValidateLesson_FieldTooLong(t *testing.T) {
	tests := []struct {
		name     string
		field    string
		maxChars int
		lesson   Lesson
		wantErr  bool
	}{
		{
			name:     "cue over 120 chars",
			field:    "cue",
			maxChars: LessonMaxCueChars,
			lesson: Lesson{
				Cue:        strings.Repeat("а", LessonMaxCueChars+1),
				Lesson:     "l",
				Evidence:   "e",
				Confidence: "high",
			},
			wantErr: true,
		},
		{
			name:     "lesson over 300 chars",
			field:    "lesson",
			maxChars: LessonMaxLessonChars,
			lesson: Lesson{
				Cue:        "c",
				Lesson:     strings.Repeat("x", LessonMaxLessonChars+1),
				Evidence:   "e",
				Confidence: "high",
			},
			wantErr: true,
		},
		{
			name:     "evidence over 300 chars",
			field:    "evidence",
			maxChars: LessonMaxEvidenceChars,
			lesson: Lesson{
				Cue:        "c",
				Lesson:     "l",
				Evidence:   strings.Repeat("x", LessonMaxEvidenceChars+1),
				Confidence: "high",
			},
			wantErr: true,
		},
		{
			name:     "confidence over 60 chars",
			field:    "confidence",
			maxChars: LessonMaxConfidenceChars,
			lesson: Lesson{
				Cue:        "c",
				Lesson:     "l",
				Evidence:   "e",
				Confidence: strings.Repeat("x", LessonMaxConfidenceChars+1),
			},
			wantErr: true,
		},
		{
			name:     "multi-byte runes count as one char, at-cap passes",
			field:    "confidence",
			maxChars: LessonMaxConfidenceChars,
			lesson: Lesson{
				Cue:        "c",
				Lesson:     "l",
				Evidence:   "e",
				Confidence: strings.Repeat("ы", LessonMaxConfidenceChars), // exactly at the cap
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateLesson(tt.lesson)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("ValidateLesson() = nil, want an error naming %q", tt.field)
				}
				if !strings.Contains(err.Error(), tt.field+" is") {
					t.Errorf("error %q does not name field %q", err, tt.field)
				}
				if !strings.Contains(err.Error(), fmt.Sprintf("limit %d", tt.maxChars)) {
					t.Errorf("error %q does not state the limit %d", err, tt.maxChars)
				}
				// Only the offending field is reported.
				for _, other := range []string{"cue", "lesson", "evidence", "confidence"} {
					if other == tt.field {
						continue
					}
					if strings.Contains(err.Error(), other+" is") {
						t.Errorf("error %q wrongly flags field %q", err, other)
					}
				}
				return
			}
			if err != nil {
				t.Fatalf("ValidateLesson() = %v, want nil", err)
			}
		})
	}
}

func TestValidateLesson_AtTheCapIsAccepted(t *testing.T) {
	atCap := Lesson{
		Cue:        strings.Repeat("c", LessonMaxCueChars),
		Lesson:     strings.Repeat("l", LessonMaxLessonChars),
		Evidence:   strings.Repeat("e", LessonMaxEvidenceChars),
		Confidence: strings.Repeat("h", LessonMaxConfidenceChars),
	}
	if err := ValidateLesson(atCap); err != nil {
		t.Fatalf("ValidateLesson(fields at cap) = %v, want nil", err)
	}
}

func TestValidateLesson_Complete(t *testing.T) {
	if err := ValidateLesson(fullLesson()); err != nil {
		t.Fatalf("ValidateLesson(full lesson) = %v, want nil", err)
	}
}

// newLessonPalace returns a Palace over an in-memory store with no embedder,
// which is enough for every lesson path: the gate and the filter never embed.
func newLessonPalace(t *testing.T) *Palace {
	t.Helper()
	store := newTestStore(t)
	return NewWithStore(store, nil)
}

// drawerCount returns the number of drawers in p's store, failing the test on
// a read error. CountDrawers lives on the store, not the facade, and the
// facade deliberately grew no counting method for these tests.
func drawerCount(t *testing.T, p *Palace) int {
	t.Helper()
	n, err := p.store.CountDrawers(context.Background())
	if err != nil {
		t.Fatalf("CountDrawers: %v", err)
	}
	return n
}

func TestAddLesson_GateRejectsIncomplete(t *testing.T) {
	tests := []struct {
		name        string
		lesson      Lesson
		wantMissing []string
	}{
		{
			name:        "empty lesson",
			lesson:      Lesson{},
			wantMissing: []string{"cue", "lesson", "evidence", "confidence"},
		},
		{
			name: "missing evidence only",
			lesson: Lesson{
				Cue:        "cue",
				Lesson:     "lesson",
				Confidence: "high",
			},
			wantMissing: []string{"evidence"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := newLessonPalace(t)
			drawer, err := p.AddLesson(context.Background(), DrawerInput{Wing: "proj", Room: "lessons"}, tt.lesson)
			if err == nil {
				t.Fatal("AddLesson() = nil error, want a gate rejection")
			}
			if drawer != nil {
				t.Fatalf("AddLesson() returned a drawer %+v despite the rejection", drawer)
			}
			for _, field := range tt.wantMissing {
				if !strings.Contains(err.Error(), field) {
					t.Errorf("error %q does not name missing field %q", err, field)
				}
			}
			// The gate fires before the store path: nothing was written.
			if n := drawerCount(t, p); n != 0 {
				t.Fatalf("palace holds %d drawers after a rejected write, want 0", n)
			}
		})
	}
}

func TestAddLesson_CompleteStoresFormattedDrawer(t *testing.T) {
	p := newLessonPalace(t)
	ctx := context.Background()

	_, err := p.AddLesson(ctx, DrawerInput{Wing: "proj", Room: "lessons", Importance: 7}, Lesson{
		Cue:        strings.Repeat("c", LessonMaxCueChars+5),
		Lesson:     fullLesson().Lesson,
		Evidence:   fullLesson().Evidence,
		Confidence: fullLesson().Confidence,
	})
	if err == nil {
		t.Fatal("AddLesson(over-long cue) = nil error, want a gate rejection")
	}
	if !strings.Contains(err.Error(), "cue") {
		t.Errorf("error %q does not name the over-long field %q", err, "cue")
	}
	if n := drawerCount(t, p); n != 0 {
		t.Fatalf("palace holds %d drawers after a rejected over-long write, want 0", n)
	}

	drawer, err := p.AddLesson(ctx, DrawerInput{Wing: "proj", Room: "lessons", Importance: 7}, fullLesson())
	if err != nil {
		t.Fatalf("AddLesson: %v", err)
	}

	got, err := p.GetDrawer(ctx, drawer.ID)
	if err != nil {
		t.Fatalf("GetDrawer: %v", err)
	}
	if got.Wing != "proj" || got.Room != "lessons" {
		t.Errorf("drawer wing/room = %q/%q, want proj/lessons", got.Wing, got.Room)
	}
	if got.Importance != 7 {
		t.Errorf("drawer importance = %d, want 7", got.Importance)
	}
	if !strings.Contains(got.Content, LessonSectionHeader) {
		t.Errorf("drawer content %q has no lesson section", got.Content)
	}

	// Issue 81, criterion (b): a complete lesson is findable by cue and by
	// lesson text through the plain palace search.
	for _, query := range []string{fullLesson().Cue, fullLesson().Lesson} {
		results, err := p.Search(ctx, SearchQuery{Query: query, Limit: 5})
		if err != nil {
			t.Fatalf("Search(%q): %v", query, err)
		}
		if len(results) == 0 {
			t.Fatalf("Search(%q) = 0 results, want the lesson drawer", query)
		}
		if results[0].Drawer.ID != drawer.ID {
			t.Errorf("Search(%q)[0].ID = %q, want %q", query, results[0].Drawer.ID, drawer.ID)
		}
	}
}

func TestLessonsBefore_PrefixOnly(t *testing.T) {
	p := newLessonPalace(t)
	ctx := context.Background()

	before := fullLesson()
	after := Lesson{
		Cue:        "created during the session",
		Lesson:     "must not leak into the session's own context",
		Evidence:   "issue 81: no leakage from parallel or later runs",
		Confidence: "high",
	}

	cutoff := time.Date(2026, 10, 23, 12, 0, 0, 0, time.UTC)

	// Insert lesson A (before the cutoff) and lesson B (after) directly, with
	// explicit CreatedAt, mirroring drawers written in different sessions.
	for _, tc := range []struct {
		lesson  Lesson
		created time.Time
	}{
		{before, cutoff.Add(-time.Hour)},
		{after, cutoff.Add(time.Hour)},
	} {
		input := DrawerInput{Wing: "proj", Room: "lessons"}
		if _, err := p.AddLesson(ctx, input, tc.lesson); err != nil {
			t.Fatalf("AddLesson: %v", err)
		}
	}

	// Rewind lesson A's creation time and push lesson B past the cutoff; the
	// store path stamps time.Now() itself, so the test pins the timestamps it
	// needs through a targeted update.
	drawers, err := p.ListDrawers(ctx, DrawerFilter{})
	if err != nil {
		t.Fatalf("ListDrawers: %v", err)
	}
	if len(drawers) != 2 {
		t.Fatalf("ListDrawers() = %d drawers, want 2", len(drawers))
	}
	for _, d := range drawers {
		l, _ := ParseLesson(d.Content)
		want := cutoff.Add(-time.Hour)
		if l == after {
			want = cutoff.Add(time.Hour)
		}
		d.CreatedAt = want
		if err := p.store.InsertDrawer(ctx, d); err != nil {
			t.Fatalf("InsertDrawer(pin CreatedAt): %v", err)
		}
	}

	lessons, err := p.LessonsBefore(ctx, cutoff)
	if err != nil {
		t.Fatalf("LessonsBefore: %v", err)
	}
	if len(lessons) != 1 {
		t.Fatalf("LessonsBefore() = %d lessons, want 1", len(lessons))
	}
	if lessons[0] != before {
		t.Fatalf("LessonsBefore()[0] = %+q, want lesson A %+q", lessons[0], before)
	}
	if lessons[0] == after {
		t.Fatal("lesson created after the cutoff leaked into the result")
	}
}

// lessonDrawerFor inserts a lesson drawer with the given wing and pinned
// creation time, mirroring a lesson written by an earlier session.
func lessonDrawerFor(t *testing.T, p *Palace, wing string, l Lesson, createdAt time.Time) {
	t.Helper()
	d := &Drawer{
		ID:         fmt.Sprintf("drawer_%s_%s", wing, l.Cue),
		Wing:       wing,
		Room:       "lessons",
		Content:    FormatLesson(l),
		Importance: 5,
		CreatedAt:  createdAt,
	}
	if err := p.store.InsertDrawer(context.Background(), d); err != nil {
		t.Fatalf("InsertDrawer(%s): %v", d.ID, err)
	}
}

func TestLessonsBefore_WingFilter(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	projLesson := Lesson{
		Cue:        "proj cue",
		Lesson:     "proj lesson body",
		Evidence:   "proj evidence",
		Confidence: "high",
	}
	otherLesson := Lesson{
		Cue:        "other cue",
		Lesson:     "other lesson body",
		Evidence:   "other evidence",
		Confidence: "high",
	}
	cutoff := time.Now().Add(time.Hour) // both lessons predate the cutoff
	lessonDrawerFor(t, newPalaceWithStoreForTest(t, store), "proj", projLesson, cutoff.Add(-2*time.Hour))
	lessonDrawerFor(t, newPalaceWithStoreForTest(t, store), "other", otherLesson, cutoff.Add(-time.Hour))

	ms := NewMemoryStack(store, nil, DefaultConfig())

	t.Run("empty wing keeps every wing", func(t *testing.T) {
		got, err := ms.WakeUpAt(ctx, "", cutoff)
		if err != nil {
			t.Fatalf("WakeUpAt: %v", err)
		}
		if !strings.Contains(got, projLesson.Cue) || !strings.Contains(got, otherLesson.Cue) {
			t.Errorf("WakeUpAt(wing=\"\") = %q, want lessons from both wings", got)
		}
	})

	t.Run("named wing excludes other wings", func(t *testing.T) {
		got, err := ms.WakeUpAt(ctx, "proj", cutoff)
		if err != nil {
			t.Fatalf("WakeUpAt: %v", err)
		}
		if !strings.Contains(got, projLesson.Cue) {
			t.Errorf("WakeUpAt(wing=\"proj\") = %q, want the proj lesson", got)
		}
		if strings.Contains(got, otherLesson.Cue) {
			t.Errorf("WakeUpAt(wing=\"proj\") leaked a lesson from another wing:\n%s", got)
		}
	})
}

func newPalaceWithStoreForTest(t *testing.T, store PalaceStore) *Palace {
	t.Helper()
	return NewWithStore(store, nil)
}

func TestLessonsBefore_EmptyPalace(t *testing.T) {
	p := newLessonPalace(t)
	lessons, err := p.LessonsBefore(context.Background(), time.Now())
	if err != nil {
		t.Fatalf("LessonsBefore: %v", err)
	}
	if len(lessons) != 0 {
		t.Fatalf("LessonsBefore() = %d lessons on an empty palace, want 0", len(lessons))
	}
}

func TestLessonsBefore_SkipsNonLessonDrawers(t *testing.T) {
	p := newLessonPalace(t)
	ctx := context.Background()

	if _, err := p.AddDrawer(ctx, DrawerInput{
		Wing:    "proj",
		Room:    "facts",
		Content: "a plain project fact, not a lesson",
	}); err != nil {
		t.Fatalf("AddDrawer: %v", err)
	}

	lessons, err := p.LessonsBefore(ctx, time.Now().Add(time.Hour))
	if err != nil {
		t.Fatalf("LessonsBefore: %v", err)
	}
	if len(lessons) != 0 {
		t.Fatalf("LessonsBefore() = %d lessons, want 0 (plain drawers are not lessons)", len(lessons))
	}
}
