package palace

import (
	"context"
	"strings"
	"testing"
)

func TestPalaceAddLessonTool_GateRejectionListsAllMissing(t *testing.T) {
	p := newLessonPalace(t)

	out, err := palaceAddLessonHandler(context.Background(), p, AddLessonToolInput{
		Wing: "proj",
		Room: "lessons",
		Cue:  "when storing a lesson",
		// lesson, evidence, confidence omitted — the tool must name all three.
	})
	if err != nil {
		t.Fatalf("palaceAddLessonHandler: %v", err)
	}
	for _, field := range []string{"lesson", "evidence", "confidence"} {
		if !strings.Contains(out.Content, field) {
			t.Errorf("tool output %q does not name missing field %q", out.Content, field)
		}
	}
	if out.ID != "" {
		t.Errorf("tool output ID = %q, want empty on rejection", out.ID)
	}

	if n := drawerCount(t, p); n != 0 {
		t.Fatalf("palace holds %d drawers after a rejected tool write, want 0", n)
	}
}

func TestPalaceAddLessonTool_WingRoomRequired(t *testing.T) {
	p := newLessonPalace(t)

	for _, tc := range []struct {
		name  string
		input AddLessonToolInput
		want  string
	}{
		{"no wing", AddLessonToolInput{Room: "r"}, "wing is required"},
		{"no room", AddLessonToolInput{Wing: "w"}, "room is required"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out, err := palaceAddLessonHandler(context.Background(), p, tc.input)
			if err != nil {
				t.Fatalf("palaceAddLessonHandler: %v", err)
			}
			if !strings.Contains(out.Content, tc.want) {
				t.Errorf("output = %q, want it to contain %q", out.Content, tc.want)
			}
		})
	}
}

func TestPalaceAddLessonTool_CompleteStoresLesson(t *testing.T) {
	p := newLessonPalace(t)

	out, err := palaceAddLessonHandler(context.Background(), p, AddLessonToolInput{
		Wing:       "proj",
		Room:       "lessons",
		Cue:        "before pushing a release tag",
		Lesson:     "verify the tag is signed",
		Evidence:   "v0.5.7 lint failure forced a tag re-push",
		Confidence: "high",
	})
	if err != nil {
		t.Fatalf("palaceAddLessonHandler: %v", err)
	}
	if out.ID == "" {
		t.Fatalf("tool output ID = %q, want the new drawer id", out.ID)
	}
	if !strings.Contains(out.Content, "Lesson stored") {
		t.Errorf("output = %q, want a success message", out.Content)
	}

	drawer, err := p.GetDrawer(context.Background(), out.ID)
	if err != nil {
		t.Fatalf("GetDrawer: %v", err)
	}
	got, ok := ParseLesson(drawer.Content)
	if !ok {
		t.Fatalf("stored content %q does not parse as a lesson", drawer.Content)
	}
	want := Lesson{
		Cue:        "before pushing a release tag",
		Lesson:     "verify the tag is signed",
		Evidence:   "v0.5.7 lint failure forced a tag re-push",
		Confidence: "high",
	}
	if got != want {
		t.Errorf("stored lesson = %+q, want %+q", got, want)
	}
}

func TestPalaceAddLessonTool_OverLongFieldRejected(t *testing.T) {
	p := newLessonPalace(t)

	out, err := palaceAddLessonHandler(context.Background(), p, AddLessonToolInput{
		Wing:       "proj",
		Room:       "lessons",
		Cue:        strings.Repeat("c", LessonMaxCueChars+1),
		Lesson:     "body",
		Evidence:   "evidence",
		Confidence: "high",
	})
	if err != nil {
		t.Fatalf("palaceAddLessonHandler: %v", err)
	}
	if !strings.Contains(out.Content, "cue") || !strings.Contains(out.Content, "120") {
		t.Errorf("tool output %q does not name the over-long field and its limit", out.Content)
	}
	if out.ID != "" {
		t.Errorf("tool output ID = %q, want empty on rejection", out.ID)
	}
	if n := drawerCount(t, p); n != 0 {
		t.Fatalf("palace holds %d drawers after a rejected over-long write, want 0", n)
	}
}

func TestPalaceTools_IncludesAddLesson(t *testing.T) {
	p := newLessonPalace(t)
	tools, err := PalaceTools(p)
	if err != nil {
		t.Fatalf("PalaceTools: %v", err)
	}
	found := false
	for _, tl := range tools {
		if runnable, ok := tl.(interface{ Name() string }); ok && runnable.Name() == "palace-add-lesson" {
			found = true
			break
		}
	}
	if !found {
		t.Error("PalaceTools() does not advertise palace-add-lesson")
	}
}
