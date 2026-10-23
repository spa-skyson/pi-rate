package palace

import (
	"context"
	"fmt"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// AddLessonToolInput defines parameters for the palace-add-lesson tool.
type AddLessonToolInput struct {
	// Wing name (project or domain).
	Wing string `json:"wing"`
	// Room name (topic or subdirectory).
	Room string `json:"room"`
	// When the lesson applies.
	Cue string `json:"cue"`
	// What to do.
	Lesson string `json:"lesson"`
	// What the lesson is based on; required.
	Evidence string `json:"evidence"`
	// Confidence annotation.
	Confidence string `json:"confidence"`
}

// AddLessonToolOutput contains the result of adding a lesson.
type AddLessonToolOutput struct {
	Content string `json:"content"`
	ID      string `json:"id,omitempty"`
}

func newPalaceAddLessonTool(p *Palace) (tool.Tool, error) {
	return functiontool.New(functiontool.Config{
		Name: "palace-add-lesson",
		Description: "Store a transferable lesson as a structured palace drawer. " +
			"Requires wing, room, cue (when the lesson applies), lesson (what to do), " +
			"and evidence (what it is based on) — the write is rejected naming every " +
			"missing field, and confidence is the free-form annotation.",
	}, func(ctx agent.Context, input AddLessonToolInput) (AddLessonToolOutput, error) {
		return palaceAddLessonHandler(ctx, p, input)
	})
}

func palaceAddLessonHandler(ctx context.Context, p *Palace, input AddLessonToolInput) (AddLessonToolOutput, error) {
	if input.Wing == "" {
		return AddLessonToolOutput{Content: "Error: wing is required"}, nil
	}
	if input.Room == "" {
		return AddLessonToolOutput{Content: "Error: room is required"}, nil
	}

	drawer, err := p.AddLesson(ctx, DrawerInput{
		Wing:    input.Wing,
		Room:    input.Room,
		AddedBy: "agent",
	}, Lesson{
		Cue:        input.Cue,
		Lesson:     input.Lesson,
		Evidence:   input.Evidence,
		Confidence: input.Confidence,
	})
	if err != nil {
		// The gate error already names every missing field; surface it as-is
		// so the agent can fix the whole note in one retry.
		return AddLessonToolOutput{Content: fmt.Sprintf("Error: %v", err)}, nil
	}

	return AddLessonToolOutput{
		Content: fmt.Sprintf("Lesson stored: %s (wing=%s, room=%s)", drawer.ID, drawer.Wing, drawer.Room),
		ID:      drawer.ID,
	}, nil
}
