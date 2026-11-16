package session

import (
	"context"
	"strings"
	"testing"

	"google.golang.org/adk/v2/session"
	"google.golang.org/genai"
)

// newSynthSession creates a persisted session seeded with the given events,
// returning the service and the session ID — the same shape the synthesize
// sweep runs against in production.
func newSynthSession(t *testing.T, events []*session.Event) (*FileService, string) {
	t.Helper()
	svc, err := NewFileService(t.TempDir())
	if err != nil {
		t.Fatalf("NewFileService: %v", err)
	}
	ctx := t.Context()
	resp, err := svc.Create(ctx, &session.CreateRequest{AppName: "pi-go", UserID: "local"})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	for i, ev := range events {
		if ev.ID == "" {
			ev.ID = "ev-" + string(rune('a'+i))
		}
		if err := svc.AppendEvent(ctx, resp.Session, ev); err != nil {
			t.Fatalf("AppendEvent: %v", err)
		}
	}
	return svc, resp.Session.ID()
}

// loadSynthEvents reads every persisted event back through the service.
func loadSynthEvents(t *testing.T, svc *FileService, id string) []*session.Event {
	t.Helper()
	resp, err := svc.Get(context.Background(), &session.GetRequest{AppName: "pi-go", UserID: "local", SessionID: id})
	if err != nil || resp == nil || resp.Session == nil {
		t.Fatalf("reading session %s: %v", id, err)
	}
	var events []*session.Event
	for ev := range resp.Session.Events().All() {
		events = append(events, ev)
	}
	return events
}

// synthOrphanIDs reports ids of function calls no function response answers.
func synthOrphanIDs(events []*session.Event) []string {
	answered := make(map[string]struct{})
	var ids []string
	for i := len(events) - 1; i >= 0; i-- {
		ev := events[i]
		if ev == nil || ev.Content == nil {
			continue
		}
		for _, part := range ev.Content.Parts {
			if part == nil {
				continue
			}
			if fr := part.FunctionResponse; fr != nil && fr.ID != "" {
				answered[fr.ID] = struct{}{}
			}
			if fc := part.FunctionCall; fc != nil && fc.ID != "" {
				if _, ok := answered[fc.ID]; !ok {
					ids = append(ids, fc.ID)
				}
			}
		}
	}
	return ids
}

func TestSynthesizeStoppedResponses_HealsOrphan(t *testing.T) {
	svc, id := newSynthSession(t, []*session.Event{callEvent("pi-go", "bash", "call_dangling", map[string]any{"command": "echo hi"})})

	SynthesizeStoppedResponses(context.Background(), svc, "pi-go", "local", id, "process killed", nil)

	events := loadSynthEvents(t, svc, id)
	if orphans := synthOrphanIDs(events); len(orphans) != 0 {
		t.Fatalf("orphaned calls after sweep: %v", orphans)
	}
	// The synthetic response must carry the call's id, name and the stop
	// reason in the error key — the shape ADK pairs and the model reads.
	var synth *genai.FunctionResponse
	for _, ev := range events {
		if ev.Content == nil {
			continue
		}
		for _, part := range ev.Content.Parts {
			if part == nil || part.FunctionResponse == nil || part.FunctionResponse.ID != "call_dangling" {
				continue
			}
			synth = part.FunctionResponse
		}
	}
	if synth == nil {
		t.Fatal("no response for call_dangling after sweep")
	}
	if synth.Name != "bash" {
		t.Errorf("synthetic response name = %q, want bash", synth.Name)
	}
	if msg, _ := synth.Response["error"].(string); !strings.Contains(msg, "process killed") {
		t.Errorf("synthetic response error = %q, want it to name the stop reason", msg)
	}
}

func TestSynthesizeStoppedResponses_PairedHistoryUntouched(t *testing.T) {
	before := []*session.Event{callEvent("pi-go", "bash", "call_ok", nil), respEvent("bash", "call_ok", map[string]any{"output": "hi"})}
	svc, id := newSynthSession(t, before)

	SynthesizeStoppedResponses(context.Background(), svc, "pi-go", "local", id, "run interrupted before the tool call completed", nil)

	events := loadSynthEvents(t, svc, id)
	if len(events) != len(before) {
		t.Fatalf("event count = %d, want %d — a paired history must gain nothing", len(events), len(before))
	}
	if orphans := synthOrphanIDs(events); len(orphans) != 0 {
		t.Fatalf("unexpected orphans: %v", orphans)
	}
	for i, ev := range events {
		if ev.ID != before[i].ID {
			t.Errorf("event %d id = %q, want untouched %q", i, ev.ID, before[i].ID)
		}
	}
}

func TestSynthesizeStoppedResponses_Idempotent(t *testing.T) {
	svc, id := newSynthSession(t, []*session.Event{callEvent("pi-go", "bash", "call_x", nil), callEvent("pi-go", "bash", "call_y", nil)})
	ctx := context.Background()

	SynthesizeStoppedResponses(ctx, svc, "pi-go", "local", id, "auto-stop", nil)
	first := loadSynthEvents(t, svc, id)

	SynthesizeStoppedResponses(ctx, svc, "pi-go", "local", id, "auto-stop", nil)
	second := loadSynthEvents(t, svc, id)

	if len(second) != len(first) {
		t.Fatalf("second sweep changed event count: %d → %d, want a no-op", len(first), len(second))
	}
	if orphans := synthOrphanIDs(second); len(orphans) != 0 {
		t.Fatalf("orphaned calls after second sweep: %v", orphans)
	}
}

func TestSynthesizeStoppedResponses_EmptyArgsNoPanic(t *testing.T) {
	// Nil service / empty session id are guarded no-ops.
	SynthesizeStoppedResponses(context.Background(), nil, "pi-go", "local", "s", "x", nil)
	svc, id := newSynthSession(t, nil)
	SynthesizeStoppedResponses(context.Background(), svc, "pi-go", "local", "", "x", nil)
	_ = id
}
