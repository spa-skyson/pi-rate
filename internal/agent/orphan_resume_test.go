package agent

import (
	"context"
	"iter"
	"strings"
	"sync"
	"testing"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/genai"
)

// recordingLLM wraps mockLLM and records every request the runner builds, so
// a test can assert on the context the model actually received rather than on
// what was persisted.
type recordingLLM struct {
	mockLLM

	mu       sync.Mutex
	requests []*model.LLMRequest
}

func (r *recordingLLM) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	r.mu.Lock()
	r.requests = append(r.requests, req)
	r.mu.Unlock()
	return r.mockLLM.GenerateContent(ctx, req, stream)
}

// snapshot returns a copy of the request slice observed so far.
func (r *recordingLLM) snapshot() []*model.LLMRequest {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]*model.LLMRequest(nil), r.requests...)
}

// TestRunStreamingHealsOrphanedToolCallOnResume drives the resume path the
// pre-turn sweep exists for: a session persisted with a tool call that never
// got its response — the state a process killed mid-tool leaves behind — then
// a fresh RunStreaming on that session. The first turn's model context must
// already contain the synthetic function response: ADK drops unpaired calls
// from every context build, so without the sweep the model never learns what
// happened to the tool, on this turn or any later one.
func TestRunStreamingHealsOrphanedToolCallOnResume(t *testing.T) {
	llm := &recordingLLM{mockLLM: mockLLM{name: "test-model", response: "Continuing."}}
	a, err := New(Config{Model: llm})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	ctx := context.Background()
	sessionID, _, err := a.CreateSession(ctx)
	if err != nil {
		t.Fatalf("CreateSession(): %v", err)
	}

	// Persist the crash state directly: one function call with no response.
	resp, err := a.sessionService.Get(ctx, &session.GetRequest{
		AppName:   AppName,
		UserID:    DefaultUserID,
		SessionID: sessionID,
	})
	if err != nil || resp == nil || resp.Session == nil {
		t.Fatalf("reading session %s: %v", sessionID, err)
	}
	orphan := &session.Event{
		Author: AgentName,
		Content: &genai.Content{
			Role: string(genai.RoleModel),
			Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
				ID:   "call_orphan",
				Name: "bash",
				Args: map[string]any{"command": "echo hi"},
			}}},
		},
	}
	if err := a.sessionService.AppendEvent(ctx, resp.Session, orphan); err != nil {
		t.Fatalf("AppendEvent(orphan call): %v", err)
	}

	for event, err := range a.RunStreaming(ctx, sessionID, "continue") {
		if err != nil {
			t.Fatalf("RunStreaming() yielded error: %v", err)
		}
		if event == nil {
			continue
		}
	}

	requests := llm.snapshot()
	if len(requests) == 0 {
		t.Fatal("the model saw no requests — RunStreaming reached no model call")
	}
	// Strictly the FIRST request after resume: the sweep runs pre-turn, so a
	// synthetic response that only shows up in a later request means the
	// resumed turn was built (and dispatched) against the degraded context —
	// exactly the regression this test guards against.
	msg, ok := syntheticResponseFor(requests[0].Contents, "call_orphan")
	if !ok {
		t.Fatalf("the first request after resume has no synthetic response for call_orphan — the orphaned call reached the model unhealed")
	}
	if !strings.Contains(msg, "turn was stopped") {
		t.Errorf("first request: synthetic response error = %q, want it to name the stop", msg)
	}
	if !strings.Contains(msg, "unknown") {
		t.Errorf("first request: synthetic response error = %q, want it to say the side-effect state is unknown", msg)
	}
}

// syntheticResponseFor reports the error payload of the function response
// answering id in the given contents, and whether one exists.
func syntheticResponseFor(contents []*genai.Content, id string) (string, bool) {
	for _, c := range contents {
		if c == nil {
			continue
		}
		for _, p := range c.Parts {
			if p == nil || p.FunctionResponse == nil || p.FunctionResponse.ID != id {
				continue
			}
			msg, _ := p.FunctionResponse.Response["error"].(string)
			return msg, true
		}
	}
	return "", false
}
