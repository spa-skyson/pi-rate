package provider

// Tests for reasoning persistence (thought.go) on the OpenAI-compatible
// paths: reasoning_content streamed by OpenAI-compatible backends (GLM,
// DeepSeek) must reach the final response as a Thought part, what is
// persisted must stay under the 64 KiB cap, a reasoning-only turn must not
// end thought-only, and the reasoning must never ride the next turn's wire.

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/openai/openai-go/v3/responses"
	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"
)

// chatSSEChunks writes chunks as a Chat Completions SSE stream and closes it.
func chatSSEChunks(w http.ResponseWriter, chunks []string) {
	w.Header().Set("Content-Type", "text/event-stream")
	for _, c := range chunks {
		_, _ = io.WriteString(w, "data: "+c+"\n\n")
	}
	_, _ = io.WriteString(w, "data: [DONE]\n\n")
}

// reasoningStreamChunks is the wire shape GLM and DeepSeek stream: reasoning
// under delta.reasoning_content, then the answer under delta.content.
func reasoningStreamChunks(reasoning, answer string) []string {
	return []string{
		`{"choices":[{"index":0,"delta":{"reasoning_content":"` + reasoning + `"}}]}`,
		`{"choices":[{"index":0,"delta":{"content":"` + answer + `"}}]}`,
		`{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":9}}`,
	}
}

// finalOf returns the TurnComplete response of a streamed run.
func finalOf(t *testing.T, m *openaiModel, contents []*genai.Content) *model.LLMResponse {
	t.Helper()
	var final *model.LLMResponse
	for resp, err := range m.GenerateContent(context.Background(), &model.LLMRequest{Contents: contents}, true) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
		if resp != nil && resp.TurnComplete {
			final = resp
		}
	}
	if final == nil {
		t.Fatal("no TurnComplete response")
	}
	return final
}

// TestOaiStreamingReasoningContentPersistsAsThought pins the GLM/DeepSeek
// streaming shape end to end: reasoning_content chunks ride the final response
// as one Thought part carrying the full reasoning, ahead of the untouched
// answer text.
func TestOaiStreamingReasoningContentPersistsAsThought(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		chatSSEChunks(w, reasoningStreamChunks("weigh the options", "The answer."))
	}))
	defer srv.Close()

	m, err := NewOpenAI(context.Background(), "glm-4.7", "test-key", srv.URL, nil)
	if err != nil {
		t.Fatalf("NewOpenAI: %v", err)
	}

	final := finalOf(t, m.(*openaiModel), []*genai.Content{
		{Role: "user", Parts: []*genai.Part{{Text: "hi"}}},
	})

	parts := final.Content.Parts
	if len(parts) != 2 {
		t.Fatalf("parts = %d, want 2 (thought + answer): %+v", len(parts), parts)
	}
	if !parts[0].Thought || parts[0].Text != "weigh the options" {
		t.Errorf("parts[0] = %+v, want the reasoning as a Thought part", parts[0])
	}
	if parts[1].Thought || parts[1].Text != "The answer." {
		t.Errorf("parts[1] = %+v, want the unflagged answer text", parts[1])
	}
}

// TestOaiStreamingReasoningOnlyTurnIsPlainAnswer pins the thinking-only
// fallback: a turn that produced nothing but reasoning keeps that reasoning as
// its plain content — flagging it Thought would make ADK read the turn as
// thought-only and re-call the model for an answer it was never going to give.
func TestOaiStreamingReasoningOnlyTurnIsPlainAnswer(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		chatSSEChunks(w, []string{
			`{"choices":[{"index":0,"delta":{"reasoning_content":"pondering"}}]}`,
			`{"choices":[{"index":0,"delta":{"reasoning_content":" deeply"}}]}`,
			`{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		})
	}))
	defer srv.Close()

	m, err := NewOpenAI(context.Background(), "glm-4.7", "test-key", srv.URL, nil)
	if err != nil {
		t.Fatalf("NewOpenAI: %v", err)
	}

	final := finalOf(t, m.(*openaiModel), []*genai.Content{
		{Role: "user", Parts: []*genai.Part{{Text: "hi"}}},
	})

	parts := final.Content.Parts
	if len(parts) != 1 {
		t.Fatalf("parts = %d, want 1: %+v", len(parts), parts)
	}
	if parts[0].Thought {
		t.Errorf("parts[0] is Thought-flagged; a thinking-only turn must end as plain text: %+v", parts[0])
	}
	if parts[0].Text != "pondering deeply" {
		t.Errorf("parts[0].Text = %q, want the reasoning as the turn's content", parts[0].Text)
	}
}

// TestPersistThoughtTextCaps pins the 64 KiB persisted cap: text at or under
// the cap passes through byte-for-byte, an over-cap text is cut on a rune
// boundary and closed with a marker naming the dropped remainder.
func TestPersistThoughtTextCaps(t *testing.T) {
	over := maxThoughtBytes + 10
	multibyte := strings.Repeat("ъ", 40000) // 2-byte runes; 80000 bytes total
	tests := []struct {
		name string
		text string
		want string
	}{
		{
			name: "under the cap passes through",
			text: "short thought",
			want: "short thought",
		},
		{
			name: "exactly at the cap passes through",
			text: strings.Repeat("a", maxThoughtBytes),
			want: strings.Repeat("a", maxThoughtBytes),
		},
		{
			name: "over the cap is cut with a marker",
			text: strings.Repeat("a", over),
			want: strings.Repeat("a", maxThoughtBytes) +
				fmt.Sprintf(thoughtTruncationMarker, over-maxThoughtBytes),
		},
		{
			// 65536 is even, so the cut lands exactly on a 2-byte rune
			// boundary: the kept text is 32768 runes, the marker names the
			// rest, and the result stays valid UTF-8.
			name: "over the cap cuts on a rune boundary",
			text: multibyte,
			want: strings.Repeat("ъ", maxThoughtBytes/2) +
				fmt.Sprintf(thoughtTruncationMarker, len(multibyte)-maxThoughtBytes),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := persistThoughtText(tt.text)
			if got != tt.want {
				t.Errorf("persistThoughtText = %d bytes, want %d", len(got), len(tt.want))
			}
			if !utf8.ValidString(got) {
				t.Error("persistThoughtText produced invalid UTF-8")
			}
		})
	}
}

// TestOaiStreamingReasoningCappedAt64KiB runs the cap through the live stream:
// a reasoning_content stream far past the cap must reach the final response as
// one Thought part holding the first 64 KiB and the remainder marker — not the
// full hundreds of kilobytes events.jsonl would otherwise store forever.
func TestOaiStreamingReasoningCappedAt64KiB(t *testing.T) {
	const reasoningLen = 70000
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		chatSSEChunks(w, reasoningStreamChunks(strings.Repeat("a", reasoningLen), "done"))
	}))
	defer srv.Close()

	m, err := NewOpenAI(context.Background(), "glm-4.7", "test-key", srv.URL, nil)
	if err != nil {
		t.Fatalf("NewOpenAI: %v", err)
	}

	final := finalOf(t, m.(*openaiModel), []*genai.Content{
		{Role: "user", Parts: []*genai.Part{{Text: "hi"}}},
	})

	var thought *genai.Part
	for _, p := range final.Content.Parts {
		if p.Thought {
			thought = p
			break
		}
	}
	if thought == nil {
		t.Fatalf("no Thought part in final: %+v", final.Content.Parts)
	}
	want := strings.Repeat("a", maxThoughtBytes) +
		fmt.Sprintf(thoughtTruncationMarker, reasoningLen-maxThoughtBytes)
	if thought.Text != want {
		t.Errorf("persisted thought = %d bytes, want the 64 KiB cap + the %d-byte remainder marker",
			len(thought.Text), reasoningLen-maxThoughtBytes)
	}
}

// TestGenaiSplitParts_SkipsThoughtParts pins the wire filter: Thought-flagged
// text parts are reasoning, not conversation, and must reach neither the text
// nor the function-call half of the split request.
func TestGenaiSplitParts_SkipsThoughtParts(t *testing.T) {
	parts := []*genai.Part{
		{Text: "secret reasoning", Thought: true},
		{Text: "the reply"},
		{Text: "", Thought: true},
		{FunctionCall: &genai.FunctionCall{ID: "c1", Name: "bash"}},
	}
	texts, calls := genaiSplitParts(parts)
	if len(texts) != 1 || texts[0] != "the reply" {
		t.Errorf("texts = %q, want only the unflagged reply", texts)
	}
	if len(calls) != 1 || calls[0].Name != "bash" {
		t.Errorf("calls = %+v, want the one function call", calls)
	}
}

// TestOaiAppendResponsesItems_SkipsThoughtParts pins the Responses-API wire
// filter: a Thought part in replayed history must not become an input item.
func TestOaiAppendResponsesItems_SkipsThoughtParts(t *testing.T) {
	content := &genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			{Text: "secret reasoning", Thought: true},
			{Text: "the answer"},
		},
	}
	items := oaiAppendResponsesItems(responses.ResponseInputParam{}, content,
		map[string]struct{}{}, map[string]struct{}{})

	b, err := json.Marshal(items)
	if err != nil {
		t.Fatalf("marshal items: %v", err)
	}
	wire := string(b)
	if strings.Contains(wire, "secret reasoning") {
		t.Errorf("reasoning reached the Responses input items: %s", wire)
	}
	if !strings.Contains(wire, "the answer") {
		t.Errorf("answer text missing from the input items: %s", wire)
	}
}

// TestOaiChatWireDropsThoughtParts is the captured-body wire filter: the
// second turn's request, whose history carries the first turn's aggregate with
// its Thought part, must not contain the reasoning text — but must keep the
// answer and the new user text.
func TestOaiChatWireDropsThoughtParts(t *testing.T) {
	var bodies []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		bodies = append(bodies, string(b))
		if len(bodies) == 1 {
			chatSSEChunks(w, reasoningStreamChunks("secret reasoning", "the answer"))
			return
		}
		chatSSEChunks(w, reasoningStreamChunks("", "second answer"))
	}))
	defer srv.Close()

	m, err := NewOpenAI(context.Background(), "glm-4.7", "test-key", srv.URL, nil)
	if err != nil {
		t.Fatalf("NewOpenAI: %v", err)
	}
	om := m.(*openaiModel)
	ctx := context.Background()

	run := func(contents []*genai.Content) {
		for resp, err := range om.GenerateContent(ctx, &model.LLMRequest{Contents: contents}, true) {
			if err != nil {
				t.Fatalf("GenerateContent: %v", err)
			}
			_ = resp
		}
	}

	// Turn 1: the reasoning arrives and is persisted on the aggregate.
	run([]*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "hi"}}}})

	// Turn 2: the history replay includes turn 1's aggregate — Thought part
	// and all — exactly as ADK assembles it from the stored session.
	run([]*genai.Content{
		{Role: "user", Parts: []*genai.Part{{Text: "hi"}}},
		{Role: "model", Parts: []*genai.Part{
			{Text: "secret reasoning", Thought: true},
			{Text: "the answer"},
		}},
		{Role: "user", Parts: []*genai.Part{{Text: "go on"}}},
	})

	if len(bodies) != 2 {
		t.Fatalf("captured %d request bodies, want 2", len(bodies))
	}
	if strings.Contains(bodies[1], "secret reasoning") {
		t.Errorf("reasoning reached the second turn's wire:\n%s", bodies[1])
	}
	for _, want := range []string{"the answer", "go on"} {
		if !strings.Contains(bodies[1], want) {
			t.Errorf("second turn's request lost %q:\n%s", want, bodies[1])
		}
	}
}

// completedResponseJSON is a minimal non-streaming Responses body.
func completedResponseJSON(text string) string {
	return `{"id":"resp_1","object":"response","created_at":1.0,"status":"completed",` +
		`"model":"gpt-5-codex","error":null,"incomplete_details":null,` +
		`"instructions":null,"metadata":{},"tool_choice":"auto","tools":[],"parallel_tool_calls":true,` +
		`"output":[{"type":"message","id":"msg_1","role":"assistant","status":"completed",` +
		`"content":[{"type":"output_text","text":"` + text + `","annotations":[]}]}],` +
		`"usage":{"input_tokens":3,"output_tokens":5,"total_tokens":8}}`
}

// TestOaiResponsesWireDropsThoughtParts drives the same captured-body filter
// through the Responses API path, where replayed history goes through
// oaiAppendResponsesItems.
func TestOaiResponsesWireDropsThoughtParts(t *testing.T) {
	var bodies []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		bodies = append(bodies, string(b))
		w.Header().Set("Content-Type", "application/json")
		if len(bodies) == 1 {
			_, _ = io.WriteString(w, completedResponseJSON("the answer"))
			return
		}
		_, _ = io.WriteString(w, completedResponseJSON("second answer"))
	}))
	defer srv.Close()

	m, err := NewOpenAI(context.Background(), "gpt-5-codex", "sk-test", srv.URL, nil)
	if err != nil {
		t.Fatalf("NewOpenAI: %v", err)
	}
	om := m.(*openaiModel)
	ctx := context.Background()

	run := func(contents []*genai.Content) {
		for resp, err := range om.GenerateContent(ctx, &model.LLMRequest{Contents: contents}, false) {
			if err != nil {
				t.Fatalf("GenerateContent: %v", err)
			}
			_ = resp
		}
	}

	run([]*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "hi"}}}})
	run([]*genai.Content{
		{Role: "user", Parts: []*genai.Part{{Text: "hi"}}},
		{Role: "model", Parts: []*genai.Part{
			{Text: "secret reasoning", Thought: true},
			{Text: "the answer"},
		}},
		{Role: "user", Parts: []*genai.Part{{Text: "go on"}}},
	})

	if len(bodies) != 2 {
		t.Fatalf("captured %d request bodies, want 2", len(bodies))
	}
	if strings.Contains(bodies[1], "secret reasoning") {
		t.Errorf("reasoning reached the second turn's Responses input:\n%s", bodies[1])
	}
	for _, want := range []string{"the answer", "go on"} {
		if !strings.Contains(bodies[1], want) {
			t.Errorf("second turn's Responses input lost %q:\n%s", want, bodies[1])
		}
	}
}
