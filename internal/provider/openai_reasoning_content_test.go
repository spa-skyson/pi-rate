package provider

// Tests for the standard delta.reasoning_content streaming field on the
// OpenAI-compatible Chat Completions path. Reasoning models behind such
// endpoints (z.ai GLM, DeepSeek, LiteLLM-style proxies) stream their whole
// thinking phase there before the first content delta; the openai-go SDK has
// no field for it, so the chunks previously produced no yield at all — a live
// stream was indistinguishable from silence to the idle watcher, which
// aborted it mid-thinking at the idle budget. The wire fixtures below are the
// literal SSE bodies those endpoints emit; the assertions run the real SDK
// decoder and the real idleStreamModel wrapper against them.

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"
)

// streamChunkFixture is the wire shape of one chat.completion.chunk, with the
// reasoning_content delta the SDK cannot decode.  Fixtures are marshaled from
// these values rather than written as JSON literals, so a field-name drift in
// the fixture fails loudly here instead of silently testing a shape no
// endpoint emits.
type streamChunkFixture struct {
	ID      string                  `json:"id"`
	Object  string                  `json:"object"`
	Model   string                  `json:"model"`
	Choices []streamChunkChoiceFixt `json:"choices"`
	Usage   *streamChunkUsageFixt   `json:"usage,omitempty"`
}

type streamChunkChoiceFixt struct {
	Index  int                  `json:"index"`
	Delta  streamChunkDeltaFixt `json:"delta"`
	Finish string               `json:"finish_reason,omitempty"`
}

type streamChunkDeltaFixt struct {
	Role             string `json:"role,omitempty"`
	Content          string `json:"content,omitempty"`
	ReasoningContent string `json:"reasoning_content,omitempty"`
}

type streamChunkUsageFixt struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

func marshalChunk(c streamChunkFixture) string {
	b, err := json.Marshal(c)
	if err != nil {
		// A fixture that cannot encode is a broken test, not a broken stream.
		panic(fmt.Sprintf("fixture chunk does not marshal: %v", err))
	}
	return string(b)
}

// reasoningChunk renders one streaming chunk carrying only a
// reasoning_content delta, in the shape GLM-class endpoints emit.
func reasoningChunk(text string) string {
	return marshalChunk(streamChunkFixture{
		ID: "s1", Object: "chat.completion.chunk", Model: "glm-4.7",
		Choices: []streamChunkChoiceFixt{{
			Index: 0,
			Delta: streamChunkDeltaFixt{ReasoningContent: text},
		}},
	})
}

// answerChunk renders one plain answer-delta chunk.
func answerChunk(text string) string {
	return marshalChunk(streamChunkFixture{
		ID: "s1", Object: "chat.completion.chunk", Model: "glm-4.7",
		Choices: []streamChunkChoiceFixt{{
			Index: 0,
			Delta: streamChunkDeltaFixt{Content: text},
		}},
	})
}

// finishChunk renders the terminating chunk with usage, as the usage-bearing
// final chunk arrives on every real stream.
func finishChunk() string {
	return marshalChunk(streamChunkFixture{
		ID: "s1", Object: "chat.completion.chunk", Model: "glm-4.7",
		Choices: []streamChunkChoiceFixt{{
			Index:  0,
			Finish: "stop",
		}},
		Usage: &streamChunkUsageFixt{PromptTokens: 5, CompletionTokens: 9, TotalTokens: 14},
	})
}

// streamReasoningSSE writes chunks as SSE data lines, pausing gap between
// lines and flushing after each so the client sees the pacing. Without the
// flushes the whole body would arrive as one burst and the paced-stream
// tests would test nothing.
func streamReasoningSSE(t *testing.T, w http.ResponseWriter, chunks []string, gap time.Duration) {
	t.Helper()
	flusher, ok := w.(http.Flusher)
	if !ok {
		t.Error("response writer cannot flush; paced streaming needs it")
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.WriteHeader(http.StatusOK)
	for _, c := range chunks {
		if gap > 0 {
			time.Sleep(gap)
		}
		if _, err := w.Write([]byte("data: " + c + "\n\n")); err != nil {
			return
		}
		flusher.Flush()
	}
	_, _ = w.Write([]byte("data: [DONE]\n\n"))
	flusher.Flush()
}

// collectReasoningStream drains one streaming turn, accumulating the
// thinking-role partials and the answer partials separately and recording the
// stream failures (both Go errors and content-less error responses).
func collectReasoningStream(t *testing.T, seq iter.Seq2[*model.LLMResponse, error]) (thinking, answer string, thinkingPartials, failures int, last *model.LLMResponse) {
	t.Helper()
	for resp, err := range seq {
		if err != nil {
			failures++
			continue
		}
		if resp.ErrorCode != "" {
			failures++
			continue
		}
		last = resp
		if resp.Content == nil || len(resp.Content.Parts) == 0 || !resp.Partial {
			continue
		}
		if resp.Content.Role == "thinking" {
			thinkingPartials++
			thinking += resp.Content.Parts[0].Text
		} else {
			answer += resp.Content.Parts[0].Text
		}
	}
	return thinking, answer, thinkingPartials, failures, last
}

func TestOaiDeltaReasoningContent(t *testing.T) {
	tests := []struct {
		name string
		raw  string
		want string
	}{
		{
			name: "reasoning delta",
			raw:  `{"choices":[{"index":0,"delta":{"reasoning_content":"thinking "}}]}`,
			want: "thinking ",
		},
		{
			name: "reasoning delta alongside empty content",
			raw:  `{"choices":[{"index":0,"delta":{"content":"","reasoning_content":"deep thought"}}]}`,
			want: "deep thought",
		},
		{name: "answer-only chunk", raw: `{"choices":[{"index":0,"delta":{"content":"hello"}}]}`},
		{name: "role-only chunk", raw: `{"choices":[{"index":0,"delta":{"role":"assistant"}}]}`},
		{name: "usage chunk without choices", raw: `{"usage":{"prompt_tokens":1}}`},
		{name: "empty choices", raw: `{"choices":[]}`},
		{name: "malformed JSON", raw: `{"choices":[`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := oaiDeltaReasoningContent(tt.raw); got != tt.want {
				t.Errorf("oaiDeltaReasoningContent() = %q, want %q", got, tt.want)
			}
		})
	}
}

// A reasoning stream followed by an answer must reach the consumer as
// thinking-role partials on the same path the OpenRouter hook uses, with the
// answer intact and the final response carrying only the answer (the
// reasoning already streamed).
func TestOpenAIStreamingReasoningContentBecomesThinking(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		chunks := []string{
			reasoningChunk("first "),
			reasoningChunk("second"),
			answerChunk("Hello"),
			finishChunk(),
		}
		streamReasoningSSE(t, w, chunks, 0)
	}))
	defer srv.Close()

	llm, err := NewOpenAI(context.Background(), "gpt-4o", "sk-test", srv.URL, nil)
	if err != nil {
		t.Fatalf("NewOpenAI: %v", err)
	}
	req := &model.LLMRequest{
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "hi"}}}},
	}
	thinking, answer, thinkingPartials, failures, last := collectReasoningStream(t,
		llm.GenerateContent(context.Background(), req, true))

	if failures != 0 {
		t.Errorf("stream reported %d failures, want none", failures)
	}
	if thinkingPartials != 2 {
		t.Errorf("got %d thinking partials, want one per reasoning delta (2)", thinkingPartials)
	}
	if thinking != "first second" {
		t.Errorf("streamed thinking = %q, want %q", thinking, "first second")
	}
	if answer != "Hello" {
		t.Errorf("streamed answer = %q, want %q", answer, "Hello")
	}
	if last == nil || !last.TurnComplete {
		t.Fatal("expected a final TurnComplete response")
	}
	if len(last.Content.Parts) != 1 || last.Content.Parts[0].Text != "Hello" {
		t.Errorf("final content = %+v, want the answer text alone", last.Content.Parts)
	}
}

// A turn that is all reasoning and never answers must still end complete,
// with the reasoning re-sent as the turn's content — the same fallback the
// OpenRouter path applies. Without the accumulation into state.thinking this
// turn would end empty.
func TestOpenAIStreamingReasoningOnlyTurnSurfacesThinking(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		chunks := []string{
			reasoningChunk("only "),
			reasoningChunk("thinking"),
			finishChunk(),
		}
		streamReasoningSSE(t, w, chunks, 0)
	}))
	defer srv.Close()

	llm, err := NewOpenAI(context.Background(), "gpt-4o", "sk-test", srv.URL, nil)
	if err != nil {
		t.Fatalf("NewOpenAI: %v", err)
	}
	req := &model.LLMRequest{
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "hi"}}}},
	}
	thinking, answer, thinkingPartials, failures, last := collectReasoningStream(t,
		llm.GenerateContent(context.Background(), req, true))

	if failures != 0 {
		t.Errorf("stream reported %d failures, want none", failures)
	}
	if thinkingPartials != 2 {
		t.Errorf("got %d thinking partials, want 2", thinkingPartials)
	}
	if thinking != "only thinking" {
		t.Errorf("streamed thinking = %q, want %q", thinking, "only thinking")
	}
	if answer != "" {
		t.Errorf("answer = %q, want none for a reasoning-only turn", answer)
	}
	if last == nil || !last.TurnComplete {
		t.Fatal("expected a final TurnComplete response")
	}
	if len(last.Content.Parts) != 1 || last.Content.Parts[0].Text != "only thinking" {
		t.Errorf("final content = %+v, want the reasoning as the turn's content", last.Content.Parts)
	}
}

// A provider that installs its own deltaThinking extractor keeps exclusive
// ownership of reasoning extraction: the default reasoning_content path must
// stay out entirely, or OpenRouter-class providers would see their reasoning
// yielded twice.
func TestOpenAIStreamingInstalledHookOwnsReasoningExtraction(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		chunks := []string{reasoningChunk("secret reasoning"), finishChunk()}
		streamReasoningSSE(t, w, chunks, 0)
	}))
	defer srv.Close()

	client := openai.NewClient(option.WithAPIKey("sk-test"), option.WithBaseURL(srv.URL))
	hookCalls := 0
	hooks := oaiExtractHooks{
		deltaThinking: func(string) string {
			hookCalls++
			// A real extractor reports only what it found in the raw chunk;
			// this one finds nothing, so nothing may reach the consumer.
			return ""
		},
	}
	var thinking []string
	params := openai.ChatCompletionNewParams{
		Model:    "gpt-4o",
		Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("hi")},
	}
	oaiRunStreamingHooks(context.Background(), &client, params,
		func(resp *model.LLMResponse, err error) bool {
			if resp != nil && resp.Content != nil && resp.Content.Role == "thinking" {
				thinking = append(thinking, resp.Content.Parts[0].Text)
			}
			return true
		}, hooks)

	if hookCalls == 0 {
		t.Fatal("the installed hook was never called")
	}
	// The chunk carried reasoning_content, yet with a hook installed none of
	// it arrives: the default extractor stays out entirely. If it ever ran
	// alongside the hook, the reasoning text shows up here and this fails.
	if len(thinking) != 0 {
		t.Errorf("thinking yields = %v, want none: the installed hook owns extraction", thinking)
	}
}

// The invariant the fix exists for: reasoning deltas are yields upstream, and
// every yield upstream rearms idleStreamModel's budget (deliver resets the
// timer). A stream that keeps producing reasoning_content for several times
// the idle budget must not be aborted; a stream that yields nothing at all
// still must. If the reasoning yield ever disappears from the stream loop,
// the first subtest fails with the idle abort this bug shipped with.
func TestIdleTimerSeesReasoningStreamAsLive(t *testing.T) {
	t.Run("reasoning-fed stream is not aborted", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			// 14 reasoning deltas 30ms apart: ~420ms of thinking-only stream
			// against a 200ms idle budget, with the answer at the end.
			chunks := make([]string, 0, 17)
			for i := 0; i < 14; i++ {
				chunks = append(chunks, reasoningChunk(strings.Repeat("x", i+1)+"\n"))
			}
			chunks = append(chunks, answerChunk("done"), finishChunk())
			streamReasoningSSE(t, w, chunks, 30*time.Millisecond)
		}))
		defer srv.Close()

		llm, err := NewOpenAI(context.Background(), "gpt-4o", "sk-test", srv.URL, nil)
		if err != nil {
			t.Fatalf("NewOpenAI: %v", err)
		}
		watched := idleStreamModel{inner: llm, timeout: 200 * time.Millisecond, tick: 30 * time.Millisecond}

		req := &model.LLMRequest{
			Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "hi"}}}},
		}
		thinking, answer, thinkingPartials, failures, last := collectReasoningStream(t,
			watched.GenerateContent(context.Background(), req, true))

		if failures != 0 {
			t.Errorf("live reasoning stream was aborted or failed %d times; idle budget saw silence", failures)
		}
		if thinkingPartials != 14 {
			t.Errorf("got %d thinking partials, want 14", thinkingPartials)
		}
		if !strings.HasPrefix(thinking, "x\nxx\n") || !strings.HasSuffix(thinking, "xxxxxxxxxxxxxx\n") {
			t.Errorf("thinking text incomplete: %d bytes received", len(thinking))
		}
		if answer != "done" {
			t.Errorf("answer = %q, want %q", answer, "done")
		}
		if last == nil || !last.TurnComplete {
			t.Error("expected a final TurnComplete response")
		}
	})

	t.Run("a stream with no yields at all is still aborted", func(t *testing.T) {
		inner := &stallModel{chunks: 3, delay: 120 * time.Millisecond}
		watched := idleStreamModel{inner: inner, timeout: 40 * time.Millisecond, tick: 10 * time.Millisecond}

		_, errs := drainWatch(watched.GenerateContent(context.Background(), nil, true))
		if len(errs) != 1 || !strings.Contains(errs[0].Error(), "llm stream idle") {
			t.Errorf("errs = %v, want exactly the idle abort", errs)
		}
	})
}

func TestOaiMessageReasoningContent(t *testing.T) {
	tests := []struct {
		name string
		raw  string
		want string
	}{
		{
			name: "message reasoning content",
			raw:  `{"choices":[{"index":0,"message":{"reasoning_content":"deep thought"}}]}`,
			want: "deep thought",
		},
		{
			name: "message reasoning alongside answer content",
			raw:  `{"choices":[{"index":0,"message":{"content":"answer","reasoning_content":"deep thought"}}]}`,
			want: "deep thought",
		},
		{name: "answer-only message", raw: `{"choices":[{"index":0,"message":{"content":"hello"}}]}`},
		{name: "role-only message", raw: `{"choices":[{"index":0,"message":{"role":"assistant"}}]}`},
		{name: "usage-only response", raw: `{"usage":{"prompt_tokens":1}}`},
		{name: "empty choices", raw: `{"choices":[]}`},
		{name: "malformed JSON", raw: `{"choices":[`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := oaiMessageReasoningContent(tt.raw); got != tt.want {
				t.Errorf("oaiMessageReasoningContent() = %q, want %q", got, tt.want)
			}
		})
	}
}

// A non-streaming completion that carries its thinking in
// choices[0].message.reasoning_content must surface that thinking as a
// thinking-role part in the final response, prepended before the answer —
// the same shape the streaming path emits.
func TestOpenAINonStreamingReasoningContentBecomesThinking(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		body := `{"id":"chatcmpl-ns1","object":"chat.completion","model":"glm-4.7",` +
			`"choices":[{"index":0,"message":{"role":"assistant","content":"Hello",` +
			`"reasoning_content":"deep "},"finish_reason":"stop"}],` +
			`"usage":{"prompt_tokens":5,"completion_tokens":9,"total_tokens":14}}`
		if _, err := w.Write([]byte(body)); err != nil {
			return
		}
	}))
	defer srv.Close()

	llm, err := NewOpenAI(context.Background(), "gpt-4o", "sk-test", srv.URL, nil)
	if err != nil {
		t.Fatalf("NewOpenAI: %v", err)
	}
	req := &model.LLMRequest{
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "hi"}}}},
	}
	var last *model.LLMResponse
	for resp, err := range llm.GenerateContent(context.Background(), req, false) {
		if err != nil {
			t.Fatalf("non-streaming turn returned error: %v", err)
		}
		if resp.ErrorCode != "" {
			t.Fatalf("non-streaming turn returned error response: %+v", resp)
		}
		last = resp
	}
	// non-streaming path prepends reasoning as the first part of the model content,
	// matching the Mistral non-streaming convention (no separate "thinking" role).
	if last == nil || !last.TurnComplete {
		t.Fatal("expected a final TurnComplete response")
	}
	if last.Content == nil {
		t.Fatal("expected a final response with content")
	}
	parts := last.Content.Parts
	if len(parts) != 2 {
		t.Fatalf("model content parts = %d, want 2 (reasoning then answer)", len(parts))
	}
	if parts[0].Text != "deep " {
		t.Errorf("reasoning part = %q, want %q", parts[0].Text, "deep ")
	}
	if parts[1].Text != "Hello" {
		t.Errorf("answer part = %q, want %q", parts[1].Text, "Hello")
	}
}

// A provider that installs its own messageThinking extractor keeps exclusive
// ownership of reasoning extraction: the default message.reasoning_content path
// must stay out entirely, or Mistral/OpenRouter-class providers would see their
// reasoning extracted twice by competing hooks.
func TestOpenAINonStreamingInstalledHookOwnsReasoningExtraction(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		body := `{"id":"chatcmpl-ns2","object":"chat.completion","model":"gpt-4o",` +
			`"choices":[{"index":0,"message":{"role":"assistant","content":"answer",` +
			`"reasoning_content":"secret reasoning"},"finish_reason":"stop"}],` +
			`"usage":{"prompt_tokens":5,"completion_tokens":9,"total_tokens":14}}`
		if _, err := w.Write([]byte(body)); err != nil {
			return
		}
	}))
	defer srv.Close()

	client := openai.NewClient(option.WithAPIKey("sk-test"), option.WithBaseURL(srv.URL))
	hookCalls := 0
	hooks := oaiExtractHooks{
		messageThinking: func(string) string {
			hookCalls++
			return ""
		},
	}
	var partTexts []string
	params := openai.ChatCompletionNewParams{
		Model:    "gpt-4o",
		Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("hi")},
	}
	oaiRunNonStreamingHooks(context.Background(), &client, params,
		func(resp *model.LLMResponse, err error) bool {
			if resp != nil && resp.TurnComplete && resp.Content != nil {
				for _, p := range resp.Content.Parts {
					if p.Text != "" {
						partTexts = append(partTexts, p.Text)
					}
				}
			}
			return true
		}, hooks)

	if hookCalls == 0 {
		t.Fatal("the installed hook was never called")
	}
	if len(partTexts) != 1 || partTexts[0] != "answer" {
		t.Errorf("final parts = %v, want [answer]: the installed hook owns extraction", partTexts)
	}
}
