package provider

// The kimi coding-plan temperature guard (the endpoint rejects every sampling
// temperature but its own — a 400 the agent never survives) is pinned in
// openai_reasoning_test.go; the tests here cover the other half of the same
// incident: the openai paths reported such failures as the bare status
// summary — "OpenAI API error: 400 Bad Request" — dropping the body that
// names what to fix.

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/openai/openai-go/v3"
	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"

	"github.com/spa-skyson/pi-rate/internal/retry"
)

// ---------------------------------------------------------------------------
// oaiErrorText
// ---------------------------------------------------------------------------

func TestOaiErrorTextCarriesAPIMessage(t *testing.T) {
	err := &openai.Error{StatusCode: 400, Message: "invalid temperature: only 1 is allowed for this model"}
	got := oaiErrorText(err)
	// The safe summary stays the prefix — the retry classifier matches on it
	// ("bad request" is a terminal pattern).
	if !strings.Contains(got, "OpenAI API error: 400 Bad Request") {
		t.Errorf("oaiErrorText = %q, want the status summary as prefix", got)
	}
	if !strings.Contains(got, "invalid temperature: only 1 is allowed for this model") {
		t.Errorf("oaiErrorText = %q, want the provider's message", got)
	}

	// Wrapped failures enrich the same way — errors.As must reach the API
	// error through fmt.Errorf's %w.
	wrapped := fmt.Errorf("stream died: %w", err)
	if got := oaiErrorText(wrapped); !strings.Contains(got, "invalid temperature") {
		t.Errorf("oaiErrorText(wrapped) = %q, want the provider's message", got)
	}
}

func TestOaiErrorTextMasksLongTokens(t *testing.T) {
	key := "sk-" + strings.Repeat("a1Bc", 10) // 43 chars of secret alphabet
	err := &openai.Error{StatusCode: 401, Message: "bad key " + key + " rejected"}
	got := oaiErrorText(err)
	if strings.Contains(got, key) {
		t.Errorf("oaiErrorText = %q, want the key masked", got)
	}
	if !strings.Contains(got, "***") {
		t.Errorf("oaiErrorText = %q, want the mask marker", got)
	}
	if !strings.Contains(got, "bad key") {
		t.Errorf("oaiErrorText = %q, want the prose around the mask kept", got)
	}
	// Dated model ids sit below the mask threshold — diagnostics quoting one
	// ("invalid model …") must survive verbatim.
	err = &openai.Error{StatusCode: 404, Message: "model gpt-4o-mini-2024-07-18 not found"}
	got = oaiErrorText(err)
	if !strings.Contains(got, "gpt-4o-mini-2024-07-18") {
		t.Errorf("oaiErrorText = %q, want the dated model id kept", got)
	}
}

func TestOaiErrorTextClipsLongMessages(t *testing.T) {
	// Prose with spaces: a repeated single word would read as one long token
	// and be masked before the clip ever ran.
	err := &openai.Error{StatusCode: 500, Message: strings.Repeat("word ", 80)}
	got := oaiErrorText(err)
	runes := len([]rune(got))
	// Summary + separator + maxServerTextRunes + the truncation marker.
	if runes > len("OpenAI API error: 500 Internal Server Error: ")+maxServerTextRunes+1 {
		t.Errorf("oaiErrorText length = %d runes, want clipped to the cap", runes)
	}
	if !strings.HasSuffix(got, "…") {
		t.Errorf("oaiErrorText = %q, want a marked truncation", got)
	}
}

func TestOaiErrorTextPassthrough(t *testing.T) {
	// Not an API error: nothing to add, nothing to mask.
	err := errors.New("connection reset by peer")
	if got := oaiErrorText(err); got != err.Error() {
		t.Errorf("oaiErrorText = %q, want the error unchanged", got)
	}
	// An API error with no decoded message and no captured body: the summary
	// alone, with no dangling separator.
	err = &openai.Error{StatusCode: 502}
	want := "OpenAI API error: 502 Bad Gateway"
	if got := oaiErrorText(err); got != want {
		t.Errorf("oaiErrorText = %q, want %q", got, want)
	}
}

func TestOaiErrorTextRawJSONFallback(t *testing.T) {
	// The house way to build an error with a captured body (see
	// openai_responses_coverage_test.go): UnmarshalJSON fills Message and the
	// raw envelope together, so feed it an envelope without a message field
	// to drive the fallback.
	var e openai.Error
	if err := e.UnmarshalJSON([]byte(`{"type":"bad_gateway"}`)); err != nil {
		t.Fatalf("UnmarshalJSON: %v", err)
	}
	got := oaiErrorText(&e)
	if !strings.Contains(got, `"type":"bad_gateway"`) {
		t.Errorf("oaiErrorText = %q, want the raw envelope quoted", got)
	}
}

// The enriched text is classification input: the appended body joins the
// summary in retry.IsTransient, where terminal patterns beat transient ones.
// The summary alone already carried the status code; these pin the body's
// vote.
func TestOaiErrorTextRetryClassification(t *testing.T) {
	tests := []struct {
		name      string
		status    int
		body      string
		transient bool
	}{
		{"429 body asks to retry", 429, "rate limit exceeded, retry later", true},
		{"429 body is quota exhaustion", 429, "You exceeded your current quota", false},
		{"400 kimi temperature", 400, kimi400Message, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			enriched := oaiErrorText(&openai.Error{StatusCode: tt.status, Message: tt.body})
			if got := retry.IsTransient(errors.New(enriched)); got != tt.transient {
				t.Errorf("IsTransient(%q) = %v, want %v", enriched, got, tt.transient)
			}
		})
	}
}

func TestOaiRenderedErrorKeepsErrorsAs(t *testing.T) {
	apiErr := &openai.Error{StatusCode: 400, Message: "invalid temperature: only 1 is allowed for this model"}
	err := newOaiRenderedError("OpenAI chat completion failed", fmt.Errorf("outer: %w", apiErr))
	// The wrap's own prose rides the summary, as it did before the enrichment.
	want := "OpenAI chat completion failed: outer: OpenAI API error: 400 Bad Request: " + kimi400Message
	if err.Error() != want {
		t.Errorf("Error() = %q, want %q", err.Error(), want)
	}
	var reached *openai.Error
	if !errors.As(err, &reached) || reached != apiErr {
		t.Errorf("errors.As lost the original API error through %v", err)
	}
	if newOaiRenderedError("intro", nil) != nil {
		t.Error("newOaiRenderedError(nil) != nil")
	}
}

// ---------------------------------------------------------------------------
// End to end: a kimi 400 reaches the session record with its message
// ---------------------------------------------------------------------------

// kimi400Message is the body the live coding endpoint returns for a
// non-1 temperature (issue-proven via curl matrix).
const kimi400Message = "invalid temperature: only 1 is allowed for this model"

// kimiStubServer answers every request with that 400.
func kimiStubServer(t *testing.T) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"message":"` + kimi400Message + `","type":"invalid_request_error"}}`))
	}))
	t.Cleanup(srv.Close)
	return srv
}

func newKimi400Model(t *testing.T) *openaiModel {
	t.Helper()
	llm, err := NewOpenAI(context.Background(), "kimi-for-coding", "sk-test", kimiStubServer(t).URL,
		&LLMOptions{Temperature: temperatureOpt(0.3)})
	if err != nil {
		t.Fatalf("NewOpenAI: %v", err)
	}
	return llm.(*openaiModel)
}

func TestKimiCoding400SurfacesAPIMessage(t *testing.T) {
	req := &model.LLMRequest{
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "hi"}}}},
	}
	for _, tt := range []struct {
		name   string
		stream bool
	}{
		{"streaming", true},
		{"non-streaming", false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			resps, errs := collectResponses(t, newKimi400Model(t), req, tt.stream)
			if tt.stream {
				// Streaming failures ride the STREAM_ERROR response.
				if len(errs) != 0 || len(resps) != 1 {
					t.Fatalf("want one STREAM_ERROR response, got %d responses and %d errors", len(resps), len(errs))
				}
				if resps[0].ErrorCode != "STREAM_ERROR" {
					t.Errorf("ErrorCode = %q, want STREAM_ERROR", resps[0].ErrorCode)
				}
				if msg := resps[0].ErrorMessage; !strings.Contains(msg, kimi400Message) {
					t.Errorf("ErrorMessage = %q, want the provider's message", msg)
				}
				return
			}
			// Non-streaming failures ride the yielded error.
			if len(errs) != 1 {
				t.Fatalf("want one error, got %d", len(errs))
			}
			if msg := errs[0].Error(); !strings.Contains(msg, kimi400Message) {
				t.Errorf("error = %q, want the provider's message", msg)
			}
		})
	}
}
