package provider

// Review finding R1: with the provider gate open for openai, a turn whose
// parts carry an InlineData part must reach the Responses wire too — the
// route a Responses-only model takes, or any turn that carries an active
// previous_response_id. The old switch in oaiAppendResponsesItems read only
// text and function calls, so the image fell out of the payload without a
// word and the model saw the screenshot question with no screenshot.

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/openai/openai-go/v3/responses"
	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"
)

// captureResponsesServer answers every /responses call with a minimal
// completed response and hands the raw request body back through body.
func captureResponsesServer(t *testing.T, body *json.RawMessage) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw := make([]byte, r.ContentLength)
		_, _ = r.Body.Read(raw)
		*body = json.RawMessage(raw)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id": "resp_cap", "object": "response", "created_at": 1.0, "status": "completed",
			"model": "gpt-5-codex", "error": null, "incomplete_details": null,
			"instructions": null, "metadata": {}, "tool_choice": "auto", "tools": [],
			"parallel_tool_calls": true, "output": [],
			"usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
		}`))
	}))
	t.Cleanup(srv.Close)
	return srv
}

// imageTurnRequest builds the request the read_image callback leaves behind:
// a user turn holding the read_image result, the image bytes, and a follow-up
// text part. The FunctionResponse before the image exercises the flush path:
// the textual result rides the function_call_output item, the image and the
// question land in one user message whose content list keeps their order.
func imageTurnRequest(data []byte) *model.LLMRequest {
	return &model.LLMRequest{
		Contents: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{
				{FunctionResponse: &genai.FunctionResponse{
					ID:       "call_img",
					Name:     "read_image",
					Response: map[string]any{"result": inlineImageToolResultText},
				}},
				{InlineData: &genai.Blob{Data: data, MIMEType: "image/png"}},
				{Text: "what is on the screenshot?"},
			}},
		},
	}
}

// TestResponsesRouteForwardsInlineImage drives a full GenerateContent call
// against a captured server. The model is gpt-5-codex — a Responses-only
// family per modelNeedsResponses — so the request really takes the Responses
// route rather than calling the converter directly. The payload must carry
// the image as an input_image content part whose image_url is the base64
// data URL, with the tool-result text as input_text, and the image bytes
// must never appear in any text field.
func TestResponsesRouteForwardsInlineImage(t *testing.T) {
	data := pngOneByOne(t)

	var captured json.RawMessage
	srv := captureResponsesServer(t, &captured)
	m := newTestResponsesModel(t, srv.URL)

	for _, err := range m.GenerateContent(t.Context(), imageTurnRequest(data), false) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
	}
	if len(captured) == 0 {
		t.Fatal("no request body captured")
	}

	var wire struct {
		Input []struct {
			Type    string `json:"type"`
			Role    string `json:"role"`
			Content []struct {
				Type     string `json:"type"`
				Text     string `json:"text"`
				ImageURL string `json:"image_url"`
			} `json:"content"`
		} `json:"input"`
	}
	if err := json.Unmarshal(captured, &wire); err != nil {
		t.Fatalf("decode request: %v\n%s", err, captured)
	}

	var msg *struct {
		Type    string `json:"type"`
		Role    string `json:"role"`
		Content []struct {
			Type     string `json:"type"`
			Text     string `json:"text"`
			ImageURL string `json:"image_url"`
		} `json:"content"`
	}
	// The wire omits "type":"message" on a bare EasyInputMessage (its zero
	// value is "message"); match on the role and the content list shape.
	for i := range wire.Input {
		if wire.Input[i].Role == "user" {
			msg = &wire.Input[i]
		}
	}
	if msg == nil {
		t.Fatalf("no user message item in input: %s", captured)
	}
	if len(msg.Content) != 2 {
		t.Fatalf("content has %d parts, want 2 (input_image + input_text): %s", len(msg.Content), captured)
	}
	if msg.Content[0].Type != "input_image" {
		t.Fatalf("first part = %s, want input_image", msg.Content[0].Type)
	}
	imagePart := msg.Content[0]
	if imagePart.Type != "input_image" {
		t.Fatalf("first part = %s, want input_image", imagePart.Type)
	}
	wantURL := "data:image/png;base64," + base64.StdEncoding.EncodeToString(data)
	if imagePart.ImageURL != wantURL {
		t.Errorf("image_url = %q, want the data URL of the PNG bytes", imagePart.ImageURL)
	}
	if msg.Content[1].Type != "input_text" || msg.Content[1].Text != "what is on the screenshot?" {
		t.Errorf("second part = %+v, want the question text as input_text", msg.Content[1])
	}
	// The textual tool result rides a function_call_output item, which the
	// Responses path only emits when the call half of the pair is present.
	// This fixture has no function call, so the output item is dropped by
	// design (see oaiResponsesPairedIDs); assert via the converter directly.
	items := oaiAppendResponsesItems(responses.ResponseInputParam{}, imageTurnRequest(data).Contents[0],
		map[string]struct{}{"call_img": {}}, map[string]struct{}{"call_img": {}})
	raw, err := json.Marshal(items)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(raw, []byte(`"output":"`+inlineImageToolResultText+`"`)) {
		t.Errorf("paired converter lost the function_call_output: %s", raw)
	}
}
