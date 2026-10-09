package provider

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/openai/openai-go/v3"
	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"
)

// pngOneByOne returns the bytes of a 1x1 PNG. Tests use real PNG bytes rather
// than arbitrary ones so the MIME-type expectations stay honest.
func pngOneByOne(t *testing.T) []byte {
	t.Helper()
	var buf bytes.Buffer
	img := image.NewRGBA(image.Rect(0, 0, 1, 1))
	img.Set(0, 0, color.RGBA{R: 255, A: 255})
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// pngTwoByOne returns the bytes of a 2x1 PNG — a second, byte-distinct image
// for tests that must tell two images apart.
func pngTwoByOne(t *testing.T) []byte {
	t.Helper()
	var buf bytes.Buffer
	img := image.NewRGBA(image.Rect(0, 0, 2, 1))
	img.Set(0, 0, color.RGBA{R: 255, A: 255})
	img.Set(1, 0, color.RGBA{G: 255, A: 255})
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// inlineImageToolResultText is the textual tool result the fixtures carry,
// asserted on by the conversion tests.
const inlineImageToolResultText = "shot taken"

// inlineImageTurn builds the turn the read_image callback leaves behind: a
// read_image FunctionResponse followed by the image bytes as InlineData, all
// on one user Content.
func inlineImageTurn(t *testing.T, data []byte, mimeType string, withText bool) *genai.Content {
	t.Helper()
	parts := []*genai.Part{
		{FunctionResponse: &genai.FunctionResponse{
			ID:       "call_img",
			Name:     "read_image",
			Response: map[string]any{"result": inlineImageToolResultText},
		}},
		{InlineData: &genai.Blob{Data: data, MIMEType: mimeType}},
	}
	if withText {
		parts = append(parts, &genai.Part{Text: "what is on the screenshot?"})
	}
	return &genai.Content{Role: "user", Parts: parts}
}

// TestGenaiTextImageSegmentsForwardableMIME pins the shared routing the
// adapters actually run: images come out in their parts' positions, an empty
// MIME type falls back to image/png, non-image parts ride as text (or not at
// all), and a MIME type the wire formats cannot carry is skipped rather than
// sent to be rejected by the provider.
func TestGenaiTextImageSegmentsForwardableMIME(t *testing.T) {
	data := pngOneByOne(t)

	t.Run("images ride in position, empty MIME falls back to png", func(t *testing.T) {
		parts := []*genai.Part{
			nil,
			{Text: "look:"},
			{InlineData: &genai.Blob{Data: data, MIMEType: "image/jpeg"}},
			{InlineData: &genai.Blob{Data: data}}, // empty MIME type
		}
		segments := genaiTextImageSegments(parts, true)
		if len(segments) != 3 {
			t.Fatalf("got %d segments, want 3 (text, jpeg image, png-fallback image)", len(segments))
		}
		if segments[0].text != "look:" {
			t.Errorf("segment 0 = %q, want the caption text", segments[0].text)
		}
		if segments[1].image.MIMEType != "image/jpeg" || string(segments[1].image.Data) != string(data) {
			t.Errorf("segment 1 = {%q %d bytes}, want {image/jpeg %d bytes}", segments[1].image.MIMEType, len(segments[1].image.Data), len(data))
		}
		if segments[2].image.MIMEType != "image/png" {
			t.Errorf("empty MIME type = %q, want image/png fallback", segments[2].image.MIMEType)
		}
		if hasImages(genaiTextImageSegments(nil, true)) {
			t.Error("nil parts must yield no images")
		}
	})

	// The MIME whitelist: the four types the wire formats can carry pass,
	// everything else — including the octet-stream a failed sniff leaves —
	// is skipped instead of earning a 400 from the provider.
	for _, mime := range []string{"image/png", "image/jpeg", "image/gif", "image/webp"} {
		if !forwardableImageMIME(mime) {
			t.Errorf("forwardableImageMIME(%q) = false, want true", mime)
		}
	}
	for _, mime := range []string{"application/octet-stream", "image/tiff", "text/plain", "", "IMAGE/PNG"} {
		if forwardableImageMIME(mime) {
			t.Errorf("forwardableImageMIME(%q) = true, want false", mime)
		}
	}

	t.Run("off-list MIME is skipped between its neighbours", func(t *testing.T) {
		// The skipped image must vanish without reordering what stays — the
		// same shape an adapter receives for a BMP that sniffed its way past
		// the image/* check. Captions on both sides stay in position with
		// the forwardable png riding between them.
		parts := []*genai.Part{
			{Text: "A"},
			{InlineData: &genai.Blob{Data: data, MIMEType: "application/octet-stream"}},
			{InlineData: &genai.Blob{Data: data, MIMEType: "image/png"}},
			{Text: "B"},
		}
		segments := genaiTextImageSegments(parts, true)
		if len(segments) != 3 {
			t.Fatalf("got %d segments, want A, png, B: %+v", len(segments), segments)
		}
		if segments[0].text != "A" || segments[2].text != "B" {
			t.Errorf("captions = %q, %q; want A then B in their original positions", segments[0].text, segments[2].text)
		}
		if segments[1].image.MIMEType != "image/png" || string(segments[1].image.Data) != string(data) {
			t.Errorf("image = {%q %d bytes}, want the png through untouched", segments[1].image.MIMEType, len(segments[1].image.Data))
		}
	})
}

// TestOaiContentsToMessagesInlineImage exercises the Chat Completions
// conversion end to end: the image lands in the content array as an
// image_url part with a base64 data URL, the text survives as a text part in
// its original position, and the image never leaks into any text field.
func TestOaiContentsToMessagesInlineImage(t *testing.T) {
	data := pngOneByOne(t)
	msgs, _ := oaiContentsToMessages(
		[]*genai.Content{inlineImageTurn(t, data, "image/png", true)}, nil)
	if len(msgs) != 1 {
		t.Fatalf("got %d messages, want 1 (user with image)", len(msgs))
	}
	user := msgs[0].OfUser
	if user == nil {
		t.Fatal("message is not a user message")
	}
	raw, err := json.Marshal(user)
	if err != nil {
		t.Fatal(err)
	}
	var wire struct {
		Content []map[string]any `json:"content"`
	}
	if err := json.Unmarshal(raw, &wire); err != nil {
		t.Fatal(err)
	}
	if len(wire.Content) != 2 {
		t.Fatalf("content has %d parts, want 2 (image + question): %s", len(wire.Content), raw)
	}
	imagePart := wire.Content[0]
	if imagePart["type"] != "image_url" {
		t.Fatalf("first part = %v, want type image_url", imagePart)
	}
	imageURL, _ := imagePart["image_url"].(map[string]any)
	wantURL := "data:image/png;base64," + base64.StdEncoding.EncodeToString(data)
	if imageURL["url"] != wantURL {
		t.Errorf("image url = %q, want %q", imageURL["url"], wantURL)
	}
	if wire.Content[1]["type"] != "text" || wire.Content[1]["text"] != "what is on the screenshot?" {
		t.Errorf("second part = %v, want the question text in its original position", wire.Content[1])
	}
}

// TestOaiContentsToMessagesInlineImageNoText pins the image-only turn: a bare
// function result that gained an image but has no text of its own still
// produces its user message, with the image as the only content part.
func TestOaiContentsToMessagesInlineImageNoText(t *testing.T) {
	data := pngOneByOne(t)
	msgs, _ := oaiContentsToMessages(
		[]*genai.Content{inlineImageTurn(t, data, "image/png", false)}, nil)
	if len(msgs) != 1 {
		t.Fatalf("got %d messages, want 1 (user with image)", len(msgs))
	}
	user := msgs[0].OfUser
	if user == nil {
		t.Fatal("message is not a user message")
	}
	raw, err := json.Marshal(user)
	if err != nil {
		t.Fatal(err)
	}
	var wire struct {
		Content []map[string]any `json:"content"`
	}
	if err := json.Unmarshal(raw, &wire); err != nil {
		t.Fatal(err)
	}
	if len(wire.Content) != 1 {
		t.Fatalf("content has %d parts, want 1: %s", len(wire.Content), raw)
	}
	if wire.Content[0]["type"] != "image_url" {
		t.Errorf("only part = %v, want image_url", wire.Content[0])
	}
}

// TestAntContentsToMessagesInlineImage exercises the Anthropic conversion end
// to end: the image lands as an image block with a base64 source, the text
// survives as a text block in its original position, and the image never
// leaks into any text field.
func TestAntContentsToMessagesInlineImage(t *testing.T) {
	data := pngOneByOne(t)
	msgs, _ := antContentsToMessages(
		[]*genai.Content{inlineImageTurn(t, data, "image/png", true)}, nil)
	if len(msgs) != 1 {
		t.Fatalf("got %d messages, want 1 (user with image)", len(msgs))
	}
	raw, err := json.Marshal(msgs[0])
	if err != nil {
		t.Fatal(err)
	}
	var wire struct {
		Role    string           `json:"role"`
		Content []map[string]any `json:"content"`
	}
	if err := json.Unmarshal(raw, &wire); err != nil {
		t.Fatal(err)
	}
	if len(wire.Content) != 2 {
		t.Fatalf("content has %d blocks, want 2 (image + question): %s", len(wire.Content), raw)
	}
	imageBlock := wire.Content[0]
	if imageBlock["type"] != "image" {
		t.Fatalf("first block = %v, want type image", imageBlock)
	}
	source, _ := imageBlock["source"].(map[string]any)
	if source["type"] != "base64" || source["media_type"] != "image/png" {
		t.Fatalf("image source = %v, want base64/image/png", source)
	}
	if source["data"] != base64.StdEncoding.EncodeToString(data) {
		t.Error("image source data does not match the PNG bytes")
	}
	if wire.Content[1]["type"] != "text" || wire.Content[1]["text"] != "what is on the screenshot?" {
		t.Errorf("second block = %v, want the question text in its original position", wire.Content[1])
	}
}

// TestAntContentsToMessagesInlineImageNoText pins the image-only turn for the
// Anthropic path.
func TestAntContentsToMessagesInlineImageNoText(t *testing.T) {
	data := pngOneByOne(t)
	msgs, _ := antContentsToMessages(
		[]*genai.Content{inlineImageTurn(t, data, "image/png", false)}, nil)
	if len(msgs) != 1 {
		t.Fatalf("got %d messages, want 1 (user with image)", len(msgs))
	}
	raw, err := json.Marshal(msgs[0])
	if err != nil {
		t.Fatal(err)
	}
	var wire struct {
		Content []map[string]any `json:"content"`
	}
	if err := json.Unmarshal(raw, &wire); err != nil {
		t.Fatal(err)
	}
	if len(wire.Content) != 1 || wire.Content[0]["type"] != "image" {
		t.Fatalf("blocks = %v, want a single image block", wire.Content)
	}
}

// TestPairToolCallWithImageTurn pins the full round trip the agent actually
// makes: model calls read_image, the user turn carries the result and the
// image. The tool result must stay textual while the image rides separately.
func TestPairToolCallWithImageTurn(t *testing.T) {
	data := pngOneByOne(t)
	b64Prefix := base64.StdEncoding.EncodeToString(data)[:20]

	call := genai.NewPartFromFunctionCall("read_image", map[string]any{"path": "/tmp/shot.png"})
	call.FunctionCall.ID = "call_img"

	t.Run("openai chat completions", func(t *testing.T) {
		msgs, _ := oaiContentsToMessages([]*genai.Content{
			{Role: "user", Parts: []*genai.Part{{Text: "look"}}},
			{Role: "model", Parts: []*genai.Part{call}},
			inlineImageTurn(t, data, "image/png", false),
		}, nil)
		// user, assistant(tool_calls), tool, user(image)
		if len(msgs) != 4 {
			t.Fatalf("got %d messages, want 4", len(msgs))
		}
		tool := msgs[2].OfTool
		if tool == nil {
			t.Fatal("third message is not a tool message")
		}
		toolRaw, err := json.Marshal(tool)
		if err != nil {
			t.Fatal(err)
		}
		var toolWire struct {
			Content string `json:"content"`
		}
		if err := json.Unmarshal(toolRaw, &toolWire); err != nil {
			t.Fatal(err)
		}
		if toolWire.Content != inlineImageToolResultText {
			t.Errorf("tool content = %q, want %q", toolWire.Content, inlineImageToolResultText)
		}
		if bytes.Contains(toolRaw, []byte(b64Prefix)) {
			t.Error("image base64 leaked into the tool message")
		}
		if msgs[3].OfUser == nil {
			t.Fatal("fourth message is not a user message")
		}
		userRaw, err := json.Marshal(msgs[3].OfUser)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Contains(userRaw, []byte("image_url")) || !bytes.Contains(userRaw, []byte(b64Prefix)) {
			t.Errorf("user payload lost the image: %s", userRaw)
		}
	})

	t.Run("anthropic", func(t *testing.T) {
		msgs, _ := antContentsToMessages([]*genai.Content{
			{Role: "user", Parts: []*genai.Part{{Text: "look"}}},
			{Role: "model", Parts: []*genai.Part{call}},
			inlineImageTurn(t, data, "image/png", false),
		}, nil)
		// user, assistant(tool_use), user(tool_result), user(image)
		if len(msgs) != 4 {
			t.Fatalf("got %d messages, want 4", len(msgs))
		}
		toolResultRaw, err := json.Marshal(msgs[2])
		if err != nil {
			t.Fatal(err)
		}
		if bytes.Contains(toolResultRaw, []byte(b64Prefix)) {
			t.Error("image base64 leaked into the tool_result message")
		}
		imageRaw, err := json.Marshal(msgs[3])
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Contains(imageRaw, []byte(`"type":"image"`)) || !bytes.Contains(imageRaw, []byte(b64Prefix)) {
			t.Errorf("fourth message lost the image: %s", imageRaw)
		}
	})
}

// TestOllamaContentsToMessagesInlineImage pins the ollama conversion
// (review finding A2): the image rides the message's Images field as raw
// base64 — no data-URL prefix, no content array — while the text stays in
// Content, and no image bytes ever leak into the text field.
func TestOllamaContentsToMessagesInlineImage(t *testing.T) {
	data := pngOneByOne(t)
	b64 := base64.StdEncoding.EncodeToString(data)

	msgs, _ := ollamaContentsToMessages(
		[]*genai.Content{inlineImageTurn(t, data, "image/png", true)}, nil)
	if len(msgs) != 1 {
		t.Fatalf("got %d messages, want 1", len(msgs))
	}
	raw, err := json.Marshal(msgs[0])
	if err != nil {
		t.Fatal(err)
	}
	var wire struct {
		Role    string   `json:"role"`
		Content string   `json:"content"`
		Images  []string `json:"images"`
	}
	if err := json.Unmarshal(raw, &wire); err != nil {
		t.Fatal(err)
	}
	if wire.Content != "what is on the screenshot?" {
		t.Errorf("content = %q, want only the question text", wire.Content)
	}
	if len(wire.Images) != 1 {
		t.Fatalf("images = %v, want exactly one raw-base64 entry: %s", wire.Images, raw)
	}
	if wire.Images[0] != b64 {
		t.Errorf("images[0] = %q, want the raw base64 of the PNG bytes (no data-URL prefix)", wire.Images[0])
	}
	if !strings.HasPrefix(wire.Images[0], "iVBORw0KGgo") {
		t.Errorf("images[0] does not start with the PNG base64 signature: %q", wire.Images[0])
	}
}

// interleavedTurn builds a user turn whose parts alternate text and images —
// [text A, img1, text B, img2] — the shape where part order carries meaning:
// each caption belongs to the picture before or after it. data1 and data2
// must be distinct byte slices so the test can tell the images apart.
func interleavedTurn(data1, data2 []byte) *genai.Content {
	return &genai.Content{Role: "user", Parts: []*genai.Part{
		{Text: "caption A"},
		{InlineData: &genai.Blob{Data: data1, MIMEType: "image/png"}},
		{Text: "caption B"},
		{InlineData: &genai.Blob{Data: data2, MIMEType: "image/png"}},
	}}
}

// TestInterleavedTextAndImageOrder pins review finding R3 for both
// image-aware adapters: the wire must walk the turn's parts in order —
// caption A, image 1, caption B, image 2 — not coalesce the text into one
// run and lose which caption belongs to which picture.
func TestInterleavedTextAndImageOrder(t *testing.T) {
	// Two distinct images: a 1x1 and a 2x1 PNG, so a swapped pair cannot
	// pass by accident. Deterministic encoders produce identical bytes for
	// identical inputs, so the shapes must differ.
	data1 := pngOneByOne(t)
	data2 := pngTwoByOne(t)
	if bytes.Equal(data1, data2) {
		t.Fatal("test premise broken: the two PNG fixtures are byte-identical")
	}

	t.Run("openai chat completions", func(t *testing.T) {
		msgs, _ := oaiContentsToMessages(
			[]*genai.Content{interleavedTurn(data1, data2)}, nil)
		if len(msgs) != 1 || msgs[0].OfUser == nil {
			t.Fatalf("want one user message, got %d", len(msgs))
		}
		raw, err := json.Marshal(msgs[0].OfUser)
		if err != nil {
			t.Fatal(err)
		}
		var wire struct {
			Content []map[string]any `json:"content"`
		}
		if err := json.Unmarshal(raw, &wire); err != nil {
			t.Fatal(err)
		}
		if len(wire.Content) != 4 {
			t.Fatalf("content has %d parts, want 4 interleaved: %s", len(wire.Content), raw)
		}
		wantURL1 := "data:image/png;base64," + base64.StdEncoding.EncodeToString(data1)
		wantURL2 := "data:image/png;base64," + base64.StdEncoding.EncodeToString(data2)
		if wire.Content[0]["type"] != "text" || wire.Content[0]["text"] != "caption A" {
			t.Errorf("part 0 = %v, want caption A", wire.Content[0])
		}
		if wire.Content[1]["type"] != "image_url" {
			t.Errorf("part 1 = %v, want image_url", wire.Content[1])
		}
		if got, _ := wire.Content[1]["image_url"].(map[string]any)["url"]; got != wantURL1 {
			t.Errorf("part 1 url = first image, got %v", got)
		}
		if wire.Content[2]["type"] != "text" || wire.Content[2]["text"] != "caption B" {
			t.Errorf("part 2 = %v, want caption B", wire.Content[2])
		}
		if wire.Content[3]["type"] != "image_url" {
			t.Errorf("part 3 = %v, want image_url", wire.Content[3])
		}
		if got, _ := wire.Content[3]["image_url"].(map[string]any)["url"]; got != wantURL2 {
			t.Errorf("part 3 url = second image, got %v", got)
		}
	})

	t.Run("anthropic", func(t *testing.T) {
		msgs, _ := antContentsToMessages(
			[]*genai.Content{interleavedTurn(data1, data2)}, nil)
		if len(msgs) != 1 {
			t.Fatalf("want one message, got %d", len(msgs))
		}
		raw, err := json.Marshal(msgs[0])
		if err != nil {
			t.Fatal(err)
		}
		var wire struct {
			Content []map[string]any `json:"content"`
		}
		if err := json.Unmarshal(raw, &wire); err != nil {
			t.Fatal(err)
		}
		if len(wire.Content) != 4 {
			t.Fatalf("content has %d blocks, want 4 interleaved: %s", len(wire.Content), raw)
		}
		wantB641 := base64.StdEncoding.EncodeToString(data1)
		wantB642 := base64.StdEncoding.EncodeToString(data2)
		if wire.Content[0]["type"] != "text" || wire.Content[0]["text"] != "caption A" {
			t.Errorf("block 0 = %v, want caption A", wire.Content[0])
		}
		if wire.Content[1]["type"] != "image" {
			t.Errorf("block 1 = %v, want image", wire.Content[1])
		}
		if got, _ := wire.Content[1]["source"].(map[string]any)["data"]; got != wantB641 {
			t.Errorf("block 1 data = first image, got %v", got)
		}
		if wire.Content[2]["type"] != "text" || wire.Content[2]["text"] != "caption B" {
			t.Errorf("block 2 = %v, want caption B", wire.Content[2])
		}
		if wire.Content[3]["type"] != "image" {
			t.Errorf("block 3 = %v, want image", wire.Content[3])
		}
		if got, _ := wire.Content[3]["source"].(map[string]any)["data"]; got != wantB642 {
			t.Errorf("block 3 data = second image, got %v", got)
		}
	})
}

// TestInterleavedTextOnlyTurnUnchanged pins the no-image fast path review
// finding R3 promised to keep byte-for-byte: a turn without an InlineData
// part still coalesces its text parts into one message exactly as before.
func TestInterleavedTextOnlyTurnUnchanged(t *testing.T) {
	contents := []*genai.Content{{Role: "user", Parts: []*genai.Part{
		{Text: "A"}, {Text: "B"},
	}}}

	msgs, _ := oaiContentsToMessages(contents, nil)
	if len(msgs) != 1 || msgs[0].OfUser == nil {
		t.Fatalf("want one user message, got %d", len(msgs))
	}
	raw, err := json.Marshal(msgs[0].OfUser)
	if err != nil {
		t.Fatal(err)
	}
	var oaiWire struct {
		Content string `json:"content"`
	}
	if err := json.Unmarshal(raw, &oaiWire); err != nil {
		t.Fatal(err)
	}
	if oaiWire.Content != "A\nB" {
		t.Errorf("openai content = %q, want %q (joined string, not an array)", oaiWire.Content, "A\nB")
	}

	antMsgs, _ := antContentsToMessages(contents, nil)
	if len(antMsgs) != 1 || len(antMsgs[0].Content) != 1 {
		t.Fatalf("want one message with one text block, got %+v", antMsgs)
	}
	antRaw, err := json.Marshal(antMsgs[0].Content[0])
	if err != nil {
		t.Fatal(err)
	}
	if string(antRaw) != `{"text":"A\nB","type":"text"}` {
		t.Errorf("anthropic block = %s, want a single joined text block", antRaw)
	}
}

// TestAssistantTurnImageDropped pins review finding R4 for both adapters: a
// model turn whose parts include an image (a shape that should not occur —
// the injector only touches user turns — but must degrade predictably) keeps
// its text on the assistant message and never leaks the image bytes.
func TestAssistantTurnImageDropped(t *testing.T) {
	data := pngOneByOne(t)
	b64Prefix := base64.StdEncoding.EncodeToString(data)[:20]
	assistantTurn := &genai.Content{Role: "model", Parts: []*genai.Part{
		{Text: "looking at it"},
		{InlineData: &genai.Blob{Data: data, MIMEType: "image/png"}},
	}}

	t.Run("openai chat completions", func(t *testing.T) {
		msgs, _ := oaiContentsToMessages([]*genai.Content{assistantTurn}, nil)
		if len(msgs) != 1 || msgs[0].OfAssistant == nil {
			t.Fatalf("want one assistant message, got %d", len(msgs))
		}
		raw, err := json.Marshal(msgs[0])
		if err != nil {
			t.Fatal(err)
		}
		var wire struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		}
		if err := json.Unmarshal(raw, &wire); err != nil {
			t.Fatal(err)
		}
		if wire.Role != "assistant" || wire.Content != "looking at it" {
			t.Errorf("message = %s, want the assistant text only", raw)
		}
		if bytes.Contains(raw, []byte(b64Prefix)) || bytes.Contains(raw, []byte("image_url")) {
			t.Errorf("image leaked into the assistant message: %s", raw)
		}
	})

	t.Run("anthropic", func(t *testing.T) {
		msgs, _ := antContentsToMessages([]*genai.Content{assistantTurn}, nil)
		if len(msgs) != 1 || msgs[0].Role != anthropic.MessageParamRoleAssistant {
			t.Fatalf("want one assistant message, got %+v", msgs)
		}
		raw, err := json.Marshal(msgs[0])
		if err != nil {
			t.Fatal(err)
		}
		var wire struct {
			Content []map[string]any `json:"content"`
		}
		if err := json.Unmarshal(raw, &wire); err != nil {
			t.Fatal(err)
		}
		if len(wire.Content) != 1 || wire.Content[0]["type"] != "text" || wire.Content[0]["text"] != "looking at it" {
			t.Errorf("blocks = %v, want the assistant text only", wire.Content)
		}
		if bytes.Contains(raw, []byte(b64Prefix)) || bytes.Contains(raw, []byte(`"image"`)) {
			t.Errorf("image leaked into the assistant message: %s", raw)
		}
	})
}

// Compile-time guard that the wire types used above keep their shape.
var (
	_ = openai.ImageContentPart
	_ = anthropic.NewImageBlockBase64
)

// mistralImageWireMessage is one Chat Completions message as it lands on the
// wire: the content array's image parts carry image_url as an object with a
// url member.
type mistralImageWireMessage struct {
	Role    string `json:"role"`
	Content []struct {
		Type     string `json:"type"`
		Text     string `json:"text"`
		ImageURL struct {
			URL string `json:"url"`
		} `json:"image_url"`
	} `json:"content"`
}

// TestMistralForwardsInlineImageOnWire pins review finding A1 end to end:
// mistralModel rides oaiContentsToMessages, so an image-bearing turn must
// reach Mistral's Chat Completions endpoint as a user message whose content
// array carries the question text and an image_url part with a base64 data
// URL — the form Mistral's vision models document for base64 images. The
// mock server answers like Mistral does and records the request body.
func TestMistralForwardsInlineImageOnWire(t *testing.T) {
	data := pngOneByOne(t)

	var captured json.RawMessage
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw := make([]byte, r.ContentLength)
		_, _ = r.Body.Read(raw)
		captured = json.RawMessage(raw)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id": "chat-img", "object": "chat.completion", "model": "mistral-small-latest",
			"choices": [{"index": 0, "message": {"role": "assistant", "content": "a cat"},
				"finish_reason": "stop"}],
			"usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}
		}`))
	}))
	defer srv.Close()

	m, err := NewMistral(context.Background(), "mistral-small-latest", "test-key", srv.URL, "", nil)
	if err != nil {
		t.Fatalf("NewMistral: %v", err)
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{inlineImageTurn(t, data, "image/png", true)},
	}
	for _, err := range m.GenerateContent(context.Background(), req, false) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
	}
	if len(captured) == 0 {
		t.Fatal("no request body captured")
	}

	// image_url arrives as an object ({\"url\": ...}) — the Chat Completions
	// shape openai-go marshals.
	var wire struct {
		Messages []mistralImageWireMessage `json:"messages"`
	}
	if err := json.Unmarshal(captured, &wire); err != nil {
		t.Fatalf("decode request: %v\n%s", err, captured)
	}
	var user *mistralImageWireMessage
	for i := range wire.Messages {
		if wire.Messages[i].Role == "user" {
			user = &wire.Messages[i]
		}
	}
	if user == nil {
		t.Fatalf("no user message in payload: %s", captured)
	}
	if len(user.Content) != 2 {
		t.Fatalf("content has %d parts, want 2 (image_url + text): %s", len(user.Content), captured)
	}
	// The fixture orders the image before the trailing question; the payload
	// must keep that order (see TestInterleavedTextAndImageOrder).
	if user.Content[0].Type != "image_url" {
		t.Fatalf("first part = %+v, want type image_url", user.Content[0])
	}
	wantURL := "data:image/png;base64," + base64.StdEncoding.EncodeToString(data)
	if user.Content[0].ImageURL.URL != wantURL {
		t.Errorf("image_url.url = %q, want the PNG data URL", user.Content[0].ImageURL.URL)
	}
	if user.Content[1].Type != "text" || user.Content[1].Text != "what is on the screenshot?" {
		t.Errorf("second part = %+v, want the question text", user.Content[1])
	}
}
