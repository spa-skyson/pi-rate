package provider

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/openai/openai-go/v3"
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

// TestGenaiInlineImages pins the shared extractor: images come out in order,
// an empty MIME type falls back to image/png, non-image parts yield nothing,
// and a MIME type the wire formats cannot carry is skipped rather than sent
// to be rejected by the provider.
func TestGenaiInlineImages(t *testing.T) {
	data := pngOneByOne(t)
	parts := []*genai.Part{
		nil,
		{Text: "not an image"},
		{InlineData: &genai.Blob{Data: data, MIMEType: "image/jpeg"}},
		{InlineData: &genai.Blob{}},
		{InlineData: &genai.Blob{Data: data}},
		{FunctionCall: &genai.FunctionCall{ID: "c", Name: "f"}},
	}
	images := genaiInlineImages(parts)
	if len(images) != 2 {
		t.Fatalf("got %d images, want 2", len(images))
	}
	if images[0].MIMEType != "image/jpeg" || string(images[0].Data) != string(data) {
		t.Errorf("first image = {%q %d bytes}, want {image/jpeg %d bytes}", images[0].MIMEType, len(images[0].Data), len(data))
	}
	if images[1].MIMEType != "image/png" {
		t.Errorf("empty MIME type = %q, want image/png fallback", images[1].MIMEType)
	}
	if len(genaiInlineImages(nil)) != 0 {
		t.Error("nil parts must yield no images")
	}

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
	offList := genaiInlineImages([]*genai.Part{
		{InlineData: &genai.Blob{Data: data, MIMEType: "application/octet-stream"}},
	})
	if len(offList) != 0 {
		t.Errorf("off-list MIME yielded %d images, want 0", len(offList))
	}
}

// TestOaiContentsToMessagesInlineImage exercises the Chat Completions
// conversion end to end: the image lands in the content array as an
// image_url part with a base64 data URL, the text survives as text parts,
// and the image never leaks into any text field.
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
		t.Fatalf("content has %d parts, want 2 (question + image): %s", len(wire.Content), raw)
	}
	if wire.Content[0]["type"] != "text" || wire.Content[0]["text"] != "what is on the screenshot?" {
		t.Errorf("first part = %v, want the question text", wire.Content[0])
	}
	imagePart := wire.Content[1]
	if imagePart["type"] != "image_url" {
		t.Fatalf("third part = %v, want type image_url", imagePart)
	}
	imageURL, _ := imagePart["image_url"].(map[string]any)
	wantURL := "data:image/png;base64," + base64.StdEncoding.EncodeToString(data)
	if imageURL["url"] != wantURL {
		t.Errorf("image url = %q, want %q", imageURL["url"], wantURL)
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
// survives as text blocks, and the image never leaks into any text field.
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
		t.Fatalf("content has %d blocks, want 2 (question + image): %s", len(wire.Content), raw)
	}
	if wire.Content[0]["type"] != "text" || wire.Content[0]["text"] != "what is on the screenshot?" {
		t.Errorf("first block = %v, want the question text", wire.Content[0])
	}
	imageBlock := wire.Content[1]
	if imageBlock["type"] != "image" {
		t.Fatalf("third block = %v, want type image", imageBlock)
	}
	source, _ := imageBlock["source"].(map[string]any)
	if source["type"] != "base64" || source["media_type"] != "image/png" {
		t.Fatalf("image source = %v, want base64/image/png", source)
	}
	if source["data"] != base64.StdEncoding.EncodeToString(data) {
		t.Error("image source data does not match the PNG bytes")
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

// TestSharedConvertersKeepTheirWireWithoutGate pins the status quo for the
// converter whose gate stays closed: the ollama conversion still drops an
// InlineData part, and it never reaches its text.
func TestSharedConvertersKeepTheirWireWithoutGate(t *testing.T) {
	data := pngOneByOne(t)
	b64Prefix := base64.StdEncoding.EncodeToString(data)[:20]

	msgs, _ := ollamaContentsToMessages(
		[]*genai.Content{inlineImageTurn(t, data, "image/png", true)}, nil)
	raw, err := json.Marshal(msgs)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(raw, []byte("base64,")) || bytes.Contains(raw, []byte(b64Prefix)) {
		t.Errorf("image data leaked into the ollama payload: %s", raw)
	}
}

// Compile-time guard that the wire types used above keep their shape.
var (
	_ = openai.ImageContentPart
	_ = anthropic.NewImageBlockBase64
)
