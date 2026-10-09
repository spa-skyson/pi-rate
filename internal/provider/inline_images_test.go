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
