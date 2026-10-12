package extension

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"

	"github.com/spa-skyson/pi-rate/internal/notice"
	"github.com/spa-skyson/pi-rate/internal/tools"
)

// writePNG writes a small 2x1 PNG to path.
func writePNG(t *testing.T, path string) {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, 2, 1))
	img.Set(0, 0, color.RGBA{R: 255, A: 255})
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := png.Encode(f, img); err != nil {
		t.Fatal(err)
	}
}

// newTestSandbox returns a sandbox rooted at a fresh temp dir.
func newTestSandbox(t *testing.T) *tools.Sandbox {
	t.Helper()
	sb, err := tools.NewSandbox(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { sb.Close() })
	return sb
}

// imageReadCallbackRequest builds the request whose user turn carries one
// read_image FunctionResponse pointing at path.
func imageReadCallbackRequest(path string) *model.LLMRequest {
	return &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					{
						FunctionResponse: &genai.FunctionResponse{
							Name: "read_image",
							Response: map[string]any{
								"path": path,
							},
						},
					},
				},
			},
		},
	}
}

// TestReadImagePartsRechecksFile pins the security re-check: the injector
// re-reads the file on every step, long after the read_image tool applied
// its own size and MIME checks, so a file that changed between the tool call
// and the injection must not ride into the prompt.
func TestReadImagePartsRechecksFile(t *testing.T) {
	t.Run("grown past the cap is skipped with a warning", func(t *testing.T) {
		sb := newTestSandbox(t)
		path := filepath.Join(sb.Dir(), "shot.png")
		writePNG(t, path)

		// One warning per path for the session.
		var notices []string
		prev := notice.SetSink(func(msg string) { notices = append(notices, msg) })
		t.Cleanup(func() { notice.SetSink(prev) })

		cb := BuildReadImageCallback(sb, "gemini")
		req := imageReadCallbackRequest(path)
		if _, err := cb(&mockReadonlyContext{}, req); err != nil {
			t.Fatalf("callback: %v", err)
		}
		if len(req.Contents[0].Parts) != 2 {
			t.Fatal("precondition failed: the small image was not injected")
		}

		// The file grows past MaxImageBytes between the tool call and the
		// next step's re-read — the race this check exists for.
		if err := sb.WriteFile(path, bytes.Repeat([]byte{0x00}, tools.MaxImageBytes+1), 0o644); err != nil {
			t.Fatalf("grow file: %v", err)
		}

		req2 := imageReadCallbackRequest(path)
		if _, err := cb(&mockReadonlyContext{}, req2); err != nil {
			t.Fatalf("callback: %v", err)
		}
		if len(req2.Contents[0].Parts) != 1 {
			t.Fatalf("oversized file was injected: %d extra parts", len(req2.Contents[0].Parts)-1)
		}
		if len(notices) != 1 {
			t.Fatalf("got %d notices (%v), want exactly one for the path", len(notices), notices)
		}
		if !strings.Contains(notices[0], "image limit") {
			t.Errorf("notice = %q, want it to name the size limit", notices[0])
		}
	})

	t.Run("file that no longer sniffs as an image is skipped with a warning", func(t *testing.T) {
		sb := newTestSandbox(t)
		path := filepath.Join(sb.Dir(), "shot.png")
		writePNG(t, path)

		var notices []string
		prev := notice.SetSink(func(msg string) { notices = append(notices, msg) })
		t.Cleanup(func() { notice.SetSink(prev) })

		cb := BuildReadImageCallback(sb, "gemini")

		// The image is replaced by text bytes between steps: the extension
		// fallback would call it image/png, but the sniff sees text/plain.
		if err := sb.WriteFile(path, []byte("definitely not a png"), 0o644); err != nil {
			t.Fatalf("rewrite file: %v", err)
		}

		req := imageReadCallbackRequest(path)
		if _, err := cb(&mockReadonlyContext{}, req); err != nil {
			t.Fatalf("callback: %v", err)
		}
		if len(req.Contents[0].Parts) != 1 {
			t.Fatalf("non-image file was injected: %d extra parts", len(req.Contents[0].Parts)-1)
		}
		if len(notices) != 1 || !strings.Contains(notices[0], "no longer sniffs as an image") {
			t.Fatalf("notices = %v, want one no-longer-an-image warning", notices)
		}
	})

	t.Run("over-cap file is detected through the bounded read, not a full allocation", func(t *testing.T) {
		sb := newTestSandbox(t)
		path := filepath.Join(sb.Dir(), "shot.png")
		writePNG(t, path)

		var notices []string
		prev := notice.SetSink(func(msg string) { notices = append(notices, msg) })
		t.Cleanup(func() { notice.SetSink(prev) })

		cb := BuildReadImageCallback(sb, "gemini")

		// The file never goes through sb.ReadFile here: readBoundedImage stops
		// at MaxImageBytes+1 and reports the cap by name. Drive the same
		// helper the injector uses so the bounded path itself is covered.
		_, err := readBoundedImage(sb, path)
		if err != nil {
			t.Fatalf("bounded read of an in-cap file: %v", err)
		}

		if err := sb.WriteFile(path, bytes.Repeat([]byte{0x00}, tools.MaxImageBytes+1), 0o644); err != nil {
			t.Fatalf("grow file: %v", err)
		}
		_, err = readBoundedImage(sb, path)
		if err == nil {
			t.Fatal("bounded read of an over-cap file returned no error")
		}
		if !strings.Contains(err.Error(), "image limit") {
			t.Errorf("error = %v, want it to name the size limit", err)
		}

		// End to end: the injector turns that error into exactly one warning
		// and injects nothing.
		req := imageReadCallbackRequest(path)
		if _, err := cb(&mockReadonlyContext{}, req); err != nil {
			t.Fatalf("callback: %v", err)
		}
		if len(req.Contents[0].Parts) != 1 {
			t.Fatalf("oversized file was injected: %d extra parts", len(req.Contents[0].Parts)-1)
		}
		if len(notices) != 1 || !strings.Contains(notices[0], "image limit") {
			t.Fatalf("notices = %v, want one image-limit warning", notices)
		}
	})

	t.Run("unchanged image still injects after a rejected sibling", func(t *testing.T) {
		sb := newTestSandbox(t)
		good := filepath.Join(sb.Dir(), "good.png")
		bad := filepath.Join(sb.Dir(), "bad.png")
		writePNG(t, good)
		if err := sb.WriteFile(bad, []byte("not an image at all"), 0o644); err != nil {
			t.Fatalf("write bad: %v", err)
		}

		prev := notice.SetSink(func(string) {})
		t.Cleanup(func() { notice.SetSink(prev) })

		cb := BuildReadImageCallback(sb, "gemini")
		req := &model.LLMRequest{Contents: []*genai.Content{{
			Role: "user",
			Parts: []*genai.Part{
				{FunctionResponse: &genai.FunctionResponse{Name: "read_image", Response: map[string]any{"path": bad}}},
				{FunctionResponse: &genai.FunctionResponse{Name: "read_image", Response: map[string]any{"path": good}}},
			},
		}}}
		if _, err := cb(&mockReadonlyContext{}, req); err != nil {
			t.Fatalf("callback: %v", err)
		}
		parts := req.Contents[0].Parts
		if len(parts) != 3 {
			t.Fatalf("got %d parts, want 2 responses + 1 injected image for the good path", len(parts))
		}
		if parts[2].InlineData == nil || parts[2].InlineData.MIMEType != "image/png" {
			t.Errorf("injected part = %+v, want the good image", parts[2])
		}
	})
}

// TestReadImagePartsForwardableMIME pins the forwardable-MIME gate at the
// injector: a BMP passes the image/* sniff and then must stop with a warning
// — the provider wire formats cannot carry it, so injecting it would earn a
// 400 that aborts the whole turn — while a PNG injects as before.
func TestReadImagePartsForwardableMIME(t *testing.T) {
	t.Run("bmp is skipped with a forwardability warning", func(t *testing.T) {
		sb := newTestSandbox(t)
		path := filepath.Join(sb.Dir(), "shot.bmp")
		// "BM" + byte-sized header fields: enough for http.DetectContentType
		// to name image/bmp, which is the exact case the warning exists for.
		bmp := append([]byte("BM"), make([]byte, 26)...)
		if got := tools.DetectImageMIMEType(bmp); got != "image/bmp" {
			t.Fatalf("precondition failed: fixture sniffs as %q, want image/bmp", got)
		}
		if err := sb.WriteFile(path, bmp, 0o644); err != nil {
			t.Fatalf("write bmp: %v", err)
		}

		var notices []string
		prev := notice.SetSink(func(msg string) { notices = append(notices, msg) })
		t.Cleanup(func() { notice.SetSink(prev) })

		cb := BuildReadImageCallback(sb, "gemini")
		req := imageReadCallbackRequest(path)
		if _, err := cb(&mockReadonlyContext{}, req); err != nil {
			t.Fatalf("callback: %v", err)
		}
		if len(req.Contents[0].Parts) != 1 {
			t.Fatalf("unforwardable image was injected: %d extra parts", len(req.Contents[0].Parts)-1)
		}
		if len(notices) != 1 {
			t.Fatalf("got %d notices (%v), want exactly one", len(notices), notices)
		}
		if !strings.Contains(notices[0], "image/bmp") || !strings.Contains(notices[0], "will not see it") {
			t.Errorf("notice = %q, want it to name the format and say the model will not see it", notices[0])
		}
	})

	t.Run("png still injects", func(t *testing.T) {
		sb := newTestSandbox(t)
		path := filepath.Join(sb.Dir(), "shot.png")
		writePNG(t, path)

		prev := notice.SetSink(func(string) {})
		t.Cleanup(func() { notice.SetSink(prev) })

		cb := BuildReadImageCallback(sb, "gemini")
		req := imageReadCallbackRequest(path)
		if _, err := cb(&mockReadonlyContext{}, req); err != nil {
			t.Fatalf("callback: %v", err)
		}
		parts := req.Contents[0].Parts
		if len(parts) != 2 || parts[1].InlineData == nil || parts[1].InlineData.MIMEType != "image/png" {
			t.Fatalf("parts = %d, want the FunctionResponse plus the injected image/png", len(parts))
		}
	})
}

func TestBuildReadImageCallback(t *testing.T) {
	sb := newTestSandbox(t)
	path := filepath.Join(sb.Dir(), "shot.png")
	writePNG(t, path)

	t.Run("injects inline data part", func(t *testing.T) {
		cb := BuildReadImageCallback(sb, "gemini")
		// Build a request whose user-turn Content contains a read_image
		// FunctionResponse referencing the screenshot path.
		req := &model.LLMRequest{
			Contents: []*genai.Content{
				{
					Role: "user",
					Parts: []*genai.Part{
						{
							FunctionResponse: &genai.FunctionResponse{
								Name: "read_image",
								Response: map[string]any{
									"path": path,
								},
							},
						},
					},
				},
			},
		}
		ctx := &mockReadonlyContext{}
		resp, err := cb(ctx, req)
		if err != nil {
			t.Fatalf("callback: %v", err)
		}
		if resp != nil {
			t.Error("expected nil response (callback must not short-circuit the model)")
		}
		parts := req.Contents[0].Parts
		if len(parts) != 2 {
			t.Fatalf("expected 2 parts (FunctionResponse + InlineData), got %d", len(parts))
		}
		imgPart := parts[1]
		if imgPart.InlineData == nil {
			t.Fatal("expected an InlineData part to be injected")
		}
		if imgPart.InlineData.MIMEType != "image/png" {
			t.Errorf("expected image/png mime, got %q", imgPart.InlineData.MIMEType)
		}
		if len(imgPart.InlineData.Data) == 0 {
			t.Error("expected non-empty image data")
		}
	})

	t.Run("ignores non-read_image tools", func(t *testing.T) {
		cb := BuildReadImageCallback(sb, "gemini")
		req := &model.LLMRequest{
			Contents: []*genai.Content{
				{
					Role: "user",
					Parts: []*genai.Part{
						{
							FunctionResponse: &genai.FunctionResponse{
								Name:     "bash",
								Response: map[string]any{"output": "hi"},
							},
						},
					},
				},
			},
		}
		ctx := &mockReadonlyContext{}
		if _, err := cb(ctx, req); err != nil {
			t.Fatalf("callback: %v", err)
		}
		if len(req.Contents[0].Parts) != 1 {
			t.Error("expected no parts to be added for a non-read_image tool")
		}
	})

	t.Run("missing path is skipped", func(t *testing.T) {
		cb := BuildReadImageCallback(sb, "gemini")
		req := &model.LLMRequest{
			Contents: []*genai.Content{
				{
					Role: "user",
					Parts: []*genai.Part{
						{
							FunctionResponse: &genai.FunctionResponse{
								Name:     "read_image",
								Response: map[string]any{},
							},
						},
					},
				},
			},
		}
		ctx := &mockReadonlyContext{}
		if _, err := cb(ctx, req); err != nil {
			t.Fatalf("callback: %v", err)
		}
		if len(req.Contents[0].Parts) != 1 {
			t.Error("expected no part added when path is missing")
		}
	})

	t.Run("nil sandbox is a no-op", func(t *testing.T) {
		cb := BuildReadImageCallback(nil, "gemini")
		req := &model.LLMRequest{
			Contents: []*genai.Content{
				{Role: "user", Parts: []*genai.Part{{Text: "hi"}}},
			},
		}
		if _, err := cb(&mockReadonlyContext{}, req); err != nil {
			t.Fatalf("callback: %v", err)
		}
		if len(req.Contents[0].Parts) != 1 {
			t.Error("expected no change with nil sandbox")
		}
	})

	t.Run("injects real mascot png", func(t *testing.T) {
		// Resolve repo root from this test file: .../pi-go/internal/extension
		repoRoot, err := filepath.Abs(filepath.Join("..", ".."))
		if err != nil {
			t.Fatalf("resolve repo root: %v", err)
		}
		relPath := filepath.Join("docs", "screen", "pi-go-mascot.png")
		absPath := filepath.Join(repoRoot, relPath)
		info, err := os.Stat(absPath)
		if err != nil {
			t.Skipf("mascot screenshot not present at %s (skipping): %v", absPath, err)
		}
		rawBytes, err := os.ReadFile(absPath)
		if err != nil {
			t.Fatalf("read mascot: %v", err)
		}

		// Sandbox rooted at the repo root so the relative path the agent would
		// pass to read_image resolves cleanly.
		sb, err := tools.NewSandbox(repoRoot)
		if err != nil {
			t.Fatalf("NewSandbox(%s): %v", repoRoot, err)
		}
		t.Cleanup(func() { sb.Close() })

		cb := BuildReadImageCallback(sb, "gemini")
		req := &model.LLMRequest{
			Contents: []*genai.Content{
				{
					Role: "user",
					Parts: []*genai.Part{
						{
							FunctionResponse: &genai.FunctionResponse{
								Name:     "read_image",
								Response: map[string]any{"path": relPath},
							},
						},
					},
				},
			},
		}
		if _, err := cb(&mockReadonlyContext{}, req); err != nil {
			t.Fatalf("callback: %v", err)
		}

		parts := req.Contents[0].Parts
		if len(parts) != 2 {
			t.Fatalf("expected 2 parts (FunctionResponse + InlineData), got %d", len(parts))
		}
		img := parts[1].InlineData
		if img == nil {
			t.Fatal("expected InlineData part to be injected for real mascot")
		}
		if img.MIMEType != "image/png" {
			t.Errorf("MIMEType = %q, want image/png", img.MIMEType)
		}
		if int64(len(img.Data)) != info.Size() {
			t.Errorf("InlineData size = %d, want %d (file on disk)", len(img.Data), info.Size())
		}
		if !bytes.Equal(img.Data, rawBytes) {
			t.Error("InlineData bytes do not match the on-disk mascot file (callback must re-read, not synthesize)")
		}
	})
}

func TestDetectImageMIMEType(t *testing.T) {
	png := pngData(t)

	t.Run("sniffs real png bytes", func(t *testing.T) {
		if got := detectImageMIMEType(png, "shot.png"); got != "image/png" {
			t.Errorf("detectImageMIMEType = %q, want image/png", got)
		}
	})

	t.Run("falls back to extension when sniff is ambiguous", func(t *testing.T) {
		garbage := []byte{0xff, 0xfe, 0x01} // sniffs as application/octet-stream
		cases := []struct {
			path string
			want string
		}{
			{"screenshot.png", "image/png"},
			{"shot.jpg", "image/jpeg"},
			{"shot.jpeg", "image/jpeg"},
			{"anim.gif", "image/gif"},
			{"frame.webp", "image/webp"},
			{"no_extension", "application/octet-stream"},
			{"uppercase.PNG", "application/octet-stream"}, // case-sensitive fallback
		}
		for _, tc := range cases {
			if got := detectImageMIMEType(garbage, tc.path); got != tc.want {
				t.Errorf("detectImageMIMEType(garbage, %q) = %q, want %q", tc.path, got, tc.want)
			}
		}
	})
}

func TestHasExt(t *testing.T) {
	cases := []struct {
		path string
		ext  string
		want bool
	}{
		{"shot.png", ".png", true},
		{"dir/shot.png", ".png", true},
		{"shot.jpg", ".jpg", true},
		{"shot.png", ".jpg", false},
		{"png", ".png", false},       // path shorter than extension
		{"shot.PNG", ".png", false},  // case-sensitive
		{"shot.pngx", ".png", false}, // suffix must match exactly
	}
	for _, tc := range cases {
		if got := hasExt(tc.path, tc.ext); got != tc.want {
			t.Errorf("hasExt(%q, %q) = %v, want %v", tc.path, tc.ext, got, tc.want)
		}
	}
}

// pngData returns the bytes of a small 2x1 PNG for sniffing tests.
func pngData(t *testing.T) []byte {
	t.Helper()
	var buf bytes.Buffer
	img := image.NewRGBA(image.Rect(0, 0, 2, 1))
	img.Set(0, 0, color.RGBA{R: 255, A: 255})
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// Ensure mockReadonlyContext satisfies agent.Context for the callback signature.
var _ agent.Context = (*mockReadonlyContext)(nil)
