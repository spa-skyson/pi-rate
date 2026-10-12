// Package forwardmime holds the set of image MIME types the provider wire
// formats can carry. It is a leaf package — standard library only — so the
// three consumers that must answer identically can all import it without
// dragging each other into their build graphs.
package forwardmime

// forwardableImageMIMEs is the set of image MIME types the provider wire
// formats can carry: the Anthropic base64 source spells the media types out,
// and the OpenAI image_url endpoint rejects the rest. One set, declared once,
// because three consumers must answer identically — the read_image tool that
// produces the bytes (internal/tools), the injector that re-reads them
// (internal/extension), and the adapters that put them on the wire
// (internal/provider).
var forwardableImageMIMEs = map[string]bool{
	"image/png":  true,
	"image/jpeg": true,
	"image/gif":  true,
	"image/webp": true,
}

// ForwardableImageMIME reports whether mimeType is one the provider wire
// formats can carry. read_image produces image/* types outside this set (the
// byte sniffer names BMP image/bmp), so callers use this to skip — and warn
// about — an image the model would never see rather than send it to be
// rejected with a 400 that aborts the whole turn.
func ForwardableImageMIME(mimeType string) bool {
	return forwardableImageMIMEs[mimeType]
}
