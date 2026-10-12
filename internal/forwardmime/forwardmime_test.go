package forwardmime

import "testing"

func TestForwardableImageMIME(t *testing.T) {
	for _, tm := range []struct {
		mimeType string
		want     bool
	}{
		{"image/png", true},
		{"image/jpeg", true},
		{"image/gif", true},
		{"image/webp", true},
		{"image/bmp", false},
		{"image/svg+xml", false},
		{"text/plain", false},
		{"", false},
	} {
		if got := ForwardableImageMIME(tm.mimeType); got != tm.want {
			t.Errorf("ForwardableImageMIME(%q) = %v, want %v", tm.mimeType, got, tm.want)
		}
	}
}
