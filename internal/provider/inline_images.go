package provider

import (
	"encoding/base64"
	"strings"

	"google.golang.org/genai"
)

// fallbackImageMIMEType is the MIME type assumed for an InlineData part whose
// sniffer could not name a type. Every consumer here wants an image.
const fallbackImageMIMEType = "image/png"

// genaiInlineImages returns the base64 payloads of an InlineData part, in the
// order the parts appear. A part that is not an InlineData image yields
// nothing; callers keep handling text and function calls as before.
func genaiInlineImages(parts []*genai.Part) []genai.Blob {
	var images []genai.Blob
	for _, part := range parts {
		if part == nil || part.InlineData == nil || len(part.InlineData.Data) == 0 {
			continue
		}
		mimeType := part.InlineData.MIMEType
		if strings.TrimSpace(mimeType) == "" {
			mimeType = fallbackImageMIMEType
		}
		images = append(images, genai.Blob{
			Data:     part.InlineData.Data,
			MIMEType: mimeType,
		})
	}
	return images
}

// base64ImageDataURL renders image bytes as the data URL the Chat Completions
// image_url content part accepts.
func base64ImageDataURL(image genai.Blob) string {
	return "data:" + image.MIMEType + ";base64," + base64.StdEncoding.EncodeToString(image.Data)
}
