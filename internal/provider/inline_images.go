package provider

import (
	"encoding/base64"
	"strings"

	"google.golang.org/genai"
)

// fallbackImageMIMEType is the MIME type assumed for an InlineData part whose
// sniffer could not name a type. Every consumer here wants an image.
const fallbackImageMIMEType = "image/png"

// forwardableImageMIME reports whether a MIME type is one the image-bearing
// wire formats here accept. Every adapter in this package can only express
// image/png, image/jpeg, image/gif and image/webp — the Anthropic base64
// source spells the media types out, and the OpenAI image_url endpoint
// rejects the rest — so anything else is skipped rather than sent to be
// rejected: the read_image tool only produces image/* types, so an off-list
// MIME here means the sniff could not name the format, and a 400 from the
// provider would abort the whole turn for bytes the model cannot see anyway.
func forwardableImageMIME(mimeType string) bool {
	switch mimeType {
	case "image/png", "image/jpeg", "image/gif", "image/webp":
		return true
	}
	return false
}

// inlineImageOf reduces one InlineData part to the image blob the adapters
// put on the wire: a blank MIME type falls back to image/png, and a type the
// wire formats cannot carry yields nothing.
func inlineImageOf(blob *genai.Blob) (genai.Blob, bool) {
	mimeType := blob.MIMEType
	if strings.TrimSpace(mimeType) == "" {
		mimeType = fallbackImageMIMEType
	}
	if !forwardableImageMIME(mimeType) {
		return genai.Blob{}, false
	}
	return genai.Blob{Data: blob.Data, MIMEType: mimeType}, true
}

// genaiInlineImages returns the image payloads of an InlineData part, in the
// order the parts appear. A part that is not an InlineData image — or one
// whose MIME type the wire formats here cannot carry — yields nothing;
// callers keep handling text and function calls as before.
func genaiInlineImages(parts []*genai.Part) []genai.Blob {
	var images []genai.Blob
	for _, part := range parts {
		if part == nil || part.InlineData == nil || len(part.InlineData.Data) == 0 {
			continue
		}
		if image, ok := inlineImageOf(part.InlineData); ok {
			images = append(images, image)
		}
	}
	return images
}

// genaiTextImageSegments splits one content into the alternating text and
// image segments its parts describe, preserving the order the parts appear.
// Adjacent text parts coalesce into one segment — what separate parts said as
// one run keeps riding as one run — while each forwardable image becomes its
// own segment, so a caption stays attached to the picture it captions:
// [A, img1, B, img2] comes back as [A, img1, B, img2], not [A B, img1, img2].
//
// An InlineData part whose MIME type the wire cannot carry is skipped, and an
// assistant turn yields no images at all: callers only route image-bearing
// user turns here, and every wire drops — rather than reorders — an image it
// cannot place.
func genaiTextImageSegments(parts []*genai.Part, allowImages bool) []genaiBlobOrText {
	if !allowImages {
		return nil
	}
	var segments []genaiBlobOrText
	for _, part := range parts {
		switch {
		case part == nil:
		case part.Text != "":
			n := len(segments) - 1
			if n >= 0 && segments[n].text != "" {
				segments[n].text += "\n" + part.Text
			} else {
				segments = append(segments, genaiBlobOrText{text: part.Text})
			}
		case part.InlineData != nil && len(part.InlineData.Data) > 0:
			if image, ok := inlineImageOf(part.InlineData); ok {
				segments = append(segments, genaiBlobOrText{image: image})
			}
		}
	}
	return segments
}

// genaiBlobOrText is one ordered segment of a turn: a caption-carrying text
// run or an image blob, in the position its parts occupied.
type genaiBlobOrText struct {
	text  string
	image genai.Blob
}

// hasImages reports whether any segment carries an image.
func hasImages(segments []genaiBlobOrText) bool {
	for _, s := range segments {
		if s.image.Data != nil {
			return true
		}
	}
	return false
}

// base64ImageDataURL renders image bytes as the data URL the Chat Completions
// image_url content part accepts.
func base64ImageDataURL(image genai.Blob) string {
	return "data:" + image.MIMEType + ";base64," + base64.StdEncoding.EncodeToString(image.Data)
}
