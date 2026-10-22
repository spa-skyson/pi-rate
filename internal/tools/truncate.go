package tools

import (
	"fmt"
	"unicode/utf8"
)

const (
	maxOutputBytes = 256 * 1024 // 256KB safety limit (smart compaction happens in AfterToolCallback)
	maxLineLength  = 500        // max chars per match/content line
)

// truncationMarker is the single cut disclosure every truncation point in
// this package appends: `… (+N bytes truncated)`, N = the number of cut
// characters. One shape everywhere means the model never has to guess
// whether a tail is an artifact of rendering or the whole answer. Points
// that stop before reading the tail (a streaming read, an entry cap) use the
// same frame with the unit they actually counted; they cannot know the
// character count without doing the reading they exist to avoid.
// truncationMarker is the single cut disclosure every truncation point in
// this package appends: `… (+N bytes truncated)`, N = the number of cut
// bytes. One shape everywhere means the model never has to guess
// whether a tail is an artifact of rendering or the whole answer. Points
// that stop before reading the tail (a streaming read, an entry cap) use the
// same frame with their own minimal estimate; they cannot know the byte
// count without doing the reading they exist to avoid.
func truncationMarker(cut int) string {
	return fmt.Sprintf("\n… (+%d bytes truncated)", cut)
}

// utf8SafeCut returns s truncated to at most n bytes, backing up to the
// nearest rune boundary so the result is never invalid UTF-8.
//
// A plain s[:n] can land in the middle of a multi-byte rune, and the resulting
// broken byte is carried all the way to the provider.
func utf8SafeCut(s string, n int) string {
	if n <= 0 {
		return ""
	}
	if n >= len(s) {
		return s
	}
	// A rune is at most 4 bytes, so at most 3 bytes need to be given back.
	for n > 0 && !utf8.RuneStart(s[n]) {
		n--
	}
	return s[:n]
}

// utf8SafeCutFrom returns the suffix of s starting at or after byte offset n,
// advanced to the nearest rune boundary.
func utf8SafeCutFrom(s string, n int) string {
	if n <= 0 {
		return s
	}
	if n >= len(s) {
		return ""
	}
	for n < len(s) && !utf8.RuneStart(s[n]) {
		n++
	}
	return s[n:]
}

func truncateOutput(s string) string {
	if len(s) <= maxOutputBytes {
		return s
	}
	cut := utf8SafeCut(s, maxOutputBytes)
	return cut + truncationMarker(len(s)-len(cut))
}

// truncateLine trims a single line to maxLineLength bytes, backing up to a
// rune boundary so the kept prefix is never invalid UTF-8.
func truncateLine(s string) string {
	if len(s) <= maxLineLength {
		return s
	}
	cut := utf8SafeCut(s, maxLineLength)
	return cut + truncationMarker(len(s)-len(cut))
}
