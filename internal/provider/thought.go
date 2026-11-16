package provider

import (
	"fmt"
	"unicode/utf8"

	"google.golang.org/genai"
)

// maxThoughtBytes caps the reasoning text one turn persists into its session
// event. Reasoning streams can run to hundreds of kilobytes on a long turn;
// events.jsonl is read whole by every dashboard page and replayed into every
// session open, so an uncapped thought would tax every reader forever to
// preserve scratch work the model itself does not need back. 64 KiB keeps a
// full plan readable while bounding the file.
const maxThoughtBytes = 64 << 10

// thoughtTruncationMarker closes a truncated reasoning text. The count tells a
// reader how much existed beyond the cap, so a short thought is never mistaken
// for a complete one.
const thoughtTruncationMarker = "\n[... reasoning truncated: %d more bytes not stored ...]"

// persistThoughtText clamps one turn's reasoning to maxThoughtBytes, cutting on
// a rune boundary and appending a marker that names the dropped remainder.
// Text at or under the cap passes through byte-for-byte.
func persistThoughtText(text string) string {
	if len(text) <= maxThoughtBytes {
		return text
	}
	cut := maxThoughtBytes
	for cut > 0 && !utf8.RuneStart(text[cut]) {
		cut--
	}
	return text[:cut] + fmt.Sprintf(thoughtTruncationMarker, len(text)-cut)
}

// thoughtPart returns the part that persists one turn's reasoning in the final
// response, or nil when there is nothing to persist. The Thought flag is the
// genai marker every downstream consumer already understands — the dashboard
// normalizer renders it as kind=thinking — and it is what lets the providers
// strip the reasoning back out of the conversation before it reaches the wire
// (see genaiSplitParts and oaiAppendResponsesItems).
//
// hasAnswer marks a turn that also produced reply text or tool calls. A turn
// with nothing but reasoning keeps that reasoning as its plain content — the
// long-standing thinking-only fallback — because flagging it Thought would make
// ADK's flow read the turn as thought-only and re-call the model for an answer
// it was never going to give.
func thoughtPart(thinking string, hasAnswer bool) *genai.Part {
	if thinking == "" || !hasAnswer {
		return nil
	}
	return &genai.Part{Text: persistThoughtText(thinking), Thought: true}
}

// partHasAnswer reports whether a part carries content the model meant to be
// read as the turn's output: reply text or a tool call. Thought text is the
// turn's scratch work, not its answer.
func partHasAnswer(p *genai.Part) bool {
	if p == nil {
		return false
	}
	if p.FunctionCall != nil {
		return true
	}
	return p.Text != "" && !p.Thought
}

// attachThought prepends one turn's reasoning to the final response parts as a
// Thought part. It is a no-op when the turn reasoned nothing, or when nothing
// else in parts counts as the answer (see thoughtPart): a thinking-only turn
// reaches the caller exactly as it did before, through its own provider's
// fallback.
func attachThought(parts []*genai.Part, thinking string) []*genai.Part {
	hasAnswer := false
	for _, p := range parts {
		if partHasAnswer(p) {
			hasAnswer = true
			break
		}
	}
	tp := thoughtPart(thinking, hasAnswer)
	if tp == nil {
		return parts
	}
	out := make([]*genai.Part, 0, len(parts)+1)
	out = append(out, tp)
	return append(out, parts...)
}

// ensureAnswerPart keeps a response made only of Thought parts readable. A
// provider that reports raw reasoning with no surfaced answer — a reasoning
// item with no message behind it — must not hand ADK a thought-only turn: the
// flow would read it as "the model is still thinking" and re-call for an
// answer. The first thought is downgraded to plain text, which is the answer
// those turns have.
func ensureAnswerPart(parts []*genai.Part) []*genai.Part {
	hasAnswer := false
	for _, p := range parts {
		if partHasAnswer(p) {
			hasAnswer = true
			break
		}
	}
	if hasAnswer {
		return parts
	}
	for i, p := range parts {
		if p != nil && p.Thought && p.Text != "" {
			parts[i] = &genai.Part{Text: p.Text}
			break
		}
	}
	return parts
}
