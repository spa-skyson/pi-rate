package extension

import "testing"

// TestProviderForwardsInlineImages pins which providers can carry an image
// part. The ADK-native Gemini path hands genai.Content to the SDK unchanged;
// the openai adapter converts InlineData into its protocol's native image
// field on both the Chat Completions and Responses routes (and mistral rides
// the same converter), the anthropic adapter sends base64 image blocks, and
// the ollama adapter fills the chat message's Images field. Every other
// adapter still rebuilds the request from text and function calls alone, so
// an InlineData part handed to it is dropped silently.
func TestProviderForwardsInlineImages(t *testing.T) {
	for _, p := range []string{"gemini", "openai", "anthropic", "mistral", "ollama"} {
		if !providerForwardsInlineImages(p) {
			t.Errorf("providerForwardsInlineImages(%q) = false; its adapter converts InlineData", p)
		}
	}
	for _, p := range []string{
		"azure", "codex",
		"openrouter", "xai", "opencode", "agentgateway", "", "unknown",
	} {
		if providerForwardsInlineImages(p) {
			t.Errorf("providerForwardsInlineImages(%q) = true; no adapter for it converts InlineData", p)
		}
	}
}
