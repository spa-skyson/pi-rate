package extension

import "testing"

// TestProviderForwardsInlineImages pins which providers can carry an image
// part. The ADK-native Gemini path hands genai.Content to the SDK unchanged;
// the openai and anthropic adapters convert InlineData into their protocol's
// native image field. Every other adapter still rebuilds the request from text
// and function calls alone, so an InlineData part handed to it is dropped
// silently.
func TestProviderForwardsInlineImages(t *testing.T) {
	for _, p := range []string{"gemini", "openai", "anthropic"} {
		if !providerForwardsInlineImages(p) {
			t.Errorf("providerForwardsInlineImages(%q) = false; its adapter converts InlineData", p)
		}
	}
	for _, p := range []string{
		"ollama", "azure", "mistral",
		"openrouter", "xai", "opencode", "agentgateway", "", "unknown",
	} {
		if providerForwardsInlineImages(p) {
			t.Errorf("providerForwardsInlineImages(%q) = true; no adapter for it converts InlineData", p)
		}
	}
}
