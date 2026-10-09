package extension

// providerForwardsInlineImages reports whether the model adapter behind a
// provider puts genai InlineData parts on the wire.
//
// The Gemini path does it natively: provider.NewGemini returns ADK's native
// gemini model, which hands genai.Content to the SDK unchanged. The openai
// and anthropic adapters convert an InlineData part into their protocol's
// native image field — a data-URL image_url content part on the Chat
// Completions wire, a base64 image content block on the Messages API wire.
// Every other provider still goes through an adapter that reads only text and
// function calls — see genaiSplitParts in internal/provider/openai_completions.go
// and its callers — and drops an InlineData part without a word.
//
// It lives here rather than beside the adapters because piagent imports this
// package and is walled off from internal/provider by TestPiagentStaysIsolated.
//
// This is not a statement about the model: a vision-capable model reached
// through an adapter not listed here still cannot be sent an image this way.
// When an adapter learns to forward images, add its provider here.
func providerForwardsInlineImages(providerName string) bool {
	switch providerName {
	case "gemini", "openai", "anthropic":
		return true
	}
	return false
}
