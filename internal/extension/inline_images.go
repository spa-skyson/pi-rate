package extension

// providerForwardsInlineImages reports whether the model adapter behind a
// provider puts genai InlineData parts on the wire.
//
// The Gemini path does it natively: provider.NewGemini returns ADK's native
// gemini model, which hands genai.Content to the SDK unchanged. The openai
// adapter converts an InlineData part into its protocol's native image
// field on both of its routes — a data-URL image_url content part on the
// Chat Completions wire, an input_image content part on the Responses wire —
// and mistral rides that same converter. The anthropic adapter sends a base64
// image content block on the Messages API wire (including its advisor beta
// route), and the ollama adapter fills the chat message's Images field with
// raw base64. Every other provider still goes through an adapter that reads
// only text and function calls — see genaiSplitParts in
// internal/provider/openai_completions.go and its callers — and drops an
// InlineData part without a word.
//
// It lives here rather than beside the adapters because piagent imports this
// package and is walled off from internal/provider by TestPiagentStaysIsolated.
//
// This is not a statement about the model: a vision-capable model reached
// through an adapter not listed here still cannot be sent an image this way.
// When an adapter learns to forward images, add its provider here.
func providerForwardsInlineImages(providerName string) bool {
	switch providerName {
	case "gemini", "openai", "anthropic", "mistral", "ollama":
		return true
	}
	return false
}
