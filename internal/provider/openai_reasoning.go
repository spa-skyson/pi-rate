package provider

import (
	"strings"

	"github.com/openai/openai-go/v3/shared"
	"google.golang.org/genai"
)

// defaultOpenAIThinkingLevel is the reasoning effort a coding agent gets when
// the caller names none.
//
// Medium, because that is OpenAI's own recommendation for all-round interactive
// coding work: the Codex prompting guide names medium as the balanced default
// and reserves higher effort for multi-file design changes and hard debugging.
// The alternative — leaving the field off — is not neutral on this family:
// gpt-5.1+ defaults to no reasoning, which is the wrong floor for an agent that
// plans, edits and verifies.
const defaultOpenAIThinkingLevel = "medium"

// oaiReasoningModels are the OpenAI model families that accept a
// `reasoning.effort` (Responses) / `reasoning_effort` (Chat Completions)
// parameter. Everything else rejects it with a 400 rather than ignoring it:
//
//	400 Unsupported parameter: 'reasoning.effort' is not supported with this model.
//
// That makes this gate load-bearing rather than cosmetic. The Responses path is
// reachable for every model with multi-turn state (see endpointMode), so a
// non-reasoning model — gpt-4.1, gpt-4o — does reach it, and sending the field
// there would break the turn outright instead of degrading the setting.
//
// GLM rides here too: z.ai exposes the same standard field, but only accepts it
// on GLM-5.2+ (docs.z.ai/guides/capabilities/thinking), so glm-4.6/4.7/5/5.1
// must stay gated out.
//
// Matched by prefix so dated and suffixed variants (gpt-5.6-luna-2026-04-23)
// are covered without an ever-growing exact list.
var oaiReasoningModels = []string{
	"o1", "o3", "o4",
	"gpt-5", "gpt-6",
	"glm-5.2", "glm-5.3",
}

// glmProviderPrefixes are the provider wrappers GLM model names travel under
// ("GLM/GLM-5.3", "zai/glm-5.2") that StripKnownProviderPrefixes does not
// list. They are stripped here, before family matching, so the gate and the
// effort mapping see the bare model id. Nested wrappers still strip first via
// StripKnownProviderPrefixes.
var glmProviderPrefixes = []string{
	"glm/", "zai/", "zhipu/", "z.ai/",
}

// glmBaseName strips the provider wrappers around a GLM model name and returns
// the lowercased bare model id ("GLM/GLM-5.3" -> "glm-5.3"). The glm/ prefix
// is cut last character-wise on purpose: cutting it before glm-5.2/ would make
// the family prefixes below unreachable.
func glmBaseName(modelName string) string {
	lower := strings.ToLower(StripKnownProviderPrefixes(strings.TrimSpace(modelName)))
	for {
		stripped := false
		for _, p := range glmProviderPrefixes {
			if strings.HasPrefix(lower, p) {
				lower = lower[len(p):]
				stripped = true
				break
			}
		}
		if !stripped {
			return lower
		}
	}
}

// oaiModelReasons reports whether a model accepts a reasoning effort parameter.
//
// The `o1`/`o3`/`o4` prefixes are matched on the family head so o1-mini,
// o3-mini and o4-mini all qualify. The gpt-5/gpt-6 prefixes cover the whole
// reasoning series (gpt-5.1 … gpt-5.6-luna, gpt-6-astra …), including the
// codex variants. The glm-5.2/glm-5.3 prefixes cover the z.ai families, which
// are the only GLM generations that accept reasoning_effort (GLM-4.x/5/5.1 do
// not); both are matched on the bare model id after glmBaseName strips any
// "GLM/"-style provider wrapper.
func oaiModelReasons(modelName string) bool {
	lower := glmBaseName(modelName)
	for _, prefix := range oaiReasoningModels {
		if strings.HasPrefix(lower, prefix) {
			return true
		}
	}
	return false
}

// glmFamily reports which z.ai generation a GLM model belongs to, since the
// two accept disjoint effort sets (docs.z.ai/guides/capabilities/thinking):
// 5.3/5.3-flash only low/high/max, 5.2 the full ladder including none.
func glmFamily(modelName string) string {
	lower := glmBaseName(modelName)
	switch {
	case strings.HasPrefix(lower, "glm-5.3"):
		return "5.3"
	case strings.HasPrefix(lower, "glm-5.2"):
		return "5.2"
	}
	return ""
}

// glmReasoningEffortFor maps pi-go's thinking level onto the reasoning effort a
// GLM model accepts, per the z.ai thinking guide
// (docs.z.ai/guides/capabilities/thinking). ok is false when the level should
// be omitted so the model default stands.
//
// GLM-5.3 / GLM-5.3-Flash: thinking cannot be disabled (thinking.type=disabled
// is an API error) and reasoning_effort accepts ONLY low | high | max — any
// other value errors. Before this mapping the gate skipped these models
// entirely, so every request ran at the provider default, which is max
// ("Deep Reasoning") no matter what the UI showed.
//
// GLM-5.2: the ladder passes through — max (default), xhigh, high, medium,
// low, and the off pair none|minimal. The off pair is ONE case by the same
// rule the OpenAI ladder applies (case "none", "minimal" → none where the
// model supports reasoning off): z.ai treats both as "the model stops
// thinking", and its own aliases step the other way — low/medium are mapped
// UP to high server-side — so stepping minimal down to low would buy MORE
// thinking than the caller asked to switch off. none is the honest wire
// value for both spellings. That is a native reasoning_effort value, not an
// extra_body knob, so it rides the same standard field. medium is sent
// as-is even though z.ai aliases it to high: the honest wire value is what
// the caller asked for, and it survives the turn on both endpoints.
//
// Older GLM families (4.6, 4.7, 5, 5.1) never reach here: they have no
// reasoning_effort support at all and stay gated out by oaiModelReasons.
func glmReasoningEffortFor(modelName, thinkingLevel string) (shared.ReasoningEffort, bool) {
	family := glmFamily(modelName)
	switch strings.ToLower(strings.TrimSpace(thinkingLevel)) {
	case "none", "minimal":
		// 5.3 cannot switch thinking off; the lowest tier it takes is low —
		// the same compromise xaiReasoningEffort documents for Grok.
		if family == "5.3" {
			return shared.ReasoningEffortLow, true
		}
		return shared.ReasoningEffortNone, true
	case "low":
		return shared.ReasoningEffortLow, true
	case "medium":
		// 5.3 has no medium tier. Stepping to low rather than high on purpose:
		// a caller who picks medium over high is asking for less thinking than
		// high, so the nearest tier *below* keeps that intent instead of
		// silently granting the more expensive one.
		if family == "5.3" {
			return shared.ReasoningEffortLow, true
		}
		return shared.ReasoningEffortMedium, true
	case "high":
		return shared.ReasoningEffortHigh, true
	case "xhigh":
		if family == "5.3" {
			return shared.ReasoningEffortMax, true
		}
		return shared.ReasoningEffortXhigh, true
	case "max":
		// max is GLM-5.2's own default tier and 5.3's top one; both take it.
		return shared.ReasoningEffortMax, true
	}
	return "", false
}

// oaiReasoningEffortFor maps pi-go's thinking level onto a reasoning effort,
// stepping down to a tier the model actually accepts.
//
// Measured against the live API, the accepted tiers are not uniform. "ok" is
// a 200 and "REJ" a 400 for that (model, effort) pair:
//
//	model             none  low  medium  high  xhigh  max
//	gpt-5.1           ok    ok   ok      ok    REJ    REJ
//	gpt-5.4 / 5.5     ok    ok   ok      ok    ok     REJ
//	gpt-5.6-luna/sol  ok    ok   ok      ok    ok     ok
//	gpt-6-astra/luna  REJ   ok   ok      ok    ok     ok
//	o1 / o3 / o4-mini REJ   ok   ok      ok    REJ    REJ
//
// low/medium/high are the only tiers every reasoning model accepts, so those
// pass through unchanged and anything outside the band steps inward to the
// nearest accepted tier. Two cases need care:
//
//   - "none" is a real request to switch reasoning off, not an extreme to
//     smooth away. gpt-5.1–5.6 accept it, and pi-go's auxiliary callers — the
//     commit-message writer, ping and the eval judge — pass it for exactly that
//     reason. It is only stepped up to "low" where the family rejects it
//     (o-series, gpt-5 base, gpt-6), which is the same compromise
//     xaiReasoningEffort documents for Grok's missing off switch.
//   - "max"/"xhigh" step down to "high", which every family accepts, because
//     the 400 a rejected top tier earns would break the turn.
//
// Stepping is silent by design: a tier the model cannot express is the model's
// limitation, and the alternative is a 400 naming a parameter the caller never
// typed, which reads as a pi-go bug.
//
// An unrecognized level returns false so the field is omitted and the model's
// own default stands, which is what ThinkingUnset means everywhere else.
func oaiReasoningEffortFor(modelName, thinkingLevel string) (shared.ReasoningEffort, bool) {
	if !oaiModelReasons(modelName) {
		return "", false
	}
	// GLM has its own tier set and its own document; the OpenAI ladder below
	// would step "none" up and "max" down for 5.3, both wrong there.
	if glmFamily(modelName) != "" {
		return glmReasoningEffortFor(modelName, thinkingLevel)
	}
	off, top := oaiSupportsReasoningOff(modelName), oaiSupportsTopEffort(modelName)
	switch strings.ToLower(strings.TrimSpace(thinkingLevel)) {
	case "none", "minimal":
		if off {
			return shared.ReasoningEffortNone, true
		}
		return shared.ReasoningEffortLow, true
	case "low":
		return shared.ReasoningEffortLow, true
	case "medium":
		return shared.ReasoningEffortMedium, true
	case "high":
		return shared.ReasoningEffortHigh, true
	case "max":
		// "max" is narrower than "xhigh": gpt-5.4/5.5 accept "xhigh" and
		// reject "max", so only the newest generations may receive it.
		if supportsMax(modelName) {
			return shared.ReasoningEffortMax, true
		}
		return shared.ReasoningEffortHigh, true
	case "xhigh":
		if top {
			return shared.ReasoningEffortXhigh, true
		}
		return shared.ReasoningEffortHigh, true
	}
	return "", false
}

// supportsMax reports whether a model accepts effort "max", which is a tier
// above "xhigh". Measured: gpt-5.4 and gpt-5.5 accept "xhigh" but reject
// "max", so treating the two as one tier would 400 those models.
func supportsMax(modelName string) bool {
	lower := strings.ToLower(StripKnownProviderPrefixes(strings.TrimSpace(modelName)))
	return strings.HasPrefix(lower, "gpt-5.6") || strings.HasPrefix(lower, "gpt-6")
}

// oaiSupportsReasoningOff reports whether a model accepts effort "none".
//
// The gpt-5.1+ series does; the o-series never did, gpt-5 base rejects it, and
// gpt-6 does not either. Narrower than "does the model reason" on purpose —
// conflating the two is what would turn an auxiliary caller's deliberate
// no-reasoning request into billed reasoning tokens.
func oaiSupportsReasoningOff(modelName string) bool {
	lower := strings.ToLower(StripKnownProviderPrefixes(strings.TrimSpace(modelName)))
	// gpt-5.1 and later accept "none"; bare "gpt-5" and gpt-5.0 do not, and
	// the family prefix alone cannot tell them apart.
	if strings.HasPrefix(lower, "gpt-6") {
		return false
	}
	if strings.HasPrefix(lower, "o1") || strings.HasPrefix(lower, "o3") || strings.HasPrefix(lower, "o4") {
		return false
	}
	return strings.HasPrefix(lower, "gpt-5.1") ||
		strings.HasPrefix(lower, "gpt-5.2") ||
		strings.HasPrefix(lower, "gpt-5.3") ||
		strings.HasPrefix(lower, "gpt-5.4") ||
		strings.HasPrefix(lower, "gpt-5.5") ||
		strings.HasPrefix(lower, "gpt-5.6")
}

// oaiSupportsTopEffort reports whether a model accepts the top tiers
// ("xhigh"/"max"). Only the newest generations do: gpt-5.1 and the o-series
// reject both, gpt-5.4/5.5 accept "xhigh" but not "max", and gpt-5.6/gpt-6
// accept both.
func oaiSupportsTopEffort(modelName string) bool {
	lower := strings.ToLower(StripKnownProviderPrefixes(strings.TrimSpace(modelName)))
	return strings.HasPrefix(lower, "gpt-5.4") ||
		strings.HasPrefix(lower, "gpt-5.5") ||
		strings.HasPrefix(lower, "gpt-5.6") ||
		strings.HasPrefix(lower, "gpt-6")
}

// oaiThinkingLevelForModel adjusts the requested level per model, before the
// clamp in oaiReasoningEffortFor.
//
// gpt-5.6-luna gets high effort even when the caller named no level, rather
// than inheriting the medium coding default. Luna is the tier pi-go drives as
// its coding model, and high is the effort that makes it useful for that: the
// work is multi-file edits with verification, which is exactly the shape
// OpenAI's guidance reserves high effort for. Medium is the right default for
// an unknown coding model, not for the one this agent actually runs on.
//
// Only an unset level is overridden — an explicit level from the caller, a
// config file or WithThinkingLevel is a deliberate request and still wins, so
// setting thinkingLevel in ~/.pirate/config.json can lower luna again if wanted.
func oaiThinkingLevelForModel(modelName, thinkingLevel string) string {
	if strings.TrimSpace(thinkingLevel) != "" {
		return thinkingLevel
	}
	if strings.HasPrefix(strings.ToLower(StripKnownProviderPrefixes(modelName)), "gpt-5.6-luna") {
		return "high"
	}
	return thinkingLevel
}

// chatReasoningEffort returns the reasoning effort for a Chat Completions
// request, using the same resolution and clamping as the Responses path so a
// turn that flips between the two protocols (see endpointMode) does not change
// effort as a side effect of which endpoint it lands on.
//
// A per-request genai thinking budget is a Responses-only channel here, because
// this path has no equivalent field to read it from; the model-level level is
// what governs it.
func (m *openaiModel) chatReasoningEffort() (shared.ReasoningEffort, bool) {
	return oaiReasoningEffortFor(m.modelName, m.thinkingLevel)
}

// oaiResponsesReasoning maps a configured thinking budget onto a Responses
// reasoning effort. Low tokens (100-500) → low effort; medium (2000-4000) →
// medium; high (8000+) → high. The second result is false when no budget is
// configured, or when the budget is not positive.
//
// The explicit per-request budget wins over the model-level thinking level: a
// caller who sets MaxOutputTokens-adjacent knobs on one request means that
// request, and the two encode the same intent at different scopes.
func oaiResponsesReasoning(config *genai.GenerateContentConfig) (shared.ReasoningParam, bool) {
	if config == nil || config.ThinkingConfig == nil || config.ThinkingConfig.ThinkingBudget == nil {
		return shared.ReasoningParam{}, false
	}
	switch bt := *config.ThinkingConfig.ThinkingBudget; {
	case bt >= 8000:
		return shared.ReasoningParam{Effort: shared.ReasoningEffortHigh}, true
	case bt >= 2000:
		return shared.ReasoningParam{Effort: shared.ReasoningEffortMedium}, true
	case bt > 0:
		return shared.ReasoningParam{Effort: shared.ReasoningEffortLow}, true
	}
	return shared.ReasoningParam{}, false
}
