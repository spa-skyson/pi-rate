package provider

import (
	"errors"
	"regexp"
	"strconv"
	"strings"

	"github.com/openai/openai-go/v3"
)

// minMaskTokenRunes is the run length above which a token from the secret
// alphabet is masked. Real keys are far longer — OpenAI keys are 51+ chars,
// bearer and JWT tokens longer still — while the longest legitimate prose run
// is a dated model id ("gpt-4o-mini-2024-07-18", 23 chars). 33 sits above
// every model id shape and below every credential shape; raise it if a
// provider ships longer model names, never lower it.
const minMaskTokenRunes = 33

// longTokenRe matches runs of minMaskTokenRunes+ characters from the secret
// alphabet (letters, digits, dash, underscore) inside server-provided error
// text — a run that long is a credential until proven otherwise.
var longTokenRe = regexp.MustCompile(`[-A-Za-z0-9_]{` + strconv.Itoa(minMaskTokenRunes) + `,}`)

// maskServerSecrets replaces every long token in s with "***", so a
// server-quoted error cannot carry a key into the session record. Model ids,
// short words, URLs and error codes pass through untouched.
func maskServerSecrets(s string) string {
	return longTokenRe.ReplaceAllString(s, "***")
}

// oaiErrorText renders a provider failure for the error a session records.
//
// openai-go's *openai.Error deliberately formats to a bare status summary
// ("OpenAI API error: 400 Bad Request") and drops the body the server sent —
// the wording that names what to fix ("invalid temperature: only 1 is allowed
// for this model"). The summary stays as the prefix so the retry classifier's
// terminal/transient matches ("bad request", "429", …) keep firing on the
// same tokens they did before; the server's own message is appended behind
// it, masked and clipped the same way failedResponseError clips server text.
//
// The appended body is classification input, not just prose: the enriched
// string flows into retry.IsTransient/isTerminal (provider/retry.go's
// streamFailure → internal/retry), and terminal patterns win over transient
// ones there — a 429 whose body says "You exceeded your current quota" stays
// terminal, a 429 saying "rate limit exceeded" retries. That is already true
// of other provider wordings that ride the same route; the pinned cases below
// hold it for this one.
//
// Errors that are not *openai.Error (connection resets, idle timeouts) pass
// through unchanged: they have no second, hidden half.
func oaiErrorText(err error) string {
	if err == nil {
		return ""
	}
	summary := err.Error()
	var apiErr *openai.Error
	if !errors.As(err, &apiErr) || apiErr == nil {
		return summary
	}
	msg := strings.TrimSpace(apiErr.Message)
	if msg == "" {
		// The body did not decode into the envelope (a proxy HTML page, an
		// error shape the SDK does not know). The raw envelope JSON, if any
		// was captured, is still the most specific thing available.
		msg = strings.TrimSpace(apiErr.RawJSON())
	}
	if msg == "" {
		return summary
	}
	return summary + ": " + clipServerText(maskServerSecrets(msg))
}

// oaiRenderedError re-renders a provider failure with oaiErrorText while
// keeping the original error reachable for errors.As — the non-streaming
// yield sites hand this to their caller in place of fmt.Errorf's %w wrap,
// which would stringify to the bare status summary again.
type oaiRenderedError struct {
	err  error
	text string
}

func (e *oaiRenderedError) Error() string { return e.text }
func (e *oaiRenderedError) Unwrap() error { return e.err }

// newOaiRenderedError builds an oaiRenderedError reading "<intro>: <rendered
// failure>", or nil for a nil err.
func newOaiRenderedError(intro string, err error) error {
	if err == nil {
		return nil
	}
	return &oaiRenderedError{err: err, text: intro + ": " + oaiErrorText(err)}
}
