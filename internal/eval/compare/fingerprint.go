package compare

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"

	"github.com/spa-skyson/pi-rate/internal/agent"
)

// Fingerprint computes the sha256 identity of a variant: the name, the
// label and the full instruction-parts inventory, hashed as canonical
// "key=value" lines. Two variants with the same fingerprint carry the same
// harness configuration — this is the reuse key for the run Store and the
// first half of every report's identity line.
func Fingerprint(v Variant) string {
	h := sha256.New()
	fmt.Fprintf(h, "name=%s\n", v.Name)
	fmt.Fprintf(h, "label=%s\n", v.Label)
	var p agent.InstructionParts
	if v.Parts != nil {
		p = *v.Parts
	}
	fmt.Fprintf(h, "base=%s\n", p.Base)
	fmt.Fprintf(h, "rules=%s\n", p.Rules)
	fmt.Fprintf(h, "skills=%s\n", p.Skills)
	fmt.Fprintf(h, "env=%s\n", p.Env)
	return hex.EncodeToString(h.Sum(nil))
}
