package tui

import (
	"fmt"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"

	"github.com/spa-skyson/pi-rate/internal/permission"
)

// The tool-approval dialog. When the permission gate resolves a call to ask,
// the CLI bridge sends a permission.ApprovalRequest on Config.ApprovalCh; the
// request blocks the agent loop's before-tool callback until the user answers
// here. The dialog is a modal in the handleKey overlay chain — y/Enter allow,
// n/Esc/q deny, a allow for the rest of the session — while every other key
// falls through, so typing continues into the prompt and nothing is lost; only
// submission is withheld, because Enter is taken by the dialog.

// approvalRequestMsg carries one tool-approval request from the gate, with
// the session tag the asker stamped on it (types.go).
type approvalRequestMsg struct {
	sa SessionApproval
}

// waitForApproval parks a reader on the approval channel and delivers the next
// request as a message. Re-armed after each delivery, like waitForSystemNotice
// — exactly one reader is ever parked, so a channel close cannot multiply
// wakeups.
func waitForApproval(ch <-chan SessionApproval) tea.Cmd {
	if ch == nil {
		return nil
	}
	return func() tea.Msg {
		sa, ok := <-ch
		if !ok {
			return nil
		}
		return approvalRequestMsg{sa: sa}
	}
}

// handleApprovalRequest shows the dialog for a new request. The agent loop is
// blocked in its before-tool callback until the user answers.
func (m *model) handleApprovalRequest(msg approvalRequestMsg) (tea.Model, tea.Cmd) {
	if m.approval != nil {
		// Tool calls are sequential inside the agent loop, and subagent
		// sessions hard-deny ask in their own process, so a second request
		// while one is on screen cannot happen by construction. If one ever
		// arrives it means a bug elsewhere: fail the stale request closed
		// (deny) rather than queue it behind a dialog the user has already
		// stopped reading, and show the new one.
		m.approval.Reply <- permission.ApprovalResult{}
	}
	req := msg.sa.Req
	m.approval = &req
	m.openOverlay(overlayApproval)
	// Attention first: the loop is blocked until the dialog is answered, so a
	// user who switched away needs the bell/notification to come back. The
	// command is nil (no-op in the Batch) when attention is off or the
	// terminal is focused.
	return m, tea.Batch(m.attentionCmd("☠ Pi-rate", "Approval required: "+req.Tool),
		waitForApproval(m.cfg.ApprovalCh))
}

// handleApprovalKey resolves the approval dialog, swallowing every key it owns
// so a stray press cannot be misread. Unowned keys fall through to the normal
// chain: typing edits the prompt as usual and is simply not submitted, because
// Enter belongs to the dialog while it is up.
func (m *model) handleApprovalKey(key tea.Key) (tea.Model, tea.Cmd, bool) {
	if m.approval == nil {
		return nil, nil, false
	}

	var res permission.ApprovalResult
	switch {
	case key.Mod == 0 && key.Code == tea.KeyEnter,
		key.Mod == 0 && (key.Code == 'y' || key.Code == 'Y'):
		res = permission.ApprovalResult{Allowed: true}
	case key.Mod == 0 && (key.Code == 'a' || key.Code == 'A'):
		res = permission.ApprovalResult{Allowed: true, Always: true}
	case key.Mod == 0 && (key.Code == 'n' || key.Code == 'N' || key.Code == 'q'),
		key.Code == tea.KeyEsc:
		res = permission.ApprovalResult{}
	default:
		// Ctrl+C deliberately falls through to handleInterruptKey: canceling
		// the whole turn is still available mid-dialog, and the canceled turn
		// answers the pending request through the bridge's ctx path.
		return nil, nil, false
	}
	model, cmd := m.answerApproval(res)
	return model, cmd, true
}

// answerApproval sends the user's answer back to the blocked gate, drops the
// dialog and records what happened in the transcript.
func (m *model) answerApproval(res permission.ApprovalResult) (tea.Model, tea.Cmd) {
	req := m.approval
	m.popOverlay(overlayApproval)
	if req != nil {
		// Reply is buffered to one and this is its only send: never blocks.
		req.Reply <- res
	}
	m.chatModel.AppendNotice(approvalFact(req, res))
	return m, nil
}

// approvalFact renders the one-line transcript record of an approval decision
// — allowed what, denied what, or which rule is now allowed for the session.
func approvalFact(req *permission.ApprovalRequest, res permission.ApprovalResult) string {
	if req == nil {
		return "approval answered"
	}
	switch {
	case res.Always:
		return fmt.Sprintf("always allowed for this session: rule %q", req.Rule)
	case res.Allowed:
		if req.Command != "" {
			return fmt.Sprintf("allowed: bash %q", truncateLabel(req.Command, 60))
		}
		return fmt.Sprintf("allowed: %s", req.Tool)
	default:
		if req.Command != "" {
			return fmt.Sprintf("denied by user: bash %q", truncateLabel(req.Command, 60))
		}
		return fmt.Sprintf("denied by user: %s", req.Tool)
	}
}

// approvalGlyph is the warning marker, shared with the transcript's warning
// notices (assistantWarningBody).
const approvalGlyph = "⚠"

// approvalKeys is the key-hint tail of the approval line.
const approvalKeys = " · [y] allow · [a] always · [n] deny"

// renderApprovalDialog renders the pending request as a single status-hint
// line in the last slot of the chat panel, above the closing rule — no border
// or title, Warning foreground like the transcript's warning notices. The
// line must fit the panel width: the command is truncated first, the rule is
// dropped next, and the narrowest form — tool + key hint — always renders.
func (m *model) renderApprovalDialog(width int) string {
	req := m.approval
	if req == nil {
		return ""
	}

	head := approvalGlyph + " approval: " + req.Tool
	rule := ""
	if req.Rule != "" {
		rule = ` · rule "` + req.Rule + `"`
	}

	if req.Command != "" {
		// ": " between head and command; cut the command to whatever the rest
		// of the line leaves, and drop it when that is nothing.
		if room := width - lipgloss.Width(head+rule+approvalKeys) - 2; room > 0 {
			return m.approvalLine(head + ": " + truncateLabel(req.Command, room) + rule + approvalKeys)
		}
	}
	if lipgloss.Width(head+rule+approvalKeys) <= width {
		return m.approvalLine(head + rule + approvalKeys)
	}
	if lipgloss.Width(head+approvalKeys) <= width {
		return m.approvalLine(head + approvalKeys)
	}
	return m.approvalLine(truncateLabel(head, max(1, width-lipgloss.Width(approvalKeys))) + approvalKeys)
}

// approvalLine styles one approval line: Warning foreground and nothing else —
// the weight of a status hint, not a titled dialog.
func (m *model) approvalLine(s string) string {
	return lipgloss.NewStyle().Foreground(m.palette.Warning).Render(s)
}
