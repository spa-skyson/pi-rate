package tui

import (
	"strings"
	"testing"
	"time"

	tea "charm.land/bubbletea/v2"

	"github.com/spa-skyson/pi-rate/internal/config"
	"github.com/spa-skyson/pi-rate/internal/permission"
)

// attentionOn builds a model with both attention channels on and the terminal
// marked unfocused — the state in which a signal is allowed to fire.
func attentionOn() *model {
	m := &model{}
	m.cfg.Attention = &config.AttentionConfig{}
	return m
}

// runCmd executes a tea.Cmd the way the Bubble Tea runtime would and returns
// the raw sequences carried by any tea.RawMsg in the result, recursing into
// BatchMsg the way the runtime's command runner does.
func runCmd(cmd tea.Cmd) []string {
	if cmd == nil {
		return nil
	}
	var seqs []string
	switch msg := cmd().(type) {
	case tea.RawMsg:
		seqs = append(seqs, msg.Msg.(string))
	case tea.BatchMsg:
		for _, c := range msg {
			seqs = append(seqs, runCmd(c)...)
		}
	}
	return seqs
}

func TestAttentionNotifySeq_ExactBytes(t *testing.T) {
	got := attentionNotifySeq("☠ Pi-rate", "Turn complete")
	want := "\033]777;notify;☠ Pi-rate;Turn complete\033\\"
	if got != want {
		t.Fatalf("notify sequence = %q, want %q", got, want)
	}
}

func TestAttentionNotifySeq_ScrubsControls(t *testing.T) {
	// A C0 control in the body must not break the OSC envelope out: the
	// scrubber replaces it with a space, same as the window-title path.
	got := attentionNotifySeq("t", "line\nESC\x1b[31mred")
	payload := strings.TrimSuffix(strings.TrimPrefix(got, "\033]777;notify;"), "\033\\")
	if strings.ContainsAny(payload, "\n\x1b\x07") {
		t.Fatalf("sequence payload leaks control characters: %q", payload)
	}
	if !strings.HasPrefix(got, "\033]777;notify;t;") || !strings.HasSuffix(got, "\033\\") {
		t.Fatalf("sequence envelope malformed: %q", got)
	}
}

func TestAttentionCmd_BothChannelsByDefault(t *testing.T) {
	m := attentionOn()
	cmd := m.attentionCmd("T", "B")
	seqs := runCmd(cmd)
	if len(seqs) != 1 {
		t.Fatalf("want exactly one RawMsg, got %d (%v)", len(seqs), seqs)
	}
	want := attentionBellSeq + attentionNotifySeq("T", "B")
	if seqs[0] != want {
		t.Fatalf("seq = %q, want %q", seqs[0], want)
	}
}

func TestAttentionCmd_ConfigFlagsGateChannels(t *testing.T) {
	off := false
	tests := []struct {
		name     string
		cfg      config.AttentionConfig
		wantSeqs []string
	}{
		{"bell only", config.AttentionConfig{Notify: &off}, []string{attentionBellSeq}},
		{"notify only", config.AttentionConfig{Bell: &off}, []string{attentionNotifySeq("T", "B")}},
		{"both off", config.AttentionConfig{Bell: &off, Notify: &off}, nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &model{}
			m.cfg.Attention = &tt.cfg
			seqs := runCmd(m.attentionCmd("T", "B"))
			if len(seqs) != len(tt.wantSeqs) {
				t.Fatalf("got %d sequences (%v), want %d", len(seqs), seqs, len(tt.wantSeqs))
			}
			for i, want := range tt.wantSeqs {
				if seqs[i] != want {
					t.Errorf("seq[%d] = %q, want %q", i, seqs[i], want)
				}
			}
		})
	}
}

func TestAttentionCmd_NilConfigOrFocusedStaysQuiet(t *testing.T) {
	// Zero Config (no attention section): no sequences at all — the quiet
	// default for directly constructed models.
	m := &model{}
	if cmd := m.attentionCmd("T", "B"); cmd != nil {
		t.Fatal("nil attention config must produce no command")
	}
	if m.attentionEnabled() {
		t.Fatal("nil attention config must not enable focus reporting either")
	}

	// Focused terminal: the user is already looking at the screen.
	m = attentionOn()
	m.focused = true
	if cmd := m.attentionCmd("T", "B"); cmd != nil {
		t.Fatal("focused terminal must not get attention signals")
	}
}

func TestTurnDoneAttention_Threshold(t *testing.T) {
	tests := []struct {
		name    string
		started time.Time
		want    bool
	}{
		{"no turn ever began", time.Time{}, false},
		{"just started", time.Now(), false},
		{"under threshold", time.Now().Add(-attentionDoneAfter + time.Second), false},
		{"over threshold", time.Now().Add(-attentionDoneAfter - time.Second), true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := attentionOn()
			m.turnStarted = tt.started
			cmd := m.turnDoneAttention()
			if got := cmd != nil; got != tt.want {
				t.Fatalf("turnDoneAttention() != nil = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestAgentDone_LongTurnSignalsShortDoesNot(t *testing.T) {
	// Short turn: no attention command, model unchanged.
	m := attentionOn()
	m.turnStarted = time.Now()
	same, cmd := m.handleAgentDone(agentDoneMsg{})
	if cmd != nil {
		t.Fatal("short turn must not ring")
	}
	if same != m {
		t.Fatal("handler must return the same model")
	}

	// Long turn: exactly one RawMsg with the done notification.
	m.turnStarted = time.Now().Add(-2 * attentionDoneAfter)
	_, cmd = m.handleAgentDone(agentDoneMsg{})
	seqs := runCmd(cmd)
	if len(seqs) != 1 || seqs[0] != attentionBellSeq+attentionNotifySeq("☠ Pi-rate", "Turn complete") {
		t.Fatalf("long turn sequences = %v, want one done signal", seqs)
	}

	// Attention off: a long turn stays quiet too.
	quiet := &model{}
	quiet.turnStarted = time.Now().Add(-2 * attentionDoneAfter)
	if _, cmd = quiet.handleAgentDone(agentDoneMsg{}); cmd != nil {
		t.Fatal("nil attention config must silence the done signal")
	}
}

func TestApprovalRequest_FiresAttentionOnce(t *testing.T) {
	m := attentionOn()
	// A spare request so the re-armed waitForApproval listener (part of the
	// returned batch) returns instead of blocking this test's goroutine the
	// way it would never block the real command runner.
	spare := make(chan SessionApproval, 2)
	m.cfg.ApprovalCh = spare
	spare <- SessionApproval{SessionID: "sess-a"}

	req := permission.ApprovalRequest{Tool: "bash", Reply: make(chan permission.ApprovalResult, 1)}
	_, cmd := m.handleApprovalRequest(approvalRequestMsg{sa: SessionApproval{Req: req, SessionID: "sess-a"}})
	seqs := runCmd(cmd)
	want := attentionBellSeq + attentionNotifySeq("☠ Pi-rate", "Approval required: bash")
	if len(seqs) != 1 || seqs[0] != want {
		t.Fatalf("approval sequences = %v, want exactly [%q]", seqs, want)
	}
	if m.approval == nil || m.approval.Tool != "bash" {
		t.Fatalf("approval dialog not set: %+v", m.approval)
	}

	// Focused: the listener still re-arms, no signal. Another spare keeps the
	// re-armed listener from blocking runCmd.
	spare <- SessionApproval{SessionID: "sess-a"}
	m.focused = true
	_, cmd = m.handleApprovalRequest(approvalRequestMsg{sa: SessionApproval{Req: req, SessionID: "sess-a"}})
	if seqs := runCmd(cmd); len(seqs) != 0 {
		t.Fatalf("focused approval produced sequences: %v", seqs)
	}
}

func TestViewReportFocus_FollowsAttention(t *testing.T) {
	// A sized model renders the main view; width 0 routes to the startup
	// splash, which carries no mode fields.
	m := attentionOn()
	m.width, m.height = 120, 30
	if v := m.View(); !v.ReportFocus {
		t.Fatal("attention enabled must set View.ReportFocus")
	}
	// Attention off: no mode flips for nothing.
	m = &model{}
	m.width, m.height = 120, 30
	if v := m.View(); v.ReportFocus {
		t.Fatal("nil attention config must leave ReportFocus off")
	}
}

func TestUpdateFocusBlur_TracksTerminalFocus(t *testing.T) {
	m := attentionOn()
	m.Update(tea.FocusMsg{})
	if !m.focused {
		t.Fatal("FocusMsg must mark the model focused")
	}
	m.Update(tea.BlurMsg{})
	if m.focused {
		t.Fatal("BlurMsg must mark the model unfocused")
	}
}
