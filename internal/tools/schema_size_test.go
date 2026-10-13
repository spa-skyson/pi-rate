package tools

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/spa-skyson/pi-rate/internal/config"
	"github.com/spa-skyson/pi-rate/internal/subagent"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/genai"
)

// buildHeadlessToolset assembles the default headless (non-interactive) tool
// set the same way internal/cli.initNonInteractiveRuntime does:
// CoreTools(WithSessionID) + BashControlTools + SubagentTools + MemoryTools.
// The orchestrator is built with no discovered agents, so the subagent
// description carries its static part only; the per-environment agent list is
// excluded on purpose.
func buildHeadlessToolset(t *testing.T) []map[string]any {
	t.Helper()

	sb, err := NewSandbox(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = sb.Close() })

	sup := NewBashSupervisor()
	t.Cleanup(sup.KillAll)

	core, err := CoreTools(sb, WithSessionID("schema-size-test"))
	if err != nil {
		t.Fatal(err)
	}
	bashCtl, err := BashControlTools(sup)
	if err != nil {
		t.Fatal(err)
	}
	orch := subagent.NewOrchestrator(&config.Config{}, "", nil)
	agentTools, err := SubagentTools(orch, nil)
	if err != nil {
		t.Fatal(err)
	}
	memTools, err := MemoryTools(&memMockStore{})
	if err != nil {
		t.Fatal(err)
	}

	all := append(append(append(core, bashCtl...), agentTools...), memTools...)

	// The wire shape: one FunctionDeclaration per tool, as ADK registers it
	// into req.Config.Tools and the providers serialize into the request body.
	decls := make([]map[string]any, 0, len(all))
	for _, tl := range all {
		decl := declaration(tl)
		if decl == nil {
			t.Fatalf("tool %s has no declaration", tl.Name())
		}
		raw, err := json.Marshal(decl)
		if err != nil {
			t.Fatalf("marshal declaration of %s: %v", tl.Name(), err)
		}
		var m map[string]any
		if err := json.Unmarshal(raw, &m); err != nil {
			t.Fatalf("unmarshal declaration of %s: %v", tl.Name(), err)
		}
		decls = append(decls, m)
	}
	return decls
}

// declaration extracts the model-facing FunctionDeclaration from a tool.
// Core tools are coercingTool wrappers that expose Declaration(); foreign
// tool types (ADK grounding, MCP) may only carry Name/Description.
func declaration(tl tool.Tool) *genai.FunctionDeclaration {
	type declarer interface {
		Declaration() *genai.FunctionDeclaration
	}
	if d, ok := tl.(declarer); ok {
		return d.Declaration()
	}
	return nil
}

// web_search is opt-in (WithWebSearch / PI_WEB_SEARCH), so it is absent from
// the default set the budget above covers; pin its description here so it
// cannot balloon unnoticed (issue #73).
func TestSchemaDiet_WebSearchDescriptionBudget(t *testing.T) {
	ws, err := newWebSearchTool()
	if err != nil {
		t.Fatal(err)
	}
	if d := ws.Description(); len(d) > 700 {
		t.Errorf("web_search description = %d chars, budget 700", len(d))
	}
}

func declSize(m map[string]any) int {
	raw, err := json.Marshal(m)
	if err != nil {
		return 0
	}
	return len(raw)
}

func declDesc(m map[string]any) string {
	if s, ok := m["description"].(string); ok {
		return s
	}
	return ""
}

// TestSchemaDiet_HeadlessBudget pins the wire size of the default headless
// tool set. Tool schemas are prefilled on every LLM call (issue #73); a
// regression here taxes every turn. The budget covers 23 tools: 16 core +
// bash_wait/bash_kill + subagent/agent_result + 3 memory tools.
func TestSchemaDiet_HeadlessBudget(t *testing.T) {
	decls := buildHeadlessToolset(t)

	total := 0
	descTotal := 0
	var heavy []string
	for _, m := range decls {
		name, _ := m["name"].(string)
		sz := declSize(m)
		ds := len(declDesc(m))
		total += sz
		descTotal += ds
		if ds > 600 {
			heavy = append(heavy, name)
		}
		t.Logf("%-16s json=%5d desc=%5d", name, sz, ds)
	}
	t.Logf("TOTAL tools=%d json=%d desc=%d", len(decls), total, descTotal)

	if len(decls) != 23 {
		t.Errorf("tool count = %d, want 23 (did the default headless set change? update the budget deliberately)", len(decls))
	}
	if total > 18000 {
		t.Errorf("tools JSON = %d chars, budget 18000 (issue #73)", total)
	}
	// The subagent description is the single heaviest schema; its mode
	// semantics must stay but not balloon again.
	for _, m := range decls {
		if m["name"] == "subagent" {
			if d := declDesc(m); len(d) > 2000 {
				t.Errorf("subagent description = %d chars, budget 2000", len(d))
			}
		}
	}
	if len(heavy) > 0 {
		t.Logf("descriptions over 600 chars: %s", strings.Join(heavy, ", "))
	}
}
