package tools

import (
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/spa-skyson/pi-rate/internal/config"
	"github.com/spa-skyson/pi-rate/internal/subagent"
)

func defaultConfigPtr() *config.Config {
	cfg := config.Defaults()
	return &cfg
}

// --- Mode detection tests ---

func TestDetectMode_Single(t *testing.T) {
	input := SubagentInput{Agent: "explore", Task: "find main.go"}
	if mode := detectMode(input); mode != "single" {
		t.Errorf("detectMode = %q, want 'single'", mode)
	}
}

func TestDetectMode_Parallel(t *testing.T) {
	input := SubagentInput{Tasks: []TaskItem{{Agent: "a", Task: "b"}}}
	if mode := detectMode(input); mode != "parallel" {
		t.Errorf("detectMode = %q, want 'parallel'", mode)
	}
}

func TestDetectMode_Chain(t *testing.T) {
	input := SubagentInput{Chain: []ChainItem{{Agent: "a", Task: "b"}}}
	if mode := detectMode(input); mode != "chain" {
		t.Errorf("detectMode = %q, want 'chain'", mode)
	}
}

func TestDetectMode_ChainPriorityOverParallel(t *testing.T) {
	input := SubagentInput{
		Chain: []ChainItem{{Agent: "a", Task: "b"}},
		Tasks: []TaskItem{{Agent: "c", Task: "d"}},
	}
	if mode := detectMode(input); mode != "chain" {
		t.Errorf("detectMode = %q, want 'chain'", mode)
	}
}

func TestDetectMode_Empty(t *testing.T) {
	input := SubagentInput{}
	if mode := detectMode(input); mode != "" {
		t.Errorf("detectMode = %q, want empty", mode)
	}
}

func TestDetectMode_SingleWithOnlyAgent(t *testing.T) {
	// Lenient: single mode should work with just agent (task might be empty)
	// This helps recover from LLM mistakes where task isn't provided
	input := SubagentInput{Agent: "explore"}
	if mode := detectMode(input); mode != "single" {
		t.Errorf("detectMode = %q, want 'single' (lenient with only agent)", mode)
	}
}

func TestDetectMode_SingleWithOnlyTask(t *testing.T) {
	// Lenient: single mode should work with just task (agent might be empty)
	// This helps recover from LLM mistakes where agent isn't provided
	input := SubagentInput{Task: "some task"}
	if mode := detectMode(input); mode != "single" {
		t.Errorf("detectMode = %q, want 'single' (lenient with only task)", mode)
	}
}

// --- Single mode tests ---

func TestSubagentSingleMode_UnknownAgent(t *testing.T) {
	agents := []subagent.AgentConfig{
		{Name: "explore", Description: "test", Role: "default"},
	}
	orch := subagent.NewOrchestrator(defaultConfigPtr(), "", agents)

	input := SubagentInput{Agent: "nonexistent", Task: "find main.go"}
	output, err := subagentHandler(nil, orch, input, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if output.Mode != "single" {
		t.Errorf("mode = %q, want 'single'", output.Mode)
	}
	if len(output.Results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(output.Results))
	}
	r := output.Results[0]
	if r.Status != "failed" {
		t.Errorf("status = %q, want 'failed'", r.Status)
	}
	if r.Error == "" {
		t.Error("expected error message for unknown agent")
	}
	if r.Agent != "nonexistent" {
		t.Errorf("agent = %q, want 'nonexistent'", r.Agent)
	}
}

func TestSubagentSingleMode_NoModeDetected(t *testing.T) {
	orch := subagent.NewOrchestrator(defaultConfigPtr(), "", nil)

	input := SubagentInput{} // empty — no mode
	_, err := subagentHandler(nil, orch, input, nil, nil)
	if err == nil {
		t.Fatal("expected error for empty input")
	}
}

// --- Parallel mode tests ---

func TestSubagentParallelMode_UnknownAgent(t *testing.T) {
	agents := []subagent.AgentConfig{
		{Name: "explore", Description: "test", Role: "default"},
	}
	orch := subagent.NewOrchestrator(defaultConfigPtr(), "", agents)

	input := SubagentInput{Tasks: []TaskItem{
		{Agent: "explore", Task: "a"},
		{Agent: "nonexistent", Task: "b"},
	}}
	output, err := subagentHandler(nil, orch, input, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if output.Mode != "parallel" {
		t.Errorf("mode = %q, want 'parallel'", output.Mode)
	}
	if len(output.Results) != 1 {
		t.Fatalf("expected 1 result (validation error), got %d", len(output.Results))
	}
	if output.Results[0].Status != "failed" {
		t.Errorf("status = %q, want 'failed'", output.Results[0].Status)
	}
	if output.Results[0].Agent != "nonexistent" {
		t.Errorf("agent = %q, want 'nonexistent'", output.Results[0].Agent)
	}
}

func TestSubagentParallelMode_TooManyTasks(t *testing.T) {
	orch := subagent.NewOrchestrator(defaultConfigPtr(), "", nil)

	tasks := make([]TaskItem, maxParallelTasks+1)
	for i := range tasks {
		tasks[i] = TaskItem{Agent: "explore", Task: "a"}
	}
	input := SubagentInput{Tasks: tasks}
	output, err := subagentHandler(nil, orch, input, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if output.Mode != "parallel" {
		t.Errorf("mode = %q, want 'parallel'", output.Mode)
	}
	if len(output.Results) != 1 {
		t.Fatalf("expected 1 result (limit error), got %d", len(output.Results))
	}
	if output.Results[0].Status != "failed" {
		t.Errorf("status = %q, want 'failed'", output.Results[0].Status)
	}
	if !strings.Contains(output.Results[0].Error, "too many") {
		t.Errorf("error should mention 'too many', got: %s", output.Results[0].Error)
	}
}

func TestSubagentParallelMode_AllUnknownAgents(t *testing.T) {
	orch := subagent.NewOrchestrator(defaultConfigPtr(), "", nil)

	// All agents unknown — fails at validation before spawning.
	input := SubagentInput{Tasks: []TaskItem{
		{Agent: "unknown1", Task: "a"},
		{Agent: "unknown2", Task: "b"},
	}}
	output, err := subagentHandler(nil, orch, input, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if output.Mode != "parallel" {
		t.Errorf("mode = %q, want 'parallel'", output.Mode)
	}
	if len(output.Results) != 1 {
		t.Fatalf("expected 1 result (validation), got %d", len(output.Results))
	}
	if output.Results[0].Status != "failed" {
		t.Errorf("status = %q, want 'failed'", output.Results[0].Status)
	}
}

// --- Chain mode tests ---

func TestSubagentChainMode_UnknownAgent(t *testing.T) {
	agents := []subagent.AgentConfig{
		{Name: "explore", Description: "test", Role: "default"},
	}
	orch := subagent.NewOrchestrator(defaultConfigPtr(), "", agents)

	input := SubagentInput{Chain: []ChainItem{
		{Agent: "explore", Task: "step 1"},
		{Agent: "nonexistent", Task: "step 2"},
	}}
	output, err := subagentHandler(nil, orch, input, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if output.Mode != "chain" {
		t.Errorf("mode = %q, want 'chain'", output.Mode)
	}
	if len(output.Results) != 1 {
		t.Fatalf("expected 1 result (validation error), got %d", len(output.Results))
	}
	if output.Results[0].Status != "failed" {
		t.Errorf("status = %q, want 'failed'", output.Results[0].Status)
	}
	if output.Results[0].Agent != "nonexistent" {
		t.Errorf("agent = %q, want 'nonexistent'", output.Results[0].Agent)
	}
}

func TestSubagentChainMode_TooManySteps(t *testing.T) {
	orch := subagent.NewOrchestrator(defaultConfigPtr(), "", nil)

	chain := make([]ChainItem, maxChainSteps+1)
	for i := range chain {
		chain[i] = ChainItem{Agent: "explore", Task: "a"}
	}
	input := SubagentInput{Chain: chain}
	output, err := subagentHandler(nil, orch, input, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if output.Mode != "chain" {
		t.Errorf("mode = %q, want 'chain'", output.Mode)
	}
	if len(output.Results) != 1 {
		t.Fatalf("expected 1 result (limit error), got %d", len(output.Results))
	}
	if output.Results[0].Status != "failed" {
		t.Errorf("status = %q, want 'failed'", output.Results[0].Status)
	}
	if !strings.Contains(output.Results[0].Error, "too many") {
		t.Errorf("error should mention 'too many', got: %s", output.Results[0].Error)
	}
}

func TestSubagentChainMode_AllUnknownAgents(t *testing.T) {
	orch := subagent.NewOrchestrator(defaultConfigPtr(), "", nil)

	input := SubagentInput{Chain: []ChainItem{
		{Agent: "unknown1", Task: "step 1"},
		{Agent: "unknown2", Task: "step 2"},
	}}
	output, err := subagentHandler(nil, orch, input, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if output.Mode != "chain" {
		t.Errorf("mode = %q, want 'chain'", output.Mode)
	}
	if len(output.Results) != 1 {
		t.Fatalf("expected 1 result (validation), got %d", len(output.Results))
	}
	if output.Results[0].Status != "failed" {
		t.Errorf("status = %q, want 'failed'", output.Results[0].Status)
	}
}

// --- Event callback tests ---

func TestEmitEvent_NilCallback(t *testing.T) {
	// Should not panic with nil callback.
	emitEvent(nil, SubagentEvent{AgentID: "test", Kind: "spawn"})
}

func TestEmitEvent_CallsCallback(t *testing.T) {
	var mu sync.Mutex
	var received []SubagentEvent

	cb := func(ev SubagentEvent) {
		mu.Lock()
		defer mu.Unlock()
		received = append(received, ev)
	}

	emitEvent(cb, SubagentEvent{AgentID: "test-1", Kind: "spawn", PipelineID: "p-1", Mode: "single", Step: 1, Total: 1})

	mu.Lock()
	defer mu.Unlock()
	if len(received) != 1 {
		t.Fatalf("expected 1 event, got %d", len(received))
	}
	ev := received[0]
	if ev.AgentID != "test-1" {
		t.Errorf("AgentID = %q, want 'test-1'", ev.AgentID)
	}
	if ev.Kind != "spawn" {
		t.Errorf("Kind = %q, want 'spawn'", ev.Kind)
	}
	if ev.PipelineID != "p-1" {
		t.Errorf("PipelineID = %q, want 'p-1'", ev.PipelineID)
	}
	if ev.Mode != "single" {
		t.Errorf("Mode = %q, want 'single'", ev.Mode)
	}
	if ev.Step != 1 {
		t.Errorf("Step = %d, want 1", ev.Step)
	}
	if ev.Total != 1 {
		t.Errorf("Total = %d, want 1", ev.Total)
	}
}

// --- expandChainTemplate tests ---

func TestExpandChainTemplate(t *testing.T) {
	tests := []struct {
		name     string
		task     string
		prev     string
		expected string
	}{
		{
			name:     "no placeholders",
			task:     "analyze the code",
			prev:     "some result",
			expected: "analyze the code",
		},
		{
			name:     "previous placeholder",
			task:     "review this: {previous}",
			prev:     "hello world",
			expected: "review this: hello world",
		},
		{
			name:     "previous_json placeholder",
			task:     `embed: "{previous_json}"`,
			prev:     "line1\nline2\t\"quoted\"",
			expected: `embed: "line1\nline2\t\"quoted\""`,
		},
		{
			name:     "both placeholders",
			task:     "text: {previous}, json: {previous_json}",
			prev:     "hello\nworld",
			expected: `text: hello` + "\n" + `world, json: hello\nworld`,
		},
		{
			name:     "empty previous",
			task:     "do {previous} stuff",
			prev:     "",
			expected: "do {previous} stuff",
		},
		{
			name:     "multiple occurrences",
			task:     "{previous} and {previous}",
			prev:     "X",
			expected: "X and X",
		},
		{
			name:     "backslash in previous",
			task:     "{previous_json}",
			prev:     `path\to\file`,
			expected: `path\\to\\file`,
		},
		{
			name:     "carriage return in previous",
			task:     "{previous_json}",
			prev:     "line1\r\nline2",
			expected: `line1\r\nline2`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := expandChainTemplate(tt.task, tt.prev)
			if result != tt.expected {
				t.Errorf("expandChainTemplate(%q, %q) = %q, want %q", tt.task, tt.prev, result, tt.expected)
			}
		})
	}
}

// --- buildSubagentDescription tests ---

func TestBuildSubagentDescription(t *testing.T) {
	agents := []subagent.AgentConfig{
		{Name: "explore", Description: "Fast codebase exploration", Role: "default"},
		{Name: "task", Description: "Complete coding tasks", Role: "default", Worktree: true},
	}
	orch := subagent.NewOrchestrator(defaultConfigPtr(), "", agents)

	desc := buildSubagentDescription(orch)
	if !strings.Contains(desc, "single:") {
		t.Error("description should mention single mode")
	}
	if !strings.Contains(desc, "parallel:") {
		t.Error("description should mention parallel mode")
	}
	if !strings.Contains(desc, "chain:") {
		t.Error("description should mention chain mode")
	}
	// Agent names should be listed (order may vary).
	if !strings.Contains(desc, "explore") {
		t.Error("description should list 'explore' agent")
	}
	if !strings.Contains(desc, "task") {
		t.Error("description should list 'task' agent")
	}
	if !strings.Contains(desc, "task [worktree]") {
		t.Error("description should mark worktree agents")
	}
	if !strings.Contains(desc, "never land in this tree") {
		t.Error("description should explain worktree edits are not automatically applied")
	}
	if !strings.Contains(desc, "exact patch or file list") {
		t.Error("description should explain the handoff needed for worktree edits")
	}
}

// --- SubagentTools registration test ---

func TestSubagentTools_Registration(t *testing.T) {
	agents := []subagent.AgentConfig{
		{Name: "explore", Description: "test", Role: "default"},
	}
	orch := subagent.NewOrchestrator(defaultConfigPtr(), "", agents)

	tools, err := SubagentTools(orch, nil)
	if err != nil {
		t.Fatalf("SubagentTools: %v", err)
	}
	if len(tools) != 2 {
		t.Fatalf("expected 2 tools, got %d", len(tools))
	}
	if tools[0].Name() != "subagent" {
		t.Errorf("expected tool[0] name 'subagent', got %q", tools[0].Name())
	}
	if tools[1].Name() != "agent_result" {
		t.Errorf("expected tool[1] name 'agent_result', got %q", tools[1].Name())
	}
}

// --- resolveContext tests ---

func TestResolveContext_Nil(t *testing.T) {
	ctx := resolveContext(nil)
	if ctx == nil {
		t.Error("expected non-nil context from nil input")
	}
}

// --- consumeAgentEvents tests ---

func TestConsumeAgentEvents_TextOnly(t *testing.T) {
	ch := make(chan subagent.Event, 3)
	ch <- subagent.Event{Type: "text_delta", Content: "Hello "}
	ch <- subagent.Event{Type: "text_delta", Content: "world"}
	close(ch)

	text, status, errMsg := consumeAgentEvents(ch)
	if status != "completed" {
		t.Errorf("status = %q, want completed", status)
	}
	if errMsg != "" {
		t.Errorf("errMsg = %q, want empty", errMsg)
	}
	if text != "Hello world" {
		t.Errorf("text = %q, want %q", text, "Hello world")
	}
}

func TestConsumeAgentEvents_WithError(t *testing.T) {
	ch := make(chan subagent.Event, 3)
	ch <- subagent.Event{Type: "text_delta", Content: "partial"}
	ch <- subagent.Event{Type: "error", Error: "timeout"}
	close(ch)

	_, status, errMsg := consumeAgentEvents(ch)
	if status != "failed" {
		t.Errorf("status = %q, want failed", status)
	}
	if errMsg != "timeout" {
		t.Errorf("errMsg = %q, want timeout", errMsg)
	}
}

func TestConsumeAgentEvents_Empty(t *testing.T) {
	ch := make(chan subagent.Event)
	close(ch)

	text, status, errMsg := consumeAgentEvents(ch)
	if status != "completed" {
		t.Errorf("status = %q, want completed", status)
	}
	if text != "" {
		t.Errorf("text = %q, want empty", text)
	}
	if errMsg != "" {
		t.Errorf("errMsg = %q, want empty", errMsg)
	}
}

// --- buildParallelSummary tests ---

func TestBuildParallelSummary_AllCompleted(t *testing.T) {
	results := []AgentResult{
		{Status: "completed"},
		{Status: "completed"},
		{Status: "completed"},
	}
	got := buildParallelSummary(results, 3, "5s")
	if got != "parallel: 3/3 completed in 5s" {
		t.Errorf("got %q", got)
	}
}

func TestBuildParallelSummary_WithFailures(t *testing.T) {
	results := []AgentResult{
		{Status: "completed"},
		{Status: "failed"},
		{Status: "completed"},
	}
	got := buildParallelSummary(results, 3, "10s")
	if !strings.Contains(got, "2/3 completed") {
		t.Errorf("missing completed count in %q", got)
	}
	if !strings.Contains(got, "1 failed") {
		t.Errorf("missing failed count in %q", got)
	}
}

// --- buildChainSummary tests ---

func TestBuildChainSummary_AllCompleted(t *testing.T) {
	results := []AgentResult{
		{Status: "completed"},
		{Status: "completed"},
	}
	got := buildChainSummary(results, 2, "3s")
	if got != "chain: 2/2 steps completed in 3s" {
		t.Errorf("got %q", got)
	}
}

func TestBuildChainSummary_StoppedEarly(t *testing.T) {
	results := []AgentResult{
		{Status: "completed"},
		{Status: "failed"},
	}
	got := buildChainSummary(results, 4, "7s")
	if !strings.Contains(got, "stopped at step 2/4") {
		t.Errorf("got %q", got)
	}
}

// --- NewSubagentTool tests ---

func TestNewSubagentTool(t *testing.T) {
	agents := []subagent.AgentConfig{
		{Name: "explore", Description: "test", Role: "default"},
	}
	orch := subagent.NewOrchestrator(defaultConfigPtr(), "", agents)

	tool, err := NewSubagentTool(orch, nil, newBackgroundRegistry())
	if err != nil {
		t.Fatalf("NewSubagentTool() error = %v", err)
	}
	if tool == nil {
		t.Fatal("NewSubagentTool() returned nil tool")
	}
	if tool.Name() != "subagent" {
		t.Errorf("Name() = %q, want 'subagent'", tool.Name())
	}
}

// --- expandChainTemplate edge cases ---

func TestExpandChainTemplate_NewlinesAndSpecialChars(t *testing.T) {
	tests := []struct {
		name     string
		task     string
		prev     string
		expected string
	}{
		{
			name:     "tab character",
			task:     "data: {previous}",
			prev:     "col1\tcol2",
			expected: "data: col1\tcol2",
		},
		{
			name:     "quote in json",
			task:     `{json: "{previous_json}"}`,
			prev:     `say "hello"`,
			expected: `{json: "say \"hello\""}`,
		},
		{
			name:     "unicode placeholder",
			task:     "result: {previous}",
			prev:     "日本語",
			expected: "result: 日本語",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := expandChainTemplate(tt.task, tt.prev)
			if result != tt.expected {
				t.Errorf("got %q, want %q", result, tt.expected)
			}
		})
	}
}

// --- detectMode edge cases ---

func TestDetectMode_ParallelOverridesSingle(t *testing.T) {
	// When both agent/task and tasks are provided, tasks should win
	input := SubagentInput{
		Agent: "explore",
		Task:  "some task",
		Tasks: []TaskItem{{Agent: "task", Task: "b"}},
	}
	if mode := detectMode(input); mode != "parallel" {
		t.Errorf("detectMode = %q, want 'parallel'", mode)
	}
}

func TestDetectMode_ChainOverridesBoth(t *testing.T) {
	input := SubagentInput{
		Agent: "explore",
		Task:  "some task",
		Tasks: []TaskItem{{Agent: "task", Task: "b"}},
		Chain: []ChainItem{{Agent: "task", Task: "c"}},
	}
	if mode := detectMode(input); mode != "chain" {
		t.Errorf("detectMode = %q, want 'chain'", mode)
	}
}

// --- SubagentEvent tests ---

func TestSubagentEvent(t *testing.T) {
	ev := SubagentEvent{
		AgentID:    "agent-123",
		Kind:       "spawn",
		Content:    "explore",
		PipelineID: "pipe-456",
		Mode:       "single",
		Step:       1,
		Total:      1,
	}

	if ev.AgentID != "agent-123" {
		t.Errorf("AgentID = %q", ev.AgentID)
	}
	if ev.Kind != "spawn" {
		t.Errorf("Kind = %q", ev.Kind)
	}
	if ev.PipelineID != "pipe-456" {
		t.Errorf("PipelineID = %q", ev.PipelineID)
	}
	if ev.Mode != "single" {
		t.Errorf("Mode = %q", ev.Mode)
	}
	if ev.Step != 1 {
		t.Errorf("Step = %d", ev.Step)
	}
	if ev.Total != 1 {
		t.Errorf("Total = %d", ev.Total)
	}
}

func TestSubagentOutput(t *testing.T) {
	output := SubagentOutput{
		Mode: "single",
		Results: []AgentResult{
			{
				Agent:    "explore",
				AgentID:  "id-1",
				Status:   "completed",
				Result:   "analysis complete",
				Duration: "5s",
			},
		},
		Summary: "explore completed in 5s",
	}

	if output.Mode != "single" {
		t.Errorf("Mode = %q", output.Mode)
	}
	if len(output.Results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(output.Results))
	}
	if output.Results[0].Status != "completed" {
		t.Errorf("Status = %q", output.Results[0].Status)
	}
	if output.Results[0].Result != "analysis complete" {
		t.Errorf("Result = %q", output.Results[0].Result)
	}
}

func TestAgentResult(t *testing.T) {
	result := AgentResult{
		Agent:     "task",
		AgentID:   "id-2",
		Status:    "failed",
		Result:    "",
		Error:     "connection refused",
		Duration:  "1s",
		SessionID: "session-123",
	}

	if result.Agent != "task" {
		t.Errorf("Agent = %q", result.Agent)
	}
	if result.Status != "failed" {
		t.Errorf("Status = %q", result.Status)
	}
	if result.Error != "connection refused" {
		t.Errorf("Error = %q", result.Error)
	}
	if result.SessionID != "session-123" {
		t.Errorf("SessionID = %q", result.SessionID)
	}
}

// --- subagentHandler error mode tests ---

func TestSubagentHandler_AmbiguousInput(t *testing.T) {
	orch := subagent.NewOrchestrator(defaultConfigPtr(), "", nil)

	// Both agent AND tasks provided - should be parallel mode
	input := SubagentInput{
		Agent: "someagent",
		Tasks: []TaskItem{{Agent: "a", Task: "b"}},
	}
	_, err := subagentHandler(nil, orch, input, nil, nil)
	// This should error for unknown agent "someagent" (in parallel validation)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

// --- Background mode tests ---

func TestSubagentHandler_BackgroundSingle(t *testing.T) {
	agents := []subagent.AgentConfig{
		{Name: "explore", Description: "test", Role: "default"},
	}
	orch := subagent.NewOrchestrator(defaultConfigPtr(), "", agents)
	bg := newBackgroundRegistry()

	// Background single mode should return immediately with agent_id.
	input := SubagentInput{Agent: "explore", Task: "do something", Background: true}
	output, err := subagentHandler(nil, orch, input, nil, bg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if output.Mode != "single" {
		t.Errorf("mode = %q, want 'single'", output.Mode)
	}
	if len(output.Results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(output.Results))
	}
	r := output.Results[0]
	if r.Status != "background" {
		t.Errorf("status = %q, want 'background'", r.Status)
	}
	if r.AgentID == "" {
		t.Error("expected non-empty agent_id")
	}
	if !strings.Contains(output.Summary, "agent_result") {
		t.Error("summary should mention agent_result")
	}
}

func TestSubagentHandler_BackgroundRejectsParallel(t *testing.T) {
	orch := subagent.NewOrchestrator(defaultConfigPtr(), "", nil)
	bg := newBackgroundRegistry()

	input := SubagentInput{
		Background: true,
		Tasks:      []TaskItem{{Agent: "explore", Task: "a"}},
	}
	output, err := subagentHandler(nil, orch, input, nil, bg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if output.Mode != "parallel" {
		t.Errorf("mode = %q, want 'parallel'", output.Mode)
	}
	if len(output.Results) != 1 || output.Results[0].Status != "failed" {
		t.Fatalf("expected failed result, got %+v", output.Results)
	}
	if !strings.Contains(output.Results[0].Error, "single-mode") {
		t.Errorf("error should mention single-mode: %q", output.Results[0].Error)
	}
}

func TestSubagentHandler_BackgroundRejectsChain(t *testing.T) {
	orch := subagent.NewOrchestrator(defaultConfigPtr(), "", nil)
	bg := newBackgroundRegistry()

	input := SubagentInput{
		Background: true,
		Chain:      []ChainItem{{Agent: "explore", Task: "a"}},
	}
	output, err := subagentHandler(nil, orch, input, nil, bg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if output.Results[0].Status != "failed" {
		t.Fatalf("expected failed result, got %+v", output.Results)
	}
}

func TestSubagentHandler_BackgroundUnknownAgent(t *testing.T) {
	agents := []subagent.AgentConfig{
		{Name: "explore", Description: "test", Role: "default"},
	}
	orch := subagent.NewOrchestrator(defaultConfigPtr(), "", agents)
	bg := newBackgroundRegistry()

	input := SubagentInput{Agent: "nonexistent", Task: "do something", Background: true}
	output, err := subagentHandler(nil, orch, input, nil, bg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if output.Results[0].Status != "failed" {
		t.Errorf("status = %q, want 'failed'", output.Results[0].Status)
	}
}

// --- Background registry tests ---

func TestBackgroundRegistry_StartFinishGet(t *testing.T) {
	bg := newBackgroundRegistry()

	_ = bg.start("agent-1", "explore")

	// Verify running state.
	status, _, _, _, ok := bg.get("agent-1")
	if !ok {
		t.Fatal("get returned not found")
	}
	if status != "running" {
		t.Errorf("status = %q, want 'running'", status)
	}

	// Finish with success.
	bg.finish("agent-1", "completed", "analysis result", "")
	status, result, errMsg, done, ok := bg.get("agent-1")
	if !ok {
		t.Fatal("get returned not found after finish")
	}
	if status != "completed" {
		t.Errorf("status = %q, want 'completed'", status)
	}
	if result != "analysis result" {
		t.Errorf("result = %q", result)
	}
	if errMsg != "" {
		t.Errorf("error = %q, want empty", errMsg)
	}

	// done channel should be closed after finish.
	select {
	case <-done:
	default:
		t.Error("done channel should be closed")
	}
}

func TestBackgroundRegistry_FinishWithError(t *testing.T) {
	bg := newBackgroundRegistry()

	bg.start("agent-2", "task")
	bg.finish("agent-2", "failed", "", "timeout")
	status, _, errMsg, _, ok := bg.get("agent-2")
	if !ok {
		t.Fatal("get returned not found")
	}
	if status != "failed" {
		t.Errorf("status = %q, want 'failed'", status)
	}
	if errMsg != "timeout" {
		t.Errorf("error = %q, want 'timeout'", errMsg)
	}
}

func TestBackgroundRegistry_FinishUnknown(t *testing.T) {
	bg := newBackgroundRegistry()
	// Must not panic.
	bg.finish("nonexistent", "completed", "", "")
}

func TestBackgroundRegistry_GetUnknown(t *testing.T) {
	bg := newBackgroundRegistry()
	_, _, _, _, ok := bg.get("nonexistent")
	if ok {
		t.Error("expected not found")
	}
}

func TestBackgroundRegistry_List(t *testing.T) {
	bg := newBackgroundRegistry()
	bg.start("a-1", "explore")
	bg.start("a-2", "task")

	ids := bg.list()
	if len(ids) != 2 {
		t.Fatalf("expected 2 IDs, got %d: %v", len(ids), ids)
	}
}

// --- agent_result tool tests ---

func TestAgentResult_UnknownID(t *testing.T) {
	bg := newBackgroundRegistry()

	out, err := agentResultHandler(bg, agentResultInput{AgentID: "nonexistent"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out.Status != "unknown" {
		t.Errorf("status = %q, want 'unknown'", out.Status)
	}
	if !strings.Contains(out.Summary, "nonexistent") {
		t.Error("summary should mention the unknown ID")
	}
}

func TestAgentResult_RunningNoWait(t *testing.T) {
	bg := newBackgroundRegistry()
	bg.start("r-1", "explore")

	out, err := agentResultHandler(bg, agentResultInput{AgentID: "r-1"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out.Status != "running" {
		t.Errorf("status = %q, want 'running'", out.Status)
	}
	if out.Hint == "" {
		t.Error("expected hint for running agent with no wait")
	}
}

func TestAgentResult_RunningWithWait(t *testing.T) {
	bg := newBackgroundRegistry()
	bg.start("rw-1", "explore")

	// With wait_seconds=0.1 and fast completion in a goroutine.
	go func() {
		time.Sleep(50 * time.Millisecond)
		bg.finish("rw-1", "completed", "done", "")
	}()

	out, err := agentResultHandler(bg, agentResultInput{AgentID: "rw-1", WaitSeconds: 1})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out.Status != "completed" {
		t.Errorf("status = %q, want 'completed'", out.Status)
	}
	if out.Result != "done" {
		t.Errorf("result = %q, want 'done'", out.Result)
	}
}

func TestAgentResult_Done(t *testing.T) {
	bg := newBackgroundRegistry()
	bg.start("d-1", "explore")
	bg.finish("d-1", "completed", "full result text", "")

	out, err := agentResultHandler(bg, agentResultInput{AgentID: "d-1"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out.Status != "completed" {
		t.Errorf("status = %q, want 'completed'", out.Status)
	}
	if out.Result != "full result text" {
		t.Errorf("result = %q", out.Result)
	}
}

func TestAgentResult_WaitTimeout(t *testing.T) {
	bg := newBackgroundRegistry()
	bg.start("wt-1", "explore")

	// Very short wait, agent never finishes.
	out, err := agentResultHandler(bg, agentResultInput{AgentID: "wt-1", WaitSeconds: 0.05})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out.Status != "running" {
		t.Errorf("status = %q, want 'running' (timed out)", out.Status)
	}
}

// --- Concurrent registry tests ---

func TestBackgroundRegistry_ConcurrentAccess(t *testing.T) {
	bg := newBackgroundRegistry()
	var wg sync.WaitGroup

	for i := range 10 {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			id := fmt.Sprintf("c-%d", n)
			bg.start(id, "explore")
			bg.finish(id, "completed", "result", "")
			bg.get(id)
		}(i)
	}
	wg.Wait()

	ids := bg.list()
	if len(ids) != 10 {
		t.Errorf("expected 10 IDs, got %d", len(ids))
	}
}
