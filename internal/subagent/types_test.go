package subagent

import (
	"strings"
	"testing"
)

func TestBundledAgents_AllDefined(t *testing.T) {
	agents, err := LoadBundledAgents()
	if err != nil {
		t.Fatalf("LoadBundledAgents failed: %v", err)
	}

	// Should have exactly 5 bundled agents
	expected := []string{"captain", "first-mate", "skipper", "cabin-boy", "memory-compressor"}
	if len(agents) != len(expected) {
		t.Errorf("expected %d bundled agents, got %d: %v", len(expected), len(agents), agentNames(agents))
	}

	// Check all expected agents exist
	for _, name := range expected {
		found := false
		for _, a := range agents {
			if a.Name == name {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("expected agent %q not found", name)
		}
	}
}

func TestBundledAgents_RoleMappings(t *testing.T) {
	validRoles := map[string]bool{
		"default": true,
		"smol":    true,
		"slow":    true,
		"plan":    true,
	}

	agents, err := LoadBundledAgents()
	if err != nil {
		t.Fatalf("LoadBundledAgents failed: %v", err)
	}

	for _, agent := range agents {
		if agent.Role == "" {
			t.Errorf("agent %q has empty role", agent.Name)
		}
		if !validRoles[agent.Role] {
			t.Errorf("agent %q maps to unknown role %q", agent.Name, agent.Role)
		}
	}
}

func TestBundledAgents_HaveInstructions(t *testing.T) {
	agents, err := LoadBundledAgents()
	if err != nil {
		t.Fatalf("LoadBundledAgents failed: %v", err)
	}

	for _, agent := range agents {
		if agent.Instruction == "" {
			t.Errorf("agent %q has empty instruction", agent.Name)
		}
		if len(agent.Tools) == 0 {
			t.Errorf("agent %q has no tools", agent.Name)
		}
	}
}

func TestBundledAgents_WorktreeTypes(t *testing.T) {
	// cabin-boy is the only bundled agent that requires a worktree.
	worktreeTypes := map[string]bool{"cabin-boy": true}

	agents, err := LoadBundledAgents()
	if err != nil {
		t.Fatalf("LoadBundledAgents failed: %v", err)
	}

	for _, agent := range agents {
		if worktreeTypes[agent.Name] && !agent.Worktree {
			t.Errorf("agent %q should require worktree", agent.Name)
		}
		if !worktreeTypes[agent.Name] && agent.Worktree {
			t.Errorf("agent %q should NOT require worktree", agent.Name)
		}
	}
}

func TestBundledAgents_WorktreeInstructionsDescribeHandoff(t *testing.T) {
	agents, err := LoadBundledAgents()
	if err != nil {
		t.Fatalf("LoadBundledAgents failed: %v", err)
	}

	for _, agent := range agents {
		if !agent.Worktree {
			continue
		}
		for _, phrase := range []string{
			"local to this worktree",
			"main working tree",
			"changed files",
			"verification commands",
		} {
			if !strings.Contains(agent.Instruction, phrase) {
				t.Errorf("worktree agent %q instruction should contain %q", agent.Name, phrase)
			}
		}
	}
}

func TestAgentInput_ToSpawnInput(t *testing.T) {
	// Test valid agent type
	input := AgentInput{Type: "skipper", Prompt: "test prompt"}
	spawnInput, err := input.ToSpawnInput()
	if err != nil {
		t.Fatalf("ToSpawnInput failed: %v", err)
	}
	if spawnInput.Agent.Name != "skipper" {
		t.Errorf("expected agent name 'skipper', got %q", spawnInput.Agent.Name)
	}
	if spawnInput.Prompt != "test prompt" {
		t.Errorf("expected prompt 'test prompt', got %q", spawnInput.Prompt)
	}

	// Test invalid agent type
	inputInvalid := AgentInput{Type: "nonexistent", Prompt: "test"}
	_, err = inputInvalid.ToSpawnInput()
	if err == nil {
		t.Fatal("expected error for invalid agent type")
	}
}
