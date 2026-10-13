package tools

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/spa-skyson/pi-rate/internal/config"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool"
)

// validTodoStatuses lists the allowed values for TodoItem.Status.
var validTodoStatuses = map[string]bool{
	"pending":     true,
	"in_progress": true,
	"completed":   true,
}

// TodoItem is one item in a plan.
type TodoItem struct {
	Content string `json:"content"`
	Status  string `json:"status"`
}

// TodoWriteInput is the write-todo argument struct.
type TodoWriteInput struct {
	Todos []TodoItem `json:"todos"`
}

// TodoWriteOutput is the write-todo result.
type TodoWriteOutput struct {
	Total      int    `json:"total"`
	Done       int    `json:"done"`
	InProgress string `json:"in_progress,omitempty"` // content of first in_progress item
	Saved      bool   `json:"saved"`
	Truncated  bool   `json:"truncated,omitempty"` // ponytail: always false, exists for compactor
}

// TodoReadOutput is the read-todo result.
type TodoReadOutput struct {
	Todos []TodoItem `json:"todos"`
}

// TodoState is the aggregated state the TUI subscribes to (sent via notifier).
type TodoState struct {
	Total      int        `json:"total"`
	Done       int        `json:"done"`
	InProgress string     `json:"in_progress,omitempty"` // content of first in_progress item
	Items      []TodoItem `json:"items"`                 // full list for the popup
}

// todoSessionDir returns the session's todos directory under ~/.pirate/sessions/.
func todoSessionDir(sessionID string) string {
	if _, err := os.UserHomeDir(); err != nil {
		return ""
	}
	return filepath.Join(config.PirateHome(), "sessions", sessionID)
}

// todoFilePath returns the path to the todos.json file for a session.
func todoFilePath(sessionID string) string {
	return filepath.Join(todoSessionDir(sessionID), "todos.json")
}

// loadTodos reads todos from disk. Returns an empty list when the file does not exist.
func loadTodos(sessionID string) ([]TodoItem, error) {
	path := todoFilePath(sessionID)
	b, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("reading todos: %w", err)
	}
	var items []TodoItem
	if err := json.Unmarshal(b, &items); err != nil {
		// Corrupt file — treat as empty rather than failing the tool call.
		return nil, nil
	}
	return items, nil
}

// saveTodos writes todos atomically (tmp+rename).
func saveTodos(sessionID string, items []TodoItem) error {
	path := todoFilePath(sessionID)
	dir := todoSessionDir(sessionID)
	if dir == "" {
		return fmt.Errorf("cannot determine session directory")
	}
	// Ensure directory exists.
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("creating session dir: %w", err)
	}
	// Write to tmp file, then rename.
	tmp, err := os.CreateTemp(dir, "todos.*.tmp")
	if err != nil {
		return fmt.Errorf("creating temp file: %w", err)
	}
	tmpName := tmp.Name()
	if err := json.NewEncoder(tmp).Encode(items); err != nil {
		_ = tmp.Close()
		_ = os.Remove(tmpName)
		return fmt.Errorf("encoding todos: %w", err)
	}
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmpName)
		return fmt.Errorf("closing temp file: %w", err)
	}
	if err := os.Rename(tmpName, path); err != nil {
		_ = os.Remove(tmpName)
		return fmt.Errorf("renaming todos file: %w", err)
	}
	return nil
}

// computeTodoState derives the aggregated TodoState from the full items list.
func computeTodoState(items []TodoItem) TodoState {
	s := TodoState{Total: len(items)}
	inProgress := ""
	for _, it := range items {
		switch it.Status {
		case "completed":
			s.Done++
		case "in_progress":
			if inProgress == "" {
				inProgress = it.Content
			}
		}
	}
	s.InProgress = inProgress
	s.Items = items
	return s
}

// validateTodos checks that every item has a non-empty content and a valid status.
func validateTodos(items []TodoItem) error {
	for i, it := range items {
		if strings.TrimSpace(it.Content) == "" {
			return fmt.Errorf("item %d: content must not be empty", i+1)
		}
		if !validTodoStatuses[it.Status] {
			return fmt.Errorf("item %d: invalid status %q; must be one of: pending, in_progress, completed", i+1, it.Status)
		}
	}
	return nil
}

// newTodoWriteTool creates the todo_write tool. sessionID must be non-empty.
// notifier, when non-nil, is called after every successful write with the new state.
func newTodoWriteTool(sessionID string, notifier func(TodoState)) (tool.Tool, error) {
	return newTool("todo_write",
		`Create or replace the session's todo list. Each item: {content, status: pending|in_progress|completed}. FULL REPLACEMENT — omitted items are deleted. Plan multi-step work first, then update statuses as you go.`,
		func(_ agent.Context, input TodoWriteInput) (TodoWriteOutput, error) {
			if err := validateTodos(input.Todos); err != nil {
				return TodoWriteOutput{}, err
			}
			if err := saveTodos(sessionID, input.Todos); err != nil {
				return TodoWriteOutput{}, fmt.Errorf("saving todos: %w", err)
			}
			st := computeTodoState(input.Todos)
			if notifier != nil {
				notifier(st)
			}
			return TodoWriteOutput{
				Total:      st.Total,
				Done:       st.Done,
				InProgress: st.InProgress,
				Saved:      true,
			}, nil
		})
}

// newTodoReadTool creates the todo_read tool. sessionID must be non-empty.
func newTodoReadTool(sessionID string) (tool.Tool, error) {
	return newTool("todo_read",
		`Read the current session's todo/plan list. Returns the full list.`,
		func(_ agent.Context, _ struct{}) (TodoReadOutput, error) {
			items, err := loadTodos(sessionID)
			if err != nil {
				return TodoReadOutput{}, fmt.Errorf("loading todos: %w", err)
			}
			if items == nil {
				items = []TodoItem{}
			}
			return TodoReadOutput{Todos: items}, nil
		})
}
