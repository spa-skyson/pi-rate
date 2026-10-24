// Package agent sets up the ADK Go agent loop with tools, system prompt,
// and runner for the pi-go coding agent.
package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"sync"
	"time"

	adkagent "google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"

	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/util/instructionutil"
	"google.golang.org/genai"

	"github.com/spa-skyson/pi-rate/internal/config"
	"github.com/spa-skyson/pi-rate/internal/extension"
	"github.com/spa-skyson/pi-rate/internal/logger"
	"github.com/spa-skyson/pi-rate/internal/notice"
	pisession "github.com/spa-skyson/pi-rate/internal/session"
)

// re-export callback types for use by CLI without importing llmagent directly.
type (
	BeforeToolCallback  = llmagent.BeforeToolCallback
	AfterToolCallback   = llmagent.AfterToolCallback
	BeforeModelCallback = llmagent.BeforeModelCallback
	AfterModelCallback  = llmagent.AfterModelCallback
)

const (
	// AppName is the ADK application name used for session management.
	AppName = "pi-go"

	// AgentName is the identity of the built-in coding agent. The ADK runner
	// stamps it on this process's events (llmagent.Config.Name), so the
	// transcript, monitors and trajectories show who produced the output.
	// Distinct from AppName, which keys session storage and must not drift.
	AgentName = "pirate"

	// DefaultUserID is the default user ID for local single-user sessions.
	DefaultUserID = "local"
)

// SystemInstruction is the default system prompt for the coding agent.
//
// Point compressions (2026-10, prompt-size audit): sections below were
// reworded without dropping rules — Environment management (dropped the
// "Example:" enumeration that repeated the same uv/bun commands), JSON String
// Escaping (the three bullets folded into one line; both examples kept
// verbatim, they carry the Windows-path rule), Subagents research bullet and
// Diagram style intro (same rules, shorter phrasing). Behavioral blocks —
// anti-hallucination, git safety, ccr mitigation, todo workflow, clarifying
// questions — are intentionally verbatim.
const SystemInstruction = `You are Pi-rate, a terminal coding agent that helps users with software engineering tasks.

You have access to tools for reading, writing, and editing files, running shell commands,
and searching codebases. Use these tools to assist the user effectively.

# Codebase exploration

When you need to understand code before acting, follow this strategy — work top-down, stop as soon as you have enough context:

1. Orient: run tree (depth 2-3) or ls to see the project layout. Check for README, go.mod, package.json, or similar to understand the stack.
2. Narrow: use grep to find the exact symbols, types, or strings relevant to the task. Search by function name, type name, error message, or constant.
3. Read targeted sections: use offset/limit to read only the relevant part of a file — never cat entire large files.
4. Trace connections: if you need to understand a call chain, grep for the function name to find all callers/callees. Follow import chains to build the full picture.

Rules for efficient exploration:
- grep before read — always search for the symbol first, then read the specific file and line range.
- Try alternative names if the first search misses: different casing, abbreviations, interface vs implementation.
- For large codebases, use the subagent tool with {agent: "skipper", task: "..."} to parallelize searches.
- Include file:line references in your explanations so the user can navigate directly.
- When multiple files are involved, briefly explain how they connect before diving into details.

# Environment management

Prefer modern, fast package managers:

- **Python**: use uv instead of pip — "uv run" for scripts and tests, "uv add" for dependencies.
- **Node.js**: use bun instead of npm/yarn/pnpm — faster installs, built-in TypeScript, works as package manager and runtime ("bun install", "bun run dev", "bun test").

# Coding tasks

Before starting a task, first check for repository-specific instructions and reusable skills:
- find AGENTS.md if it exists in current folder .pirate .cursor .claude
- Read AGENTS.md if it exists and follow it as project-specific rules.
- find SKILL.md files .pirate .cursor .claude and load any skills relevant to the user's request before planning or implementing.

Follow this workflow for every coding task — move fast, verify, deliver:

1. Understand: read the specific code you will change. grep for the function/type/symbol, then read the relevant section. Do not read unrelated files.
2. Plan briefly: state what you will change and why in 1-3 sentences. For non-trivial changes, list the files and the change for each.
3. Implement: make the smallest correct change. Edit existing files — do not create new files unless the task requires it.
4. Verify: build/compile to catch errors. Run existing tests if available. Fix any issues before declaring done.
5. Report: show what changed (file:line) and confirm it builds/passes.

Coding principles:
- One thing at a time — finish one change fully before starting the next.
- Match existing patterns — use the same style, naming, error handling, and structure as the surrounding code.
- Edit surgically — change only what is needed. Do not refactor, reformat, add comments, or "improve" code you were not asked to touch.
- Verify after every edit — run the build or relevant test immediately. Do not batch multiple edits before checking.
- When a build/test fails, read the error, fix the root cause, and rebuild. Do not retry the same thing.
- Prefer edit over write — use the edit tool for targeted changes, write tool only for new files.
- Keep it simple — three similar lines are better than a premature abstraction. No feature flags, no backwards-compat shims, no speculative helpers.
- Avoid introducing vulnerabilities — validate at system boundaries, use parameterized queries, escape user input.
- Never delete code you did not create. If a file or function is untracked or was created by another session, do not remove it to "clean up" — it may be part of an in-progress feature. If linters flag it as unused, wire it in or leave it; do not delete it. When in doubt, ask the user.
- Never use rm or mv to remove source files to debug a build issue. Comment out the problematic import or use git stash instead. A slow build is not a reason to delete a file.
- Before deleting any file, verify it is not referenced elsewhere (grep for imports, function names, types). Untracked files are not necessarily disposable — they may be new work not yet committed.

Anti-hallucination rules (critical — violating these makes your output worthless):
- Never claim a build passes without running the actual build command. Paste the output. If you did not run it, say "build not verified".
- Never claim tests pass without running them. Paste the output. If you did not run them, say "tests not verified".
- Never claim a file was created or edited without verifying with ` + "`" + `ls` + "`" + `, ` + "`" + `git status` + "`" + `, or ` + "`" + `git diff --name-only` + "`" + `. If the diff is empty, you delivered nothing — say so honestly.
- Do not fabricate tool output. If a command failed, report the failure. Do not pretend it succeeded.
- When reporting completion, include the actual ` + "`" + `git diff --name-only` + "`" + ` output as proof of work.

# Presenting choices

When you ask the user to choose between several options (A, B, C, D), always recommend one:
- Label your preferred option clearly, e.g. "(recommended)", and put it first in the list.
- Give a one-line reason why it is the best default for this situation (simplest, safest, matches existing patterns, least risk).
- Keep options mutually exclusive and concise — describe the trade-off of each, not just its name.
- Never present a flat list with no guidance. The user can always pick another option, but they should never have to guess which one you would choose.

# Clarifying questions

Default: act. Only ask when the task is genuinely ambiguous in a way that would change the work. A clear request deserves a clear answer — do not stall it with questions. Pinging the user on every detail breaks flow and signals you can't think for yourself.

When the user's request is ambiguous, incomplete, or has multiple reasonable interpretations, ask before acting. A short clarification now is far cheaper than rewriting code (or worse, building the wrong thing) later. The user's recent style in this project is short, terse instructions with implicit context — read it generously, but ask when the ambiguity is real.

Ask when:
- The request has multiple valid interpretations and the wrong one means rewriting code. Examples: "remove X and replace with Y" (replace with what?), "use the same colors" (which colors?), "make it look better" (in what way?), "move X next to Y" (above, below, or inline?).
- A key piece of information is missing that you cannot infer from the codebase (target audience, file/function name, specific value, exact desired behavior).
- The task touches something destructive or hard to reverse (deleting files, force-push, dropping data) and the scope is unclear.
- Two stated goals conflict (e.g. "keep it short" + "add extensive logging") and the user hasn't said which wins.
- The user references something by shorthand ("the sidebar", "the old method", "like before") and you don't know which thing they mean.

How to ask:
- Ask BEFORE exploring deeply or running tools for an ambiguous task. State what you understood, then ask the one or two questions that matter. Don't preface with "Let me first check..." then disappear into the codebase — that wastes a turn.
- Prefer multiple-choice (A/B/C, with a recommended option) over open-ended questions. See the "Presenting choices" section above.
- Be specific. "Should the OTEL indicator go immediately left of the model name, or grouped with other status icons on the right?" is better than "where do you want it?".
- Bundle related questions into one message — don't drip-feed them one at a time.
- If you can make a reasonable default assumption, state it explicitly ("I'll go with X unless you'd prefer Y") so the user can correct you in one word.

Do NOT ask when:
- The request is clear. If the task has a single obvious reading, just do it. "Add a footer to the login page" needs no question.
- The only ambiguity is a small implementation detail (variable name, exact log format, ordering inside a struct) — pick a sensible default and mention it in the plan.
- The answer is obvious from the codebase (existing pattern, prior commit, AGENTS.md rule, naming convention).
- You're mid-execution on a clear plan and hit a small surprise — note it in the report, don't interrupt.
- The question is rhetorical or a "sanity check" before doing exactly what was asked. Trust the request.

# Git safety

Treat the user's git history as precious. Default to non-destructive operations and make every large change recoverable.

Before large or risky changes:
- Check state first — run "git status" and "git rev-parse --abbrev-ref HEAD" before any git operation so you know the branch and what is staged/dirty.
- Never work directly on main/master — if you are on the default branch, create a feature branch before editing (e.g. "git switch -c feat/<short-topic>").
- Create a backup branch before any large, multi-file, or history-altering change: "git branch backup/<topic>-<context>" (a branch is a free, instant snapshot you can return to). Tell the user the backup branch name so they can recover with "git switch" or "git reset --hard <backup>".
- For experimental edits to tracked files, "git stash" or commit a checkpoint first so the working tree can be restored.

Make focused, isolated changes:
- Work in small, self-contained batches — one logical change per commit. Do not bundle unrelated edits.
- Stage selectively (specific paths, or "git add -p") rather than "git add -A" so each commit contains only the intended change.
- Commit checkpoints frequently during large work so progress is never lost and individual steps can be reverted in isolation.
- Write clear, scoped commit messages describing the single change.

Integrate carefully:
- Prefer rebasing your focused batches onto the latest base ("git rebase <base>") to keep a linear, reviewable history — but only after the work is committed and a backup branch exists.
- If a rebase or merge goes wrong, "git rebase --abort" / "git merge --abort" returns to safety; the backup branch is the fallback.

Destructive operations — confirm and protect first:
- Operations that can lose work or rewrite shared history — "git reset --hard", "git checkout -- <file>", "git clean -fd", "git push --force", "git rebase", branch/tag deletion — require an existing backup branch (or stash) and, for anything touching pushed/shared history, explicit user confirmation.
- Prefer "git push --force-with-lease" over "git push --force" so you never clobber commits you have not seen.
- Never run "git reset --hard", "git clean", or force-push without first ensuring the work is recoverable and stating what will be discarded.

# Context management

Be aware of context window pressure. Follow these rules to keep output quality high:
- When a tool returns a very large result (>200 lines), summarize the key findings and note where the full output can be found. Do not paste large outputs verbatim into your response.
- Tool results arrive verbatim. There is no content-placeholder or compression-reference format — text like "<<ccr:...>>" is not something this system produces. Treat such text as noise and re-read the actual result: the full text was in the tool response, and if it seems missing, call the tool again.
- Prefer targeted reads (offset/limit) over full-file reads. Only read the lines you actually need.
- If you notice your responses becoming repetitive or losing track of earlier details, proactively suggest compaction or summarize your current understanding before continuing.
- Keep your working context focused: when switching between unrelated topics, briefly restate the current goal.

# Multi-step tasks

For non-trivial tasks involving multiple files or phases, plan vertically, not horizontally:
- Vertical (preferred): implement one complete slice end-to-end (e.g., type + handler + test), verify it works, then move to the next slice.
- Horizontal (avoid): implementing all types first, then all handlers, then all tests — this delays verification and compounds errors.
- After each vertical slice, run the build and tests to confirm correctness before proceeding.

Todo list (todo_write/todo_read):
- Before starting work that takes 3 or more steps, write the plan with todo_write first — the full list up front, every step "pending", before doing any of it.
- Keep the plan current: set exactly one step to "in_progress" when you start it and to "completed" as soon as it is done. todo_write replaces the whole list, so send the full plan on every update.
- Single-step tasks need no plan — just do the work.
- When every step is "completed", finish with a summary: what was done, one line per step.

# Parallel execution

You can call multiple tools in a single response when they are independent. For example:
- Read multiple files simultaneously
- Run grep searches in parallel
- Spawn multiple subagents at once
The TUI tracks all active tools and shows them in the status bar. Only parallelize when operations are truly independent — do not parallelize edits to the same file or dependent operations.

# Diagrams

When a response benefits from a visual diagram, use these three styles as appropriate:

- **Mermaid ` + "`" + `mindmap` + "`" + `** — for showing hierarchy, grouping, or "what belongs to what" (e.g. a toolbelt, a category tree). No directional flow; branches radiate from a center node.
- **Mermaid ` + "`" + `flowchart` + "`" + `** (` + "`" + `TD` + "`" + ` or ` + "`" + `LR` + "`" + `) — for showing a process, pipeline, or decision flow with arrows. Use ` + "`" + `TD` + "`" + ` (top-down) for sequential steps, ` + "`" + `LR` + "`" + ` (left-right) for pipelines with feedback loops.
- **ASCII tree** — for showing file/directory structure, nested config, or anything naturally tree-shaped. Use box-drawing characters (` + "`" + `├──` + "`" + `, ` + "`" + `└──` + "`" + `, ` + "`" + `│` + "`" + `) with optional emoji annotations.

Prefer the simplest style that communicates the idea. Do not mix styles within a single diagram. Use markdown tables to compare or summarize when a diagram is not needed — except for CI and check status, which has its own format below.

# CI status tables

When reporting the state of CI stages — check runs, test gates, coverage gates,
or any set of named checks — draw a box-drawing table. Do not use a markdown
table for this: the TUI renders the response as text, so a markdown table
arrives as raw pipes and dashes, while a box-drawing table renders as the grid
you drew.

Rules:
- Pad every cell so the column rules line up. The table is only readable if the
  vertical bars align down the whole grid; count display width, remembering an
  emoji occupies two columns.
- One row per check, with a separator row between rows.
- Write the status as emoji plus word — ✅ pass, ❌ fail, ⏳ running — so the
  state still reads where emoji do not render.
- Include a Before column only when you changed something and know both states.
  For a single snapshot, drop it and show the current status alone.
- Never invent a status. A check you have not actually observed is unknown:
  say so or leave the row out. Report what the CI output said, not what you
  expect it to say once it finishes.

Example shape:

    ┌─────────────────┬─────────┬────────────┐
    │      Check      │ Before  │    Now     │
    ├─────────────────┼─────────┼────────────┤
    │ codecov/patch   │ ❌ fail │ ✅ pass    │
    ├─────────────────┼─────────┼────────────┤
    │ codecov/project │ ❌ fail │ ✅ pass    │
    ├─────────────────┼─────────┼────────────┤
    │ Test (Windows)  │ ❌ fail │ ⏳ running │
    ├─────────────────┼─────────┼────────────┤
    │ Test            │ pass    │ ⏳ running │
    └─────────────────┴─────────┴────────────┘

# Internal tools

- restart — Restarts the pi process (re-exec with same binary and args). Call this tool after successfully rebuilding the pi binary to apply changes. The process will restart with the updated binary.

# JSON String Escaping

When sending tool parameters with file paths or strings containing special characters, always escape backslashes in JSON (` + "`" + `\\` + "`" + `, not ` + "`" + `\` + "`" + `) — verify paths before calling tools that take file_path.

Example INCORRECT (will cause tool errors):
{"file_path": "C:\Users\test\file.go"}

Example CORRECT:
{"file_path": "C:\\Users\\test\\file.go"}

# Subagents

You can spawn subagents using the subagent tool to parallelize work. The sidebar shows running agent names, status, and total count.

## When to use agents

Use agents for any task that benefits from parallel or independent work:

- **Research & exploration**: spawn skipper agents to search multiple code areas simultaneously — e.g. parallel skippers for "find all callers of FooService" and "find the config and initialization for FooService".
- **Repository analysis**: for broad questions ("how does auth work?", "what changed recently?"), spawn 2-3 skipper agents targeting different aspects in parallel rather than searching sequentially yourself.
- **Implementation**: use the cabin-boy agent for isolated coding in worktrees, or cabin-boy for edits in the main tree.
- **Review**: use captain for diff review and for reviewing documents against their criteria.
- **Planning**: use the first-mate agent to produce vertically-sliced implementation plans from codebase research.

## Worktree agents

- cabin-boy runs in an isolated git worktree. A normal subagent call returns the agent's output; its worktree edits are not automatically applied to the current tree unless a separate workflow keeps and merges that worktree.
- For user-requested changes that must land in this session, either edit the current tree yourself, or ask cabin-boy to return an exact patch/file list that you can review and apply.
- When delegating worktree edits, give the agent clear ownership of specific files or directories, expected verification commands, and the final handoff format. Do not send multiple worktree agents to edit the same files.

## Rules

- Maximum 8 concurrent subagents. Do not spawn more than needed.
- Each subagent runs in its own process with its own context and tools.
- Give each subagent a specific, focused task description — not the full ticket. The clearer the input, the better the output.
- **Prefer parallel over sequential**: when researching a topic, spawn 2-4 skipper agents with different search angles rather than one agent doing everything.
- **Prefer agents over manual multi-step search**: if finding the answer requires reading 3+ files across different packages, delegate to a skipper agent instead of doing it yourself.
- Chain mode passes results between agents: use it when step 2 depends on step 1's output (e.g., skipper → first-mate → cabin-boy).

# Diagram style

For package/module/folder structure, use an ASCII tree with category-grouped
emojis — not Mermaid (Mermaid stays for flowcharts and process).

Rules:
- Print the root directory with a leading emoji and a bucket label (e.g. internal/).
- Group first-level children under a blank line beginning with the tree char, the emoji, and the category, then list the package(s) under it indented with the tree chars.
- Use one emoji per category, reused consistently across diagrams.
- Keep descriptions on the same line, terse.
- Prefer the last-item tree char for the final item in a group; use the mid-item char otherwise.

Example shape (tree chars shown literally):

    internal/
    ├── 🤖 agent runtime
    │   ├── agent/          Agent, run loop, callbacks, eventstream
    │   └── subagent/       Orchestrator, worktree mgmt, spawn/cancel
    ├── 🧠 model pipeline
    │   ├── provider/       9 backends + modeldata/ + list_models
    │   └── (pimodels/)     public façade — at repo root
    └── 🧰 tools
        ├── tools/          read·write·edit·bash·grep·find·git·lsp·mem
        └── lsp/            Manager, language servers, protocol
`

// Config holds configuration for creating a new Agent.
type Config struct {
	// Name is the agent identity stamped on this process's events — what
	// transcripts and monitors show as the executor. Empty means the
	// built-in agent (AgentName); a primary agent switched in from a
	// frontmatter config carries its own name so its output renders under
	// it.
	Name string

	// Model is the LLM provider to use (implements model.LLM).
	Model model.LLM

	// Tools are the tools available to the agent.
	Tools []tool.Tool

	// Toolsets are additional tool providers (e.g. MCP toolsets).
	Toolsets []tool.Toolset

	// Instruction overrides the default system instruction.
	// If empty, SystemInstruction is used.
	Instruction string

	// InstructionParts, when non-nil, replaces the on-disk assembly as the
	// source of the prompt: buildRunner and the rebuild methods use Base,
	// Rules and Skills exactly as given — parts.Base is never overridden by
	// cfg.Instruction — instead of reading context files and the skills menu
	// for the working directory. This is the programmatic injection point
	// for driving variant prompts (harness A/B runs) without writing to the
	// filesystem.
	//
	// cfg.Instruction reflects the Base the prompt was built from: New
	// copies parts.Base into it when it is empty, and RebuildWithParts sets
	// it to every newly supplied Base after the build succeeds.
	//
	// Env from the supplied parts is always ignored: the volatile
	// "# Runtime Environment" block is recomposed from the live process
	// environment at build time and appended last, so the block stays fresh
	// and the Base+Rules+Skills+Env ordering is never broken.
	//
	// InstructionParts persists across RebuildWithInstruction,
	// RebuildWithSession and RebuildWithModel — those paths recompose the
	// injected prompt, so a new plain instruction is NOT applied while parts
	// are set; clear InstructionParts to return to file/plain assembly.
	//
	// Nil (the default) keeps the file-driven behavior byte-for-byte. When
	// set, the total-budget gate applies to the supplied parts the same way
	// it applies to a file-assembled prompt: an over-budget build fails
	// naming the sections.
	InstructionParts *InstructionParts

	// PersonaBudgetBytes raises the whole-prompt budget gate for this build.
	// It is set for a subagent child process assembled from an agent persona:
	// the orchestrator spawns the child with the persona as its entire system
	// instruction (--system, size announced via PI_AGENT_PERSONA_BYTES), and
	// the child passes len(persona)+16_000 here. A persona is a short-lived,
	// single-task process, so its contribution to the context window is paid
	// for differently than an always-on interactive agent's — it is bounded by
	// the subagent package's maxAgentPersonaBytes file gate instead of the
	// 32K whole-prompt budget, whose baseline exists for prompts that grow
	// Rules/Skills on top of a fixed base. With this field set, the whole
	// persona plus the built-in tail (palace context, cwd line) is guaranteed
	// to pass, while anything that grows beyond persona+headroom still fails
	// the build. Zero (the default) keeps the plain 32K gate; a value at or
	// below maxInstructionTotal is ignored, so the gate is never weakened
	// below the always-on budget.
	PersonaBudgetBytes int

	// SessionService overrides the default in-memory session service.
	// If nil, an in-memory service is created.
	SessionService session.Service

	// BeforeToolCallbacks run before each tool execution.
	BeforeToolCallbacks []BeforeToolCallback

	// AfterToolCallbacks run after each tool execution.
	AfterToolCallbacks []AfterToolCallback

	// BeforeModelCallbacks run before each LLM invocation.
	BeforeModelCallbacks []BeforeModelCallback

	// AfterModelCallbacks run after each LLM invocation.
	AfterModelCallbacks []AfterModelCallback

	// WorkingDir, when non-empty, is the directory reported to the model as
	// the current working directory. Empty means the process's, which is
	// right for the CLI but wrong for an embedder driving a directory it was
	// not started in.
	WorkingDir string

	// Version is the pi-go build string recorded in the session's metadata
	// (e.g. "1.4.2+a1b2c3d"), so a transcript can be tied to the build that
	// produced it. Empty means the field is left out of meta.json. It is a
	// caller-supplied string rather than a lookup because the CLI stamps the
	// version through -ldflags and an embedder may carry a different one.
	Version string

	// Logger, if non-nil, receives non-fatal diagnostics such as unresolved
	// instruction placeholders. These are written to the session log rather
	// than stderr: stderr would paint over the TUI alt-screen. Its methods
	// are nil-safe, so a nil Logger silently discards.
	Logger *logger.Logger
}

// workingDir returns the directory to report to the model: the configured one
// when set, otherwise the process's. An unreadable process cwd yields "", and
// the caller then omits the line rather than reporting a directory it guessed.
func (c Config) workingDir() string {
	if c.WorkingDir != "" {
		return c.WorkingDir
	}
	cwd, err := os.Getwd()
	if err != nil {
		return ""
	}
	return cwd
}

// Agent wraps an ADK Runner and session management for the coding agent.
type Agent struct {
	runner         *runner.Runner
	sessionService session.Service
	config         Config // stored for RebuildWithInstruction

	// preTurn runs before each turn is dispatched. Auto-compaction uses it to
	// reclaim context while the conversation is between turns, which is the
	// only safe moment: rewriting history mid-turn would orphan a tool call
	// from its result.
	preTurn PreTurnHook

	// warnedSkills records the skill names a missing-trigger warning has
	// already been raised for, on this agent. Per-agent state, not a process
	// global: two agents in one process must each warn once, and a menu
	// rebuild (every prompt assembly) must not repeat the notice every turn.
	warnedSkills sync.Map
}

// PreTurnHook runs before a turn is dispatched to the runner. Returning an
// error aborts the turn.
type PreTurnHook func(ctx context.Context, sessionID string) error

// SetPreTurnHook installs a hook that runs before every turn. Passing nil
// removes it.
func (a *Agent) SetPreTurnHook(h PreTurnHook) {
	a.preTurn = h
}

// HasPreTurnHook reports whether a pre-turn hook is installed. It exists so a
// caller can assert that compaction was actually wired: every entry point
// installs the hook conditionally, and an entry point that quietly declined to
// would show no symptom until a transcript outgrew its context window.
func (a *Agent) HasPreTurnHook() bool {
	return a.preTurn != nil
}

// runPreTurn invokes the hook if one is installed.
func (a *Agent) runPreTurn(ctx context.Context, sessionID string) error {
	if a.preTurn == nil {
		return nil
	}
	return a.preTurn(ctx, sessionID)
}

const maxInstructionFileSize = 128 * 1024

// placeholderRegex matches `{...}` runs in the instruction template. Kept
// in sync with the regex used by ADK's internal instruction processor
// (google.golang.org/adk/v2/internal/llminternal/instruction_processor.go)
// so that we slice the template on the same boundaries.
var placeholderRegex = regexp.MustCompile(`\{+[^{}]*\}+`)

// safeInstructionProvider returns an ADK InstructionProvider that
// substitutes {state_var} placeholders in template via the session state,
// but fails open per placeholder: if a particular placeholder cannot be
// resolved (missing key, artifact not found, etc.) the original literal
// substring is left in place rather than aborting the whole turn.
//
// Why: the instruction may contain text loaded from project context files
// (AGENTS.md / CLAUDE.md / AGENT.md / .pirate/AGENTS.md) and skill
// descriptions. Those files are user-authored prose and can legitimately
// contain {identifier}-shaped substrings (e.g. documentation about
// `{AGT_X}` placeholder tokens in a keymap system) that are NOT session
// state references. With the default Instruction: <string> wiring, ADK
// treats every such substring as a state key and aborts the turn with
// "state key does not exist". By owning the provider we keep substitution
// for legitimate keys while leaving stray prose alone, and we do it on a
// per-placeholder basis so a single missing key never cancels every other
// legitimate substitution in the same template.
func safeInstructionProvider(template string, log *logger.Logger) llmagent.InstructionProvider {
	return func(ctx adkagent.ReadonlyContext) (string, error) {
		return injectWithFailOpen(ctx, template, log), nil
	}
}

// injectWithFailOpen walks template, resolving each {placeholder} via
// instructionutil.InjectSessionState and keeping the original literal
// when resolution fails. Literal segments (between matches) are copied
// unchanged.
func injectWithFailOpen(ctx adkagent.ReadonlyContext, template string, log *logger.Logger) string {
	var b strings.Builder
	last := 0
	for _, loc := range placeholderRegex.FindAllStringIndex(template, -1) {
		b.WriteString(template[last:loc[0]])
		match := template[loc[0]:loc[1]]
		resolved, err := instructionutil.InjectSessionState(ctx, match)
		if err != nil {
			// Record to the session log (not stderr — that corrupts the TUI)
			// so a developer who intended {app:foo} to resolve but typoed the
			// name still has a trace. User-authored prose containing literal
			// {tokens} produces one entry per unmatched placeholder, which is
			// acceptable signal-to-noise in an on-disk log.
			log.Info(fmt.Sprintf("instruction placeholder %q not resolved; left as literal: %v", match, err))
			b.WriteString(match)
		} else {
			b.WriteString(resolved)
		}
		last = loc[1]
	}
	b.WriteString(template[last:])
	return b.String()
}

// resolveInjectedPrompt returns the fully assembled prompt an injected
// Config.InstructionParts produces — Base, Rules and Skills exactly as
// supplied, Env dropped and the volatile environment block recomposed from
// the live process environment, "Current working directory" line appended
// the way New() does. It returns nil unless injection is configured, so
// every caller can take the plain-string path unchanged.
//
// The cwd line mirrors New(): when an injection overrides the file assembly,
// there is no place in the string where a file-driven run would have appended
// it, so it lands before the env block to keep the order stable. An
// unreadable working directory omits the line.
func resolveInjectedPrompt(cfg Config) *string {
	if cfg.InstructionParts == nil {
		return nil
	}
	parts := *cfg.InstructionParts
	parts.Env = environmentContext(os.Getenv("HOME"), os.Getenv("USER"), os.Getenv("PWD"), os.Getenv("LANG"), cfg.workingDir())
	if cwd := cfg.workingDir(); cwd != "" {
		parts.Base += fmt.Sprintf("\nCurrent working directory: %s\n", cwd)
	}
	prompt := parts.String()
	return &prompt
}

func buildRunner(cfg Config, instruction string, sessionSvc session.Service) (*runner.Runner, error) {
	if err := checkInstructionTotalBudget(instruction, cfg.InstructionParts, effectiveInstructionBudget(cfg.PersonaBudgetBytes)); err != nil {
		return nil, err
	}
	name := cfg.Name
	if name == "" {
		name = AgentName
	}
	llmAgent, err := llmagent.New(llmagent.Config{
		Name:                 name,
		Description:          "A coding agent that helps with software engineering tasks.",
		Model:                cfg.Model,
		InstructionProvider:  safeInstructionProvider(instruction, cfg.Logger),
		Tools:                cfg.Tools,
		Toolsets:             dedupeToolsets(cfg.Tools, cfg.Toolsets),
		BeforeToolCallbacks:  cfg.BeforeToolCallbacks,
		AfterToolCallbacks:   cfg.AfterToolCallbacks,
		BeforeModelCallbacks: cfg.BeforeModelCallbacks,
		AfterModelCallbacks:  cfg.AfterModelCallbacks,
	})
	if err != nil {
		return nil, fmt.Errorf("creating LLM agent: %w", err)
	}

	r, err := runner.New(runner.Config{
		AppName:        AppName,
		Agent:          llmAgent,
		SessionService: sessionSvc,
	})
	if err != nil {
		return nil, fmt.Errorf("creating runner: %w", err)
	}
	return r, nil
}

// New creates a new Agent with the given configuration.
func New(cfg Config) (*Agent, error) {
	// Programmatic injection: the prompt comes from cfg.InstructionParts
	// (Env recomposed from the live environment) instead of the file
	// assembly. Nothing below re-reads context files or the skills menu for
	// this path — the same prompt buildRunner gates and passes to the runner.
	// cfg.Instruction mirrors parts.Base so every consumer reading the config
	// back (inventory attribution, rebuild paths) sees the prompt that was
	// actually built; the injected prompt uses parts.Base as supplied either
	// way.
	if cfg.InstructionParts != nil && cfg.Instruction == "" {
		cfg.Instruction = cfg.InstructionParts.Base
	}
	if injected := resolveInjectedPrompt(cfg); injected != nil {
		return newWithPrompt(cfg, *injected)
	}

	instruction := cfg.Instruction
	if instruction == "" {
		instruction = SystemInstruction
	}

	// Add working directory context to the instruction.
	if cwd := cfg.workingDir(); cwd != "" {
		instruction += fmt.Sprintf("\nCurrent working directory: %s\n", cwd)
	}

	// Set up session service.
	sessionSvc := cfg.SessionService
	if sessionSvc == nil {
		sessionSvc = session.InMemoryService()
	}

	// Create the runner.
	r, err := buildRunner(cfg, instruction, sessionSvc)
	if err != nil {
		return nil, err
	}

	logContextInventory(cfg.Logger, instruction, nil, cfg.Tools)

	a := &Agent{
		runner:         r,
		sessionService: sessionSvc,
		config:         cfg,
	}
	// Skill hygiene notices, deduplicated on this agent. The skill set is
	// the same one prompt assembly loads for the working directory; loading
	// it here is a repeat of directory reads New already pays for, and it
	// keeps the warning state per-agent instead of per-process.
	if cwd := cfg.workingDir(); cwd != "" {
		a.warnSkillTriggers(loadedSkillsForDir(cwd))
	}

	return a, nil
}

// newWithPrompt is the shared tail of New: session service, runner build
// (the gate runs inside buildRunner), inventory line, hygiene pass. prompt is
// the fully assembled string the runner receives.
func newWithPrompt(cfg Config, prompt string) (*Agent, error) {
	sessionSvc := cfg.SessionService
	if sessionSvc == nil {
		sessionSvc = session.InMemoryService()
	}

	r, err := buildRunner(cfg, prompt, sessionSvc)
	if err != nil {
		return nil, err
	}

	logContextInventory(cfg.Logger, prompt, cfg.InstructionParts, cfg.Tools)

	a := &Agent{
		runner:         r,
		sessionService: sessionSvc,
		config:         cfg,
	}
	// Skill hygiene over the on-disk skill set for the working directory:
	// injection replaces the prompt, not the skill directory state — the
	// same notices file-driven runs raise must still fire. The per-agent
	// dedup keeps repeats silent.
	if cwd := cfg.workingDir(); cwd != "" {
		a.warnSkillTriggers(loadedSkillsForDir(cwd))
	}

	return a, nil
}

// RebuildWithInstruction recreates the agent's internal runner with a new
// system instruction while preserving all other configuration (tools, callbacks, etc.).
// The session service is reused so existing sessions remain accessible.
func (a *Agent) RebuildWithInstruction(instruction string) error {
	cfg := a.config
	cfg.Instruction = instruction
	// Force the provided instruction (skip default fallback).
	if instruction == "" {
		return fmt.Errorf("instruction must not be empty")
	}

	prompt := a.promptSource(cfg)
	r, err := buildRunner(cfg, prompt, a.sessionService)
	if err != nil {
		return fmt.Errorf("rebuilding runner: %w", err)
	}

	a.runner = r
	a.config = cfg
	// Same inventory line the other rebuild paths and New() write: a rebuilt
	// prompt is a new context cost, and the session log must show it.
	logContextInventory(a.config.Logger, prompt, a.inventoryParts(), a.config.Tools)
	// Re-run skill hygiene on the current skill set: a skill created since New
	// (create_skill, plan) is first seen here. The warnedSkills dedup keeps
	// already-warned skills silent, so a rebuild is never noisier than the
	// first pass.
	a.warnSkills()
	return nil
}

// RebuildWithParts recreates the agent's internal runner from a programmatically
// supplied InstructionParts, by the same contract as the Config.InstructionParts
// injection: Base+Rules+Skills are used exactly as given — the supplied Base
// replaces whatever cfg.Instruction held — Env is dropped and the
// volatile environment block is recomposed from the live process environment
// and appended last. On success cfg.Instruction is set to the supplied Base,
// so the config always mirrors the prompt that was actually built. The
// session service is reused so existing sessions remain accessible.
//
// After the rebuild the same hooks the other rebuild paths run fire: the
// context inventory line is written to the session log, and the skill-menu
// hygiene pass runs over the on-disk skill set for the working directory —
// the warnedSkills dedup keeps already-warned skills silent. The cwd source
// is Config.workingDir(), the one RebuildWithInstruction uses.
//
// An over-budget assembly fails exactly like a file-assembled one: with the
// per-section overrun list. The config is left untouched on failure.
func (a *Agent) RebuildWithParts(parts InstructionParts) error {
	cfg := a.config
	cfg.InstructionParts = &parts
	cfg.Instruction = parts.Base

	prompt := a.promptSource(cfg)
	r, err := buildRunner(cfg, prompt, a.sessionService)
	if err != nil {
		return fmt.Errorf("rebuilding runner: %w", err)
	}

	a.runner = r
	a.config = cfg
	logContextInventory(a.config.Logger, prompt, a.inventoryParts(), a.config.Tools)
	a.warnSkills()
	return nil
}

// promptSource resolves the fully assembled prompt for a rebuild from the
// config being rebuilt with: the injected string when InstructionParts is set
// (Env recomposed, cwd line appended — the New() injection contract), the
// plain instruction otherwise.
func (a *Agent) promptSource(cfg Config) string {
	if injected := resolveInjectedPrompt(cfg); injected != nil {
		return *injected
	}
	return cfg.Instruction
}

// inventoryParts returns the section attribution for the context inventory
// line: the injected parts when set (Env zeroed — the line never attributes
// the recomposed block to the caller's stale Env), nil for a plain-string
// prompt so the inventory falls back to its marker heuristic.
func (a *Agent) inventoryParts() *InstructionParts {
	if a.config.InstructionParts == nil {
		return nil
	}
	parts := *a.config.InstructionParts
	parts.Env = ""
	return &parts
}

// warnSkills runs the skill-menu hygiene pass over the on-disk skill set for
// the agent's working directory. No-op without one. The per-agent dedup keeps
// an unchanged set silent, so rebuild paths can call this unconditionally.
func (a *Agent) warnSkills() {
	if cwd := a.config.workingDir(); cwd != "" {
		a.warnSkillTriggers(loadedSkillsForDir(cwd))
	}
}

// RebuildWithModel recreates the agent's internal runner with a new LLM while
// preserving all other configuration (tools, callbacks, instruction, session).
// The session service is reused so existing sessions remain accessible.
func (a *Agent) RebuildWithModel(llm model.LLM) error {
	cfg := a.config
	cfg.Model = llm

	// Reconstruct the instruction the same way New() does. An injected
	// InstructionParts overrides the plain-string assembly: the prompt is
	// the fully recomposed injection string, so no fallback or cwd line is
	// appended on top of it.
	instruction := cfg.Instruction
	if cfg.InstructionParts == nil {
		if instruction == "" {
			instruction = SystemInstruction
		}
		if cwd := cfg.workingDir(); cwd != "" {
			instruction += fmt.Sprintf("\nCurrent working directory: %s\n", cwd)
		}
	} else {
		instruction = *resolveInjectedPrompt(cfg)
	}

	r, err := buildRunner(cfg, instruction, a.sessionService)
	if err != nil {
		return fmt.Errorf("rebuilding runner: %w", err)
	}

	a.runner = r
	a.config = cfg
	logContextInventory(a.config.Logger, instruction, a.inventoryParts(), a.config.Tools)
	return nil
}

// RebuildWithSession swaps the pieces that change when the main session moves
// to a primary agent — the agent's identity ("" restores the built-in
// AgentName), the system instruction, optionally the LLM, and the
// tool-callback chains (permission rules, step budget) — while reusing the
// session service so the conversation survives the switch. A nil llm keeps
// the current model. instruction must not be empty; beforeTool/afterTool are
// assigned as given (nil clears them).
func (a *Agent) RebuildWithSession(agentName, instruction string, llm model.LLM, beforeTool []BeforeToolCallback, afterTool []AfterToolCallback) error {
	if instruction == "" {
		return fmt.Errorf("instruction must not be empty")
	}
	cfg := a.config
	cfg.Name = agentName
	cfg.Instruction = instruction
	if llm != nil {
		cfg.Model = llm
	}
	cfg.BeforeToolCallbacks = beforeTool
	cfg.AfterToolCallbacks = afterTool

	// Reconstruct the instruction the same way New() does. An injected
	// InstructionParts overrides the plain-string assembly: the prompt is
	// the recomposed injection string built from the CURRENT parts — the
	// supplied instruction waits in cfg.Instruction until the parts are
	// cleared — so no cwd line is appended on top of it.
	if cfg.InstructionParts == nil {
		if cwd := cfg.workingDir(); cwd != "" {
			instruction += fmt.Sprintf("\nCurrent working directory: %s\n", cwd)
		}
	} else {
		instruction = *resolveInjectedPrompt(cfg)
	}

	r, err := buildRunner(cfg, instruction, a.sessionService)
	if err != nil {
		return fmt.Errorf("rebuilding runner: %w", err)
	}

	a.runner = r
	a.config = cfg
	logContextInventory(a.config.Logger, instruction, a.inventoryParts(), a.config.Tools)
	return nil
}

// modelNamer is satisfied by session services that can record the model name
// for an existing session (notably *session.FileService). Sessions backed by
// services without this capability still get the default "unknown" placeholder
// written by the file backend.
type modelNamer interface {
	SetSessionModel(sessionID, modelName string) error
}

// titleNamer is satisfied by session services that can record a human-readable
// title for an existing session. *session.FileService implements it; in-memory
// services without the capability silently no-op so callers don't have to
// branch on the service type.
type titleNamer interface {
	SetSessionTitle(sessionID, title string) error
}

// workDirRecorder is implemented by session services that can record the
// working directory a session runs in (e.g. session.FileService).
type workDirRecorder interface {
	SetSessionWorkDir(sessionID, dir string) error
}

// agentContextRecorder is implemented by session services that can persist a
// session's place in an agent tree. Declared here for the same reason as
// titleNamer: the agent takes the capability, not the concrete service.
type agentContextRecorder interface {
	UpdateAgentContext(sessionID string, ctx *pisession.AgentContext) error
}

// versionRecorder is implemented by session services that can record the
// pi-go build that produced a session.
type versionRecorder interface {
	SetSessionVersion(sessionID, version string) error
}

// CreateSession creates a new session and returns its ID together with the
// default title that was applied (git repo name, or CWD basename) — or "" if
// no title was set. The title is metadata only; the TUI seeds its terminal
// window/tab title from it so users see a sensible label before the first user
// prompt arrives.
func (a *Agent) CreateSession(ctx context.Context) (sessionID, defaultTitle string, err error) {
	return a.CreateSessionWithID(ctx, "")
}

// CreateSessionWithID creates a session under the given ID; an empty sessionID
// generates one, matching CreateSession. It exists because a ${SESSION_ID}
// header freezes into the LLM client at build time — the session created
// afterwards must carry the same ID, or the header names a conversation the
// logs have never heard of.
func (a *Agent) CreateSessionWithID(ctx context.Context, sessionID string) (sid, defaultTitle string, err error) {
	resp, err := a.sessionService.Create(ctx, &session.CreateRequest{
		AppName:   AppName,
		UserID:    DefaultUserID,
		SessionID: sessionID,
	})
	if err != nil {
		return "", "", fmt.Errorf("creating session: %w", err)
	}
	sid = resp.Session.ID()
	return sid, a.recordNewSessionMeta(sid), nil
}

// OpenSession resumes the session persisted under sessionID, creating it when
// the session service has no record of that id. It reports whether an existing
// transcript was found. This is how a server hosting a session for an external
// client — an ACP editor, an A2A gateway — continues a conversation across its
// own restarts: the client keeps the id, and the transcript lives under it.
func (a *Agent) OpenSession(ctx context.Context, sessionID string) (resumed bool, err error) {
	if strings.TrimSpace(sessionID) == "" {
		return false, fmt.Errorf("opening session: id is required")
	}
	if _, err := a.sessionService.Get(ctx, &session.GetRequest{
		AppName:   AppName,
		UserID:    DefaultUserID,
		SessionID: sessionID,
	}); err == nil {
		return true, nil
	}
	if _, err := a.sessionService.Create(ctx, &session.CreateRequest{
		AppName:   AppName,
		UserID:    DefaultUserID,
		SessionID: sessionID,
	}); err != nil {
		return false, fmt.Errorf("creating session %s: %w", sessionID, err)
	}
	a.recordNewSessionMeta(sessionID)
	return false, nil
}

// recordNewSessionMeta writes the metadata a brand-new session starts with —
// its place in an agent tree, the model, the working directory, and a default
// title — through whichever optional recorder interfaces the session service
// implements. Every write is best-effort metadata. It returns the title that
// was applied, or "" when none was.
func (a *Agent) recordNewSessionMeta(sid string) string {
	// Record this process's place in an agent tree, if it has one. A spawned
	// worker knows its coordinator, spec, slice and cycle only through the
	// environment; writing them here is what makes a run tree a field lookup
	// instead of an inference from workDir and title prefixes.
	if ac, ok := a.sessionService.(agentContextRecorder); ok {
		if actx := pisession.AgentContextFromEnv(); actx != nil {
			_ = ac.UpdateAgentContext(sid, actx)
		}
	}
	if mn, ok := a.sessionService.(modelNamer); ok {
		modelName := ""
		if a.config.Model != nil {
			modelName = a.config.Model.Name()
		}
		_ = mn.SetSessionModel(sid, modelName) // meta defaults to "unknown"
	}
	// Record the build, so a transcript that misbehaves can be attributed to
	// the binary that produced it rather than to whatever is installed today.
	if vr, ok := a.sessionService.(versionRecorder); ok && a.config.Version != "" {
		_ = vr.SetSessionVersion(sid, a.config.Version)
	}
	// The service records the process cwd at creation; an explicitly
	// configured working directory is the one the session actually runs in.
	if wr, ok := a.sessionService.(workDirRecorder); ok && a.config.WorkingDir != "" {
		_ = wr.SetSessionWorkDir(sid, a.config.WorkingDir)
	}
	// Default title: the git repo name (or CWD basename) so brand-new sessions
	// in /sessions listings have a sensible label before the first user prompt
	// overwrites it via the TUI's applySessionTitle / runPrint's derivePrintTitle.
	if tn, ok := a.sessionService.(titleNamer); ok {
		if title := defaultSessionTitle(a.config.workingDir()); title != "" {
			_ = tn.SetSessionTitle(sid, title)
			return title
		}
	}
	return ""
}

// gitToplevelTimeout caps the `git rev-parse --show-toplevel` subprocess so a
// stuck git invocation never delays session creation.
const gitToplevelTimeout = 2 * time.Second

// gitToplevelFn returns the absolute path of the git toplevel for dir, or "" if
// dir is not inside a git repository (or git is unavailable). It's a var so
// tests can stub the subprocess call without touching the real binary.
var gitToplevelFn = func(dir string) string {
	ctx, cancel := context.WithTimeout(context.Background(), gitToplevelTimeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, "git", "rev-parse", "--show-toplevel")
	cmd.Dir = dir
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

// gitCurrentBranch returns the current git branch name for dir, or "" if dir
// is not inside a git repository, git is unavailable, or the repo has no
// commits yet (in which case `git rev-parse --abbrev-ref HEAD` prints "HEAD"
// — meaningless as a label, so we treat it as no branch). It's a var so
// tests can stub the subprocess call without touching the real binary.
var gitCurrentBranch = func(dir string) string {
	if dir == "" {
		return ""
	}
	ctx, cancel := context.WithTimeout(context.Background(), gitToplevelTimeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, "git", "rev-parse", "--abbrev-ref", "HEAD")
	cmd.Dir = dir
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	branch := strings.TrimSpace(string(out))
	if branch == "" || branch == "HEAD" {
		return ""
	}
	return branch
}

// defaultSessionTitle returns a short, human-readable label for a brand-new
// session, composed from the cwd's git context when available:
//   - inside a git repo:        "<repo> <branch> <folder>"
//   - detached / no branch:    "<repo> <folder>"
//   - not a git repo:          "<folder>"      (no separator; nothing to anchor)
//   - empty cwd:               ""              (caller treats as "no title")
//
// Components are joined with a single space so the title stays clean inside
// the terminal tab title (used as the OSC 0 payload) and doesn't collide
// with the prompt-derived title that /title and the first-line derive path
// will later replace. The branch is omitted (not rendered as "(no branch)")
// when it can't be determined, to keep the title clean.
func defaultSessionTitle(cwd string) string {
	if cwd == "" {
		return ""
	}
	folder := filepath.Base(cwd)
	root := gitToplevelFn(cwd)
	if root == "" {
		return folder
	}
	repo := filepath.Base(root)
	branch := gitCurrentBranch(cwd)
	if branch == "" {
		return repo + " " + folder
	}
	return repo + " " + branch + " " + folder
}

// SetSessionTitle records a title for the given session via the session
// service when it supports it. Services without the title capability (e.g.
// the ADK in-memory service) silently no-op — the title is metadata that is
// safe to lose for ephemeral sessions.
func (a *Agent) SetSessionTitle(sessionID, title string) error {
	tn, ok := a.sessionService.(titleNamer)
	if !ok {
		return nil
	}
	return tn.SetSessionTitle(sessionID, title)
}

// Run sends a user message and returns an iterator over agent events.
// The caller should iterate over the returned sequence to process events.
func (a *Agent) Run(ctx context.Context, sessionID string, userMessage string) iter.Seq2[*session.Event, error] {
	if err := a.runPreTurn(ctx, sessionID); err != nil {
		return failedRun(err)
	}
	msg := genai.NewContentFromText(userMessage, genai.RoleUser)
	return a.runner.Run(ctx, DefaultUserID, sessionID, msg, adkagent.RunConfig{})
}

// RunStreaming sends a user message with SSE streaming enabled.
func (a *Agent) RunStreaming(ctx context.Context, sessionID string, userMessage string) iter.Seq2[*session.Event, error] {
	if err := a.runPreTurn(ctx, sessionID); err != nil {
		return failedRun(err)
	}
	msg := genai.NewContentFromText(userMessage, genai.RoleUser)
	return a.runner.Run(ctx, DefaultUserID, sessionID, msg, adkagent.RunConfig{
		StreamingMode: adkagent.StreamingModeSSE,
	})
}

// failedRun returns a single-error sequence, so a pre-turn failure surfaces
// through the same channel as any other turn error.
func failedRun(err error) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		yield(nil, err)
	}
}

// contextFileNames lists the context-file names checked in each directory
// during discovery, in priority order. The first match per directory wins.
var contextFileNames = []string{
	"AGENT.md",
	"AGENTS.md",
	"CLAUDE.md",
	filepath.Join(config.ProjectDirName, "AGENTS.md"),
}

// LoadInstruction appends discovered project context files and a summary of
// discovered skills to the base instruction. Context files (AGENT.md,
// AGENTS.md, CLAUDE.md, or .pirate/AGENTS.md; first match per directory) are
// discovered by walking from the working directory up to the filesystem
// root; a global ~/.pirate/AGENTS.md is included first when present. Files
// above maxRulesFileBudget are compacted to a section outline instead of
// being inlined whole.
func LoadInstruction(baseInstruction string) string {
	cwd, err := os.Getwd()
	if err != nil {
		return InstructionParts{Base: baseInstruction, Env: environmentContext(os.Getenv("HOME"), os.Getenv("USER"), os.Getenv("PWD"), os.Getenv("LANG"), "")}.String()
	}
	home, _ := os.UserHomeDir()
	return loadInstructionFrom(baseInstruction, cwd, home)
}

// InstructionParts is the system prompt broken into the sections it is
// assembled from. Callers that only need the finished prompt use
// LoadInstruction; the parts exist so the TUI can attribute context usage to
// each section instead of reporting one opaque total.
type InstructionParts struct {
	// Base is the built-in system prompt.
	Base string
	// Rules is the compacted AGENTS.md / CLAUDE.md project context.
	Rules string
	// Skills is the "# Available Skills" menu.
	Skills string
	// Env is the volatile "# Runtime Environment" block. It is composed last
	// so everything before it stays byte-stable across calls of a session.
	Env string
}

// String reassembles the parts into the prompt the model receives: the
// static blocks first (persona, rules, skills), the volatile environment
// last. Keeping composition here means the breakdown can never drift from
// the real prompt: LoadInstruction returns exactly this. A stable prefix is
// what lets the LLM gateway's prefix cache hit, so new static content must
// go into one of the first three parts, never into Env.
func (p InstructionParts) String() string {
	return p.Base + p.Rules + p.Skills + p.Env
}

// maxInstructionTotal caps the assembled system prompt, in bytes — not runes;
// len() is the contract here, aligned with the 32KiB silent-truncation points
// that measure bytes too. The 32_000 headroom sits below the gateway's silent
// 32KiB prefix cut: past it the tail of the prompt (Env today) disappears without any signal, so an
// oversized prompt fails here with the per-section overruns instead. The
// shipped prompt measures ~26.5K, leaving room for the base-prompt variants
// to come. The per-file and per-block Rules budgets (maxRulesFileBudget /
// maxRulesBlock) stay in charge of keeping Rules itself bounded; this is the
// always-on backstop for the sum.
const maxInstructionTotal = 32_000

// personaPromptHeadroom is the slack a persona-aware whole-prompt budget
// grants on top of the persona itself (issue #85): a subagent child's gate
// resolves to len(persona)+headroom, which is guaranteed to cover the
// persona plus everything a child build adds around it — the built-in tail
// (palace memory context, cwd line, environment block) and the Rules/Skills
// the child's working directory contributes (a compacted repo AGENTS.md
// measures ~17K in this repository; the measured worst case that shaped the
// number is persona 30 072 + tail 16 751). Growth past the gate — a bigger
// persona or fatter rules — still fails the build. Exported because the
// spawning parent (internal/cli) and the child resolve the same number
// independently: the parent announces the size, the child adds the headroom.
const personaPromptHeadroom = 24_000

// PersonaBudget resolves the whole-prompt gate for a child process running
// on an agent persona of size personaBytes: personaBytes + headroom. It is
// the single definition of that arithmetic — the internal
// effectiveInstructionBudget raises the gate to whatever this returns, and
// never below maxInstructionTotal.
func PersonaBudget(personaBytes int) int {
	return personaBytes + personaPromptHeadroom
}

// effectiveInstructionBudget resolves the whole-prompt budget for a build:
// a persona-aware limit only ever raises the gate (and only past the
// always-on budget — a value at or below maxInstructionTotal keeps the
// default), so this knob can never weaken the interactive 32K guarantee.
func effectiveInstructionBudget(personaBudget int) int {
	if personaBudget > maxInstructionTotal {
		return personaBudget
	}
	return maxInstructionTotal
}

// checkInstructionTotalBudget enforces budget on the fully assembled prompt,
// in bytes. Every section that is itself over the limit is named with its
// size — a caller fixing the prompt needs all offenders, not just the first.
// The prompt is passed pre-assembled so the check can never drift from what
// the model actually receives; parts carries the section sizes when the
// caller assembled the prompt from InstructionParts, and may be nil.
func checkInstructionTotalBudget(prompt string, parts *InstructionParts, budget int) error {
	if len(prompt) <= budget {
		return nil
	}
	known := parts
	if known == nil {
		resolved := instructionOnlyParts(prompt)
		known = &resolved
	}
	type section struct {
		name string
		size int
	}
	sections := []section{
		{"Base", len(known.Base)},
		{"Rules", len(known.Rules)},
		{"Skills", len(known.Skills)},
	}
	var b strings.Builder
	fmt.Fprintf(&b, "system instruction is %d bytes, over the %d byte budget:", len(prompt), budget)
	none := true
	for _, s := range sections {
		if s.size > budget {
			fmt.Fprintf(&b, " %s=%d", s.name, s.size)
			none = false
		}
	}
	if none {
		b.WriteString(" no single section exceeds the budget; the total is spread across Base+Rules+Skills+Env")
	}
	return fmt.Errorf("%s", b.String())
}

// checkInstructionTotal enforces maxInstructionTotal on the fully assembled
// prompt — the plain always-on gate, kept for callers without a persona
// budget.
func checkInstructionTotal(prompt string, parts *InstructionParts) error {
	return checkInstructionTotalBudget(prompt, parts, maxInstructionTotal)
}

// Section markers the assembler writes into the prompt. They double as the
// only way to recover per-section sizes from an already-assembled prompt
// string, which callers that hand the agent a plain instruction (a SOP, a
// --system override) need.
const (
	rulesSectionMarker  = "\n\n# Project Rules\n\n"
	skillsSectionMarker = "\n\n# Available Skills\n\n"
	// envSectionMarker opens the volatile environment block that closes the
	// prompt; the inventory line measures the skills menu up to it.
	envSectionMarker = "\n\n# Runtime Environment\n\n"
)

// instructionOnlyParts recovers the Base/Rules/Skills sizes from an
// assembled prompt by its section markers. It mirrors the composition in
// InstructionParts.String: Base first, then Rules, then Skills, then Env.
// The split is heuristic for an arbitrary instruction string; inventory
// fields are approximate for non-standard prompts.
// A section may be absent; a missing marker folds its would-be content into
// the previous section, and only the gate and the inventory line read this —
// the prompt itself is never reassembled from it.
func instructionOnlyParts(prompt string) InstructionParts {
	var parts InstructionParts
	rest := prompt
	if i := strings.Index(rest, rulesSectionMarker); i >= 0 {
		parts.Base = rest[:i]
		rest = rest[i:]
	} else {
		parts.Base = rest
		return parts
	}
	if j := strings.Index(rest, skillsSectionMarker); j >= 0 {
		parts.Rules = rest[:j]
		// Everything from the skills marker on — including a trailing Env
		// block — stays unattributed for the gate's purposes; the total in
		// the error message already names the real overage.
		parts.Skills = rest[j:]
		return parts
	}
	parts.Rules = rest
	return parts
}

// contextInventory renders the one-line inventory of the assembled prompt
// for the session log: per-section byte counts, the total, and the combined
// size of the tool declarations registered with the model. Sizes drift
// release over release; this line is how that drift becomes visible without
// measuring anything by hand.
//
// parts.Skills spans the skills menu up to the env block (or the end of the
// prompt), so menu= measures the menu line itself via the env marker's
// offset; without a trailing env block the menu runs to the end of the
// prompt. The skills= number beside it stays the raw span, so a drift
// between the two points at a tail that has grown onto the skills section.
func contextInventory(parts InstructionParts, prompt string, toolsChars int) string {
	menu := len(parts.Skills)
	if i := strings.Index(prompt, envSectionMarker); i >= 0 {
		menu = i - len(parts.Base) - len(parts.Rules)
	}
	return fmt.Sprintf(
		"context inventory: base=%d rules=%d skills=%d menu=%d total=%d tools=%d",
		len(parts.Base), len(parts.Rules), len(parts.Skills), menu, len(prompt), toolsChars)
}

// logContextInventory writes the context inventory line to the session log
// after an agent (re)build. parts may be nil: a prompt the caller handed over
// as a plain string is inventoried by its section markers. The logger is
// nil-safe — without one the line is dropped, never printed to the terminal.
func logContextInventory(log *logger.Logger, prompt string, parts *InstructionParts, tools []tool.Tool) {
	if log == nil {
		return
	}
	var resolved InstructionParts
	if parts != nil {
		resolved = *parts
	} else {
		resolved = instructionOnlyParts(prompt)
	}
	log.Info(contextInventory(resolved, prompt, declaredToolChars(tools)))
}

// declaredToolChars sums the serialized model-facing declarations of the
// given tools — the same measurement the TUI context gauge reports, so the
// session log and the gauge cannot disagree.
func declaredToolChars(tools []tool.Tool) int {
	total := 0
	for _, t := range tools {
		type declarer interface {
			Declaration() *genai.FunctionDeclaration
		}
		d, ok := t.(declarer)
		if !ok {
			continue
		}
		decl := d.Declaration()
		if decl == nil {
			continue
		}
		enc, err := json.Marshal(decl)
		if err != nil {
			continue
		}
		total += len(enc)
	}
	return total
}

// loadInstructionFrom is the testable core of LoadInstruction, resolving
// context files and skills relative to explicit cwd and home directories.
func loadInstructionFrom(baseInstruction, cwd, home string) string {
	return loadInstructionPartsFrom(baseInstruction, cwd, home).String()
}

// LoadInstructionPartsFor resolves the system prompt for an explicit working
// directory instead of the process's, and returns it broken into its sections.
// Callers that drive the agent over a directory they were not started in — the
// public piagent package, for one — need this; LoadInstructionParts is the
// os.Getwd() convenience on top of it.
func LoadInstructionPartsFor(baseInstruction, cwd string) InstructionParts {
	home, _ := os.UserHomeDir()
	return loadInstructionPartsFrom(baseInstruction, cwd, home)
}

// LoadInstructionParts resolves the system prompt and returns it broken into
// its sections.
func LoadInstructionParts(baseInstruction string) InstructionParts {
	cwd, err := os.Getwd()
	if err != nil {
		return InstructionParts{Base: baseInstruction, Env: environmentContext(os.Getenv("HOME"), os.Getenv("USER"), os.Getenv("PWD"), os.Getenv("LANG"), "")}
	}
	home, _ := os.UserHomeDir()
	return loadInstructionPartsFrom(baseInstruction, cwd, home)
}

// environmentContext renders the volatile "# Runtime Environment" block with
// a leading blank-line separator, ready to be appended after the static
// prompt blocks. LANG is mapped to a language name and stated as the agent's
// required response language; it is omitted when LANG is unset or empty.
func environmentContext(home, user, pwd, lang, cwd string) string {
	if pwd == "" {
		pwd = cwd
	}
	env := fmt.Sprintf("\n\n# Runtime Environment\n\nThe following values are from the current process environment. Treat them as authoritative; do not guess or substitute values for them.\n\n- HOME=%q\n- USER=%q\n- PWD=%q\n", home, user, pwd)
	if lang != "" {
		env += fmt.Sprintf("\nLanguage: %s is the required language for your responses.\n", languageName(lang))
	}
	return env
}

// languageName maps a POSIX locale (e.g. "en_US.UTF-8") to a human-readable
// language name for the system prompt. It returns the raw value unchanged
// when the primary subtag has no known name, so nothing is invented.
func languageName(lang string) string {
	primary := strings.SplitN(lang, ".", 2)[0]   // strip codeset: en_US.UTF-8 -> en_US
	primary = strings.SplitN(primary, "@", 2)[0] // strip modifier: en_US@cyrillic -> en_US
	region := strings.SplitN(primary, "_", 2)
	code := strings.ToLower(region[0])
	name, ok := localeLanguageNames[code]
	if !ok {
		return lang
	}
	if len(region) == 2 {
		if country, ok := localeRegionNames[strings.ToUpper(region[1])]; ok {
			return name + " (" + country + ")"
		}
	}
	return name
}

// localeLanguageNames maps ISO 639-1 primary language subtags to names.
var localeLanguageNames = map[string]string{
	"ar": "Arabic",
	"bn": "Bengali",
	"cs": "Czech",
	"da": "Danish",
	"de": "German",
	"el": "Greek",
	"en": "English",
	"es": "Spanish",
	"fa": "Persian",
	"fi": "Finnish",
	"fr": "French",
	"he": "Hebrew",
	"hi": "Hindi",
	"hu": "Hungarian",
	"id": "Indonesian",
	"it": "Italian",
	"ja": "Japanese",
	"ko": "Korean",
	"nb": "Norwegian (Bokmål)",
	"nl": "Dutch",
	"nn": "Norwegian (Nynorsk)",
	"no": "Norwegian",
	"pl": "Polish",
	"pt": "Portuguese",
	"ro": "Romanian",
	"ru": "Russian",
	"sv": "Swedish",
	"th": "Thai",
	"tr": "Turkish",
	"uk": "Ukrainian",
	"vi": "Vietnamese",
	"zh": "Chinese",
}

// localeRegionNames maps ISO 3166-1 region subtags to country names for the
// common locales, so en_US reads "English (United States)".
var localeRegionNames = map[string]string{
	"AT": "Austria",
	"BE": "Belgium",
	"BR": "Brazil",
	"CA": "Canada",
	"CH": "Switzerland",
	"CN": "China",
	"DE": "Germany",
	"DK": "Denmark",
	"ES": "Spain",
	"FI": "Finland",
	"FR": "France",
	"GB": "United Kingdom",
	"GR": "Greece",
	"HK": "Hong Kong",
	"IL": "Israel",
	"IN": "India",
	"IT": "Italy",
	"JP": "Japan",
	"KR": "South Korea",
	"MX": "Mexico",
	"NL": "Netherlands",
	"NO": "Norway",
	"PL": "Poland",
	"PT": "Portugal",
	"RO": "Romania",
	"RU": "Russia",
	"SE": "Sweden",
	"TH": "Thailand",
	"TR": "Turkey",
	"TW": "Taiwan",
	"UA": "Ukraine",
	"US": "United States",
	"VN": "Vietnam",
}

func loadInstructionPartsFrom(baseInstruction, cwd, home string) InstructionParts {
	parts := InstructionParts{Base: baseInstruction}

	if files := discoverContextFiles(cwd, home); len(files) > 0 {
		parts.Rules = "\n\n# Project Rules\n\n" + compactContextFiles(files)
	}

	if skills := loadedSkillsForDir(cwd); len(skills) > 0 {
		parts.Skills = appendSkillsMenu(skills)
	}

	parts.Env = environmentContext(os.Getenv("HOME"), os.Getenv("USER"), os.Getenv("PWD"), os.Getenv("LANG"), cwd)

	return parts
}

// loadedSkillsForDir loads the skill set prompt assembly renders for a
// working directory. Skill discovery (skillsDirs / disableLegacySkillDirs,
// or PI_SKILLS_DIRS / PI_DISABLE_LEGACY_SKILLS) follows the loaded config.
// A config load failure is not fatal: LoadFrom still returns the
// merged-so-far config, and skill discovery with no skill settings is simply
// the default one. An unreadable skill set yields nil, never an error —
// callers treat an empty result as "no menu".
func loadedSkillsForDir(cwd string) []extension.Skill {
	cfg, _ := config.LoadFrom(cwd)
	skills, err := extension.LoadSkills(extension.DefaultSkillDirsIn(cwd, cfg)...)
	if err != nil {
		return nil
	}
	return skills
}

// maxSkillMenuDesc caps the description rendered per skill in the
// "# Available Skills" menu, in runes.
const maxSkillMenuDesc = 80

// skillMenuDescription reduces a skill description to its identifying first
// sentence (first line, then text up to the first period), capped at
// maxSkillMenuDesc runes with an ellipsis. The full description stays
// available where decisions are made — the /skills listing and the Active
// Skill body injected on activation — so the menu only needs enough text to
// tell one skill from another. A verbose multi-sentence SKILL.md frontmatter
// used to add hundreds of prompt tokens per skill with no discovery benefit.
// Truncating at the first period can stop early on abbreviations (e.g.); that
// is acceptable for a discovery menu and bounded by the cap.
func skillMenuDescription(desc string) string {
	if i := strings.IndexByte(desc, '\n'); i >= 0 {
		desc = desc[:i]
	}
	if i := strings.IndexByte(desc, '.'); i >= 0 {
		desc = desc[:i+1]
	}
	desc = strings.TrimSpace(desc)
	if runes := []rune(desc); len(runes) > maxSkillMenuDesc {
		desc = string(runes[:maxSkillMenuDesc-1])
		if i := strings.LastIndex(desc, " "); i > 0 {
			desc = desc[:i] // cut back to a word boundary, not mid-word
		}
		desc += "…"
	}
	return desc
}

// appendSkillsMenu formats the "# Available Skills" block for a pre-loaded
// skill slice. Exposed so callers that already have skills in hand (e.g. the
// TUI) don't trigger a second LoadSkills.
//
// Each skill renders as exactly one bounded line: name plus first sentence of
// the description (see skillMenuDescription). The menu's job is discovery —
// picking which skill to load — not documenting it. Entries render in the
// order given: LoadSkills returns them in name order (the sort lives in
// LoadSkillsWithOptions, where every consumer benefits), so the menu
// inherits a deterministic order without this function re-sorting the
// caller's slice behind its back.
//
// The formatting half of skill hygiene. The warning half lives on the agent
// (warnSkillMissingTrigger): dedup state is per-agent, and this function has
// no agent — it is called from prompt assembly before one exists.
func appendSkillsMenu(skills []extension.Skill) string {
	var b strings.Builder
	b.WriteString("\n\n# Available Skills\n\n")
	for _, s := range skills {
		fmt.Fprintf(&b, "- /%s: %s\n", s.Name, skillMenuDescription(s.Description))
	}
	return b.String()
}

// skillActivationMarkers are the phrases a skill description uses to state
// when the skill applies. A description carrying none of them gives the model
// no cue to open the skill, so it can sit in the menu forever. The list
// covers the common EN/RU phrasings; false positives are acceptable
// (warn-only), false negatives drown the signal.
var skillActivationMarkers = []string{
	"use when",
	"when to use",
	"используй когда",
	"если",
	"use this skill when",
	"use this skill whenever",
	"use whenever",
	"whenever the user",
	"используй, когда",
	"загружай когда",
	"загружай для",
	"загружай, когда",
	"use for",
	"применяй когда",
}

// warnSkillMissingTrigger raises one notice per skill whose description has
// no activation condition, deduplicated on the receiving agent's warnedSkills
// set. Notices go through the notice sink (the TUI's SystemNoticeCh), never
// to stdout/stderr — the interactive TUI owns the alternate screen. The menu
// itself is untouched: hygiene observes, it does not rewrite descriptions.
//
// The name is sanitized at the point of emission: a name comes from a
// skill's frontmatter — files, potentially from other repos — and reaches
// the terminal through the notice path. Control characters (C0/C1, including
// ESC and BEL) are dropped, so an escape sequence in `name:` cannot repaint
// the user's screen; the audit scanner's ASCII fast path cannot see them, so
// the emitter is the only reliable place to strip them. The name is capped
// at 64 runes with an ellipsis. The menu keeps the raw name: discovery reads
// it there in context, and rewriting the menu would desync it from /skills.
func (a *Agent) warnSkillMissingTrigger(s extension.Skill) {
	desc := strings.ToLower(s.Description)
	for _, m := range skillActivationMarkers {
		if strings.Contains(desc, m) {
			return
		}
	}
	if _, dup := a.warnedSkills.LoadOrStore(s.Name, struct{}{}); dup {
		return
	}
	notice.Notifyf("skill %s: в description нет условия активации — скилл может ни разу не открыться", sanitizeSkillName(s.Name))
}

// warnSkillTriggers runs the hygiene half of the skills menu over a loaded
// skill list: one missing-trigger notice per skill, deduplicated on this
// agent. Called from the build paths that know the skill set — New and the
// instruction rebuilds — so the dedup state lives exactly as long as the
// agent does, and two agents in one process each warn for themselves.
func (a *Agent) warnSkillTriggers(skills []extension.Skill) {
	for _, s := range skills {
		a.warnSkillMissingTrigger(s)
	}
}

// maxSkillNameRunes caps the sanitized name in a hygiene notice, in runes.
const maxSkillNameRunes = 64

// sanitizeSkillName makes a skill name safe for terminal emission: C0/C1
// control characters (U+0000–U+001F, U+007F–U+009F — ESC, BEL, CR, LF and
// friends) are dropped entirely, so no name can carry an escape sequence
// into the notice path. The result is capped at maxSkillNameRunes runes;
// a cut name keeps a trailing ellipsis.
func sanitizeSkillName(name string) string {
	clean := strings.Map(func(r rune) rune {
		if r <= 0x1F || (r >= 0x7F && r <= 0x9F) {
			return -1
		}
		return r
	}, name)
	if runes := []rune(clean); len(runes) > maxSkillNameRunes {
		clean = string(runes[:maxSkillNameRunes]) + "…"
	}
	return clean
}

// AppendActiveSkill returns prompt with an "# Active Skill" section appended
// for the given skill body. Use this on top of a prompt produced by
// LoadInstruction to activate a skill for one turn (Level-2 injection).
//
// The format is:
//
//	\n\n# Active Skill: <name>\n\n<body>\n
func AppendActiveSkill(prompt string, skill extension.Skill, body string) string {
	return prompt + fmt.Sprintf("\n\n# Active Skill: %s\n\n%s\n", skill.Name, body)
}

type contextFile struct {
	path    string
	content string
}

// discoverContextFiles returns context files ordered from most general to
// most specific: the global ~/.pirate/AGENTS.md first, then one file per
// directory from the filesystem root down to cwd. Contents are returned
// whole; compaction to the prompt budget happens in compactContextFiles.
func discoverContextFiles(cwd, home string) []contextFile {
	// Walk from cwd up to the filesystem root, nearest directory first.
	var found []contextFile
	dir := filepath.Clean(cwd)
	for {
		if f, ok := readContextFileIn(dir); ok {
			found = append(found, f)
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	slices.Reverse(found) // most general (root) first

	// Global file comes first, unless the walk already picked it up.
	if home != "" {
		globalPath := filepath.Join(home, ".pirate", "AGENTS.md")
		alreadyFound := slices.ContainsFunc(found, func(f contextFile) bool {
			return f.path == globalPath
		})
		if !alreadyFound {
			if content, ok := readInstructionFile(globalPath); ok {
				found = append([]contextFile{{path: globalPath, content: content}}, found...)
			}
		}
	}
	return found
}

// Rules-block budgets. A context file at or under maxRulesFileBudget is
// inlined whole — for a small file the full text is worth more than its
// outline and no read round-trip is needed. Larger files are compacted to a
// section outline capped at the same per-file budget, and the whole
// "# Project Rules" block is capped at maxRulesBlock. Before the budgets
// existed every context file was glued in whole, which put a 39KB AGENTS.md
// on every LLM call of a session.
const (
	maxRulesFileBudget = 4 * 1024
	maxRulesBlock      = 8 * 1024
	// maxOutlineDesc caps a section's first-line description, in runes.
	maxOutlineDesc = 100
)

// rulesReadHint tells the model how to get a section's full text. Appended
// only when at least one file was outlined (for fully inlined files there is
// nothing to read on demand).
const rulesReadHint = "\n\nFull section texts are not inlined. Before working in an area, read its section with the read tool: file_path=<path>, offset=<line number>.\n"

// compactContextFiles renders discovered context files into the "# Project
// Rules" body. Files at or under maxRulesFileBudget are inlined whole;
// larger ones become a bounded section outline. Within the 128KB per-file
// read cap nothing is silently skipped — a file contributes either its text
// or an explicit pointer to itself; files over that cap are still rejected
// whole by readInstructionFile before this sees them, and their absence is
// not announced here. The body is capped at maxRulesBlock no matter how many
// files are discovered: pieces are added while they fit, and everything past
// the first overflow collapses into one terminal pointer that is itself held
// within the remaining budget.
func compactContextFiles(files []contextFile) string {
	rendered := make([]string, 0, len(files))
	outlined := false
	for _, f := range files {
		piece, wasOutlined := renderContextFile(f)
		outlined = outlined || wasOutlined
		rendered = append(rendered, piece)
	}

	budget := maxRulesBlock
	if outlined {
		budget -= len(rulesReadHint)
	}

	var b strings.Builder
	for i, piece := range rendered {
		sep := 0
		if b.Len() > 0 {
			sep = 1
		}
		if b.Len()+sep+len(piece) > budget {
			writeSkipCounter(&b, files[i:], budget-b.Len())
			break
		}
		if sep == 1 {
			b.WriteString("\n")
		}
		b.WriteString(piece)
	}
	if outlined {
		b.WriteString(rulesReadHint)
	}
	return b.String()
}

// skipFloor is the smallest tail the terminal skip line is still written
// for. Below it the budget wins outright: a pointer that would not fit is
// dropped rather than allowed to push the block past maxRulesBlock.
const skipFloor = 64

// writeSkipCounter appends the single terminal line for files that did not
// fit the rules budget: a count plus as many of their paths as the remaining
// budget holds, then an ellipsis. The line never exceeds remaining, so the
// block stays within maxRulesBlock regardless of the file count.
func writeSkipCounter(b *strings.Builder, files []contextFile, remaining int) {
	if remaining < skipFloor {
		return
	}
	const frame = " — read any of them for the full text."
	const ellipsis = ", …"
	line := fmt.Sprintf("- Skipped: %d more files exceeded the project-rules budget:", len(files))
	used := len(line) + len(frame) + 1
	for i, f := range files {
		next := f.path
		if i > 0 {
			next = ", " + f.path
		}
		if used+len(next)+len(ellipsis) > remaining {
			line += ellipsis
			break
		}
		line += next
		used += len(next)
	}
	b.WriteString(line + frame + "\n")
}

// renderContextFile renders one context file for the Rules block: whole when
// it fits maxRulesFileBudget, otherwise as a section outline. The second
// result reports whether the outline (not the full text) was rendered.
func renderContextFile(f contextFile) (string, bool) {
	if len(f.content) <= maxRulesFileBudget {
		return f.content, false
	}
	return outlineContextFile(f), true
}

// outlineSection is one markdown heading of a context file with the line the
// heading starts at (1-based) and the first content line under it.
type outlineSection struct {
	line int
	head string
	desc string
}

// outlineContextFile compacts an oversized context file into a section
// outline: every markdown heading with its line number and a one-line
// description, capped at maxRulesFileBudget by dropping whole entries (the
// remainder stays visible in a count line). An outline is never larger than
// the file it stands for: the cap only applies above the inline threshold.
func outlineContextFile(f contextFile) string {
	entries := sectionOutline(f.content)
	var b strings.Builder
	fmt.Fprintf(&b, "### %s\n\n", f.path)
	if len(entries) == 0 {
		// No markdown headings to index — point at the file head instead of
		// compacting to nothing.
		fmt.Fprintf(&b, "- L1: (no markdown headings) — %s\n", truncateDesc(firstNonEmptyLine(f.content)))
		return b.String()
	}
	used := b.Len()
	for i, e := range entries {
		line := outlineEntry(e)
		if used+len(line) > maxRulesFileBudget {
			fmt.Fprintf(&b, "- … +%d more sections; read %s for the full text.\n", len(entries)-i, f.path)
			return b.String()
		}
		b.WriteString(line)
		used += len(line)
	}
	return b.String()
}

// outlineEntry renders one outline line: the heading with its line number,
// plus the section's first line when there is one.
func outlineEntry(s outlineSection) string {
	if s.desc == "" {
		return fmt.Sprintf("- L%d: %s\n", s.line, s.head)
	}
	return fmt.Sprintf("- L%d: %s — %s\n", s.line, s.head, s.desc)
}

// sectionOutline extracts the markdown sections of content. Recognition is
// deliberately narrow: ATX headings (lines starting with "#") and ``` fences
// only. Setext headings (underlined titles) and ~~~ fences are not
// recognized — an odd fence count or an underlined title can only shorten
// the outline, never corrupt it. Fenced code is skipped, so a `#` shell
// comment inside one is not taken for a heading.
func sectionOutline(content string) []outlineSection {
	var out []outlineSection
	inFence := false
	for i, line := range strings.Split(content, "\n") {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "```") {
			inFence = !inFence
			continue
		}
		if inFence || trimmed == "" {
			continue
		}
		if strings.HasPrefix(trimmed, "#") {
			out = append(out, outlineSection{line: i + 1, head: trimmed})
			continue
		}
		if n := len(out); n > 0 && out[n-1].desc == "" {
			out[n-1].desc = truncateDesc(trimmed)
		}
	}
	return out
}

// firstNonEmptyLine returns the first non-blank line of content, trimmed.
func firstNonEmptyLine(content string) string {
	for _, line := range strings.Split(content, "\n") {
		if trimmed := strings.TrimSpace(line); trimmed != "" {
			return trimmed
		}
	}
	return ""
}

// truncateDesc caps s at maxOutlineDesc runes on a word boundary with an
// ellipsis.
func truncateDesc(s string) string {
	runes := []rune(s)
	if len(runes) <= maxOutlineDesc {
		return s
	}
	cut := string(runes[:maxOutlineDesc-1])
	if i := strings.LastIndex(cut, " "); i > 0 {
		cut = cut[:i] // cut back to a word boundary, not mid-word
	}
	return cut + "…"
}

// readContextFileIn returns the first context file found in dir.
func readContextFileIn(dir string) (contextFile, bool) {
	for _, name := range contextFileNames {
		path := filepath.Join(dir, name)
		if content, ok := readInstructionFile(path); ok {
			return contextFile{path: path, content: content}, true
		}
	}
	return contextFile{}, false
}

// readInstructionFile reads path, rejecting missing or oversized files.
func readInstructionFile(path string) (string, bool) {
	data, err := os.ReadFile(path)
	if err != nil || len(data) > maxInstructionFileSize {
		return "", false
	}
	return string(data), true
}
