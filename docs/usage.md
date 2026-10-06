# Usage

How to drive Pi-rate: interactive sessions, one-shot and machine-readable
modes, agents, subagents, permissions, and the TUI's commands and hotkeys.

- [Running](#running) — interactive, one-shot, model selection
- [Model roles](#model-roles)
- [API keys](#api-keys) — env vars, `pirate setup`
- [Agents](#agents) — markdown files with frontmatter, primary agents
- [Prompt editor](#prompt-editor) — multiline input, paste summaries, `@`-mentions
- [Subagents](#subagents) — monitor, steering, background runs
- [Permissions](#permissions) — `allow`/`deny`/`ask`, the approval dialog
- [Todo plans](#todo-plans)
- [The question tool](#the-question-tool)
- [JSON mode](#json-mode)
- [Slash commands](#slash-commands)
- [Hotkeys](#hotkeys)
- [Plugin marketplaces](#plugin-marketplaces)
- [Memory Palace](#memory-palace)
- [Security audit](#security-audit)

## Running

```bash
# Default interactive mode
pirate

# Select a model by prefix
pirate --model claude:sonnet
pirate --model openai:gpt-4o
pirate --model gemini:gemini-2.5-pro
pirate --model mistral-large-latest
pirate --model grok-4.6
pirate --model azure/my-gpt5-deployment
pirate --model openrouter/google/gemini-3.7-flash
pirate --model ollama/gemma4:12b-mlx
pirate --model opencode/kimi-k3
pirate --model corp-claude/claude-opus-5   # a provider declared in config.json
pirate --model agentgateway/deepseek-v4-flash:0731-cloud
pirate --model minimax-m3:cloud # automatically detect ollama if :cloud

# Use model roles
pirate --smol          # fast, cheap model
pirate --slow          # most capable model
pirate --plan          # planning-oriented model

# Additional options
pirate --continue      # continue last session
pirate --session <id>  # resume specific session
pirate --system "..."  # custom system instructions
pirate --url "..."     # custom API endpoint URL

# Non-interactive modes
pirate --mode print "explain this codebase"
pirate --mode json "list all TODO comments"
pirate --mode json --json-deltas full "..."      # one event per model chunk, not per sentence
pirate --mode socket --socket /tmp/pi-go.sock    # JSON-RPC 2.0 over a Unix socket
pirate --mode rpc                                # pi-compatible NDJSON over stdio (for pi-acp)
```

## Model roles

Roles are named model configurations in `~/.pirate/config.json` — `default`
(the fallback), `smol` (fast and cheap), `slow` (most capable), `plan`
(planning-oriented) ship as names, and you can add your own. The built-in
three are selected with `--smol`, `--slow` and `--plan`; `/model` in the TUI
lists the roles configured on your machine and switches between them. See
[Configuration → Model roles](configuration.md#model-roles).

## API keys

Set the API key for your provider as an environment variable. The provider is
inferred from the model name, so `--model` is usually the only routing you need.

| Provider | Model prefix | API key env var | Base URL env var |
|---|---|---|---|
| Anthropic | `claude-*` | `ANTHROPIC_API_KEY` (or `ANTHROPIC_AUTH_TOKEN`) | `ANTHROPIC_BASE_URL` |
| OpenAI | `gpt-*` | `OPENAI_API_KEY` | `OPENAI_BASE_URL` |
| Google Gemini | `gemini-*` | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | `GEMINI_BASE_URL` |
| Mistral | `mistral-*`, `magistral-*` | `MISTRAL_API_KEY` | `MISTRAL_BASE_URL` |
| xAI (Grok) | `grok-*` | `XAI_API_KEY` | `XAI_BASE_URL` |
| OpenRouter | `openrouter/<model>` | `OPENROUTER_API_KEY` | `OPENROUTER_BASE_URL` |
| agentgateway | `agentgateway/<model>` | none (optional `AGENTGATEWAY_API_KEY`) | `AGENTGATEWAY_BASE_URL` (default `http://localhost:4000`) |
| Azure OpenAI | `azure/<deployment>` | `AZUREOPENAI_API_KEY` | — |
| OpenCode | `opencode/<model>` | `OPENCODE_API_KEY` | `OPENCODE_BASE_URL` |
| Ollama (local) | `ollama/<model>` | none | `OLLAMA_HOST` (default `http://localhost:11434`) |
| Ollama Cloud | `<model>:cloud` | `OLLAMA_API_KEY` | `https://api.ollama.com`, or the local daemon when no key is set |

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="..."
export MISTRAL_API_KEY="..."     # optional — only if you use Mistral models
export XAI_API_KEY="..."         # optional — only if you use Grok models
export OPENROUTER_API_KEY="..."  # optional — only if you use OpenRouter models
export OPENCODE_API_KEY="..."
export OLLAMA_API_KEY="..."      # optional — only to reach Ollama Cloud directly
```

A name with no recognized prefix is rejected rather than guessed at — reach for
the `ollama/` prefix or the `:cloud` suffix to name an Ollama model explicitly.

To configure a provider interactively instead of exporting the variable yourself:

```bash
pirate setup
```

The wizard asks which provider to use, for its API key, and which model should
be the default. It writes the key to `~/.pirate/.env` and the provider and model
to the `default` role in `~/.pirate/config.json` — the same places a hand-exported
variable and a hand-edited config land. Providers that need no credential
(Ollama, agentgateway) skip the key step; providers Pi-rate has no offline catalog
for (Ollama, agentgateway, Azure, OpenCode) have their model typed in, since only
you know what your daemon or deployment serves.

A `:cloud` tag names a model, not a destination. With `OLLAMA_API_KEY` set the
request goes straight to `api.ollama.com`; without one it goes to the local
daemon, which has served cloud models on your `ollama signin` identity since
Ollama 0.12 — so `pirate --model deepseek-v3.1:671b-cloud` works with no key at all.
The `ollama/` prefix always means the local daemon, tag notwithstanding, and an
explicit `OLLAMA_HOST` overrides both.

## Agents

Agents are markdown files with YAML frontmatter, discovered from
`~/.pirate/agents/` (global) and `.pirate/agents/` (project). Same-named
files resolve project > user > bundled; a set of agents ships bundled
(captain, first-mate, skipper, cabin-boy, memory-compressor).

```markdown
---
name: reviewer
description: Reviews diffs for correctness and style
mode: all                      # primary | subagent | all (default: subagent)
model: mycompany/model-x       # overrides role; may carry a provider prefix
role: smol                     # config role used when model: is absent
fallback-models: mycompany/model-x-lite, mycompany/model-x-mini
tools: read, grep, find        # comma-separated; empty = all tools
temperature: 0.2
reasoningEffort: high          # none | minimal | low | medium | high | max
steps: 150                     # tool-call iterations cap; 0 = unlimited
timeout: 600000                # ms; accepts 1h/30m/90s suffixes (>=1s)
streamIdleTimeout: 90s         # stream-silence abort; 0 disables it
worktree: true                 # run in an isolated git worktree
lsp: full                      # off | min | full
permission:                    # agent-scoped rules, layered over the global ones
  edit: deny
  bash:
    "git *": allow
---
System prompt body in markdown.
```

Frontmatter keys and their defaults:

| Key | Values | Default |
|---|---|---|
| `name` | agent identifier | filename stem (`<name>.md` → `<name>`) |
| `description` | one-line description shown to the calling model | — |
| `mode` | `primary`, `subagent` or `all` | `subagent` |
| `model` | model name, optionally `provider/model`; overrides `role` | — |
| `role` | config role name (`smol`, `plan`, `slow`, …) used when `model` is absent | — |
| `fallback-models` | comma-separated models, same format as `model`, tried in order when the primary is unavailable (exhausted quota, auth errors, repeated 5xx); at most the first 3 are used | the role's `fallbackModels` (falling back to the `default` role's) |
| `tools` | comma-separated tool names | all tools |
| `temperature` | float | inherited |
| `reasoningEffort` | `none`, `minimal` (same as `none`), `low`, `medium`, `high`, `max`; unknown values warn and inherit | inherited |
| `steps` | cap on the child's tool-call iterations; `0` = unlimited | unlimited |
| `timeout` | absolute agent timeout, ms or with `1h`/`30m`/`90s` suffixes; values under 1 second are ignored with a warning | 20 minutes |
| `streamIdleTimeout` | stream-silence timeout, ms or the same suffixes; `0` disables the idle abort; values under 1 second are ignored with a warning | inherited from the child's config (`90s` when unset there) |
| `worktree` | `true` or `false` | `false` |
| `lsp` | `off`, `min` or `full` | the child's default (`min`) |
| `permission` | block of rules — per-tool rules plus a nested `bash:` block of command patterns — layered over the global rules; a scalar value is ignored with a warning | global rules only |

`mode:` decides where an agent may run: `subagent` (default — spawned as a
subagent only), `primary` (reserved for the main session), `all` (both).
Primary-capable agents are switchable with **Shift+Tab** or `/agent`
mid-conversation; `defaultAgent` in `config.json` names the one an
interactive session starts in (an explicit `--model` keeps the flag's model
and takes only the agent's prompt). Unknown or subagent-only `defaultAgent`
names fall back to the built-in agent with a notice.

See [AGENTS-HOWTO.md](AGENTS-HOWTO.md) for the extension system (skills, hooks,
agent instructions) explained against the Claude Code `.claude/` setup.

## Prompt editor

- **Shift+Enter** inserts a newline; **Enter** submits. The area grows to a
  visual-row cap.
- **Paste collapsing** — a large paste (more than 2 KiB or 30 lines) folds
  into a summary placeholder (`[вставка: N строк / N КБ]` — "pasted: N lines /
  N KiB") in the input; the full text is restored on submit. **Alt+V** opens
  the paste nearest the cursor fullscreen to read it before sending.
- **`@`-mentions** — typing `@` opens a fuzzy file picker; each mentioned
  path is appended to the submitted prompt as a `[Referenced file: …]`
  annotation, across lines too.

## Subagents

The `subagent` tool spawns subagents as child processes. While they run:

- **Monitor** — **Ctrl+T** (or `/subagents`) opens a live view with one card
  per run: agent, model, elapsed timer.
- **Steer** — press `s` on a running agent in the monitor to send it a
  follow-up instruction mid-run.
- **Background** — long runs detach; the model collects their output later
  with the `agent_result` tool, optionally waiting for completion.
- **Prompt queue** — input typed while a response is active is queued, not
  cancelled; queued prompts run in order when the turn ends.

## Permissions

The `permission` block in `config.json` gates tool calls before execution —
the opencode-compatible format: directives on tool-name globs plus bash
command patterns. The full rule syntax and examples live in
[Configuration → Permission rules](configuration.md#permission-rules).

```json
{
  "permission": {
    "edit": "deny",
    "serena*": "ask",
    "bash": {
      "*": "ask",
      "git *": "allow"
    }
  }
}
```

- **Directives**: `allow` passes through; `deny` blocks with an error the
  model sees; `ask` opens the approval dialog in the TUI — and denies in
  headless and subagent runs, where no one can answer.
- **Matching**: `*` matches any run of characters; the most specific pattern
  wins (`edit` beats `ed*`). For bash the command line is matched first, so
  `"git *": allow` allows `git status` even when the bash tool itself is
  `ask`.
- **Layering**: agent frontmatter `permission:` blocks (see
  [Agents](#agents)) replace same-named tool keys and append bash patterns
  over the global rules, and are forwarded to spawned subagents
  automatically.
- **Approval dialog**: answer once, or "always" — the latter records an
  allow override on that rule for the rest of the session, in memory only.
  Persistence is always a manual edit of `config.json`.

## Todo plans

The `todo_write`/`todo_read` tools let the agent maintain a visible plan:
a checklist shown live in the TUI sidebar below the MCP tools, browsable with
`/todos`. The sidebar line reads `Plan — done/total`. Plans survive the turn
they were written in and are the fork's answer to opencode's plan mode — the
agent works through a checklist you can watch tick off, instead of narrating
progress in prose.

## The question tool

The `question` tool is the reverse of the approval dialog: the model asks
*you* a multiple-choice question (up to 8 options, free text allowed by
default) and waits for the answer. In headless, one-shot and subagent runs
there is no one to answer, so the tool returns a note telling the model to
decide itself and proceed — it never hangs a run.

## JSON mode

`--mode json` writes one JSON object per line to stdout. Streamed assistant text
is grouped by sentence, so one `text_delta` carries a whole sentence rather than
a single SSE chunk — a three-sentence reply is 5 lines instead of 76, and the
`delta` fields still concatenate to exactly the reply. Concatenate `delta` to
reconstruct the text; that is how every consumer uses it.

```
{"type":"message_start","agent":"pi","role":"model","session_id":"..."}
{"type":"thinking_delta","agent":"pi","delta":"..."}
{"type":"text_delta","agent":"pi","delta":"Go is a compiled language. "}
{"type":"tool_call","agent":"pi","tool_name":"bash","tool_input":{...}}
{"type":"tool_result","agent":"pi","tool_name":"bash","content":"{...}"}
{"type":"message_end"}
```

`--json-deltas full` switches to one event per model chunk instead of per
sentence, for consumers that want raw deltas.

## Slash commands

The TUI autocomplete and the `/help` dialog (a searchable popup grouping every
command and hotkey by category) are generated from one registry, so what is
listed here is what ships.

| Command                | Description                                                     |
|------------------------|-----------------------------------------------------------------|
| `/help`                | Show help (searchable dialog of commands and keys)              |
| `/clear`               | Clear conversation                                              |
| `/copy`                | Copy conversation to clipboard                                  |
| `/model`               | Show or switch model (no args opens the picker)                 |
| `/agent`               | Show or switch the session agent (no args opens the picker)     |
| `/session`             | Show session info                                               |
| `/context`             | Show context usage                                              |
| `/branch`              | Manage branches                                                 |
| `/compact`             | Compact context                                                 |
| `/subagents`           | Monitor subagents (no args opens the monitor popup)             |
| `/history`             | Command history                                                 |
| `/login`               | Configure API keys (codex, openai, anthropic, gemini)           |
| `/commit`              | Create commit from staged changes                               |
| `/diff`                | Fullscreen diff viewer (working tree / last commit)             |
| `/plan`                | Start PDD planning session                                      |
| `/run`                 | Execute a spec with the cabin-boy agent                          |
| `/pr-autofix`          | Watch a GitHub PR's checks and fix them until it is green       |
| `/retry`               | Re-send the prompt of a turn that failed                        |
| `/skills`              | List skills (create, load)                                      |
| `/theme`               | Switch theme or list themes                                     |
| `/todos`               | Show todo/plan list                                             |
| `/ping`                | Test LLM connectivity                                           |
| `/model-price-refresh` | Refresh model prices from models.dev                            |
| `/rtk`                 | Output compaction stats                                         |
| `/mcp`                 | List MCP servers and tool status                                |
| `/exit`, `/quit`       | Exit                                                            |

`/skill-list`, `/skill-load` and `/skill-create` also work (via `/skills`)
but stay out of autocomplete to keep it concise.

## Hotkeys

| Key         | Action                                                        |
|-------------|---------------------------------------------------------------|
| `Ctrl+T`    | Toggle the subagent monitor                                   |
| `Shift+Tab` | Cycle the session agent                                       |
| `Ctrl+R`    | Re-send the prompt of a turn that failed                      |
| `Ctrl+O`    | Toggle compact tool output                                    |
| `Ctrl+B`    | Toggle the branch popup                                       |
| `Ctrl+H`    | Open history search                                           |
| `Ctrl+Z`    | Suspend the process                                           |
| `Ctrl+C`    | Cancel the running turn; press twice to quit                  |
| `Esc`       | Dismiss an overlay or cancel the running turn                 |
| `Up` / `Down` | Prompt history window / scroll chat; at the bottom `Down` opens the subagent monitor |
| `PgUp` / `PgDn` | Scroll the chat by a page                                 |
| `s`         | Steer the selected running subagent (inside the monitor)      |
| `Alt+V`     | View the collapsed paste nearest the cursor fullscreen        |

## Plugin marketplaces

Install skill bundles from a **plugin marketplace**. Pi-rate reads the same
`.claude-plugin/marketplace.json` manifest other coding agents use, so existing
marketplaces work unchanged:

```bash
# Register the Superpowers marketplace
pirate plugin marketplace add obra/superpowers-marketplace

# Install a plugin from it
pirate plugin install superpowers@superpowers-marketplace

# See what is installed
pirate plugin list
```

| Command                                          | Description                                             |
|--------------------------------------------------|---------------------------------------------------------|
| `pirate plugin marketplace add <source>`         | Register a marketplace (GitHub `owner/repo`, git URL, or local directory) |
| `pirate plugin marketplace list`                 | List registered marketplaces                            |
| `pirate plugin install <plugin[@marketplace]>`   | Install a plugin and make its skills available          |
| `pirate plugin list`                             | List installed plugins, versions, and commits           |
| `pirate plugin update [plugin]`                  | Update one plugin, or all of them                       |
| `pirate plugin uninstall <plugin>`               | Remove a plugin and its files                           |

Installed plugins live in `~/.pirate/plugins/`, and their skills are discovered
automatically. **Plugin skills have lower precedence than your own**: a skill in
`~/.pirate/skills` or `.pirate/skills` with the same name always wins, so
installing a plugin can never silently replace a skill you wrote or customized.

Sources may be a GitHub shorthand (`owner/repo`), a full git URL, or a local
directory. Plugin names come from the marketplace manifest, which is untrusted
input, so they are validated before being used as directory names.

## Memory Palace

A 4-layer contextual memory system that gives the agent persistent awareness across sessions.

**Layers:**

| Layer | Name | Description |
|-------|------|-------------|
| L0 | Identity | Static identity file |
| L1 | Essential Story | Top-15 drawers by importance, injected into system prompt |
| L2 | On-Demand Recall | Context-filtered drawer chunks |
| L3 | Search | Semantic (embedding) or keyword (FTS5) search |

**CLI commands:**

```bash
# Setup
pirate memory model download         # download all-MiniLM-L6-v2 embedding model
pirate memory model status           # check model path and status
pirate memory init [dir]             # create palace.db + generate mempalace.yaml

# Ingest
pirate memory mine <dir>             # mine source files into drawers
pirate memory mine --convos <dir>    # mine conversation files (JSONL/text)

# Query
pirate memory status                 # palace overview (drawers, wings, rooms, KG)
pirate memory search <query>         # semantic or keyword search
pirate memory wake-up                # print L0+L1 context for system prompt
pirate memory recent [project]       # recent memory observations

# Knowledge Graph
pirate memory kg query <entity>      # query triples involving an entity
pirate memory kg add <s> <p> <o>     # add a fact triple
pirate memory kg timeline <entity>   # chronological timeline of facts
```

**Configuration** via `mempalace.yaml` in the project root:

```yaml
wing: my-project
rooms:
  - name: auth
    patterns: ["internal/auth/**"]
    keywords: [jwt, token, session]
  - name: api
    patterns: ["internal/api/**"]
    keywords: [handler, endpoint, route]
```

When the Palace is enabled, the agent also gains tool access: `palace-search`, `palace-add-drawer`, `palace-kg-query`, `palace-kg-add`, `palace-diary-write`, `palace-traverse`, and more.

## Security audit

```bash
# Scan all skill files for hidden Unicode characters
pirate audit

# Scan with verbose output (include info-level findings)
pirate audit -v

# Output as JSON for CI pipelines
pirate audit --format json --output report.json

# Auto-remove dangerous characters (creates .bak backups)
pirate audit --strip

# Preview what would be removed
pirate audit --strip --dry-run

# Scan a specific file
pirate audit --file path/to/SKILL.md
```

Skills are automatically scanned on load — skills with critical findings (Unicode tags, BiDi overrides, variation selector attacks) are blocked from loading.
