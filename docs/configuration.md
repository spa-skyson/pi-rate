# Configuration

Everything configurable in Pi-rate: where config lives, what the file accepts,
and the individual subsystems. The commented, copy-paste-ready reference is
[`config.example.jsonc`](../config.example.jsonc); the machine-readable
description of every field is
[`schemas/config.schema.json`](../schemas/config.schema.json) — this page
explains and links rather than duplicating them.

- [Config file locations](#config-file-locations)
- [JSONC comments and `$schema`](#jsonc-comments-and-schema)
- [Model roles](#model-roles)
- [Declared providers](#declared-providers) — any OpenAI-compatible or Anthropic endpoint, by name
- [Provider base URLs](#provider-base-urls)
- [Custom OpenAI-compatible provider (env-var form)](#custom-openai-compatible-provider-env-var-form)
- [Permission rules](#permission-rules)
- [Attention](#attention) — bell and desktop notifications
- [Custom themes](#custom-themes)
- [Ollama generation tuning](#ollama-generation-tuning)
- [Web search](#web-search) — local daemon and cloud
- [MCP servers](#mcp-servers)
- [Hooks, memory and other fields](#hooks-memory-and-other-fields)

## Config file locations

Pi-rate reads configuration from `~/.pirate/config.json` (global) and
`.pirate/config.json` (project-local). An existing `~/.pi-go` directory is
migrated to `~/.pirate` automatically on first run; the old directory is left
untouched.

## JSONC comments and `$schema`

Config files accept `//` line comments — anything from `//` to the end of a line
is ignored, except inside string values (so `"https://…"` is safe). Note that
Pi-rate itself rewrites config files as plain JSON when saving, so comments in
sections it edits (e.g. `roles`, via `/model`) do not survive a save. A
commented, copy-paste-ready example lives in
[`config.example.jsonc`](../config.example.jsonc).

For editor autocompletion and validation, add:

```json
{
  "$schema": "https://raw.githubusercontent.com/spa-skyson/pi-rate/main/schemas/config.schema.json"
}
```

## Model roles

Roles map names to model configurations, so a role flag or `/model` entry picks
a whole setup rather than one string. `"default"` is the fallback role; `smol`,
`slow` and `plan` are the names the `--smol`/`--slow`/`--plan` flags select. A
model prefixed with a declared provider's name (`corp-claude/…`) routes to that
provider automatically.

```json
{
  "roles": {
    "default": { "model": "gpt-5.6-sol", "thinkingLevel": "high" },
    "smol":    { "model": "ollama/gemma-4-e4b:latest" },
    "slow":    { "model": "corp-claude/claude-opus-5" },
    "commit":  { "model": "gpt-5.6-sol", "advisorModel": "claude-opus-4-7", "advisorMaxUses": 3 }
  }
}
```

`defaultAgent` names the primary agent (`.pirate/agents/*.md` with frontmatter
`mode: primary` or `all`) that an interactive session starts in; unknown names
fall back to the built-in agent. See [Usage → Agents](usage.md#agents).

The API keys a role's provider needs are environment variables — see
[Usage → API keys](usage.md#api-keys).

## Declared providers

Beyond the built-ins, any OpenAI-compatible or Anthropic endpoint can be
declared by name in the `providers` section. The entry name becomes a
model-name prefix — `corp-claude` serves `corp-claude/claude-opus-5`:

```json
{
  "providers": {
    "corp-claude": {
      "type": "anthropic",
      "baseURL": "https://llm.corp.example/v1",
      "apiKey": "${CORP_LLM_KEY}",
      "headers": {
        "x-opencode-session": "${SESSION_ID}"
      },
      "models": {
        "claude-opus-5": { "contextWindow": 200000 }
      }
    }
  }
}
```

- **`type`** — `openai-compatible` (default) or `anthropic`. For
  openai-compatible endpoints the `baseURL` is the full endpoint:
  `/chat/completions` and `/models` are appended to it, no `/v1` is added.
- **Credentials** — `apiKey` expands `${VAR}` from the environment or
  `~/.pirate/.env` at load (so the secret stays out of config.json); the
  literal value wins over `apiKeyEnv`, which names the variable to read
  instead.
- **`${SESSION_ID}`** in `headers` is not an env var: it survives load
  verbatim and is replaced with the current session's id when the client is
  built, so session-scoped headers (like prompt-cache routing) stay stable
  per conversation.
- **`models`** — optional per-model metadata; `contextWindow` enables
  auto-compaction for models absent from the embedded catalog.
- Names must not collide with a built-in provider, and every entry is
  validated at load — a bad `baseURL` or `type` fails fast, naming the
  entry.

## Provider base URLs

Self-hosted or LAN endpoints can be declared in config instead of exported in every shell:

```json
{
  "roles": {
    "default": { "model": "ollama/gemma-4-e4b:latest", "provider": "ollama" }
  },
  "baseURLs": {
    "ollama": "http://192.168.1.10:11434"
  }
}
```

Precedence is `--url` flag, then environment variable, then `baseURLs` config. The matching env vars are
`ANTHROPIC_BASE_URL`, `OPENAI_BASE_URL`, `GEMINI_BASE_URL`, `MISTRAL_BASE_URL`, `XAI_BASE_URL`, `OPENROUTER_BASE_URL`, `OPENCODE_BASE_URL`, and `OLLAMA_HOST`. A per-shell or
CI override still takes effect. An empty env var does not mask a configured value.

## Custom OpenAI-compatible provider (env-var form)

For OpenAI-compatible APIs with model names that Pi-rate cannot infer from a prefix, explicitly set the role provider to
`openai` and point `OPENAI_BASE_URL` at the custom endpoint:

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.example.com/v1"
```

```json
{
  "roles": {
    "default": {
      "model": "Qwen3.5-397B-A17B-FP8",
      "provider": "openai"
    }
  }
}
```

Then run Pi-rate normally:

```bash
pirate
```

You can also pass the endpoint per invocation:

```bash
OPENAI_API_KEY="your-api-key" pirate --model Qwen3.5-397B-A17B-FP8 --url https://api.example.com/v1
```

When `--url` or `OPENAI_BASE_URL` is set, unknown model names are treated as custom OpenAI-compatible models. Setting
`provider: "openai"` in config avoids relying on model-prefix detection.

For a provider you want to keep around with a name of its own, declare it in
the `providers` section instead — see [Declared providers](#declared-providers).

## Permission rules

The `permission` block gates tool calls before execution — the
opencode-compatible format: a directive per tool-name glob, plus bash
command-line patterns under the reserved `bash` key. The behaviour of the
directives, the matching rules and the approval dialog is described in
[Usage → Permissions](usage.md#permissions); this section is the config shape.

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

The `bash` rules also accept an array form in which each entry is a
`{ "pattern", "directive" }` object and the later rule wins a specificity tie:

```json
{
  "permission": {
    "edit": "ask",
    "bash": [
      { "pattern": "git status", "directive": "allow" },
      { "pattern": "git push*", "directive": "deny" },
      { "pattern": "rm -rf *", "directive": "deny" },
      { "pattern": "*", "directive": "ask" }
    ]
  }
}
```

The same keys can be scoped to one agent in its frontmatter — see
[Usage → Agents](usage.md#agents).

## Attention

When the TUI needs you back — a tool-approval dialog opening, or a long turn
finishing — it rings the terminal bell and sends an OSC 777 desktop
notification. Both channels are on by default; `attention` turns them off
individually:

```json
{
  "attention": {
    "bell": true,
    "notify": false
  }
}
```

## Custom themes

Custom themes are JSON files dropped into `~/.pirate/themes/` (global) or
`.pirate/themes/` (project — the nearest one up the directory tree wins). The
file name, lowercased and without the `.json` extension, becomes the theme
name; pick it with `/theme <name>` like any built-in — custom entries are
tagged `(custom)` in the `/theme` list, and the choice persists in the config.
A theme that shares a name with a built-in overrides it; a project file
overrides a global file of the same name. Partial files are fine: missing
color roles are filled in from the built-in theme of the same name, or from
the default theme. [`themes/example.json`](../themes/example.json) shows the full
format. The 13 color roles under `colors`: `text`, `base` and `secondary`
paint body text, background and muted text; `primary`, `info`, `warning`,
`error` and `success` color prompts, notices and statuses; `tool` names tool
calls; `diffAdded`/`diffRemoved` tint added and removed diff lines and
`diffAddedText`/`diffRemovedText` the text on them. A file that is not valid
JSON or declares no color roles is reported and skipped; the running theme is
never affected.

## Ollama generation tuning

Ollama's per-request options are left at the server's own defaults, except for an
output cap. Each knob below is opt-in: unset means the option is not sent at all,
so Ollama's default stays in force. An unparseable value is ignored rather than
fatal — a typo in an env var should not take down an otherwise healthy session.

| Env var | Ollama option | Ollama default | Purpose |
|---|---|---|---|
| `PI_OLLAMA_NUM_PREDICT` | `num_predict` | unlimited | Max tokens generated per turn. Pi-rate defaults this to `16384`; `0` or less removes the cap. |
| `PI_OLLAMA_REPEAT_PENALTY` | `repeat_penalty` | `1.1` | How strongly repeated tokens are penalised. `1.0` disables. |
| `PI_OLLAMA_REPEAT_LAST_N` | `repeat_last_n` | `64` | How many recent tokens the penalty looks back over. `0` disables, `-1` uses the full context. |
| `PI_OLLAMA_PRESENCE_PENALTY` | `presence_penalty` | `0.0` | Flat penalty for tokens already used. |
| `PI_OLLAMA_FREQUENCY_PENALTY` | `frequency_penalty` | `0.0` | Penalty scaled by how often a token was used. |

These matter for models prone to repetition collapse, where a turn stops making
progress and restates the same phrase until it hits a limit. `num_predict` only
bounds how far such a turn runs; it does not stop it degenerating. The penalty
window is the knob that targets the cause, and the default window is narrow:
Ollama penalises repeats across the last 64 tokens only, while observed
degenerate turns cycle on phrases of roughly 25–55 tokens, so a full cycle can
fall outside the window the penalty can see.

```bash
# Widen the repetition window and penalise repeats harder.
export PI_OLLAMA_REPEAT_LAST_N=512
export PI_OLLAMA_REPEAT_PENALTY=1.2
```

Both apply to local Ollama and Ollama Cloud — they share one request path. Raising
these trades diversity for repetition control, and a value that helps one model can
degrade another, so tune per model rather than setting them globally.

## Web search

The `web_search` tool lets the agent look up things that are not in the
repository — a library's current release, a recent API change, an error message
it has not seen before.

**Off by default.** It needs a backend that is frequently absent, so a session
without one would advertise a tool the model calls and then fails on:

```
web_search  error: Post "https://ollama.com:443/api/web_search": net/http: TLS handshake timeout
```

Turn it on with the flag, or with the environment variable:

```bash
pirate --web-search-enabled "what changed in the latest Ollama release?"   # flag
PI_WEB_SEARCH=1 pirate "what changed in the latest Ollama release?"        # env
```

Prefer `PI_WEB_SEARCH` when subagents matter: a subagent runs as a child `pirate`
process whose command line carries only model, url, headers and `--lsp`, so a
flag does not reach it, while the `PI_` prefix is forwarded to every child.

It uses one of two Ollama endpoints, tried in order:

1. **A local daemon** (`OLLAMA_HOST`, default `http://localhost:11434`), on
   `/api/experimental/web_search`. No key is needed: the daemon searches using
   the identity from `ollama signin`. This path spends no quota, so it is tried
   first.
2. **`https://ollama.com/api/web_search`**, when `OLLAMA_API_KEY` is set. This is
   the fallback for environments with no daemon — CI runners, dev containers, VS
   Code remotes. It draws on the account's monthly search quota.

With neither available, the tool returns that as an ordinary result naming what
to set, rather than failing the turn.

The two endpoints are different *paths*, not the same path on two hosts, which is
why Pi-rate calls them directly instead of through the Ollama Go SDK: the SDK's
`WebSearchExperimental` posts to `/api/experimental/web_search`, which 404s on
`api.ollama.com`.

Each result keeps its URL and is capped at 4000 bytes of content, so a large page
— a GitHub repository page comes back as the whole rendered README — spends a
bounded part of the context window, and the model can still fetch the full page
with `bash` when a snippet is not enough.

```bash
# Local daemon (no key needed)
pirate --web-search-enabled "what changed in the latest Ollama release?"

# Headless / CI, using the cloud search API
export OLLAMA_API_KEY="..."
PI_WEB_SEARCH=1 pirate "what changed in the latest Ollama release?"
```

## MCP servers

Pi-rate supports the [Model Context Protocol](https://modelcontextprotocol.io/). Use it to extend the agent with external tools. Configure servers in
`~/.pirate/config.json`:

```json
{
  "mcp": {
    "servers": [
      {
        "name": "tavily-search",
        "url": "https://mcp.tavily.com/mcp/?tavilyApiKey=${TAVILY_API_KEY}"
      },
      {
        "name": "filesystem",
        "command": "npx",
        "args": [
          "-y",
          "@modelcontextprotocol/server-filesystem",
          "/tmp"
        ]
      }
    ]
  }
}
```

Or in standalone `~/.pirate/mcp.json` (Claude Desktop compatible format):

```json
{
  "mcpServers": {
    "tavily-search": {
      "url": "https://mcp.tavily.com/mcp/?tavilyApiKey=${TAVILY_API_KEY}"
    },
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/tmp"
      ]
    }
  }
}
```

**Supported transports:**

- **HTTP/Streamable** — `url` field for cloud-based MCP servers
- **Stdio** — `command` + `args` for local subprocess servers

**Environment variable substitution:** Pi-rate automatically expands `${ENV_VAR}` patterns in server URLs using `.pirate/.env`

`/mcp` in the TUI lists the configured servers and their tool status.

**Headless runs start no MCP servers by default.** `--mode print` and
`--mode json` are one-shot: they pay the startup of every configured server
per prompt, so a server is lifted only when the run names one — pass
`--mcp <name1>,<name2>` (server names, case-insensitive) or `--mcp all` to
restore the previous behavior; naming a server in `--tools` keeps working
too. Interactive and server modes (`rpc`, `socket`) always start every
configured server.

## Hooks, memory and other fields

Every field this page does not cover — hooks (shell callbacks on tool events),
the persistent-memory settings (`memory.enabled`, `token_budget`,
`compressor`, …), budgets and guards (`maxDailyTokens`, `insecureSkipTLS`,
`caCertPath`, `traceHTTP`, …) and the rest — is documented inline in
[`config.example.jsonc`](../config.example.jsonc) and enumerated in
[`schemas/config.schema.json`](../schemas/config.schema.json). The skills,
hooks and agent-instructions side of the extension system has its own guide:
[AGENTS-HOWTO.md](AGENTS-HOWTO.md).
