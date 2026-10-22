# AGENTS.md — pi-rate

Guidance for coding agents working in this repo. Applies to Claude Code and to
pi-rate's own agent (`pirate`); where the two differ, both are described.

pi-rate is a fork of `dimetron/pi-go`. History older than the fork point
belongs to upstream; this file describes *this* repository.

## Repo and release layout

**Remotes.** `gitlab` (`git.home.fwz.ru/spa/pi-rate`) is the primary: branches
and merge requests live there. `origin` (`github.com/spa-skyson/pi-rate`) is a
mirror that carries CI and releases: GitHub Actions (`.github/workflows/`)
runs against it, and GitHub Releases are where binaries, the SBOM, the kagent
manifest and the VSIX are published. `upstream` (`dimetron/pi-go`) exists only
to track the fork point — never push to it.

**Merges.** Work lands through a GitLab MR merged with a **merge commit**
(`Merge branch 'x' into 'main'`). There is no squash and no rebase-merge; the
GitLab-created merge commit itself is unsigned — expected, see "Commits: all
commits must be signed".

**Releases.** A release is a **signed annotated tag** `vX.Y.Z` (semver) on
`main`, pushed to GitHub. The tag triggers the `Release` workflow
(`.github/workflows/release.yml`): Lint → Test → Docker image → GoReleaser →
kagent manifest → VS Code → Publish. The release is created as a **draft** and
published only by the last job, after every asset is attached — published
releases are immutable and accept no new assets. Consequence: the tag can be
**moved before publication** (delete and re-push it, re-run the workflow; the
draft-exists checks read drafts). Once published, the release is sealed —
never retag a published version. This is not theoretical: v0.5.7's first tag
push failed the Lint job, the tag was re-pushed after the fix, and the release
shipped from the moved tag. Latest release: `v0.5.7`.

## Before starting work

Use an isolated git worktree for every task that edits tracked files. Before
reading or changing code, enumerate the repository's instruction files and read
the applicable ones:

```bash
rg --files -g 'AGENTS.md' -g 'CLAUDE.md' -g '!tmp/**' -g '!.git/**' ..
```

Read the root `AGENTS.md` first, then any instruction file in each parent
directory of the files you will touch. Instructions closer to a file add
constraints to the repository guidance; do not skip them because a task appears
small.

## Work in a git worktree, not the primary checkout

**Do not make uncommitted edits in the primary checkout and leave them there.**
The primary checkout has its branch switched frequently, and `git checkout`
discards uncommitted changes in tracked files without warning. Work has been
lost to this. A worktree gives each task its own working directory and its own
branch, so a switch in one cannot destroy another.

### Check first: are you already in a worktree?

**Do not create a worktree when you are already inside one.** Nesting a
worktree below another puts the new branch's base at the enclosing worktree's
HEAD, and leaves two worktrees whose branches shadow each other on the same
task. If you are already in one, work in the current directory.

The check is a comparison, because `git rev-parse --git-dir` and
`--git-common-dir` differ in exactly the case that matters:

```bash
# Already in a worktree? Then the two paths differ.
if [ "$(git rev-parse --git-dir)" != "$(git rev-parse --git-common-dir)" ]; then
  echo "already in a worktree: $(git rev-parse --show-toplevel) — work here"
else
  echo "primary checkout — create a worktree for this task"
fi
```

In the primary checkout both report `.git`. Inside a linked worktree `--git-dir`
is `<repo>/.git/worktrees/<name>` while `--git-common-dir` stays `<repo>/.git`.

Create one only when that check says you are in the primary checkout:

```bash
git worktree add -b fix/<topic> .worktrees/fix-<topic> HEAD
cd .worktrees/fix-<topic>
```

Remove it when the branch is merged or abandoned:

```bash
git worktree remove .worktrees/fix-<topic>
git worktree list          # verify; prune stale metadata with `git worktree prune`
```

### Where worktrees live

Two conventions coexist. Match the one that fits who is doing the work.

| Creator | Path | Branch | Notes |
|---|---|---|---|
| Human / Claude Code | `<repo>/.worktrees/<branch-with-dashes>` | `fix/…`, `feat/…` | Inside the repo; `.worktrees/` is gitignored |
| pi-rate agent (`/run`, subagents) | `<repo>/.pirate/tasks/<pathID>` | `pi-agent-<shortID>`, or the sanitized requested name | Created by `internal/subagent/worktree.go:339`; `.pirate/` is gitignored |

`.pirate/` (and the legacy `.pi-go/` spelling) are both gitignored, so agent
worktrees never show up as untracked noise in `git status`.

### What pi-rate's agent already does, and why it matters

`WorktreeManager.Create` (`internal/subagent/worktree.go:277`) **stashes
uncommitted changes before `git worktree add` and pops them afterwards**, with
a unique stash message so the pop is deterministic
(`stashMessage`, `popStashByMessage` at `internal/subagent/worktree.go:85,129`).
It does this because `worktree add` from HEAD fails on a dirty tree.

The consequence worth knowing: if a pi-rate subagent runs while you have
uncommitted work in the primary checkout, your changes take a round trip
through the stash. That is safe, but it is one more reason not to keep
long-lived uncommitted work in the primary checkout.

Agents marked `[worktree]` edit an isolated tree. Their edits do **not** land in
the caller's tree — ask for an explicit patch or file list to apply, or use a
non-worktree editing agent (`internal/tools/subagent.go:238`).

## Commits: all commits must be signed

**Rule: every commit must be cryptographically signed and carry a
`Signed-off-by` trailer.** There is no exception — not for merge commits, not
for reverts, not for WIP or "just this once". An unsigned commit is a broken
commit; fix it before pushing (see the pre-push hook below).

The signing command:

```bash
git commit -s -S -m "..."     # -s = Signed-off-by trailer, -S = sign
```

`-S` is redundant when `commit.gpgsign` is enabled, but **signing here is
explicit — nothing signs for you**: the `commit-msg` hook adds a missing
`Signed-off-by` trailer but does not sign. A bare `git commit` lands unsigned.
Pass `-S` every time. The other hooks back the rule up: `post-commit` warns on
unsigned commits, and `pre-push` hard-fails any push containing an unsigned
commit or one missing a matching `Signed-off-by` trailer — so an unsigned
commit cannot reach a remote.

Branch commits stay signed through the merge: GitLab merges MRs with a merge
commit, and the first-parent merge commit it creates is itself unsigned
(see below). The commits that carry the change must be signed.

Signing here is SSH-format, not GPG (`gpg.format = ssh`, key from
`user.signingkey`).

### Verify signatures with `git verify-commit`, never with the `gpgsig` header

**A `gpgsig` header does not mean the signature is valid.** It only means signing
was *attempted* at the time the commit was created. Rewriting the commit object
afterwards keeps the header and silently invalidates the signature. That is not
hypothetical: appending a trailer, dropping a duplicate one, or re-writing the
message with `git hash-object -t commit -w` all produce a commit that passes a
header check and fails verification.

```bash
# WRONG — passes on invalid signatures
git cat-file commit HEAD | grep -q '^gpgsig' && echo signed || echo UNSIGNED

# RIGHT — actually verifies the signature
git verify-commit HEAD && echo VALID || echo INVALID
```

One-time setup, without which verification cannot run at all
(`gpg.ssh.allowedSignersFile needs to be configured and exist`, and `%G?`
reports `N` for signed and unsigned commits alike):

```bash
echo "$(git config user.email) $(git config user.signingkey)" > ~/.config/git/allowed_signers
git config --global gpg.ssh.allowedSignersFile ~/.config/git/allowed_signers
```

Then check a range — every commit an MR would publish:

```bash
git log --format='%h' origin/main..HEAD | while read c; do
  printf '%s ' "$c"
  git verify-commit "$c" >/dev/null 2>&1 && echo VALID || echo ">>> INVALID <<<"
done
git log --format='%h %(trailers:key=Signed-off-by,valueonly,separator=;) %s' -5
```

GitHub's verdict on the mirror agrees with `git verify-commit`:

```bash
gh api repos/spa-skyson/pi-rate/commits/<sha> \
  --jq '.commit.verification | "\(.verified) \(.reason)"'
```

**To fix commits that carry an invalid signature**, re-sign them — do not
hand-edit the object:

```bash
git rebase --exec 'git commit --amend --no-edit -S' origin/main
```

`-S` signs. Do **not** pass `-s` as well when the message already ends in a
`Signed-off-by` line: you get a duplicate trailer, and "fixing" that by
rewriting the message with `git hash-object` is what breaks the signature in the
first place. Prefer letting the `commit-msg` hook add the trailer, or run
`git commit -S` alone.

### Two kinds of commit that fail a local check and are not your problem

**GitLab-created merge commits on `main` are unsigned** (`%G?` = `N`) and fail
`git verify-commit` locally. GitLab builds the merge commit server-side and
does not sign it with a key your `allowed_signers` knows. They are never in a
feature-branch push range (`rev-list <sha> --not --remotes=gitlab` is empty for
them), so `pre-push` does not reject them. Do not "fix" them.

**History inherited from upstream pi-go contains commits that do not verify**
— unsigned ones, GitHub-rewritten ones (`E`: signature present but the key is
unknown locally), even a couple with `B` status. All predate the fork and are
shared published history; rewriting them would invalidate every clone and
desync from upstream. Record, don't repair.

### Mirror drift and the converge merge

The mirror can drift from `gitlab/main`: a PR merged through the GitHub web
UI lands on `origin/main` only, and gitlab never sees it. Converging means
merging `origin/main` into gitlab as an **empty converge merge** and pushing
the result back to origin — a push that carries the unsigned GitLab-created
merge commits and is therefore rejected by `pre-push`. Do not reach for
`--no-verify`; run the one push with the narrow escape hatch, in the spirit
of `PI_ALLOW_LARGE_FILES`:

```bash
PI_PREPUSH_ALLOW_SERVER_MERGES=1 git push origin main
```

The variable passes **merge commits only** (two or more parents). Unsigned
ones warn and go through, since server-side merges are legal per the
paragraph above; the Signed-off-by requirement is waived for them too,
because a server-built merge carries neither a user signature nor a user
Signed-off-by. A merge may also carry a **valid signature without the
trailer** — a local converge merge created with `-S` and no `-s`, already
published on gitlab: the signature proves authorship more strictly than the
trailer would, so it is waved under the same env with its own warning.
Unsigned **non-merge** commits still fail both checks unconditionally, and
the size check is unchanged for every commit.

### Never use `--no-verify`

**Do not pass `--no-verify` to `git commit`, ever.** Not to unblock a failing
hook, not "just this once", not with a note in the commit message. There is no
case in this repo where it is the right answer.

`--no-verify` skips *all* hooks, including the signing path — so a bypassed
commit lands unsigned. Set up `gpg.ssh.allowedSignersFile` (above) so that
`git verify-commit` catches problems immediately rather than leaving it for a
reviewer to notice.

The hook that usually tempts this is `golangci-lint` in `pre-commit`, which
runs repo-wide and so can fail on issues outside the files you touched. When
that happens, stop and report it — the fix is to clear the lint failure or to
have the user decide, not to bypass the hook. See "Lint is repo-wide" below.

## Never commit a GIF or a screen recording

**Recordings go on a release, never into git history.** A GIF of a test
run or a TUI session is large, write-once, and stale within a week — but every
revision of one is downloaded by every clone forever, and a blob cannot be
un-pushed without rewriting history for everyone. GitHub itself warns above
50 MiB and rejects a push above 100 MiB.

Attach one to a GitHub release as an asset instead of committing it.

`.githooks/check-large-files` enforces it from both `pre-commit` (staged blobs)
and `pre-push` (blobs the push would upload), with two limits:

| What | Limit | Why |
|---|---|---|
| `.gif .gifv .apng .mp4 .m4v .mov .webm .mkv .avi .ogv .cast` | 1 MiB | Recordings belong on a release |
| Anything else | 10 MiB | Far below GitHub's 100 MiB hard reject; the largest non-media file in the tree is ~70 KiB |

The check runs twice on purpose. `pre-commit` catches it early, when unstaging
is the whole fix; `pre-push` is the last reversible moment for a commit that
reached the branch some other way — a rebase, a cherry-pick, another tool.

One path is grandfathered in the script's `allow_re`: `docs/screen/pi-go.gif`
(5.4 MiB), committed before the hook existed and referenced by nothing in the
tree. It should move to a release asset and take its allowlist entry with it.

If a file genuinely has to live in the tree, shrink it, or as a last resort:

```bash
PI_ALLOW_LARGE_FILES=1 git commit -sS -m "..."
```

**Do not reach for `--no-verify`** — it skips the signing hooks too, and lands
an unsigned commit (see above).

## Merge requests and review

**Work lands through a GitLab MR, and the MR is the review track.** Push the
branch to the `gitlab` remote, open the MR, and have it reviewed — by a human
reviewer or a review agent — before merging. Keep every finding, fix, and
resolution in MR threads so the record stays in one place; a local terminal
dump of findings is lost context, MR threads are not.

The flow:

1. **Finish the branch and open the MR.** All gates — build, tests, lint, vet —
   run *before* pushing; opening early does not skip them.
2. **Have it reviewed** (a reviewer, or a review agent run against the MR
   diff). Findings arrive as comments; fix accepted ones in normal signed
   commits pushed to the branch and resolve the thread after verification.
   For a dismissed finding, reply in the thread with the reason before
   resolving it — never silently ignore a finding.
3. **Merge with a merge commit** via GitLab. No squash, no fast-forward:
   history keeps the branch shape (`Merge branch 'x' into 'main'`).

```bash
git add -A
git commit -s -S -m "feat(scope): ..."   # Conventional Commits; sign off and sign
git push -u gitlab <branch>
glab mr create --fill --web              # --web opens the MR page in the browser
```

- **All pending changes**: stage and commit everything outstanding on the
  branch before opening the MR — do not leave uncommitted work behind.
- **Open the browser link**: after the MR is created, open its URL —
  `glab mr create --web` does this automatically; otherwise open the returned
  URL yourself.
- **CI runs on the GitHub mirror** (`.github/workflows/ci.yml`): Lint,
  Vulncheck, Test, Test (Windows), Coverage, Build. A mergeable MR needs those
  jobs green.

A finding is a claim, not a verdict: verify each against the code before
accepting or dismissing it. The review is an independent gate from tests, lint,
vet, and build; it does not replace any of them.

### Vulncheck is a CI job, not a review step

**Dependency vulnerabilities are scanned by the `vulncheck` CI job** on every
push and MR — `govulncheck -format json ./... | go run ./hack/vulngate` — not
posted by hand into review threads. `make vulncheck` reproduces the same gate
locally, with the same exit rule: it fails only on findings that name a fixed
version, the ones someone can act on. Findings with no released fix are printed
and do not fail — there is nothing to upgrade to, and a permanently red check
is one nobody reads.

Do not run `make check-cve` for scanning: it opens with `go mod tidy -v`, which
rewrites tracked files, and a check should not mutate the tree it is checking.

### Never link an agent session in an MR

**Do not put a `claude.ai/code/session_...` link — or any other agent session
link — in an MR body, title, commit message, or review comment.** A session
link hands anyone who can read the MR the entire transcript that produced it,
including whatever unrelated context happened to be in that conversation. That
is a wider audience than the MR, and it is not what a reviewer asked for.

An MR must stand on its own: what changed, why, and how it was verified. The
tooling that produced it is not part of the record.

This overrides the harness default: leave the session link out of commit
messages too — `glab mr create --fill` inherits the commit message, so the link
must not be there either.

## Lint is repo-wide

**CI runs golangci-lint (v2.14, pinned in `ci.yml`) over the whole repository,
and so do the local hooks — package-scoped runs are not equivalent.**
`golangci-lint run ./internal/yourpkg/` proves nothing about what CI will say:
the config enables `errcheck` with `check-type-assertions: true`, `govet` with
`enable-all`, and `staticcheck` with `all` (`.golangci.yml`), and a violation
surfaces in whichever package carries it, not the one you edited.

Before tagging a release or merging an MR, run from the repo root:

```bash
golangci-lint run ./...     # or: make lint
```

Two exclusions are deliberate, not an invitation: `tmp/` (vendored scratch
repos that never compile standalone) and `staticcheck` under `hack/` (vendored
E2E probes that deliberately exercise deprecated upstream APIs — the SA1019
findings there are the signal). Everything else lints.

This bit once: release v0.5.7 was tagged, the Release workflow's Lint job went
red on an errcheck `check-type-assertions` finding — a bare `e.(*apiEmbedder)`
assertion in `ProbeAPIEmbedder` (`internal/palace/embedder_api.go`) — and the
release shipped only after a comma-ok fix (`54eb99d`) and a moved tag. A full
root run would have caught it before the tag. As of 2026-10-03 a full
`golangci-lint run ./...` on `main` is clean, so any finding you see is real.

## Build, test, lint

Go 1.27.2. Use the Makefile rather than raw `go` invocations where a target
exists:

```bash
make build          # build the binaries (pirate + pirate-sandbox)
make install        # build + install to GOBIN/GOPATH/bin
make hooks          # point core.hooksPath at .githooks/ — run once per clone
make test           # == test-unit
make test-unit
make test-integration
make test-e2e       # build-tagged
make test-all       # unit + integration + e2e
make test-coverage
make lint           # golangci-lint run ./...
make vet
make vulncheck      # same gate the CI vulncheck job runs
make check-cve      # note: runs `go mod tidy -v`, mutates the tree
```

### Two failures macOS cannot catch, but Windows CI will

CI runs a `Test (Windows)` job and a `Build` matrix that cross-compiles
`(windows, amd64)`. Both classes below pass on macOS and fail only there, so a
green local run is not evidence they are fixed.

- **An open directory handle breaks `t.TempDir()` cleanup.** Windows refuses to
  remove a directory whose entries are still open: `TempDir RemoveAll cleanup:
  unlinkat ...: The process cannot access the file because it is being used by
  another process`. `tools.NewSandbox` holds an open root, so a test that builds
  one must release it:

  ```go
  sb, err := NewSandbox(t.TempDir())
  if err != nil {
      t.Fatal(err)
  }
  t.Cleanup(func() { _ = sb.Close() })
  ```

  The failure surfaces during *cleanup*, so the test's own assertions pass and the
  report points at `testing.go` rather than at your test. `internal/agent` wraps
  this in `testSandbox`; use that helper, or close explicitly.

- **`rg` is not installed on the Windows runner.** `newGrepTool` self-names by
  capability — `"ripgrep"` when `rg` is present, `"grep"` when it is not
  (`rgAvailable` at `internal/tools/grep.go:96`, naming at `:128-133`) — so a
  test that requires `"ripgrep"` by name fails on any
  host without `rg`. Assert *exactly one* of the two is registered and that both
  spellings route, and cover both branches by toggling the `rgAvailable` package
  variable (`TestCompactorRouting_WithRipgrep` / `_WithoutRipgrep` show the
  pattern). Do not hard-code either name.

When CI reports a Windows failure, read the job log directly rather than guessing:

```bash
gh run list --branch <branch> --limit 5 \
  --json databaseId,conclusion,headSha --jq '.[].databaseId' -R spa-skyson/pi-rate
gh api repos/spa-skyson/pi-rate/actions/runs/<run>/jobs \
  --jq '.jobs[] | select(.name|test("Windows")) | .id'
gh api --allow-escape-sequences repos/spa-skyson/pi-rate/actions/jobs/<job>/logs \
  | tr -d '\r' | grep -E 'FAIL|panic'
```

`--allow-escape-sequences` is required; without it the log API refuses to print.

### VS Code extension (`vscode/`)

The extension is built and installed from `vscode/` with its own Makefile:

```bash
cd vscode && make install   # bun compile → vsce package → install
```

Local build output goes to `$HOME/.vscode-ext/` (`pirate-vscode.vsix`), never
into the repo — worktree branches must not accumulate VSIX artifacts, and
`.vscode-ext` output is kept out of `git status`. The `install` target
auto-detects the VS Code CLI: `code` on PATH first, then the binary inside the
Insiders app bundle, then stable VS Code. CI (`release.yml`) uses `make release`
only, which keeps the versioned `pirate-vscode-<version>.vsix` at the repo root
for the release upload glob — do not move that output.

## TUI output safety: never write to stdout/stderr

The interactive TUI runs on the terminal's alternate screen. **Any write to
stdout or stderr from inside the TUI corrupts the display** — stray `fmt.Print*`,
`log.Print*`, `os.Stdout.Write`, or `os.Stderr.Write` calls render as garbage
over the UI and break the session.

Rules:

- **Never** use `fmt.Print*`, `log.Print*`, `stdlog`, `os.Stdout`, or
  `os.Stderr` to emit diagnostics or output from code that runs while the TUI
  is active (agent loop, callbacks, hooks, commands, model/tool callbacks).
- **Route diagnostics through the session logger** (`m.cfg.Logger` /
  `logger.Logger`) instead — `Info`, `Error`, `Errorf`, etc. These write to the
  session log file, never the terminal.
- **Allowed TUI outputs** are the only sanctioned ways to surface text to the
  user: the chat transcript, `SystemNoticeCh` (short system notices like
  auto-compaction outcomes), and the TUI's own status/error rendering. If a
  message must reach the user, deliver it through one of these, not a raw
  stdout/stderr write.
- A panic handler may write to stderr only as a last resort before the process
  dies; it must never be used for routine logging.

When in doubt, grep for `Printf|Println|os.Stdout|os.Stderr|stdlog` in the
package you are touching and confirm every hit is either outside the TUI path
or routed through the session logger.

## Prompt budget conventions

The always-on system block is paid for on every request; everything else
should load on demand. Conventions that keep it that way:

- **Prefer the cheapest form that does the job: skill by trigger > script >
  subagent > always-on prose.** A skill's trigger line costs one line per
  request; its body is read only when triggered. Always-on prose is the
  expensive last resort — anything added there must be subtracted elsewhere.
- **Every skill `description` states its activation condition** (a
  "use when …" style phrase). A description carrying none earns a warning at
  menu build (`warnSkillMissingTrigger`, `internal/agent/agent.go`): without
  a stated trigger the skill can sit in the menu unopened forever.
- **Every cut of tool output carries the shared marker
  `… (+N chars truncated)`** — N = characters cut
  (`truncationMarker`, `internal/tools/truncate.go`). One shape everywhere,
  so the model never mistakes a rendering artifact for the whole answer.
- **The assembled always-on block is gated at 32 000 chars**
  (`maxInstructionTotal`, `internal/agent/agent.go`). Past the budget agent
  assembly fails naming every overrunning section — keep the sum under it;
  the per-block Rules budgets alone do not guarantee that.

## Tool-output compaction: one route per tool, and write what you read

`internal/tools/compactor*.go` shrinks tool results before they reach the model.
Seven of its nine pipelines were no-ops in production for a long time, and every
cause was a silent mismatch rather than a crash. The invariants below exist so
that cannot recur.

- **Route on the registered name.** `compactorPipelines` is keyed by the name a
  tool actually registers — hyphens for the git tools (`git-file-diff`), and
  `ripgrep` for the search tool, which self-names via `grepToolName` and is
  built once by `CoreTools`. The tool registers `grep` only on a host without
  `rg`. A `switch` on underscore spellings is what made three git pipelines
  unreachable.
- **A pipeline may only write the field it read.** `CompactResult` carries
  `CompactWrite{Key, Value}` pairs. Do not add a probe order that guesses a
  target field: the previous `stdout → content → output → diff → result → data`
  order wrote to the wrong key, and no tool emits `output` at all.
- **Keep the value's type.** ADK round-trips every result through
  `json.Marshal`/`json.Unmarshal` (the ADK module's
  `internal/typeutil/convert.go`, adk v2.4.0), so a
  `[]GrepMatch` arrives as `[]any` of `map[string]any` — never a typed slice.
  Cap the array; do not re-render it to a string. The TUI result summaries read
  the same keys (`internal/tui/tool_display.go`) as `[]any`.
- **Preserve the true total.** A cap sets `truncated` and leaves
  `total_matches`/`total_files`/`total_entries`/`total_hunks` as the pre-cap
  count, so a partial list is never read as the whole answer. `truncated`
  carries `omitempty`, so it is absent from a complete result — `applyCompaction`
  writes named keys unconditionally for exactly this reason.
- **Decline when there is nothing to gain.** Return `nil` when the result would
  not shrink (the rtk `never_worse` guard). Do not cap lists that mislead when
  truncated: `git-overview` caps `recent_commits` but leaves the
  staged/unstaged/untracked lists whole, because hiding a dirty file is a
  correctness problem, not a saving.
- **Test against real structs.** Build test payloads by marshalling the tool's
  actual output struct (`buildProdResult`), never a hand-written map. Synthetic
  maps carrying an `output` key are what kept the dead pipelines green.
  `TestCompactor_EveryToolCompacts` and `TestCompactorRouting_EveryRegisteredTool`
  are the guards; both fail if a route or field drifts.

### Compose the after-tool chain — a slice runs only its first callback

**ADK invokes only the first after-tool callback that returns a non-nil result.**
`Flow.invokeAfterToolCallbacks` (`adk/v2 internal/llminternal/base_flow.go`) is:

```go
for _, callback := range f.AfterToolCallbacks {
    result, err := callback(...)
    if result != nil { return result, nil }   // stops here
}
```

Every pi-rate after-tool callback returns the result map, and the OTEL tracing
callback is registered first and always returns it — so handing ADK a slice ran
*tracing only*. Dedup, the compactor, the LSP after-hook and memory recording
were all silently dead for months, and it is why the compactor's revival changed
nothing at runtime until the chain was composed.

Rules:

- **Pass the chain through `extension.ComposeAfterToolChain`.** It folds the
  callbacks into one, so each stage's effect survives. Do not hand ADK a slice of
  mutating callbacks, and do not "reorder" the slice expecting a later stage to
  run.
- **A composed stage returns `(nil, nil)` to continue**, `(m, nil)` to replace
  the result for later stages, or `(_, err)` to abort. Returning the result map
  from a non-final stage is what short-circuits the chain.
- **Test with the composed callback and a real `agent.Context`.** The tracing
  stage reads trace state off the context and panics on nil, so a test that
  passes nil cannot observe the real chain. Better still, assert end to end
  through a real agent — `internal/agent/after_chain_compose_test.go` shows the
  `toolCallingLLM` pattern.
- **A guard that walks the slice itself proves nothing.** An earlier test in this
  repo looped over the callbacks and fed each output into the next, which is the
  opposite of ADK's semantics; it passed while production stopped at the first
  entry. Model the real invocation or the test cannot fail.

### The deduper reaches fewer tools than it lists

`dedupTools` (`internal/tools/dedup.go`) names the tools whose repeats may be
elided, but the deduper hashes exactly one string field per result —
`primaryOutputField` picks from `stdout|content|output|diff|result|data`. Measured
against each tool's real output struct, only `read`, `read_image` and
`git-file-diff` can actually be elided. `ripgrep`/`grep` (`matches`), `find`
(`files`) and `ls` (`entries`) carry arrays, and `tree` puts its listing under
`tree`, which the probe does not look at. `git-hunk` and `git-overview` were
removed from the list for the same reason.

`TestDedup_ReachableFieldsAreHonest` (`dedup_compaction_test.go:183`) pins which
tools are reachable. Extending
`primaryOutputField` to serialize array payloads would make the remaining five
reachable, but that is a feature change: do it deliberately, and move each tool
into the reachable table as it starts working rather than listing it in advance.

### rtk is a CLI proxy, not a library — do not delegate to it

`rtk` (third-party, `rtk-ai/rtk`, installed on PATH) is an
independent Rust CLI. It is a **reference for design**, not a dependency: the
compactor is native Go. Measured against this repo's real shapes, delegating to
it does not work:

- `rtk pipe -f grep` was **byte-identical** on a 400-match list — the same class
  of no-op bug described above. `rtk grep` (command mode) does compact (93.8% on
  the same payload), but that is a different interface: it *runs* the search
  rather than filtering a result this repo already has.
- `rtk pipe -f go-test` on real `go test` failure output emitted
  `Go test: No tests found` and **exit code 0**, losing both the failure and its
  signal.
- Shelling out per tool result adds a process spawn on the hot path, and rtk's
  filter set is not this repo's tool set.

Use rtk's *decisions* (caps that preserve totals, the never-worse guard,
head-capping logs while keeping status lists whole); do not wire it in.

## Never trust a big win without a proof check

**A large improvement number is a claim about the code, and claims need evidence
of the same size as the number.** A 90% saving, a 10× speedup, "this fixes the
slow path" — each is exactly as likely to mean *the thing never ran* as it is to
mean the work succeeded. Treat a big win as a hypothesis to falsify, not a result
to report.

This is not hypothetical. The compaction work above reported savings per tool,
and three separate defects hid behind those numbers:

- **The shape was impossible.** A pipeline was credited with a 49–76% saving on
  payloads of 200 commits. The tool runs `git log --oneline -10`
  (`git_overview.go:70`), so that input can never occur and the pipeline never
  fires. The number was real and meaningless.
- **The input was synthetic.** Pipelines looked healthy against hand-written maps
  carrying an `output` key no tool emits. The tests passed; production did
  nothing.
- **The saving was measured per-tool, not end-to-end.** Nothing checked that the
  bytes actually left the prompt.

The rules that follow from that:

- **Bound the input by what the producer can emit.** Before measuring, find the
  cap the tool itself enforces (a constant like `maxGrepMatches`, a `-10` in the
  command, `truncateOutput`'s byte ceiling) and measure at or below it. A
  payload larger than the tool can produce measures a program that does not
  exist. State the bound next to the number.
- **Build fixtures from the real types.** Marshal the actual output struct and
  unmarshal into `map[string]any`, the way ADK does. Never a hand-written map: a
  fixture you invented encodes the contract you assumed, which is the thing under
  test.
- **Prove the saving end-to-end, not at the seam.** A per-function number shows a
  function works; it does not show the result reached the model. `AvgResultBytes`
  per tool in the eval harness (`internal/eval/metrics.go`) is the end-to-end
  version of this — it is currently *reported* in every eval but asserted only in
  one unit test, and never gated against a baseline. That gap is why seven dead
  pipelines looked healthy.
- **Check information, not just size.** Ask what the consumer needs from the
  output — a file name, a count, a total, whether the list is complete — and
  assert each survives. A cap that drops the one field a decision depends on is a
  regression that a byte count calls a win. `compactor_preservation_test.go`
  exists for this.
- **Verify the guard can fail.** Reintroduce the bug and confirm the test goes
  red, then restore. A test that has never failed proves nothing; several of the
  guards here were only trusted after being seen to break.
- **Prefer the cheap falsification first.** Before a full eval run, ask what
  result would show the win is fake, and run the smallest thing that would show
  it. Measuring at the producer's real bound is usually a few seconds of work and
  kills most false wins outright.
- **Say which numbers you did not verify.** If a figure came from an earlier
  session, a different shape, or reasoning rather than a run, label it. An
  honest gap is cheap; a confident wrong number is expensive.

When a win is large enough to be worth reporting, it is large enough to be worth
an eval. `make eval-tools` runs one headless scenario per tool family and rolls
the results into a coverage matrix (`internal/eval/scenarios/README.md`);
`make eval-run` and `make eval-judge` drive `/run` end-to-end against the pinned
`eval/base` baseline (`internal/eval/eval.md`). Report the before/after the
harness produced, not the before/after you expected.

## Profiling

`pirate --pprof true` serves `net/http/pprof` on
`http://localhost:6060/debug/pprof` — the flag is persistent, so subcommands get
it too, and the port comes from `--pprof-port` (default `6060`;
`internal/cli/cli.go:268-269`). `pirate --cpuprofile <path>` writes a CPU
profile for the process lifetime, and `make record-pgo` merges eval-suite and
TUI-render profiles into `cmd/pirate/default.pgo`, which `go build` picks up
automatically.

A single sample cannot distinguish churn from a leak — establish drift before
diagnosing, and profile the app in the state being complained about: an empty
session and an aged one are effectively different programs here.

## Session history lookup

When the user asks to check a specific session (e.g.
`sess_c3bfd0398e8e6693176f24ce`) or review session history, search under
**`$HOME/.pirate/`** — that is the session root, not the repo.

Sessions live in `$HOME/.pirate/sessions/<session-id>/`, one directory per
session (`sessionsDir()` in `internal/cli/cli.go:1241`). Each contains:

- `meta.json` — id, title, model, provider, workDir, timestamps, host info.
- `events.jsonl` — the full turn/event stream (user + assistant messages).
- `trajectory.atif.json` — the ATIF trajectory (agent tool-call trace).
- `branches.json` — session branch state.

Session ids look like `sess_<hex>`; directories with the older
`DDMMYY-HHMM-…` shape are migrated sessions from before the rename and read
the same way.

Other useful files under `$HOME/.pirate/`:

- `last-session.json` — metadata of the most recent session start.
- `history.jsonl` — command history.
- `log/` — runtime logs; see the next section.
- `config.json` — default model and settings.
- `memory/` — memory databases (see the memory map below).

Useful commands:

```bash
ls $HOME/.pirate/sessions/ | grep <session-id>   # confirm a session exists
cat $HOME/.pirate/sessions/<session-id>/meta.json
tail -n 50 $HOME/.pirate/sessions/<session-id>/events.jsonl
```

The `session-stats` tool (`internal/tools/session_stats.go`) scans these
directories for anomalies; it defaults to `$HOME/.pirate/sessions` and accepts a
`session_dir` override.

## Session and runtime logging

Two logs record every session, and they are different records:

- **Session events** — `~/.pirate/sessions/<id>/events.jsonl`: the persisted
  conversation (user + assistant messages, tool calls), plus
  `trajectory.atif.json` next to it.
- **Runtime logs** — `~/.pirate/log/YYYY-MM-DD/session-HH-MM-SS.log`: JSON
  lines with full nanosecond timestamps and typed entries — `user`,
  `llm_text`, `thinking`, `tool_call`, `tool_result`, `error`, `info`,
  `http_request`, `http_response` (`internal/logger/logger.go:32`).

**For timings, trust the runtime log.** Events that a batch persists together
carry the timestamp of the batch's first part in `events.jsonl`, so an apparent
"model gap" (a long silence before a burst of events) can be a batching
artifact rather than a real pause. The runtime log's per-entry `time` field is
the source of exact durations; correlate by session id and entry type.

```bash
grep '"type":"tool_call"' $HOME/.pirate/log/2026-10-03/session-12-25-33.log
```

## Memory subsystem map

Four things carry the name "memory" here; they are different systems.

| Subsystem | Where | Notes |
|---|---|---|
| Observation mining | `internal/memory` → `~/.pirate/memory/claude-mem.db` | Session observations and semantic search. The compressor is `"none"` by default (`CompressorNone`, `internal/config/config.go:87`): raw events, no model call. |
| MemPalace | `internal/palace` → `<project>/.pirate/palace.db` | Embeddings — Ollama, an OpenAI-compatible `/v1/embeddings` endpoint (`ProbeAPIEmbedder`), or a local ONNX backend — indexed into an SQLite FTS5 store. `RankBySimilarity` (`internal/palace/embedder.go:213`) scores only vectors of the query's dimension and drops the rest; never assume a candidate was scored. Known bug: agent and CLI resolve the DB path differently (`memory_init`/`memory_mine` anchor at the project dir, `memory_status`/`memory_search` default to cwd-relative `.pirate/palace.db`) — finding F3 in `specs/memory-fixes/research/findings.md`. |
| agentmemory MCP | External MCP server (`agentmemory` in `~/.pirate/config.json`) | Not part of this repo. It needs `AGENTMEMORY_URL` in its environment; without it the shim silently falls back to a per-process local KV (`~/.agentmemory/standalone.json`) and "memories" stop being shared. The configured wrapper script injects the env for exactly this reason. |
| Session + autocompact | `internal/session` | The conversation itself and its compaction (`autocompact.go`); distinct from the three stores above. |

## Turn-loop invariants

These behaviors are load-bearing and pinned by tests. They look like
implementation details; breaking any of them shipped real bugs. If a change
needs one to bend, change the invariant deliberately — code and tests together —
never silently.

- **The step budget resets when the ADK invocation id changes.** Each user turn
  restarts from the full budget; a resumed human-in-the-loop run reuses the
  paused invocation id on purpose (`internal/agent/step_limit.go:46-50`).
  Tests: `TestStepLimitCallback` (`step_limit_test.go`),
  `TestStepBudgetResetsOnEachUserTurn` (`step_limit_e2e_test.go`).
- **Mid-turn retry has one budget and a tool-traffic guard.** When a transient
  failure hits mid-turn, the turn may run at most `MidTurnAttempts` times in
  total (default `DefaultMidTurnAttempts = 3`, env `PI_MIDTURN_ATTEMPTS`;
  `internal/config/config.go:35`), provider-internal retries count against the
  same budget, and **an attempt that already emitted FunctionCall or
  FunctionResponse parts is never replayed** — a replay would execute tools a
  second time (`internal/tui/agent_loop.go:1220-1257`). Tests:
  `TestRunAgentLoop_Midturn*` in `internal/tui/agent_loop_midturn_retry_test.go`
  (`_NoRetryAfterToolTraffic`, `_MidturnBudgetCoversInnerRetries`, …).
- **The stream idle timeout watches streams only.** Non-streaming calls pass
  through untouched at any timeout — without a chunk cadence a budget would
  degenerate into a whole-call cap and would silently degrade unmanaged callers
  like autocompact and summarize (`internal/provider/stream_idle.go:73-79`).
  Tests: `TestStreamIdle*` in `internal/provider/stream_idle_test.go`.
- **`${VAR}` placeholders in keys survive `Config.Save()`.** A literal
  `${VAR}` stored in config must still be there after a save — expanding or
  stripping it leaks credentials into the file. Test:
  `TestPalace_EmbeddingsAPIKeyStaysPlaceholderThroughSave`
  (`internal/config/palace_config_test.go:68`).
- **`RankBySimilarity` drops foreign-dimension vectors** rather than erroring
  or padding (`internal/palace/embedder.go:208-213`). A dimension change is a
  re-index event, not a scoring concern.
- **A dead turn closes its dialogs.** When the agent run ends while a question
  or approval dialog is open, `handleAgentDone` dismisses it and notices the
  user — a tool waiting for an answer that will never come must not wedge the
  TUI (`internal/tui/agent_loop.go:2264-2285`).
- **One live loop per session slot; every slot owns its session whole.**
  The registry `model.slots` holds N background sessions in insertion order
  (the `/slots` popup's rows); N grows by sequential run+detach cycles, by
  `/new` parkings, and by every attach — which swaps a slot's session with
  the active one and parks the displaced session back into the registry, so
  switching never loses a session. A detach moves the running session WHOLE
  into its slot — agent, transcript, inbox, channel, loop — and the session
  factory (`Config.NewSessionAgent`) opens a fresh, empty active session.
  The loop keeps executing on the agent it started with; a slot's agent is
  frozen while its loop lives, and only an agent with no live loop in ITS
  slot may be rebuilt (`RebuildWithSession`/`RebuildWithModel` target the
  active slot's agent alone — that is why agent/model switches and slash
  commands work while slots run). Ownership moves, it never copies: deltas
  never cross slots, and a detached run keeps writing its own
  `events.jsonl` through the SessionService — durability is the run's, not
  the render's (`internal/tui/background.go`).
- **A run's final writes only to its own slot.** A background final goes
  through `finishSlotRun` — done flags, stashed requests denied, notice +
  attention, the slot's OWN inbox restarted — never through
  `handleAgentDone`, which would clobber the active slot's `m.running`,
  dialogs, plan state and queue mid-turn. Esc-Esc cancels the NEWEST live
  slot's run through its own cancel func and finalizes it synthetically
  (stack discipline); Ctrl+C cancels the active turn and every live slot
  and EMPTIES their inboxes first — without the drain each finalize would
  resurrect the next queued prompt and make quit unreachable. Dismissing a
  slot from the popup drains the same way: dismissal ends a session, it
  does not continue it.
- **Cross-session messages carry a session tag.** Approval and question
  requests are tagged at the per-agent asker closures; the tag is the whole
  route (`slotBySession`: active → dialog, its slot → stash + ⚑ badge). An
  untagged or unknown-tagged request is denied and logged — never routed by
  guessing who is running.
- **Channel readers never change hands — on every channel.** Each run
  channel keeps its single parked reader through detach and attach, armed as
  a `slotEnvelope` naming its channel from the start: classification is by
  CHANNEL against the registry (`slotByChannel`), never by pointer, so a
  canceled run's late events and a second detach over the same channel are
  routed to the run that owns the channel instead of being mistaken for the
  next turn's. Subagent events always re-arm `waitForSubEvent`, their own
  reader — routing, never a second goroutine. Documented exceptions: a
  `/run` cycle cannot be backgrounded (its events bypass the run channel),
  plan sessions refuse detach and attach (plan state is still global), and
  attach refuses while the active session runs — the design's
  park-the-active attach is a later increment.
