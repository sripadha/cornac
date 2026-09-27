# Tools & Workspace — Beginner Walkthrough (Week 2)

A slow, example-driven walkthrough of Week 2, in the same style as
[walking-skeleton.md](walking-skeleton.md). (A shorter design-summary version lives in
[tools-and-workspace.md](tools-and-workspace.md); this file is the teaching version.)

**What changed from Week 1:** Week 1 had one harmless tool (`get_current_time`). Week 2
adds tools that touch your real computer — read/write files, run shell commands, run
Python, search the web. The moment a tool can touch the machine, and the *model* decides
on its own which tool to call with which arguments, you need guardrails. Most of Week 2
is those guardrails.

The layers, bottom-up:

```
Workspace (the fence)            cornac/tools/workspace.py
   ↑ used by
File tools   read/write/list/grep    cornac/tools/builtin/files.py
Code tools   run_bash / run_python   cornac/tools/builtin/shell.py, python_exec.py
Web tools    web_search / web_fetch  cornac/tools/builtin/web.py
Assembler    default_tools()         cornac/tools/builtin/__init__.py
```

---

## 1. The Workspace — a fence around one folder

**The danger:** the model picks tool arguments itself. Nothing stops it from deciding
`read_file("/etc/shadow")` or `run_bash("rm -rf ~")`. So before the file tools, we build
a fence: file/shell tools may only operate inside **one root directory**.

Analogy: *"the kids can play anywhere in the backyard, but may not leave the yard."*
`Workspace` is the fence; `resolve()` is the gate that checks "still inside the yard?"

### `pathlib.Path` — two methods that matter

- `Path("/a/b/../c").resolve()` → `/a/c`  — makes a path absolute and **collapses `..`**.
- `Path("/a/b/c").parents` → `[/a/b, /a, /]` — all ancestor folders.

### The gate: `resolve()`

```python
def resolve(self, path):
    candidate = (self.root / path).resolve()        # join to root, collapse ".."
    if candidate != self.root and self.root not in candidate.parents:
        raise WorkspaceError(f"path {path!r} escapes the workspace root")
    return candidate
```

Two steps: (1) figure out where the path *actually* lands (resolving `..`), (2) reject it
unless it lands at or inside the root.

### The key subtlety: it's about WHERE you land, not whether `..` appears

`..` is not automatically bad. What matters is the final destination. With root
`/home/spv/project`:

| Input | Resolves to | Verdict | Why |
|---|---|---|---|
| `config.txt` | `/home/spv/project/config.txt` | ✅ | inside |
| `data/sales.csv` | `/home/spv/project/data/sales.csv` | ✅ | inside |
| `data/../config.txt` | `/home/spv/project/config.txt` | ✅ | down 1, up 1 — cancels, stays inside |
| `src/utils/../../config.txt` | `/home/spv/project/config.txt` | ✅ | down 2, up 2 — cancels |
| `../../../etc/passwd` | `/etc/passwd` | ❌ | climbs above root |
| `/etc/passwd` | `/etc/passwd` | ❌ | absolute, never inside |
| `data/../../secret.txt` | `/home/spv/secret.txt` | ❌ | down 1, up 2 — net up, escaped |

> **Rule:** trace `..` as "go up one level," treat each named segment as a *literal*
> folder that must exist at that spot, then check the final landing spot. A folder name
> (even one matching the root's name) does **not** teleport you home — only being in the
> right *location* does. When in doubt, resolve and check. This is why the code resolves
> first and *then* checks, instead of scanning the string for `..` (a naive blocklist
> would wrongly reject the harmless cancelling cases and could be tricked anyway).

`WorkspaceError` is a custom exception. When a tool raises it, the registry (Layer 2)
catches it and feeds it back to the model as an error result — so an escape attempt
becomes a recoverable observation, never a crash and never a breach.

---

## 2. File tools — every one delegates to the gate

`read_file`, `write_file`, `list_dir`, `grep`. These are **class-based** tools (subclass
`Tool`) — not `@tool` functions — because each needs to *remember* its Workspace. (Week 1
rule: **state → class; stateless → `@tool` function**.)

Every one has the identical skeleton:

```python
class ReadFile(Tool):
    name = "read_file"
    description = "..."          # the model reads this
    input_schema = {...}
    def __init__(self, workspace):
        self.ws = workspace                          # remember the fence
    def run(self, arguments):
        path = self.ws.resolve(arguments["path"])    # ALWAYS step 1: pass through the gate
        ... do the real work ...
```

The security model is one line, repeated in all four: **the first thing `run()` does is
`self.ws.resolve(...)`.** If the path escapes, it raises before any disk access. Add a
fifth file tool tomorrow — as long as its first line is `resolve()`, it's automatically
sandboxed. The fence is defined once, reused everywhere.

### `read_file` — the template, line by line

```python
def run(self, arguments):
    path = self.ws.resolve(arguments["path"])              # 1. THE GATE
    if not path.is_file():                                 # 2. sanity check
        return f"Error: not a file: {self.ws.relative(path)}"
    data = path.read_text(encoding="utf-8", errors="replace")   # 3. the read
    if len(data) > MAX_BYTES:                              # 4. don't flood context
        return data[:MAX_BYTES] + f"\n... [truncated at {MAX_BYTES} chars]"
    return data
```

1. **The gate.** `arguments["path"]` is the model's string (e.g. `"README.md"`). `resolve()`
   turns it into a safe absolute path inside the sandbox, or raises. Everything below this
   line is guaranteed in-sandbox.
2. **Sanity check.** `path.is_file()` — did the model point at a real file (not a folder or
   a missing path)? If not, return a *readable error string*, not a crash.
3. **The read.** `read_text(errors="replace")` reads the file. `errors="replace"` means "if
   you hit a byte you can't decode as UTF-8, substitute a placeholder instead of raising" —
   robustness against messy real-world files.
4. **Context protection.** `MAX_BYTES = 100_000`. The result goes back into the conversation,
   which costs tokens and can overflow the model's context window. So a huge file is
   truncated with a note. **General agent-design rule: tool output lands in the model's
   context, so tools must never return unbounded data.**

> **Why `resolve()` comes *before* `is_file()` (the order is not arbitrary):** `resolve()`
> does two jobs — it's the security gate *and* it produces the correct absolute `Path`.
> `is_file()` depends on both. If you checked `is_file()` first, (a) you'd be touching the
> filesystem with an *unvalidated* path — statting `is_file()` on `/etc/shadow` reveals
> whether it exists before the fence ever rejects it; and (b) you'd check the wrong
> *location* — the raw relative string `"README.md"` would be resolved against the process's
> working directory, not the workspace root. **Validate first, then act. Never touch the
> filesystem with a path you haven't run through the gate.**

### `write_file` — same skeleton, two differences

Its schema needs **two** required args — `path` *and* `content`. And its `run` writes:

```python
path = self.ws.resolve(arguments["path"])          # same gate, first thing
path.parent.mkdir(parents=True, exist_ok=True)     # NEW: create missing folders
path.write_text(arguments["content"], encoding="utf-8")
return f"Wrote {len(content)} chars to {self.ws.relative(path)}"
```

- `path.parent.mkdir(parents=True, exist_ok=True)` — if the model writes `"notes/todo.txt"`
  but `notes/` doesn't exist, plain `write_text` would fail. This creates it first.
  `parents=True` = create any missing parent folders (like `mkdir -p`); `exist_ok=True` =
  don't complain if it already exists.
- It returns a **confirmation** (`"Wrote 42 chars to notes/todo.txt"`), not the content — the
  model needs to know the write *succeeded*.
- Safe to expose *because* `resolve()` already guaranteed the path is inside the sandbox.

### `list_dir` — the sorting trick

Its schema has **no `required`** — `path` is optional, defaulting to the root.

```python
path = self.ws.resolve(arguments.get("path", "."))          # .get with default "."
if not path.is_dir():
    return f"Error: not a directory: {self.ws.relative(path)}"
entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))   # the trick
return "\n".join(e.name + ("/" if e.is_dir() else "") for e in entries)
```

- `arguments.get("path", ".")` — since `path` is optional, `.get` returns `"."` (the root)
  if the model didn't send one, instead of crashing on a missing key.
- **The sort trick:** `key=lambda p: (p.is_file(), p.name)`. `p.is_file()` is `False` (which
  Python treats as `0`) for directories and `True` (`1`) for files. Sorting by that tuple
  puts **all folders first (0), then all files (1)**, each group alphabetical by name. The
  `lambda` is just an inline "here's how to rank each item" rule.
- The output appends `/` to directory names so the model can tell folders from files at a
  glance: `cornac/`, `docs/`, then `README.md`.

### `grep` — the most involved (but same skeleton)

```python
root = self.ws.resolve(arguments.get("path", "."))     # gate, first thing
regex = re.compile(arguments["pattern"])               # compile the search pattern
hits = []
for file in sorted(root.rglob("*")):                   # walk EVERY file, recursively
    if not file.is_file():
        continue
    try:
        text = file.read_text(encoding="utf-8", errors="strict")
    except (UnicodeDecodeError, OSError):
        continue                                       # skip binary / unreadable files
    for lineno, line in enumerate(text.splitlines(), start=1):   # number lines from 1
        if regex.search(line):                         # does this line match?
            hits.append(f"{self.ws.relative(file)}:{lineno}:{line.strip()}")
            if len("\n".join(hits)) > MAX_BYTES:       # too many matches -> stop early
                hits.append("... [truncated: too many matches]")
                return "\n".join(hits)
return "\n".join(hits) if hits else f"(no matches for {arguments['pattern']!r})"
```

- `re.compile(pattern)` — `re` is Python's regular-expression module (a mini-language for
  "find text matching this shape"). The model supplies a pattern like `def add`.
- `root.rglob("*")` — the **r** is *recursive* glob: visit every file under the folder at any
  depth. (This is why grepping with root `.` dives into `.venv/` — `rglob` visits
  *everything*, so a broad root surfaces library files too.)
- `try/except (UnicodeDecodeError, OSError)` — `errors="strict"` means "raise if this isn't
  valid UTF-8," and we catch that to **skip binary files** (images, `.pyc`) rather than grep
  through a JPEG.
- `enumerate(text.splitlines(), start=1)` — split into lines and number them from 1, so we
  can report `path:line_number:line`.
- `regex.search(line)` — does the line contain the pattern? If yes, record a hit.
- The `MAX_BYTES` guard — same context-protection idea as `read_file`, but for match
  *count*: a pattern matching thousands of lines shouldn't dump megabytes into context.
- The final line returns a clear `(no matches for '...')` instead of an empty string — a
  definite "nothing found" beats ambiguous silence.

### The unifying picture

```
   ┌─ read_file  ─┐
   ├─ write_file ─┤   each one:  run(arguments)
   ├─ list_dir   ─┤      │
   └─ grep       ─┘      ├─► 1. self.ws.resolve(path)   ← THE GATE (same for all 4)
                        │        escape? → WorkspaceError → registry → error result
                        └─► 2. its own operation on the now-safe path:
                                 read_file  → read_text
                                 write_file → mkdir + write_text
                                 list_dir   → iterdir + sort
                                 grep       → rglob + regex search
```

Every file tool is **"pass the path through the gate, then do one file operation."** And
every one shares two habits: **failures become readable strings** (not a file? can't decode?
→ a message, never a crash) and **output is capped** (`MAX_BYTES`) so it never floods the
model's context. That discipline is what makes them safe to hand to an autonomous model.

**Proven live:** all four work inside the sandbox, and all four reject the same escape
attempts identically (`is_error=True, "...escapes the workspace root"`) — without crashing.

---

## 3. Code tools — the fence is NOT enough (why permissions exist)

`run_bash` and `run_python` run **arbitrary code**. Same class shape, same workspace, but a
fundamentally weaker guarantee.

### `subprocess.run` — launch an external program

```python
# run_bash:
subprocess.run(command, shell=True, cwd=self.ws.root, capture_output=True,
               text=True, timeout=self.timeout)

# run_python (the only real difference):
subprocess.run([sys.executable, "-c", code], cwd=self.ws.root, ...)
```

- `shell=True` (bash) vs `[sys.executable, "-c", code]` (Python) — "run this string through
  bash" vs "through Python." `sys.executable` is cornac's *own* venv python, so snippets
  see the same installed libraries (which is why a model's `import pandas` failed — pandas
  isn't in this venv).
- `cwd=self.ws.root` — the command *starts* inside the sandbox.
- `capture_output=True, text=True` — grab the program's stdout/stderr (instead of letting
  it print to *our* terminal) and hand it back as strings, not raw bytes — so the output can
  become a `ToolResult` fed to the model.
- `timeout` — kill a hanging command; return a readable timeout error instead of freezing.
- subprocess (not in-process `exec`) — a crash or infinite loop is contained to the child;
  cornac keeps running.

### The gap, demonstrated

The Workspace fully contains *file* tools, but **cannot** contain code tools, because
`cwd` only sets where a command *starts* — arbitrary code can use absolute paths or `cd`:

```python
run_bash("head -1 /etc/hostname")   # -> "SripV"  ← read a system file; fence did nothing
run_bash("cd / && rm -rf tmp")      # -> cd OUT of the sandbox, then act
```

This was run live: `run_bash("head -1 /etc/hostname")` printed the machine's hostname —
straight out of the sandbox. A `read_file` could *never* do this (its path goes through
`resolve()`); `run_bash` does it trivially.

The code **deliberately does not** scan the command for "bad" words like `rm`/`sudo`. A
blocklist gives false confidence — `rm` can be rewritten as `python -c "shutil.rmtree(...)"`.
An evadable filter that *feels* safe is worse than none.

> **So what actually protects against dangerous code?** Not analyzing the command —
> **asking permission before running it at all.** That gate is the **Week 3 permission
> system** (`run_bash`/`run_python` default to "ask": *"Allow run_bash('rm -rf /')? [y/N]"*).
> This is the core architectural reason permissions exist.

| Threat | File tools | Code tools |
|---|---|---|
| read/write outside sandbox | ✅ blocked by `resolve()` | ❌ NOT blocked (proven live) |
| hang forever | n/a | ✅ timeout |
| crash the agent | ✅ registry catches | ✅ subprocess isolates |
| run dangerous code at all | n/a | ⏳ needs Week 3 permissions |

---

## 4. Web tools — back to stateless `@tool` functions

`web_search`, `web_fetch` touch the **network, not the filesystem** — nothing to remember,
so they're plain `@tool` functions (like `get_current_time`), not classes. Same `Tool`
contract underneath; the registry treats them identically to the class-based tools.

- **web_search** — DuckDuckGo via the `ddgs` library (free, **no API key**, so readers can
  reproduce the benchmark). Lazy-imports `DDGS`; formats results as numbered title/url/snippet.
- **web_fetch** — `httpx.get(url, follow_redirects=True, headers={...})`, then a crude
  3-pass HTML→text cleanup (drop `<script>`/`<style>`, strip tags, collapse whitespace).
  "Good enough" beats pulling in a heavy readability dependency for a learning project.

Both wrap their network call in `try/except` and return a *readable* error string. The
registry would catch a raw exception anyway, but a hand-crafted `"Error: fetch failed
(TimeoutError: ...)"` gives the model something cleaner to act on. (Network calls fail
constantly — timeouts, rate limits, dead links — so this matters.)

---

## 5. The assembler — `default_tools(workspace)`

A convenience so demos/benchmark don't wire up 9 tools by hand:

```python
def default_tools(workspace="."):
    ws = workspace if isinstance(workspace, Workspace) else Workspace(workspace)
    return [
        ReadFile(ws), WriteFile(ws), ListDir(ws), Grep(ws),   # class-based: NEED the workspace
        RunBash(ws), RunPython(ws),                            # class-based: NEED the workspace
        web_search, web_fetch, get_current_time,               # function-based: no workspace
    ]
```

The list *visually* shows the two tool shapes: tools called with `(ws)` are class-based
(being instantiated with the fence); tools passed bare are already-built function-tools.
`isinstance` lets you pass either a `Workspace` or a plain path string.

Usage:

```python
from cornac.tools.builtin import default_tools
from cornac.tools.workspace import Workspace
registry = ToolRegistry(default_tools(Workspace("/path/to/project")))
```

---

## The payoff: multi-step chaining + self-correction (live, on Qwen 2.5 7B)

`examples/multistep_trace.py` — a sandbox with `sales.csv`, asked for total revenue:

```
iter 1: run_python(import pandas ...)  -> ModuleNotFoundError: No module named 'pandas'
iter 2: run_python(import csv ...)     -> 89.0
iter 3: "The total revenue ... is 89.0."   (no tool call -> loop returns)
```

The first approach **failed** (no pandas). Because the registry returned the error as data
(not a crash), the model **saw** it, **rewrote** its code to use the stdlib `csv` module,
and got the right answer. That try → observe failure → adapt loop is the entire difference
between a **static workflow** (a human hard-codes "use pandas"; crashes here) and a
**dynamic agent** (routes around the failure on its own). It's the behavior the whole
project exists to demonstrate — here on a weak local model, three iterations deep.

## The four big ideas of Week 2

1. **Two shapes of tool** — class (stateful, carries a workspace) vs `@tool` function
   (stateless). `default_tools` shows both side by side.
2. **The Workspace fence** fully secures *file* tools (every path through one gate) but
   **cannot** contain *code* tools (proven: `run_bash` read `/etc/hostname`) — which is
   exactly *why* permissions exist.
3. **Defense in depth** — `resolve()` gate + subprocess isolation + timeouts + context
   caps + (coming) permissions. No single guard does everything.
4. **Errors as observations** — every tool turns failures (escapes, network errors,
   crashes) into readable strings the model can act on, never a crash.

## Try it

```bash
python examples/multistep_trace.py ollama       # local Qwen
pytest -q                                        # 16 tests incl. sandbox-escape
```
