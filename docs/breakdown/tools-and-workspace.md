# Tools & Workspace — Code Breakdown (Week 2)

Week 1 gave the agent one toy tool (`get_current_time`). Week 2 builds the real tool
surface — 8 tools — and adds a **sandbox** so file/shell tools can't roam your whole
disk. The headline result: with a real toolset, the agent starts **chaining tools
across multiple loop iterations and recovering from its own mistakes** — the behavior
that separates a dynamic agent from a hard-coded workflow.

## The 8 tools

| Tool | Kind | Needs sandbox? | Notes |
|---|---|---|---|
| `read_file` | file | yes | read a text file |
| `write_file` | file | yes | write/overwrite, creates parent dirs |
| `list_dir` | file | yes | list a directory |
| `grep` | file | yes | regex search across files |
| `run_bash` | shell | yes (cwd) | run a command; **gated by permissions later** |
| `run_python` | python | yes (cwd) | run a snippet; **gated by permissions later** |
| `web_search` | web | no | DuckDuckGo, no API key |
| `web_fetch` | web | no | fetch a URL → readable text |

(`get_current_time` from Week 1 is still there, so `default_tools()` returns 9.)

## Two shapes of tool: function vs class

Week 1's `@tool` decorator turns a *plain function* into a Tool. That's perfect for
stateless tools (`get_current_time`, `web_search`, `web_fetch`) — nothing to remember
between calls.

But the file/shell tools need to know **which directory they're confined to**. A plain
function can't carry that. So those are **class-based tools** — they subclass `Tool`
directly and take a `Workspace` in their constructor:

```python
class ReadFile(Tool):
    name = "read_file"
    description = "..."
    input_schema = {...}
    def __init__(self, workspace: Workspace):
        self.ws = workspace          # remembers the sandbox
    def run(self, arguments: dict) -> str:
        path = self.ws.resolve(arguments["path"])   # validate against the sandbox
        ...
```

Both shapes satisfy the *same* `Tool` contract from Layer 2 (`name`, `description`,
`input_schema`, `run`), so the registry treats them identically. The decorator was just
a shortcut for the common case; when a tool needs to carry state, you write the class.

> **Takeaway:** `@tool` (function) and subclassing `Tool` (class) are two ways to make
> the same kind of object. Use the function form for stateless tools, the class form
> when the tool needs to remember something (like its workspace).

## The Workspace sandbox

An agent picks its own file paths and shell commands. Unconstrained, it could
`read_file("/etc/shadow")` or `run_bash("rm -rf ~")`. The `Workspace` pins every
file/shell tool to one root directory.

The heart of it is `resolve()`:

```python
def resolve(self, path):
    candidate = (self.root / path).resolve()        # join to root, collapse ".."
    if candidate != self.root and self.root not in candidate.parents:
        raise WorkspaceError(f"path {path!r} escapes the workspace root")
    return candidate
```

How it blocks an escape: `Path.resolve()` collapses `..` segments into a real absolute
path. So `"../../etc/passwd"` resolves to `/etc/passwd`, and the check "is `root` an
ancestor of this?" fails → `WorkspaceError`. Absolute paths like `/etc/passwd` are
caught the same way.

And here's where Layer 2 pays off again: that `WorkspaceError` is raised *inside*
`tool.run()`, so the **registry catches it** and returns it as an error `ToolResult`.
The model sees `"path '../../etc/passwd' escapes the workspace root"` and can correct —
the escape attempt becomes a recoverable observation, not a crash and not a breach.

```python
# tested behavior:
reg.execute(ToolCall(name="read_file", arguments={"path": "../../etc/passwd"}))
# -> ToolResult(is_error=True, content="...escapes the workspace root...")
```

> **Honest scope:** the Workspace is a *structural guardrail*, not a security sandbox.
> `run_bash`/`run_python` still execute real code that can read what your user can read.
> The real gate is the Week 3 permission system (these tools default to "ask"); the
> Workspace + timeouts + pinned cwd are defense-in-depth underneath it.

## `default_tools(workspace)`

A convenience that wires all the tools to a workspace so demos/benchmarks don't repeat
the assembly:

```python
from cornac.tools.builtin import default_tools
from cornac.tools.workspace import Workspace

registry = ToolRegistry(default_tools(Workspace("/path/to/project")))
```

File/shell/python tools get confined to that root; web + clock tools are added as-is.

## The payoff: multi-step chaining + self-correction (live)

`examples/multistep_trace.py` sets up a sandbox with a `sales.csv` and asks Qwen 2.5 7B
to compute total revenue. What happened, unedited:

```
iteration 1:
  CALLS run_python(import pandas as pd; df['units'].mul(df['price']).sum())
  tool→ ModuleNotFoundError: No module named 'pandas'
iteration 2:
  CALLS run_python(import csv; ... DictReader ... sum)
  tool→ 89.0
iteration 3:
  "The total revenue ... is 89.0."   ← no tool call -> loop returns
```

Read iteration 2 carefully. The model's first approach **failed** (pandas isn't
installed). Because the registry returned the error as data (not a crash), the model
**saw** it, **rewrote** its code to use the standard-library `csv` module instead, and
got the right answer. (Check: 10·2.50 + 4·9.00 + 7·4.00 = 89.0 ✓.)

That try → observe failure → adapt loop is the entire difference between:

- a **static workflow** (Stage 4): a human hard-codes "use pandas" — it would crash here.
- a **dynamic agent** (Stage 5): the model routes around the failure on its own.

This is the behavior the whole project exists to demonstrate, and here it is on a weak
local model, three iterations deep.

## What's still missing (next weeks)

- `run_bash`/`run_python` run unguarded. **Week 3** adds permissions (ask/allow/deny).
- We can observe the loop only by copying it into demo scripts. **Week 3** adds a hook
  system so we can watch/log the loop without copying it.
- Some tasks would benefit from delegation. **Week 4** adds sub-agents.

## Try it

```bash
python examples/multistep_trace.py ollama       # local Qwen
python examples/multistep_trace.py anthropic    # Claude (needs ANTHROPIC_API_KEY)
pytest -q                                       # 16 tests incl. sandbox-escape
```
