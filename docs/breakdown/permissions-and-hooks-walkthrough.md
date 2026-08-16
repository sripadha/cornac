# Permissions & Hooks — Beginner Walkthrough (Week 3)

The slow, example-driven walkthrough of Week 3, in the same style as
[walking-skeleton.md](walking-skeleton.md) and
[tools-and-workspace-walkthrough.md](tools-and-workspace-walkthrough.md).
(A shorter design-summary version lives in [permissions-and-hooks.md](permissions-and-hooks.md).)

**The problem Week 3 solves — you proved it yourself in Week 2:**
`run_bash("head -1 /etc/hostname")` walked straight out of the sandbox. The Workspace
fence secures *file* tools (every path goes through `resolve()`), but it **cannot**
contain a tool that runs arbitrary code. So we need a different kind of guard: not
*"where can files go?"* but **"should this tool run at all?"** — and when unsure,
**ask a human**.

Analogy: think of a bank. The Workspace is the **vault door** (controls *where* things
are stored). But for a *withdrawal*, the vault door doesn't help — you need a **teller**
who checks the rulebook: "is this allowed? do I need a manager's approval?" The `Policy`
is that rulebook. The CLI prompt is the manager being called over.

A second, smaller problem: in Weeks 1–2, the only way to *watch* the loop was to **copy
it** into demo scripts (`walking_skeleton_trace.py` and `multistep_trace.py` both
re-implement `run()`). To observe behavior we had to duplicate it — a design smell.
**Hooks** fix that.

The four pieces, in the order data flows:

```
permissions/policy.py    the rulebook   — pure decision: ALLOW / DENY / ASK
permissions/prompt.py    the human      — what happens on ASK
hooks/bus.py             the observers  — watch the loop without copying it
core/agent.py            the wiring     — both slotted in WITHOUT changing the loop's shape
```

---

## Part 1 — `policy.py`: the rulebook

### First, what this file deliberately is NOT

`policy.py` contains **no `input()`, no `print()`, no tool execution**. It is *pure
decision logic*: a `ToolCall` goes in, a verdict comes out. Asking the human lives in a
*separate* file (`prompt.py`).

Why split them? Because "decide the verdict" and "ask a person" are different jobs:

- Pure logic is **trivially testable** — no faking keyboard input in tests.
- The asker is **swappable** — a CLI prompt today; a web dialog, a Slack approval, or a
  test stub tomorrow. All of them can reuse the same unchanged rulebook.

This is the same separation instinct as the Provider seam: isolate the part that talks
to the outside world.

### New Python concept: `Enum`

```python
class Decision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"
```

An **Enum** ("enumeration") is a type with a fixed, named set of values. Instead of
passing raw strings like `"allow"` around (typo `"alow"` once and you get a silent bug),
you use `Decision.ALLOW` — guaranteed to be one of exactly these three.

The `(str, Enum)` double inheritance means each value *is also a string*:
`Decision.ALLOW == "allow"` is `True`. That's deliberate — rules are written in YAML as
plain strings, and `Decision("ask")` converts a string straight into the enum member.
Safety of an enum, convenience of a string.

What each verdict *means to the agent*:

| Verdict | Meaning |
|---|---|
| `ALLOW` | run the tool, no questions |
| `DENY` | refuse — the model gets an error result ("Permission denied") and can adapt, exactly like it adapts to a tool error |
| `ASK` | *the rulebook can't decide* — escalate to a human |

That three-way split is the heart of the design: ALLOW/DENY are decisions; ASK is an
escalation.

### The rules format — two forms

```python
DEFAULT_RULES = {
    "default": "ask",          # any tool NOT listed below -> ask
    "tools": {
        "read_file": "allow",  # SIMPLE form: a plain verdict string
        "write_file": "ask",
        "run_bash": {          # RICH form: dict with deny/allow substring lists
            "deny": ["rm -rf", "sudo", "mkfs", ":(){", "dd if="],
            "default": "ask",
        },
        "run_python": "ask",
    },
}
```

- **Simple form** — a verdict string. Read-only tools (`read_file`, `list_dir`, `grep`,
  `web_search`…) are safe → auto-allow. Mutating/code tools → ask.
- **Rich form** — for command-bearing tools like `run_bash`: deny the obviously
  destructive outright, ask about the rest. The deny list is the classic
  "destroy your system" collection (`:(){ ` starts a fork bomb; `dd if=` can wipe a disk).
- **`"default": "ask"` at the top** — an unlisted tool falls back to ask. Default to
  caution, always.

`Policy.from_yaml(path)` loads the same shape from a YAML file
(see [examples/cornac.policy.yaml](../../examples/cornac.policy.yaml)) so users can
configure permissions without touching code. And `remember(tool, decision)` records an
"always allow / never" answer in `self._session` — the "don't ask me again this session"
memory.

### `check()` — a four-step priority ladder

This is the whole point of the file. First match wins, top to bottom:

```python
def check(self, call):
    # 1. Did the human already say "always/never" for this tool? That wins.
    if call.name in self._session:
        return self._session[call.name]

    # 2. No rule at all for this tool? -> cautious global default (usually ASK).
    if call.name not in self._tools:
        return self._default

    rule = self._tools[call.name]

    # 3. Simple rule ("allow"/"deny"/"ask")? Return it.
    if isinstance(rule, str):
        return Decision(rule)

    # 4. Rich rule: match deny/allow lists against the command text.
    text = self._argument_text(call)
    for pattern in rule.get("deny", []):     # deny checked FIRST
        if pattern in text:
            return Decision.DENY
    for pattern in rule.get("allow", []):
        if pattern in text:
            return Decision.ALLOW
    return Decision(rule.get("default", "ask"))
```

Why the ordering:

1. **Session override first** — your explicit human choice has top priority.
2. **Unknown tool → global default** — nothing sneaks through by being unlisted.
3. **Simple rules** — most tools land here.
4. **Rich rules, deny-before-allow** — the security-critical detail, next.

### Why deny MUST be checked before allow

Consider `run_bash("sudo pytest")`. It matches **both** lists: `"sudo"` (deny) and
`"pytest"` (allow). If allow were checked first, a dangerous command could sneak through
by *also containing* an allowed word. Deny-first means **deny always wins** — you can
never re-enable a forbidden pattern by tacking on an allowed one. Standard security
principle: *denials take precedence.*

Proven live (all four `check()` branches, real output):

```
STEP 3 — simple rules:
  check(read_file )  args={'path': 'x.txt'}          -> ALLOW
  check(write_file)  args={'path': 'x', ...}         -> ASK
  check(run_python)  args={'code': 'print(1)'}       -> ASK

STEP 2 — a tool with NO rule -> global default:
  check(some_unknown_tool)                           -> ASK

STEP 4 — run_bash rich rule (deny checked first):
  check(run_bash)  args={'command': 'pytest tests/'} -> ASK    (matches neither list)
  check(run_bash)  args={'command': 'rm -rf /'}      -> DENY
  check(run_bash)  args={'command': 'sudo pytest'}   -> DENY   <- matches BOTH; deny wins
  check(run_bash)  args={'command': 'ls -la'}        -> ASK

STEP 1 — session override wins:
  before remember: ASK
  pol.remember("run_python", Decision.ALLOW)
  after  remember: ALLOW
```

### `_argument_text` — match against ALL the arguments

```python
@staticmethod
def _argument_text(call):
    return " ".join(str(v) for v in call.arguments.values())
```

The text the deny/allow patterns scan is **every** string argument joined — not just a
`command` field. If dangerous content hid in a second argument, checking only the first
would miss it. Joining everything means the deny list sees the whole picture.
(`@staticmethod` = a helper that doesn't need `self`; it only reads the call.)

### The honest limitation

This is **substring matching**, not command parsing. `"rm -rf" in text` just checks
characters — `rm   -rf` (extra spaces) slips past. That's *exactly why* the default for
unmatched commands is `ask`, not `allow`: the string filter catches the obvious;
**the human prompt is the real backstop** for anything clever. We never pretend the
filter is complete security — same honesty as the Week 2 `run_bash` docstring.

> **One-line summary of `policy.py`:** the rulebook that turns a tool call into
> ALLOW / DENY / ASK, checked in priority order — your "always" choices win, then
> per-tool rules, then a cautious default — and deny patterns always beat allow patterns.

---

## Part 2 — `prompt.py`: the human side of ASK

When the rulebook says ASK, the agent calls an **approver**. The default approver is
`cli_ask`, which pauses right in the terminal:

```
cornac wants to run a tool:
  run_python(code="import pandas as pd ...")
Allow? [y]es / [n]o / [a]lways / nev[e]r:
```

| Answer | Effect |
|---|---|
| `y` | allow **this once** |
| `n` (or just Enter) | deny **this once** — the model gets an error result and adapts |
| `a` | allow, **and** `policy.remember(tool, ALLOW)` → never asked again this session |
| `e` | deny, **and** `policy.remember(tool, DENY)` → auto-denied for the session |
| EOF (piped stdin ran out) | safe default: **deny** — never run unapproved code because input dried up |

The "a"/"e" answers are how Step 1 of the priority ladder gets populated: the prompt
writes into the policy's session memory, and future `check()` calls short-circuit there.

The crucial design point: `cli_ask` is *just a function* with the shape
`(call, policy) -> Decision`. The agent doesn't know or care that it's a CLI. Anything
with that shape is a valid approver — a web dialog, a Slack bot, an auto-approving test
stub. That's the payoff of keeping the rulebook I/O-free: the *decision* logic never
changes when the *asking* mechanism does.

---

## Part 3 — `hooks/bus.py`: observe without copying

### The problem it solves

Both Week 1–2 trace demos had to re-implement the agent loop just to print what was
happening. Duplicating the thing you want to observe is fragile (two copies drift apart)
and wrong in principle. The fix is the classic **observer / publish-subscribe pattern**:

- Interested parties **subscribe** a callback to a named event: `bus.on("pre_tool_use", fn)`
- The agent **publishes** at fixed points in its loop: `bus.fire("pre_tool_use", call)`
- Every subscribed callback runs. The loop neither knows nor cares who's listening.

Newspaper analogy: the agent is the publisher; hooks are subscribers. The paper doesn't
change what it prints based on who subscribes — it just delivers to whoever signed up.

### The whole implementation

```python
class HookBus:
    def __init__(self):
        self._hooks: dict[str, list[Callable]] = defaultdict(list)

    def on(self, event, callback):
        self._hooks[event].append(callback)

    def fire(self, event, *args, **kwargs):
        for callback in self._hooks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception:
                pass    # an observer must never crash the subject
```

Two things to understand:

- **`defaultdict(list)`** — a dict that auto-creates an empty list for any key you touch
  the first time. So `self._hooks["new_event"].append(cb)` just works without checking
  "does this key exist yet?" first.
- **`fire` swallows callback exceptions.** A hook is an *observer*; its failure is not
  the agent's failure. If your logging hook has a bug, the agent must not die mid-task.
  (Tested: a hook that raises `1/0` on every call — the agent still completes.)

### The events the agent fires

| Event | When | Arguments |
|---|---|---|
| `on_user_message` | user input appended | the Message |
| `on_assistant_message` | model replied | the Message |
| `pre_tool_use` | before a tool call is handled | the ToolCall |
| `on_permission_decision` | after policy/approver decided | call, Decision |
| `post_tool_use` | after the tool ran (or was refused) | call, result |
| `on_stop` | the loop is ending | final text (or None) |

---

## Part 4 — wiring it into the loop (`core/agent.py`)

The promise from Week 1's docstring — *"permissions, hooks, sub-agents slot into this
loop at well-defined points without changing its shape"* — comes due here. Compare:

```python
# Week 1:                                  # Week 3:
for call in response.tool_calls:           for call in response.tool_calls:
    result = registry.execute(call)            hooks.fire("pre_tool_use", call)
    messages.append(from_tool_result(result))  result = self._handle_call(call)      # <- gate
                                               hooks.fire("post_tool_use", call, result)
                                               messages.append(from_tool_result(result))
```

Same shape. The permission gate lives in one new method:

```python
def _handle_call(self, call):
    decision = self.policy.check(call) if self.policy else Decision.ALLOW
    if decision == Decision.ASK:
        decision = self.approver(call, self.policy) if self.approver else Decision.DENY
    self.hooks.fire("on_permission_decision", call, decision)
    if decision == Decision.DENY:
        return ToolResult(..., content="Permission denied: ...", is_error=True)
    return self.registry.execute(call)
```

Read it as four moves: **consult the rulebook → escalate ASK to the approver → announce
the decision → refuse or execute.** Two safe defaults are baked in:

- **No policy at all → ALLOW everything.** Week 1 behavior, fully preserved (all
  constructor args optional — old code keeps working).
- **ASK but no approver wired → DENY.** Never run an unapproved tool just because
  nobody was there to ask.

And the property that ties Week 3 back to everything you've learned: **a DENY becomes an
error `ToolResult`, not a crash.** The model reads "Permission denied" in its next turn
and adapts — the same *errors-are-observations* principle as the pandas failure and the
sandbox escapes.

### Proven live (Qwen 2.5 7B)

**Hooks watching the *real* `run()`** (`examples/hook_trace.py` — no copied loop):

```
[user] 'What is the total revenue ... in sales.csv?'
[assistant] wants tools: run_python(import pandas ...)
   ↳ policy says ALLOW for run_python
[assistant] wants tools: run_python(import csv ...)      <- self-corrected pandas→csv
   ↳ policy says ALLOW for run_python
[assistant] 'The total revenue ... is 61.0'
[stop] final answer ready
```

**The deny path** (`examples/permissions_demo.py`, answering `n`):

```
cornac wants to run a tool: run_python(import pandas ...)
Allow? ...: n                        -> [decision: DENY]
   (model tries a DIFFERENT run_python approach)
Allow? ...: (no input — denying)     -> [decision: DENY]
=> "Let's manually calculate the total revenue instead..."
```

Denied twice, the model **gave up on code execution and computed the answer by hand**
from the CSV text it had already read. A permission denial is just another observation
it routes around — the harness stays in control, the model stays useful.

---

## The big ideas of Week 3

1. **Decide, then ask — separately.** The rulebook (`check()`) is pure, testable logic;
   the approver is a swappable function. Neither changes when the other does.
2. **The permission gate is the real control for arbitrary-code tools** — the thing the
   Workspace fence structurally couldn't provide.
3. **Deny beats allow, always.** You can't smuggle a forbidden command past the filter
   by adding an allowed word. And the substring filter is honest about its limits: the
   human prompt is the backstop, which is why unmatched commands default to ASK.
4. **A denial is an observation, not a crash.** It flows back as an error result; the
   model adapts — same principle as tool errors, now applied to *policy*.
5. **Hooks observe without modifying.** Publish/subscribe lets loggers and demos watch
   the loop without copying it, and a broken hook can never take the agent down.

## Check yourself

**Q: Why does `policy.py` contain no `input()` or `print()`?**
*Answer:* because "decide" and "ask" are different jobs. Pure decision logic is testable
without faking a keyboard, and the asker is swappable (CLI today, web/Slack/test-stub
tomorrow) without touching the rules. The agent only needs *something* of shape
`(call, policy) -> Decision`.

**Q: `run_bash("sudo ls")` — what verdict, and why?**
*Answer:* DENY. `"sudo"` matches the deny list, and deny is checked before allow, so the
`"ls"` allow-pattern never gets a say.

## Try it

```bash
python examples/hook_trace.py ollama                  # watch the real loop via hooks
python examples/permissions_demo.py ollama            # interactive [y/n/a/e] prompt
echo a | python examples/permissions_demo.py ollama   # non-interactive (always-allow)
pytest -q                                             # 25 tests
```
