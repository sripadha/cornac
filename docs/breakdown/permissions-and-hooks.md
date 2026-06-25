# Permissions & Hooks — Walkthrough (Week 3)

Week 2 ended with a problem you saw live: `run_bash("head /etc/hostname")` walked
straight out of the sandbox, because the Workspace fence can't contain arbitrary code.
And both trace demos had to *copy* the agent loop to observe it. Week 3 fixes both:

- **Permissions** — decide, *before* a tool runs, whether to ALLOW / DENY / ASK. The
  real gate for `run_bash`/`run_python` that the fence couldn't provide.
- **Hooks** — fire named lifecycle events so observers can watch the loop *without*
  copying it.

Both slot into the **exact same loop shape** from Week 1 — the loop's logic doesn't
change, it just gains two well-defined hook/permission points.

```
permissions/policy.py   the decision engine (ALLOW / DENY / ASK)
permissions/prompt.py   the human prompt when the answer is ASK
hooks/bus.py            publish/subscribe event bus
core/agent.py           the loop, now wired to both (optional)
```

---

## 1. The policy engine — `permissions/policy.py`

### Three decisions

```python
class Decision(str, Enum):
    ALLOW = "allow"   # run it
    DENY  = "deny"    # refuse; model gets an error result and can adapt
    ASK   = "ask"     # undecided here -> the agent asks a human (prompt.py)
```

`policy.py` is **pure decision logic** — no input(), no printing. Given a `ToolCall`, it
returns one of these three. (Keeping "decide" separate from "ask the human" is what lets
you swap the human prompt for a web UI / Slack bot / automated test later.)

### Rules

Rules are a small dict (or YAML). Each tool maps to either a plain string decision, or —
for command-bearing tools — a dict with deny/allow substring lists + a default:

```python
{
  "default": "ask",                      # tools not listed -> ask
  "tools": {
    "read_file": "allow",
    "write_file": "ask",
    "run_bash": {
      "deny":  ["rm -rf", "sudo"],       # checked FIRST — deny wins
      "allow": ["pytest", "ls"],
      "default": "ask",                  # neither matched -> ask
    },
  },
}
```

### `check()` — the decision, step by step

```python
def check(self, call):
    if call.name in self._session:       # 1. "always/never" from a previous prompt wins
        return self._session[call.name]
    if call.name not in self._tools:     # 2. no rule -> global default
        return self._default
    rule = self._tools[call.name]
    if isinstance(rule, str):            # 3. simple "allow"/"deny"/"ask"
        return Decision(rule)
    text = self._argument_text(call)     # 4. rich form: match deny/allow vs arg text
    for p in rule.get("deny", []):       #    deny first — it wins
        if p in text: return Decision.DENY
    for p in rule.get("allow", []):
        if p in text: return Decision.ALLOW
    return Decision(rule.get("default", "ask"))
```

Key points:
- **deny is checked before allow** — a destructive pattern can't be re-enabled by also
  matching an allow pattern.
- **`_argument_text` joins *all* string argument values**, so a deny pattern can't be
  hidden in an unexpected argument.
- **session overrides** (`remember()`) come from "always"/"never" answers at the prompt
  and win over the configured rules for the rest of the run.

`Policy.from_yaml(path)` loads the same shape from a file (see
`examples/cornac.policy.yaml`).

> **Why substring matching, not a real parser?** It's transparent and good enough for a
> learning harness. It is **not** bulletproof — `r''m -rf` style tricks exist. That's why
> the *default* for unmatched commands is `ask`, not `allow`: the human is the backstop.

---

## 2. The human prompt — `permissions/prompt.py`

When the policy returns ASK, the agent calls an **approver**. The default is `cli_ask`:

```
cornac wants to run a tool:
  run_python(code='...')
Allow? [y]es / [n]o / [a]lways / nev[e]r:
```

- `y` allow once · `n` deny once · `a` allow + remember for the session ·
  `e` deny + remember.
- "always"/"never" call `policy.remember(tool, decision)` so you're not re-asked.
- EOF / closed stdin (piped input runs out) → safe default: **deny**.

`cli_ask` is just a function `(call, policy) -> Decision`. Any approver with that shape
works — a web prompt, a Slack approval, or a test stub. The agent only needs "given a
call, return a Decision."

---

## 3. The hook bus — `hooks/bus.py`

The Week 1-2 trace demos *copied* the loop to observe it. A hook bus removes that need.

```python
class HookBus:
    def on(self, event, callback): ...    # subscribe
    def fire(self, event, *args): ...     # publish: run every callback for that event
```

`fire` **swallows exceptions** from callbacks — a misbehaving observer must never crash
the agent (the subject). This is the classic publish/subscribe (observer) pattern, and
it's the seam real harnesses expose for logging, metrics, and tracing.

Events the agent fires:

| Event | When | Args |
|---|---|---|
| `on_user_message` | user input added | the Message |
| `on_assistant_message` | model replied | the Message |
| `pre_tool_use` | before handling a tool call | the ToolCall |
| `on_permission_decision` | after policy/approver decides | call, Decision |
| `post_tool_use` | after the tool ran (or was denied) | call, result |
| `on_stop` | loop is ending | final text (or None) |

---

## 4. Wiring it into the loop — `core/agent.py`

The loop's shape is unchanged. The new constructor args are all optional — with none of
them, `run()` behaves exactly as in Week 1.

```python
for call in response.tool_calls:
    self.hooks.fire("pre_tool_use", call)
    result = self._handle_call(call)          # <- permission gate lives here
    self.hooks.fire("post_tool_use", call, result)
    self.messages.append(Message.from_tool_result(result))
```

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

Two safe defaults worth noting:
- **No policy → ALLOW everything** (Week 1 behavior preserved).
- **ASK but no approver → DENY** (never run an unapproved tool blindly).

And the crucial design property: a **DENY becomes an error `ToolResult`**, not a crash.
The model sees "Permission denied" and adapts — exactly like it adapts to a tool error.

---

## Proven live (Qwen 2.5 7B)

**Hooks** (`examples/hook_trace.py`) — calls the *real* `agent.run()`, no copied loop:

```
[user] 'What is the total revenue ... in sales.csv?'
[assistant] wants tools: run_python(import pandas ...)
   ↳ policy says ALLOW for run_python
[assistant] wants tools: run_python(import csv ...)      # self-corrected pandas->csv
   ↳ policy says ALLOW for run_python
[assistant] 'The total revenue ... is 61.0'
[stop] final answer ready
```

**Permissions deny path** (`examples/permissions_demo.py`, answering `n`):

```
cornac wants to run a tool: run_python(import pandas ...)
Allow? ...: n        -> [decision: DENY]
   (model tries a different run_python)
Allow? ...: (no input — denying)  -> [decision: DENY]
=> "Let's manually calculate the total revenue instead..."   # model routes around the denial
```

When code execution was refused twice, the model **fell back to computing the answer by
hand** — denials are just observations the model adapts to, the same principle as tool
errors.

## The big ideas of Week 3

1. **Decide, then ask.** `policy.check()` is pure logic (ALLOW/DENY/ASK); the human
   prompt is a separate, swappable approver. This separation is why the gate works for
   any front-end.
2. **The permission gate is the real control for arbitrary-code tools** — the thing the
   Workspace fence couldn't provide.
3. **A denial is an observation, not a crash** — it flows back as an error result and the
   model adapts.
4. **Hooks observe without modifying** — publish/subscribe lets loggers/tracers/demos
   watch the loop without copying it; a broken hook can't crash the agent.

## Try it

```bash
python examples/hook_trace.py ollama            # watch the loop via hooks
python examples/permissions_demo.py ollama      # interactive y/n/a/e prompt
echo a | python examples/permissions_demo.py ollama   # non-interactive (always-allow)
pytest -q                                       # 25 tests
```
