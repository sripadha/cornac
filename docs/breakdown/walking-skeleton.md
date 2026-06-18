# Walking Skeleton — Code Breakdown (Week 1)

A slow, beginner-friendly walkthrough of cornac's Week 1 code: the four layers that
make up the agent loop, then an end-to-end trace of one real request through all of
them.

If you remember nothing else: **an agent is `ask the model → did it want a tool? →
if yes, run it and ask again → if no, return the answer`.** Everything below is the
machinery that makes those four steps possible without the loop knowing any messy
details.

The layers depend on each other bottom-up:

```
Layer 4  Agent loop        (orchestrates everything)        cornac/core/agent.py
            ↑ depends on
Layer 3  Provider          (talks to the LLM)               cornac/providers/*.py
Layer 2  Tools + Registry  (what the model can DO)          cornac/tools/*.py
            ↑ both depend on
Layer 1  Messages          (the shared vocabulary)          cornac/core/messages.py
```

---

## Layer 1 — `messages.py`: the shared vocabulary

**The one problem it solves:** Anthropic's API and Ollama's API describe the *same
conversation* with *different JSON shapes*. If the agent loop spoke one provider's
shape, swapping providers would mean rewriting the loop. So cornac defines its own
**neutral** shape that everything inside speaks; providers translate at the edges.

A conversation is just a **list of turns** — like any chat:

```
Turn 1 (you):   "What time is it in Tokyo?"
Turn 2 (me):    "Let me check..."   (I go look at a clock)
Turn 3 (clock): "2:30 PM"
Turn 4 (me):    "It's 2:30 PM in Tokyo."
```

Three classes model the three kinds of thing that flow through that list:

| Real-world thing | Class |
|---|---|
| A turn in the conversation | `Message` |
| The model saying "go run a tool for me" | `ToolCall` |
| The answer a tool gives back | `ToolResult` |

### Python concept: `@dataclass`

`@dataclass` auto-writes the boilerplate `__init__` for a data-holding class. Instead of:

```python
class ToolCall:
    def __init__(self, id, name, arguments):
        self.id = id
        self.name = name
        self.arguments = arguments
```

you write:

```python
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict
```

The `id: str` syntax is a *type hint* (documentation for humans/tools; not enforced
at runtime).

### `ToolCall` — "the model wants to run a tool"

```python
@dataclass
class ToolCall:
    id: str          # a label, e.g. "c1"
    name: str        # which tool, e.g. "get_current_time"
    arguments: dict  # what to pass it, e.g. {"timezone": "Asia/Tokyo"}
```

**The single most important idea in the whole project:** the model can never *run*
anything. It only produces text. When it "wants" a tool, all it can do is emit a
*request* — a `ToolCall`. The harness decides whether/how to actually run it.

Restaurant analogy: you (the model) can't walk into the kitchen and cook. You hand
the waiter (the harness) a written order. The `ToolCall` is that order ticket.

Why the `id`? You might order three dishes at once. The ticket number matches each
returned plate to its order. A model can request several tools in one turn; the `id`
ties each result back to its call.

### `ToolResult` — "here's what the tool produced"

```python
@dataclass
class ToolResult:
    tool_call_id: str   # "c1" — matches the ToolCall's id
    name: str
    content: str        # "2026-06-08 14:30:00 JST" — the actual answer
    is_error: bool = False
```

The plate coming back from the kitchen. `tool_call_id` carries the same ticket number
so everyone knows which order it answers.

Why `is_error`? If the kitchen is out of an ingredient, the waiter comes back and says
"sorry, here's why" instead of throwing the plate (crashing). `is_error=True` lets the
model *see* the failure and adapt. This is why an agent doesn't crash when a tool
fails — the failure is just another message. (`= False` means optional, defaults to
"no error".)

### `Message` — "one turn in the conversation"

```python
@dataclass
class Message:
    role: Role                       # "system" | "user" | "assistant" | "tool"
    text: str = ""
    tool_calls: list[ToolCall] = ... # only assistant turns fill this
    tool_call_id: str | None = None  # only "tool" turns
    name: str | None = None
    is_error: bool = False
```

`role` says *who is speaking*:

| `role` | Who | Restaurant analogy |
|---|---|---|
| `"system"` | Standing instructions | Rules posted in the kitchen |
| `"user"` | The human | The customer |
| `"assistant"` | The AI model | You, deciding what to order/say |
| `"tool"` | A tool's result | The plate returning from the kitchen |

A complete conversation as actual Python — *this is the agent's entire memory*:

```python
messages = [
    Message(role="system", text="You are a concise assistant."),
    Message(role="user",   text="What time is it in Tokyo?"),
    Message(role="assistant", tool_calls=[ToolCall(id="c1", name="get_current_time",
                                                   arguments={"timezone": "Asia/Tokyo"})]),
    Message(role="tool",   text="2026-06-08 14:30:00 JST", tool_call_id="c1"),
    Message(role="assistant", text="It's 2:30 PM in Tokyo."),
]
```

An assistant turn can **talk, act, or both**:

```python
# Only talk (final answer):
Message(role="assistant", text="It's 2:30 PM in Tokyo.", tool_calls=[])
# Only call a tool (text empty — it's placing an order, not talking to the human):
Message(role="assistant", text="", tool_calls=[ToolCall(id="c1", ...)])
# Both — think out loud AND call a tool:
Message(role="assistant", text="Let me check the clock...", tool_calls=[ToolCall(id="c1", ...)])
```

That's why `Message` has *both* a `text` field and a `tool_calls` field.

### The helpers

Shortcut constructors and one shortcut question:

```python
Message.user("hi")                  # == Message(role="user", text="hi")
Message.system("be nice")           # == Message(role="system", text="be nice")
Message.from_tool_result(result)    # turns a ToolResult (plate) into a role="tool" Message
                                    #   (placing the plate onto the conversation table)

@property
def wants_tools(self) -> bool:
    return bool(self.tool_calls)    # "did this turn ask for any tools?"  -> True/False
```

`wants_tools` is the **hinge of the entire agent**: after the model speaks — does it
want tools (run them, loop) or not (we're done)?

> **Layer 1 mental model:** a conversation is just a Python `list[Message]`. That list
> is the agent's only memory. Everything the agent does is read it or append to it.

---

## Layer 2 — Tools: what the model can actually *do*

A model alone only produces text. Tools are how it touches the world. Three files:

| File | Question it answers |
|---|---|
| `tools/base.py` | What *is* a tool? What shape must every tool have? |
| `tools/builtin/clock.py` | What does *one real tool* look like? |
| `tools/registry.py` | Who holds all the tools and runs the right one? |

### A tool is a thing with 4 parts

| Part | Restaurant analogy | Why the model needs it |
|---|---|---|
| `name` | Dish name on the menu | So it can say "I want *this* one" |
| `description` | Menu blurb | So it knows *when* to order it |
| `input_schema` | Customization options | So it knows *what details* to specify |
| `run()` | The kitchen | The code that does the real work |

### `tools/base.py` — the contract

```python
class Tool(ABC):
    name: str
    description: str
    input_schema: dict          # JSON Schema describing the arguments
    @abstractmethod
    def run(self, arguments: dict) -> str: ...
```

**Python concept: `ABC` + `@abstractmethod`.** `ABC` = Abstract Base Class = a
fill-in-the-blank template you can't instantiate directly. `@abstractmethod` on `run`
means every real tool *must* provide its own `run`, or Python refuses to create it:

```python
Tool()   # ❌ ERROR: Can't instantiate abstract class Tool
```

This is a safety net: anywhere that receives "a tool" can safely call `.run()`,
`.name`, etc., because the contract guarantees they exist.

`_FunctionTool` is a real `Tool` (fills in the blanks) that wraps a plain Python
function. `self._fn(**arguments)` unpacks the args dict into named arguments — so
`{"timezone": "Asia/Tokyo"}` becomes `fn(timezone="Asia/Tokyo")`.

### The `@tool` decorator — ergonomics

A *decorator* is a function that wraps another function. `@tool()` takes the function
below it and replaces it with a fully-built `Tool` object. It manufactures the 4 parts
by *reading the function's signature* (via the `inspect` module):

```python
@tool()
def get_current_time(timezone: str = "UTC") -> str:
    """Return the current date and time in the given IANA timezone."""
    ...
```

produces:

```python
get_current_time.name          # "get_current_time"          (the function name)
get_current_time.description   # the docstring
get_current_time.input_schema  # {"type": "object",
                              #  "properties": {"timezone": {"type": "string"}}}
                              # (no "required": timezone has a default -> optional)
get_current_time.run({"timezone": "Asia/Tokyo"})   # "2026-06-08 14:30:00 JST"
```

The rule: a parameter **without** a default is `required`; **with** a default is
optional. Type hint → JSON type via the `_PY_TO_JSON` map (`str→"string"`,
`int→"integer"`, ...).

> The docstring is **not** a comment for humans — it's the text the model reads to
> decide *when* to use the tool. Vague docstring = confused model.

### `tools/registry.py` — the switchboard / the waiter

At heart, the registry is **one dictionary: `name → Tool`**.

```python
self._tools = {"get_current_time": <the Tool object>}
```

Three jobs:

**1. `add` / `__init__`** — put tools in the dict, keyed by name. Refuses duplicate
names loudly (a silent overwrite would confuse the model).

**2. `schemas()` — hand out the menu.** Walks every tool and returns the 3 model-facing
parts (NOT `run` — the model never sees the kitchen):

```python
registry.schemas()
# [{"name": "get_current_time",
#   "description": "...",
#   "input_schema": {"type": "object", "properties": {"timezone": {"type": "string"}}}}]
```

This list is what gets sent to the LLM so it knows what's available.

**3. `execute(call)` — run an order. ALWAYS returns a `ToolResult`, never raises.**
Three cases:

```python
def execute(self, call):
    tool = self._tools.get(call.name)
    if tool is None:                                  # CASE A: unknown tool
        return ToolResult(..., content="Unknown tool: ... Available: ...", is_error=True)
    try:
        content = tool.run(call.arguments)            # CASE B: success
        return tool.to_result(call.id, content)
    except Exception as exc:                          # CASE C: tool crashed
        return tool.to_result(call.id, f"{type(exc).__name__}: {exc}", is_error=True)
```

- **Case A** — model hallucinated a tool name → helpful error listing the real tools.
- **Case B** — tool ran fine → wrap output in a `ToolResult`.
- **Case C** — tool threw (e.g. bad timezone) → catch it, wrap the error message.

The dunder methods `__len__` / `__contains__` let you write `len(registry)` and
`"name" in registry`. Pure convenience.

> **Why "never raise, always return a result" matters:** if `execute` let an exception
> fly, it would crash the whole agent — one bad tool call throws away the entire
> conversation. By returning `is_error=True` instead, the failure becomes *just another
> message* the model sees on its next turn, so it can self-correct. **In an agent,
> errors are not exceptions to crash on — they are observations to feed back.**
> The registry keeps the loop alive; the model does the deciding.

---

## Layer 3 — Providers: translating to/from a specific LLM

The **translator at the border** between cornac's neutral world and the LLM's native
world. One job: neutral conversation + neutral menu → call backend → one neutral
assistant `Message`. Neutral in, neutral out.

### `providers/base.py` — the contract

```python
class Provider(ABC):
    @abstractmethod
    def complete(self, messages: list[Message], tools: list[dict]) -> Message: ...
```

One required method. Read the signature as a sentence: take the conversation
(`list[Message]`) + the menu (`list[dict]` from `registry.schemas()`), return one
assistant `Message`.

Crucially, `base.py` imports **only** `Message` — not `anthropic`, not `httpx`. It
knows nothing about any specific backend. The agent loop depends on *this* abstract
`Provider`, so adding a backend = writing one subclass, with **zero** changes
elsewhere. That's the project's spine.

### Every provider has the same 3-part shape

1. `__init__` — setup (which model, where's the server / API key)
2. `complete` — the required method: translate OUT → call → translate BACK
3. two private helpers — `_to_X` (neutral→native) and `_from_X` (native→neutral)

### `OllamaProvider` (the simpler dialect)

`complete` is three moves:

```python
def complete(self, messages, tools):
    payload = {"model": ..., "messages": self._to_ollama(messages),     # MOVE 1: build request
               "stream": False, "options": {"temperature": 0}}
    if tools:
        payload["tools"] = [{"type": "function",
                             "function": {"name": t["name"], "description": t["description"],
                                          "parameters": t["input_schema"]}} for t in tools]
    resp = self._client.post(f"{self.host}/api/chat", json=payload)      # MOVE 2: call model (HTTP)
    resp.raise_for_status()
    return self._from_ollama(resp.json())                               # MOVE 3: translate back
```

Note Ollama's tool format differs from our neutral menu: wrapped in `"function"`, and
the schema is called `"parameters"` (not `"input_schema"`). Same info, different
packaging — that reshaping is the translator's job.

`_to_ollama` is an `if/elif` on `role`. The notable cases:
- `tool` results → `{"role": "tool", "content": ..., "tool_name": ...}` — **Ollama
  matches results to calls by ORDER**, so it doesn't even need the id.

`_from_ollama` rebuilds a neutral `Message`. Key quirk: **Ollama often omits the tool
call id**, so we synthesize one (`call_0_get_current_time`). The provider absorbs the
quirk so the rest of cornac — which assumes every `ToolCall` has an id — never cares.

### `AnthropicProvider` (same job, trickier dialect)

Same 3-part shape. Differences:
- Uses the official `anthropic` **SDK** (lazy-imported inside `__init__` so Ollama-only
  users don't need it installed) instead of raw HTTP.
- Requires `max_tokens` (Ollama didn't).
- **Quirk #1:** the system prompt is a *separate top-level parameter*, not a list item.
  So `_to_anthropic` returns `(system, messages)`.
- **Quirk #2:** Anthropic has **no `tool` role**. Tool results must ride *inside a user
  message* as `tool_result` blocks, matched by **id** (`tool_use_id`) — which is why
  every `ToolCall` carries an id even though Ollama ignored it. The "folding" logic
  merges consecutive tool results into one user turn (Anthropic requires all results
  for one assistant turn to arrive together).

### Same neutral input, totally different native output

```
Neutral (same for both):
    Message(role="tool", text="2:30 PM", tool_call_id="c1", name="get_current_time")

  ── OllamaProvider ──►                  ── AnthropicProvider ──►
  {"role": "tool",                       {"role": "user", "content": [
   "content": "2:30 PM",                     {"type": "tool_result",
   "tool_name": "get_current_time"}           "tool_use_id": "c1",
  (matched by ORDER)                           "content": "2:30 PM"}]}
                                          (matched by ID, folded into a user turn)
```

> **Why the neutral type exists (the architecture in one idea):** it's a *firewall*.
> Provider differences are quarantined inside `_to_X` / `_from_X`. Everything above the
> provider — loop, tools, permissions, hooks — lives in a world with ONE message
> format. New provider = new firewall-crossing, zero changes inside. Without it, you'd
> trend toward N×N translators and `if provider == "..."` branches leaking into the
> loop. (This is the classic **Adapter pattern** / *anti-corruption layer*.)

---

## Layer 4 — `agent.py`: the loop that ties it together

Reads almost trivially, *because Layers 1–3 did the hard work.* The whole agent is ~7
real lines; the rest is setup and a safety rail.

### `__init__` — assemble the three layers

```python
def __init__(self, provider, registry=None, system_prompt=None, max_steps=20):
    self.provider = provider                      # Layer 3 — the brain
    self.registry = registry or ToolRegistry()    # Layer 2 — the toolbox
    self.messages = []                            # Layer 1 — the memory
    if system_prompt:
        self.messages.append(Message.system(system_prompt))   # seed memory with instructions
    self.max_steps = max_steps
```

The type hint `provider: Provider` asks for **any** provider (the abstract base) — not
`AnthropicProvider` or `OllamaProvider`. *That single word is where "provider-agnostic"
lives in the agent.*

### `run` — the loop

```python
def run(self, user_message):
    self.messages.append(Message.user(user_message))          # record the human turn

    for _ in range(self.max_steps):                           # safety rail vs infinite loop
        response = self.provider.complete(self.messages, self.registry.schemas())  # ① ask model
        self.messages.append(response)                        # ② remember reply

        if not response.wants_tools:                          # ③ done?
            return response.text                              #    no tools -> final answer

        for call in response.tool_calls:                      # ④ run tools, feed results back
            result = self.registry.execute(call)
            self.messages.append(Message.from_tool_result(result))

    return "[cornac] Stopped after reaching max_steps ..."    # gave up gracefully
```

The four beats: ① ask (Layer 3 hides all translation), ② remember, ③ the `wants_tools`
hinge (return, or continue), ④ run each tool (Layer 2, never crashes) and append its
result (Layer 1). Then loop — and now the model can *see* the tool results in
`self.messages`.

`max_steps=20` caps runaway loops (a model could call tools forever). `_` is the
throwaway loop variable.

> **Why the loop is so short:** complexity was pushed *down* into well-named,
> single-purpose pieces, so the thing on top reads like plain English. The loop is the
> conductor; the layers are the orchestra. The plan's fuller loop adds permissions and
> hooks at the two marked spots (before `execute`, and around each step) *without
> changing this shape*.

---

## End-to-end trace: one real `run()` call

Scenario:

```python
agent = Agent(
    provider=OllamaProvider(),                    # Layer 3
    registry=ToolRegistry([get_current_time]),    # Layer 2
    system_prompt="You are a concise assistant. Use get_current_time when needed.",
)
answer = agent.run("What is the current time in Tokyo?")
```

Watch `self.messages` grow from 1 → 5 messages.

**FRAME 0 — construction.** System prompt seeds memory.
`messages = [system]`

**FRAME 1 — `run` records the question.**
`messages = [system, user]` → enter the loop, **iteration 1**.

**FRAME 2 — ① ask the model.** Provider gets the 2 messages + the menu. `_to_ollama`
translates to native; the HTTP request goes out. Qwen decides it needs the time and
returns a tool call. `_from_ollama` translates back (synthesizing the missing id):

```python
response = Message(role="assistant", text="",
    tool_calls=[ToolCall(id="call_0_get_current_time", name="get_current_time",
                         arguments={"timezone": "Asia/Tokyo"})])
```

**FRAME 3 — ② remember it.**
`messages = [system, user, assistant(tool_call)]`

**FRAME 4 — ③ done?** `wants_tools` is True → do NOT return. Fall through to ④.

**FRAME 5 — ④ run the tool.** `registry.execute` finds `get_current_time`, runs
`clock.py`, gets `"2026-06-08 14:30:00 JST"`, wraps it. Appended as a tool turn:
`messages = [system, user, assistant(tool_call), tool(result)]` → loop back,
**iteration 2**.

**FRAME 6 — ① ask the model AGAIN.** Now `self.messages` has 4 messages, including the
tool result — so the model *sees the answer it asked for* and writes a sentence with
**no** tool call:

```python
response = Message(role="assistant", text="It is currently 2:30 PM in Tokyo.", tool_calls=[])
```

**FRAME 7 — ② remember it.**
`messages = [system, user, assistant(tool_call), tool(result), assistant(final)]`

**FRAME 8 — ③ done?** `wants_tools` is False → **return** `response.text`. Loop ends.

### The whole motion

```
[system]                                       ← Frame 0 (construction)
[system, user]                                 ← Frame 1 (human asks)
ITERATION 1 ──────────────────────────────────
  ① ask model      → wants a tool              ← Frame 2 (Layer 3)
  [.., assistant(tool_call)]                   ← Frame 3
  ③ done? NO
  ④ run tool → [.., tool(result)]              ← Frame 5 (Layer 2 + 1)
ITERATION 2 ──────────────────────────────────
  ① ask model AGAIN (now sees result)          ← Frame 6 (Layer 3)
  [.., assistant(final)]                       ← Frame 7
  ③ done? YES → return text  ✅                ← Frame 8
```

The model "thought" across two turns: turn 1 = "I need data, fetch it"; turn 2 = "now
I have it, here's the answer." The loop stitched them together by feeding the tool
result back. **That feedback — append the result, ask again — is what makes it an agent
rather than a single Q&A.** Every layer played its part: 1 held the memory, 2 ran the
tool safely, 3 translated both ways, 4 conducted.

---

## Try it yourself

```bash
# Offline (fake brain — proves the loop without any model):
python examples/walking_skeleton_trace.py   # uses a scripted provider in tests

# Live, local model:
ollama pull qwen2.5:7b-instruct-q4_K_M
python examples/walking_skeleton_trace.py ollama

# Live, Claude:
export ANTHROPIC_API_KEY=sk-ant-...
python examples/walking_skeleton_trace.py anthropic
```

`walking_skeleton_trace.py` prints the conversation growing message by message — the
live version of Frames 0–8 above.
