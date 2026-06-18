# cornac — Project Plan

**Status:** Draft v1 — 2026-06-07
**Owner:** spv
**Target ship:** 6–7 weeks part-time

---

## 1. Goal

Build `cornac`: a from-scratch, provider-agnostic LLM **agent harness** in Python that demonstrates how a weak model becomes dramatically more capable as scaffolding is added.

The repo will contain two things:

1. **The harness package** — the published artifact (`pip install cornac`).
2. **A 6-stage capability ladder + 15-task benchmark** — the demo that motivates the harness, runnable end-to-end.

The narrative for the public post (LinkedIn / GitHub README): **"Same weak open-weight model. Different scaffolding. Big capability difference. The agent isn't the model — it's the harness."**

## 2. Non-goals (deliberately out of scope for v1)

These are deferred to keep v1 shippable:

- Competing with LangChain / AutoGen / LlamaIndex as a production framework.
- Reimplementing inference. We use **Ollama** for local serving. The harness lives strictly *above* the inference boundary. (Inference deep-dive belongs in a separate future project.)
- Streaming responses to a UI. CLI-only output is fine for v1.
- Multimodal (images, audio). Text in, text out.
- Context compaction / summarization at token-limit. Deferred to v2.
- Slash commands or interactive REPL. CLI-script invocation only.
- Persistent cross-session memory. Each `agent.run(...)` is a fresh conversation.

## 3. Scope (what's in v1)

- **Core agent loop** with clean separation between LLM call, tool dispatch, and policy.
- **Provider abstraction**: `AnthropicProvider`, `OllamaProvider`. Identical message/tool shape; provider-specific code confined to thin adapters.
- **Tool system**: registry + 8 built-in tools (see §6).
- **Permission system**: YAML policy file + interactive CLI prompt fallback.
- **Hook system**: lifecycle events with in-process callbacks.
- **Sub-agents**: a `spawn_agent` tool that instantiates a child harness with a custom system prompt.
- **6-stage demo scripts** (`examples/stage_0_bare_model.py` … `examples/stage_5_dynamic_harness.py`).
- **15-task benchmark** with grader + stage-vs-success-rate report.
- **`pip`-publishable package** with README, examples, and a small test suite.

## 4. Architecture

### 4.1 Package layout

```
cornac/
├── cornac/                       # the published package
│   ├── __init__.py
│   ├── core/
│   │   ├── agent.py               # the loop
│   │   ├── messages.py            # Message / ToolCall / ToolResult dataclasses
│   │   └── state.py               # conversation state container
│   ├── providers/
│   │   ├── base.py                # abstract Provider interface
│   │   ├── anthropic.py
│   │   └── ollama.py
│   ├── tools/
│   │   ├── base.py                # Tool ABC + @tool decorator
│   │   ├── registry.py            # name -> Tool dispatch
│   │   └── builtin/
│   │       ├── files.py           # read_file, write_file, list_dir, grep
│   │       ├── shell.py           # run_bash
│   │       ├── python_exec.py     # run_python (sandboxed)
│   │       ├── web.py             # web_search, web_fetch
│   │       └── spawn.py           # spawn_agent (sub-agent)
│   ├── permissions/
│   │   ├── policy.py              # allow / deny / ask rules from YAML
│   │   └── prompt.py              # interactive CLI approval
│   ├── hooks/
│   │   └── bus.py                 # event -> callbacks
│   └── cli.py                     # user-facing entrypoint
├── examples/
│   ├── stage_0_bare_model.py
│   ├── stage_1_system_prompt.py
│   ├── stage_2_rag.py
│   ├── stage_3_single_tool.py
│   ├── stage_4_static_workflow.py
│   └── stage_5_dynamic_harness.py
├── benchmark/
│   ├── tasks/                     # 15 task definitions
│   ├── grader.py                  # hand-graded + LLM judge cross-check
│   ├── run_all.py                 # runs all stages on all tasks
│   └── results.md                 # generated success-rate table
├── docs/
│   ├── plan.md                    # this file
│   └── architecture.md            # written in Week 6
├── tests/
├── pyproject.toml
├── README.md
└── LICENSE
```

### 4.2 The agent loop (mental model)

```python
def run(user_message, state):
    state.messages.append(user_message)
    hooks.fire("on_user_message", user_message)

    while True:
        response = provider.complete(state.messages, registry.schemas())
        state.messages.append(response)
        hooks.fire("on_assistant_message", response)

        if not response.tool_calls:
            hooks.fire("on_stop", state)
            return response.text

        for call in response.tool_calls:
            hooks.fire("pre_tool_use", call)
            decision = policy.check(call)
            if decision == ASK:
                decision = prompt.ask(call)
            if decision == DENY:
                result = ToolResult(error="denied by policy")
            else:
                result = registry.execute(call)
            hooks.fire("post_tool_use", call, result)
            state.messages.append(result)
```

This is the whole show. Everything else is layered modules.

### 4.3 The two abstractions that earn their keep

**Provider interface** — the only place that knows API-specific message shapes:

```python
class Provider(ABC):
    @abstractmethod
    def complete(self, messages: list[Message], tools: list[ToolSchema]) -> Message: ...
```

`AnthropicProvider` maps our `Message` types to Anthropic's `messages.create(...)` and back.
`OllamaProvider` hits `http://localhost:11434/api/chat` and reads the `tool_calls` field.

**Tool interface** — uniform contract for any callable:

```python
class Tool(ABC):
    name: str
    description: str
    input_schema: dict   # JSON Schema
    def run(self, input: dict) -> ToolResult: ...
```

## 5. The 6-stage capability ladder

Each stage is an `examples/stage_N_*.py` script. Each stage uses the **same weak model** so the uplift is attributable to scaffolding alone.

| Stage | Adds | New capability | Code complexity |
|---|---|---|---|
| 0 | Bare `provider.complete([user_msg])` | None — answers from training only | ~10 lines |
| 1 | System prompt | Role, format adherence | ~15 lines |
| 2 | RAG: embed docs, retrieve top-k, inject | Answers about private/recent data | ~80 lines |
| 3 | Single-turn tool call | One live action per turn (no loop) | ~40 lines |
| 4 | Static workflow (developer-orchestrated multi-step) | Reliable pipelines for known task shapes | ~60 lines |
| 5 | Dynamic harness (the `cornac` package itself) | Open-ended tasks, plans on the fly | uses package |

Stage 5 is where the published package shines. Stages 0–4 are intentionally minimal — they exist to make the uplift visible.

## 6. Tool surface

Eight built-in tools cover all benchmark domains:

| Tool | Domain | Permission default |
|---|---|---|
| `read_file` | SWE, data analysis | auto-allow |
| `write_file` | SWE | **ask** |
| `list_dir` | SWE, data analysis | auto-allow |
| `grep` | SWE | auto-allow |
| `run_bash` | SWE | **ask** |
| `run_python` | data analysis | **ask** |
| `web_search` | research | auto-allow |
| `web_fetch` | research | auto-allow |
| `spawn_agent` | (cross-domain) | auto-allow (parent agent decides delegation) |

## 7. Permissions

Two-layer system, modeled loosely on Claude Code:

1. **Policy file** (`cornac.policy.yaml`) declares per-tool rules:
   ```yaml
   tools:
     read_file: allow
     write_file: ask
     run_bash:
       allow: ["pytest", "ls", "git status"]
       deny: ["rm -rf", "sudo"]
       default: ask
   ```
2. **Interactive prompt** when policy returns `ask`:
   ```
   Allow run_bash("pytest tests/")?  [y/N/always/never]
   ```
   `always` / `never` rewrite the policy file for the session.

## 8. Hooks

In-process callbacks registered against named events. Lifecycle events:

- `on_user_message`
- `pre_tool_use(call)`
- `post_tool_use(call, result)`
- `on_assistant_message(msg)`
- `on_stop(state)`

Used internally for logging the benchmark runs. Also a public extension point.

## 9. Sub-agents

`spawn_agent` is just a tool whose `run()` instantiates a fresh `Agent` with a parent-supplied system prompt and tool subset, runs it to completion, and returns its final message. Parent's context stays clean because only the child's *result* enters the parent's history, not its scratch work.

Implementation note: ~150 lines including loop-detection safeguards (max depth, max children).

## 10. Benchmark

### 10.1 Constants

- **Model:** Qwen 2.5 7B Instruct (Q4_K_M) served by Ollama on RTX 2060.
- **Sanity check run:** same benchmark against Claude Haiku via Anthropic API (~$0.50 of API credit).
- **Temperature:** 0 for determinism.
- **Max steps per task:** 20 (dynamic harness).
- **Timeout per task:** 5 minutes.

### 10.2 Task mix (15 tasks)

| # | Domain | Example task |
|---|---|---|
| 1–5 | SWE | "Find the function that handles X and add a docstring"; "Run the tests and fix any that fail"; … |
| 6–10 | Research | "What's the current latest stable Python release and its release date?"; "Summarize the abstract of arXiv:XXXX"; … |
| 11–15 | Data analysis | "Load data.csv, report the mean and median of column Y"; "Find rows where X > 100"; … |

(Final task wording to be drafted in Week 4.)

### 10.3 Grading

Each task has a hand-written pass/fail rubric. We grade twice:

1. Hand-graded against the rubric (ground truth).
2. LLM-judge cross-check (Claude Haiku) to surface where judge ≠ human, useful for the post.

### 10.4 Output

A markdown table generated by `benchmark/run_all.py`:

| Stage | SWE | Research | Data | Overall |
|---|---|---|---|---|
| 0. Bare model | x/5 | x/5 | x/5 | x/15 |
| 1. + System prompt | … | … | … | … |
| 2. + RAG | … | … | … | … |
| 3. + Tool calling | … | … | … | … |
| 4. + Static workflow | … | … | … | … |
| 5. + Dynamic harness | … | … | … | … |

This table is the LinkedIn hero image.

## 11. Model strategy

| Slot | Model | Why |
|---|---|---|
| Primary constant | Qwen 2.5 7B Instruct Q4_K_M (Ollama, RTX 2060) | Fits 6GB VRAM; strong tool-call support; reproducible by readers with one `ollama pull` |
| Sanity check | Claude Haiku (Anthropic API) | Verifies "harness > model" pattern is provider-agnostic; consumes ~$0.50 of credit |
| Future v2 | Qwen 2.5 14B / 32B on A6000 (when available) | Bigger-model curve for a follow-on post |

## 12. Week-by-week schedule

Each week ends with a *shareable* artifact so the project is de-risked against interruption.

| Week | Deliverable | Shareable artifact |
|---|---|---|
| **1** | Ollama installed + Qwen 2.5 7B pulled. Provider interface + `AnthropicProvider` + `OllamaProvider` + core loop + 1 tool (`get_current_time`). | "Day 7: from-scratch agent called its first tool, same code, two providers." |
| **2** | Tool registry + 8 built-in tools. Both providers exercised end-to-end with multi-step tool use. | "Same agent, swapped to local Qwen. Same code." |
| **3** | Permission system (YAML policy + CLI prompt) + hook bus. | "Added Claude-Code-style approval flow in ~100 lines." |
| **4** | Sub-agents (`spawn_agent` tool). Stage scripts 0–5 written and runnable. | "Built the capability ladder, end to end." |
| **5** | 15-task benchmark + grader + run_all.py + the success-rate table. | **LinkedIn post day.** |
| **6** | Package polish: README, architecture.md, examples cleanup, basic tests, `pip` publishable. | "Open-sourced `cornac` today." |
| **7** (buffer) | A6000 rerun with larger Qwen; v2 post. | "Update: same harness, bigger model — the curve." |

## 13. Open questions to resolve before / during Week 1

- Exact Qwen variant: `qwen2.5:7b-instruct-q4_K_M` vs newer Qwen 3 if released. To verify against Ollama registry in Week 1.
- Tool-call format from Ollama: confirm the `/api/chat` response shape returns `tool_calls` consistently for Qwen 2.5. If unreliable, fall back to JSON-mode prompting.
- Hook execution model: in-process callbacks (simpler) vs subprocess (Claude-Code-style). Default: in-process. Revisit only if there's a clear win.
- `run_python` sandboxing: subprocess with resource limits, or a tighter sandbox (`RestrictedPython`, container)? Default: subprocess. Acknowledged limitation, not production-grade.

## 14. Decisions and rationale

Brief decision log so future-me / readers don't have to dig through chat to understand the why.

- **Python over TypeScript.** Audience is AI/ML. Tool implementations integrate naturally with Python data libraries. TypeScript is trending in editor-integrated agents (Claude Code, Cursor, Cline) because VS Code's extension API is TS-first — not relevant here.
- **Ollama, not from-scratch inference.** The harness is the from-scratch part; inference is a separate domain. Conflating them dilutes both. Inference learning happens in a separate future project (candidates: paged attention from scratch, speculative decoding, manual quantization).
- **Provider-agnostic from day one.** "Provider abstraction" added late is usually cosmetic — bolted-on `if provider == "anthropic"` branches. Starting with two providers (Anthropic + Ollama) forces the abstraction to be real.
- **Mid-tier scope.** Permissions + hooks + sub-agents are included because they're load-bearing for the "real harness" framing. Streaming, compaction, slash commands are excluded because they're scope-creep that doesn't change the demo's thesis.
- **Qwen 2.5 7B as the constant model.** Smallest model that does tool calling reliably and fits the RTX 2060. Open-weight reproducibility is the strongest LinkedIn hook.
- **15-task mixed benchmark.** Smaller benchmarks (5 tasks) are too noisy to support the success-rate claim; larger ones (50+) take too long to hand-grade. 15 is the sweet spot.

## 15. Success criteria

The project is "done" (shippable for the post) when **all** of:

- [ ] `pip install -e .` installs and `cornac --version` works.
- [ ] `examples/stage_5_dynamic_harness.py` runs end-to-end against both Anthropic and Ollama providers using the *same* agent code.
- [ ] `benchmark/run_all.py` produces the stage-vs-success-rate table.
- [ ] At least one sub-agent invocation succeeds in the benchmark.
- [ ] At least one `ask`-policy tool call shows the CLI approval flow.
- [ ] README has install steps, the architecture diagram, and the benchmark table.
- [ ] LinkedIn draft is written.
