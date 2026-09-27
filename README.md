# cornac

> *cornac* (French): the person who rides and directs an elephant — a **mahout**.

A from-scratch, provider-agnostic **LLM agent harness** in Python.

**The thesis:** the agent isn't the model — it's the harness. The same weak model
becomes dramatically more capable as you add scaffolding around it (a system prompt,
retrieval, tools, and finally a dynamic agent loop). cornac is that scaffolding, built
from scratch so every layer is legible.

> ⚠️ **Status: work in progress.** Built in the open as a learning project.
> Weeks 1-3 are done (agent loop, 9 tools + workspace sandbox, permissions + hooks),
> plus a hardening pass (gates that fail closed, timeouts that really stop a command,
> head+tail output trimming, `RunResult`), and sub-agents. Next up: the benchmark.

## Why "cornac"?

A mahout is small; the elephant is enormous. The mahout doesn't *become* stronger —
they *direct* a much larger force. That's exactly what a harness does to an LLM.

## What works today

- A provider-neutral conversation core (`Message` / `ToolCall` / `ToolResult`).
- A `Provider` seam with **two** implementations — Anthropic (Claude) and Ollama
  (local models) — that the agent loop is completely blind to.
- A tool system (`@tool` decorator + registry) with 11 built-in tools, confined to a
  `Workspace` sandbox. Long output is trimmed head+tail, so a traceback at the end
  of a flood of output still reaches the model. `run_python` echoes the value of a
  trailing bare expression, like a notebook cell, so a model that ends with `total`
  instead of `print(total)` still sees its answer. `edit_file` changes one snippet
  in place instead of rewriting the whole file, and both it and `write_file` refuse
  a `.py` that does not compile, leaving the file untouched, so the model hears
  about a syntax slip now rather than from the next test run.
- Permissions: every tool call passes an ALLOW / DENY / ASK policy (YAML or in-code)
  with an interactive approval prompt — and the gates fail closed.
- Lifecycle hooks to observe the loop, including a `pre_tool_use` hook that can veto
  a call.
- The agent loop itself, returning a `RunResult` (final text, stop reason, steps,
  duration, token usage). Two small pieces of scaffolding the coding spike showed
  were needed: a reply that only *announces* a step ("Let me fix that and re-run
  the tests") and stops gets one nudge to act or answer, and a tool call repeated
  with the identical result gets a note saying so.
- Sub-agents: a `spawn_agent` tool hands a self-contained sub-task to a fresh child
  agent (own empty context, a subset of the parent's tools, the *same* policy,
  approver and `pre_tool_use` veto hooks, so delegation never escalates privilege)
  and gets back only its final text, length-capped and framed as a sub-agent reply.
  Depth and count are bounded, the child's tokens roll up into the parent's
  `RunResult.usage`, and `on_spawn` / `on_spawn_done` hooks show the delegation
  tree. See [examples/subagent_demo.py](examples/subagent_demo.py).

Same agent, same tool, two brains:

```bash
# Local Qwen 3.5 4B via Ollama (the frozen benchmark model)
python examples/walking_skeleton.py ollama

# Claude Haiku via the Anthropic API
python examples/walking_skeleton.py anthropic
```

## Quick start

```python
from cornac import Agent, ToolRegistry
from cornac.providers.ollama import OllamaProvider
from cornac.tools.builtin.clock import get_current_time

agent = Agent(
    provider=OllamaProvider(),                     # or AnthropicProvider()
    registry=ToolRegistry([get_current_time]),
    system_prompt="You are a helpful assistant.",
)
result = agent.run("What time is it in Tokyo right now?")
print(result.text)
```

### Run results

`agent.run()` returns a `RunResult`, not a bare string — so a caller can tell a
real answer apart from a run that hit the `max_steps` safety rail, and can see what
the run cost:

```python
result = agent.run("What is the total revenue in sales.csv?")

result.text          # the final answer — None if the loop gave up at max_steps
result.stop_reason   # "done" or "max_steps"
result.steps         # how many model calls the run made
result.duration      # wall-clock seconds
result.usage         # Usage(input_tokens=..., output_tokens=...); .total_tokens sums them

print(result)        # str(result) is still the text, so this keeps working
```

## Setup

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"

# For the local provider:
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3.5:4b     # the frozen benchmark model: 9/9 on the coding spike,
                          # ~20 s per task, fits a 6 GB GPU (see benchmark/spike/)

# For the Claude provider:
export ANTHROPIC_API_KEY=sk-ant-...
```

## Roadmap

The project is built as a **capability ladder** — each rung is a runnable demo:

| Stage | Adds | Status |
|------|------|--------|
| 0 | Bare model | planned |
| 1 | + System prompt | planned |
| 2 | + RAG | planned |
| 3 | + Single-turn tool call | planned |
| 4 | + Static workflow | planned |
| 5 | + Dynamic harness (this package) | 🚧 in progress |

**Tooling so far:** 11 built-in tools (`read_file`, `write_file`, `edit_file`, `list_dir`,
`grep`, `run_bash`, `run_python`, `web_search`, `web_fetch`, `get_current_time`,
`spawn_agent`) confined to a `Workspace` sandbox.
The agent chains them across loop iterations and recovers from its own errors — see
[examples/multistep_trace.py](examples/multistep_trace.py) and
[docs/breakdown/tools-and-workspace.md](docs/breakdown/tools-and-workspace.md).

**Permissions & hooks:** tool calls pass through an ALLOW/DENY/ASK policy (YAML or
in-code) with an interactive `[y/n/a/e]` approval prompt; lifecycle **hooks** let you
observe the loop without copying it. A `pre_tool_use` hook can also **veto** a call by
returning `Decision.DENY` — checked before the policy and the prompt, so you can
enforce rules a static policy can't express (and a veto hook that crashes counts as a
veto: gatekeepers fail closed). See
[examples/permissions_demo.py](examples/permissions_demo.py),
[examples/hook_trace.py](examples/hook_trace.py), and
[docs/breakdown/permissions-and-hooks.md](docs/breakdown/permissions-and-hooks.md).

Then: a 15-task benchmark showing the success-rate climb across the
ladder — all on the same weak model.

See [docs/plan.md](docs/plan.md) for the full design.

## License

MIT
