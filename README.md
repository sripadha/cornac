# cornac

> *cornac* (French): the person who rides and directs an elephant — a **mahout**.

A from-scratch, provider-agnostic **LLM agent harness** in Python.

**The thesis:** the agent isn't the model — it's the harness. The same weak model
becomes dramatically more capable as you add scaffolding around it (a system prompt,
retrieval, tools, and finally a dynamic agent loop). cornac is that scaffolding, built
from scratch so every layer is legible.

> ⚠️ **Status: early / work in progress.** Built in the open as a learning project.
> Week 1 = walking skeleton (provider abstraction + agent loop + one tool).

## Why "cornac"?

A mahout is small; the elephant is enormous. The mahout doesn't *become* stronger —
they *direct* a much larger force. That's exactly what a harness does to an LLM.

## What works today (Week 1)

- A provider-neutral conversation core (`Message` / `ToolCall` / `ToolResult`).
- A `Provider` seam with **two** implementations — Anthropic (Claude) and Ollama
  (local models) — that the agent loop is completely blind to.
- A tool system (`@tool` decorator + registry) with one built-in tool.
- The agent loop itself.

Same agent, same tool, two brains:

```bash
# Local Qwen 2.5 7B via Ollama
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
print(agent.run("What time is it in Tokyo right now?"))
```

## Setup

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"

# For the local provider:
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:7b-instruct-q4_K_M

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

**Tooling so far:** 8 built-in tools (`read_file`, `write_file`, `list_dir`, `grep`,
`run_bash`, `run_python`, `web_search`, `web_fetch`) confined to a `Workspace` sandbox.
The agent chains them across loop iterations and recovers from its own errors — see
[examples/multistep_trace.py](examples/multistep_trace.py) and
[docs/breakdown/tools-and-workspace.md](docs/breakdown/tools-and-workspace.md).

**Permissions & hooks:** tool calls pass through an ALLOW/DENY/ASK policy (YAML or
in-code) with an interactive `[y/n/a/e]` approval prompt; lifecycle **hooks** let you
observe the loop without copying it. See
[examples/permissions_demo.py](examples/permissions_demo.py),
[examples/hook_trace.py](examples/hook_trace.py), and
[docs/breakdown/permissions-and-hooks.md](docs/breakdown/permissions-and-hooks.md).

Then: a permission system, lifecycle hooks, sub-agents, and a 15-task benchmark
showing the success-rate climb across the ladder — all on the same weak model.

See [docs/plan.md](docs/plan.md) for the full design.

## License

MIT
