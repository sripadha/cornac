# docs/breakdown — the learning record

Beginner-friendly walkthroughs of every layer of the harness, written as the code was
built. Each *walkthrough* teaches slowly with analogies, Python-concept asides, worked
examples, and real experiment output; the *summary* files are shorter design references.

> Naming note: the package is being renamed `cornac` → `ankus` (Week-4a batch, see
> [../plan.md](../plan.md)). These walkthroughs reference the code as it currently
> exists; identifiers will be swept in the rename batch.

## Recommended reading order

| # | File | Week | What it teaches |
|---|---|---|---|
| 1 | [walking-skeleton.md](walking-skeleton.md) | 1 | The four layers of an agent: neutral messages, tools + registry, the Provider seam, the loop — plus a frame-by-frame trace of one request |
| 2 | [how-the-model-sees-tools.md](how-the-model-sees-tools.md) | 1–2 Q&A | The two channels tools reach the model through (schema vs system prompt, with the ablation proof), and why wording wobbles at temperature 0 — grade behavior, not phrasing |
| 3 | [tools-and-workspace-walkthrough.md](tools-and-workspace-walkthrough.md) | 2 | The Workspace path-fence (with the `..` traversal lesson), the four file tools, why `run_bash`/`run_python` can't be fenced, web tools, `default_tools` |
| 4 | [permissions-and-hooks-walkthrough.md](permissions-and-hooks-walkthrough.md) | 3 | The ALLOW/DENY/ASK rulebook (priority ladder, deny-beats-allow), the human approver, the hook bus (observer pattern), and how both slot into the loop unchanged |
| 5 | [ollama-and-environment.md](ollama-and-environment.md) | setup | What Ollama actually is (weights vs engine vs server), why not raw HuggingFace on a 6GB GPU, and `uv` vs `.venv` vs `pip` |

## Visual pages

| File | What it shows |
|---|---|
| [ankus-agent-loop.html](ankus-agent-loop.html) | The whole harness on one page: the loop flowchart (with the Week-3 permission gate and hooks), the six layers, a traced request, and the permission priority ladder. Open it in a browser — from WSL: `explorer.exe "$(wslpath -w docs/breakdown/ankus-agent-loop.html)"` (the Mermaid diagram needs internet). |

## Shorter design summaries

| File | Companion to |
|---|---|
| [tools-and-workspace.md](tools-and-workspace.md) | Week 2 walkthrough |
| [permissions-and-hooks.md](permissions-and-hooks.md) | Week 3 walkthrough |

## The recurring ideas (they show up in every file)

1. **A conversation is a growing `list[Message]`** — the agent's only memory.
2. **The neutral format is a firewall** — provider quirks are quarantined in adapters.
3. **Errors are observations, not crashes** — failures (tool errors, sandbox escapes,
   permission denials) flow back as results the model adapts to.
4. **Push complexity down** — the loop stays ~15 lines because every hard thing lives in
   a well-named module underneath it.
5. **State → class; stateless → `@tool` function** — the two shapes of tool.
6. **Decide, then ask, separately** — pure policy logic + a swappable human approver.
