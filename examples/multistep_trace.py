"""Week 2 demo — multi-step tool chaining.

Week 1 showed a 2-iteration loop (call one tool, answer). This shows the agent
chaining SEVERAL tools across multiple iterations to solve a task it can't do in one
shot — the moment cornac starts behaving like a real agent.

We set up a sandbox workspace with a tiny CSV, then ask a question that forces the
model to discover the file, read it, and compute over it. Watch the iteration count
climb past 2.

    python examples/multistep_trace.py ollama
    python examples/multistep_trace.py anthropic   # needs ANTHROPIC_API_KEY
"""

import sys
import tempfile
from pathlib import Path

from cornac import Agent, Message, ToolRegistry
from cornac.tools.builtin import default_tools
from cornac.tools.workspace import Workspace


def build_provider(which: str):
    if which == "anthropic":
        from cornac.providers.anthropic import AnthropicProvider

        return AnthropicProvider()
    from cornac.providers.ollama import OllamaProvider

    return OllamaProvider()


def show(msg: Message) -> str:
    if msg.role == "assistant" and msg.tool_calls:
        calls = ", ".join(f"{c.name}({c.arguments})" for c in msg.tool_calls)
        # keep long args readable
        calls = (calls[:160] + "...") if len(calls) > 160 else calls
        return f"  [assistant] -> CALLS: {calls}"
    if msg.role == "tool":
        body = msg.text.replace("\n", " ⏎ ")
        body = (body[:140] + "...") if len(body) > 140 else body
        flag = " [ERROR]" if msg.is_error else ""
        return f"  [tool→{msg.name}]{flag} {body!r}"
    if msg.role == "assistant":
        return f"  [assistant] {msg.text!r}"
    return f"  [{msg.role}] {msg.text!r}"


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "ollama"

    # --- build a sandbox workspace with sample data ---
    tmp = Path(tempfile.mkdtemp(prefix="cornac_ws_"))
    (tmp / "sales.csv").write_text(
        "product,units,price\n"
        "widget,10,2.50\n"
        "gadget,4,9.00\n"
        "gizmo,7,4.00\n"
    )
    (tmp / "README.txt").write_text("Quarterly sales data is in sales.csv.\n")
    workspace = Workspace(tmp)

    agent = Agent(
        provider=build_provider(which),
        registry=ToolRegistry(default_tools(workspace)),
        system_prompt=(
            "You are a precise data assistant working inside a sandboxed workspace. "
            "Explore files with list_dir/read_file, and use run_python for any "
            "calculation. Do not guess numbers — compute them."
        ),
        max_steps=10,
    )

    task = (
        "What is the total revenue (units * price summed across all products) in "
        "sales.csv? Show the number."
    )
    print(f"\n=== cornac multi-step trace | provider: {agent.provider.name} ===")
    print(f"workspace: {tmp}")
    print(f"task: {task}\n")

    agent.messages.append(Message.user(task))

    step = 0
    while step < agent.max_steps:
        step += 1
        print(f"--- iteration {step} ---")
        response = agent.provider.complete(agent.messages, agent.registry.schemas())
        agent.messages.append(response)
        print(show(response))

        if not response.wants_tools:
            print(f"\n=== FINAL ANSWER ===\n{response.text}\n")
            break

        for call in response.tool_calls:
            result = agent.registry.execute(call)
            agent.messages.append(Message.from_tool_result(result))
            print(show(agent.messages[-1]))
    else:
        print(f"\n[stopped after {agent.max_steps} steps]")

    print(f"[debug] {len(agent.messages)} messages, {step} model iterations")


if __name__ == "__main__":
    main()
