"""Week 3 demo — the permission prompt in action.

This is the Claude-Code-style approval flow. The policy marks run_python as "ask", so
when the model wants to run code, cornac pauses and asks YOU in the terminal:

    cornac wants to run a tool:
      run_python(code='...')
    Allow? [y]es / [n]o / [a]lways / nev[e]r:

Answer 'y' to let it run, 'n' to refuse (the model gets an error and adapts), 'a' to
allow run_python for the rest of the session, 'e' to refuse it for the session.

    python examples/permissions_demo.py ollama

Tip: pipe an answer in for a non-interactive run, e.g.:
    echo a | python examples/permissions_demo.py ollama
"""

import sys
import tempfile
from pathlib import Path

from cornac import Agent, HookBus, Policy, ToolRegistry, cli_ask
from cornac.tools.builtin import default_tools
from cornac.tools.workspace import Workspace


def build_provider(which: str):
    if which == "anthropic":
        from cornac.providers.anthropic import AnthropicProvider

        return AnthropicProvider()
    from cornac.providers.ollama import OllamaProvider

    return OllamaProvider()


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "ollama"

    tmp = Path(tempfile.mkdtemp(prefix="cornac_perm_"))
    (tmp / "sales.csv").write_text("product,units,price\nwidget,10,2.50\ngadget,4,9.00\n")
    workspace = Workspace(tmp)

    # read_file is auto-allowed; run_python must be approved by the human.
    policy = Policy({"default": "ask", "tools": {
        "list_dir": "allow", "read_file": "allow", "run_python": "ask"}})

    # A hook so you can see the decision that each prompt produced.
    bus = HookBus()
    bus.on("on_permission_decision",
           lambda call, d: print(f"   [decision: {d.value.upper()}]"))

    agent = Agent(
        provider=build_provider(which),
        registry=ToolRegistry(default_tools(workspace)),
        system_prompt=("You are a precise data assistant in a sandboxed workspace. "
                       "Use run_python for any calculation; do not guess."),
        max_steps=10,
        policy=policy,
        approver=cli_ask,   # <-- when policy says ASK, prompt the human
        hooks=bus,
    )

    print(f"=== permissions_demo | provider: {agent.provider.name} ===")
    print(f"workspace: {tmp}\n")
    answer = agent.run("What is the total revenue (units * price) in sales.csv?")
    print(f"\n=== FINAL ANSWER ===\n{answer}")


if __name__ == "__main__":
    main()
