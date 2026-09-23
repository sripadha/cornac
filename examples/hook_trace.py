"""Week 3 demo — observe the loop with HOOKS (no copied loop).

Compare this to walking_skeleton_trace.py / multistep_trace.py, which had to *copy*
the agent loop to print what was happening. Here we register hook callbacks and then
call the real agent.run() — the loop lives in cornac, we just watch it. That's the
whole point of the hook system.

Since Week 4, hooks can do one thing beyond watching: a pre_tool_use callback that
returns Decision.DENY vetoes the call before the policy or the human is consulted.
The pre_tool_use hook below only narrates, but the comment shows where a veto goes.

    python examples/hook_trace.py ollama
    python examples/hook_trace.py anthropic   # needs ANTHROPIC_API_KEY
"""

import sys
import tempfile
from pathlib import Path

from cornac import Agent, HookBus, Policy, ToolRegistry
from cornac.tools.builtin import default_tools
from cornac.tools.workspace import Workspace


def build_provider(which: str):
    if which == "anthropic":
        from cornac.providers.anthropic import AnthropicProvider

        return AnthropicProvider()
    from cornac.providers.ollama import OllamaProvider

    return OllamaProvider()


def install_trace_hooks(bus: HookBus) -> None:
    """Register callbacks that narrate the loop. None of this touches the loop itself."""
    bus.on("on_user_message", lambda m: print(f"\n[user] {m.text!r}"))

    def on_assistant(m):
        if m.tool_calls:
            calls = ", ".join(f"{c.name}({c.arguments})" for c in m.tool_calls)
            print(f"[assistant] wants tools: {calls[:160]}")
        else:
            print(f"[assistant] {m.text!r}")
    bus.on("on_assistant_message", on_assistant)

    # Fires before the policy looks at the call. We just narrate and return None; if
    # this returned Decision.DENY instead — or crashed — the agent would refuse the
    # call right here.
    bus.on("pre_tool_use", lambda call: print(f"   ↳ about to run {call.name}"))

    bus.on("on_permission_decision",
           lambda call, decision: print(f"   ↳ policy says {decision.value.upper()} for {call.name}"))

    def on_post_tool(call, result):
        # `result` is a ToolResult (the tool's raw output), not a Message — so the
        # body lives in .content, not .text.
        body = result.content.replace("\n", " ⏎ ")
        body = (body[:120] + "...") if len(body) > 120 else body
        flag = " [ERROR]" if result.is_error else ""
        print(f"   ↳ {call.name} ->{flag} {body!r}")
    bus.on("post_tool_use", on_post_tool)

    # on_stop gets the final text — or None if the loop gave up at max_steps.
    def on_stop(text):
        print("\n[stop] final answer ready" if text is not None else "\n[stop] hit max_steps")
    bus.on("on_stop", on_stop)


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "ollama"

    tmp = Path(tempfile.mkdtemp(prefix="cornac_hook_"))
    (tmp / "sales.csv").write_text("product,units,price\nwidget,10,2.50\ngadget,4,9.00\n")
    workspace = Workspace(tmp)

    bus = HookBus()
    install_trace_hooks(bus)

    # Allow the read-only + python tools this task needs, so it runs non-interactively.
    policy = Policy({"default": "ask", "tools": {
        "list_dir": "allow", "read_file": "allow", "run_python": "allow"}})

    agent = Agent(
        provider=build_provider(which),
        registry=ToolRegistry(default_tools(workspace)),
        system_prompt=("You are a precise data assistant in a sandboxed workspace. "
                       "Use run_python for any calculation; do not guess."),
        max_steps=10,
        policy=policy,
        hooks=bus,
    )

    print(f"=== hook_trace | provider: {agent.provider.name} | workspace: {tmp} ===")
    result = agent.run("What is the total revenue (units * price) in sales.csv?")
    print(f"\n=== FINAL ANSWER ===\n{result.text}")
    print(f"\n[run] stop_reason={result.stop_reason} steps={result.steps} "
          f"duration={result.duration:.1f}s tokens={result.usage.total_tokens} "
          f"(in {result.usage.input_tokens} / out {result.usage.output_tokens})")


if __name__ == "__main__":
    main()
