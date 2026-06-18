"""Week 1 walking skeleton — TRACE edition.

Same agent, same single tool, but this version PRINTS the conversation growing
message by message so you can watch the loop happen — the live version of the
frame-by-frame trace (Frames 0-8).

    # Local model (needs `ollama serve` running + the model pulled):
    python examples/walking_skeleton_trace.py ollama

    # Claude (needs ANTHROPIC_API_KEY in the environment):
    python examples/walking_skeleton_trace.py anthropic

How it works: instead of calling agent.run() (which hides the loop), we run the
loop ourselves here and print after each step. This is exactly the loop from
cornac/core/agent.py — copied out so we can narrate it. In Week 3 we'll add a
real "hook" system so we can observe the loop WITHOUT copying it.
"""

import sys

from cornac import Agent, Message, ToolRegistry
from cornac.tools.builtin.clock import get_current_time


def build_provider(which: str):
    if which == "anthropic":
        from cornac.providers.anthropic import AnthropicProvider

        return AnthropicProvider()
    elif which == "ollama":
        from cornac.providers.ollama import OllamaProvider

        return OllamaProvider()
    raise SystemExit(f"unknown provider {which!r}; use 'anthropic' or 'ollama'")


def show(msg: Message) -> str:
    """One-line, human-readable rendering of a neutral Message."""
    if msg.role == "assistant" and msg.tool_calls:
        calls = ", ".join(f"{c.name}({c.arguments})" for c in msg.tool_calls)
        body = f"(no text) -> CALLS: {calls}"
    elif msg.role == "tool":
        flag = " [ERROR]" if msg.is_error else ""
        body = f"{msg.text!r}{flag}  (answering {msg.tool_call_id})"
    else:
        body = repr(msg.text)
    return f"  [{msg.role:9}] {body}"


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "ollama"
    provider = build_provider(which)

    agent = Agent(
        provider=provider,
        registry=ToolRegistry([get_current_time]),
        system_prompt=(
            "You are a concise assistant. When a question depends on the current "
            "date or time, "#use the get_current_time tool rather than guessing."
        ),
    )
    question = "What is the current time in France? Answer in one sentence."

    print(f"\n=== cornac walking-skeleton trace | provider: {provider.name} ===\n")
    print("FRAME 0-1  seed the conversation")
    agent.messages.append(Message.user(question))
    for m in agent.messages:
        print(show(m))

    # --- the agent loop, unrolled so we can print between steps ---
    step = 0
    while step < agent.max_steps:
        step += 1
        print(f"\nITERATION {step}")
        print("  (1) asking the model...")
        response = agent.provider.complete(agent.messages, agent.registry.schemas())
        agent.messages.append(response)
        print("  (2) model replied:")
        print(show(response))

        if not response.wants_tools:
            print("  (3) no tool calls -> model is DONE.")
            print(f"\n=== FINAL ANSWER ===\n{response.text}\n")
            break

        print(f"  (3) model wants {len(response.tool_calls)} tool(s) -> running them:")
        for call in response.tool_calls:
            result = agent.registry.execute(call)
            tool_msg = Message.from_tool_result(result)
            agent.messages.append(tool_msg)
            print("  (4)", show(tool_msg).strip())
    else:
        print(f"\n[stopped after {agent.max_steps} steps without a final answer]")

    print(f"[debug] conversation ended with {len(agent.messages)} messages")


if __name__ == "__main__":
    main()
