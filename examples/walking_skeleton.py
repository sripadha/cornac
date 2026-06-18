"""Week 1 walking skeleton.

Proves the whole loop works end to end with a single tool, on either provider —
without changing a single line of agent code between them. Run it like:

    # Local model (needs `ollama serve` running + the model pulled):
    python examples/walking_skeleton.py ollama

    # Claude (needs ANTHROPIC_API_KEY in the environment):
    python examples/walking_skeleton.py anthropic

The question deliberately needs the current wall-clock time, which the model cannot
know from training — so a correct answer is proof the tool round-trip happened.
"""

import sys

from cornac import Agent, ToolRegistry
from cornac.tools.builtin.clock import get_current_time


def build_provider(which: str):
    if which == "anthropic":
        from cornac.providers.anthropic import AnthropicProvider

        return AnthropicProvider()
    elif which == "ollama":
        from cornac.providers.ollama import OllamaProvider

        return OllamaProvider()
    raise SystemExit(f"unknown provider {which!r}; use 'anthropic' or 'ollama'")


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "ollama"
    provider = build_provider(which)

    agent = Agent(
        provider=provider,
        registry=ToolRegistry([get_current_time]),
        system_prompt=(
            "You are a concise assistant. When a question depends on the current "
            "date or time, use the get_current_time tool rather than guessing."
        ),
    )

    question = "What is the current time in Tokyo? Answer in one sentence."
    print(f"\n[provider] {provider.name}")
    print(f"[user]     {question}\n")

    answer = agent.run(question)

    print(f"[agent]    {answer}\n")
    print(f"[debug]    conversation had {len(agent.messages)} messages "
          f"(system + user + assistant turns + tool results)")


if __name__ == "__main__":
    main()
