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

    result = agent.run(question)

    # run() hands back a RunResult, not a string. The answer is result.text; the run's
    # cost (model calls, seconds, tokens) and why it stopped ride along with it.
    print(f"[agent]    {result.text}\n")
    print(f"[run]      stop_reason={result.stop_reason} steps={result.steps} "
          f"duration={result.duration:.1f}s tokens={result.usage.total_tokens} "
          f"(in {result.usage.input_tokens} / out {result.usage.output_tokens})")
    print(f"[debug]    conversation had {len(agent.messages)} messages "
          f"(system + user + assistant turns + tool results)")


if __name__ == "__main__":
    main()
