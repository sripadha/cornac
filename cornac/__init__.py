"""cornac — a from-scratch, provider-agnostic LLM agent harness.

The agent isn't the model; it's the harness. cornac wraps any LLM (Claude via the
Anthropic API, or a local model via Ollama) in an agent loop with tools, and — as the
project grows — permissions, hooks, and sub-agents.

Quick start:

    from cornac import Agent, ToolRegistry
    from cornac.providers.ollama import OllamaProvider
    from cornac.tools.builtin.clock import get_current_time

    agent = Agent(
        provider=OllamaProvider(),
        registry=ToolRegistry([get_current_time]),
        system_prompt="You are a helpful assistant.",
    )
    result = agent.run("What time is it in Tokyo right now?")
    print(result.text)            # or just print(result); str() gives the text
    print(result.stop_reason)     # "done", or "max_steps" if the loop gave up
    print(result.steps, result.usage.total_tokens)
"""

from cornac.core.agent import Agent
from cornac.core.messages import Message, ToolCall, ToolResult, Usage
from cornac.core.result import RunResult
from cornac.hooks.bus import HookBus
from cornac.permissions.policy import Decision, Policy
from cornac.permissions.prompt import cli_ask
from cornac.tools.base import Tool, tool
from cornac.tools.registry import ToolRegistry

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "Message",
    "ToolCall",
    "ToolResult",
    "Usage",
    "RunResult",
    "Tool",
    "tool",
    "ToolRegistry",
    "Policy",
    "Decision",
    "cli_ask",
    "HookBus",
    "__version__",
]
