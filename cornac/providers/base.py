"""The Provider interface.

This is the seam between cornac and any LLM backend. A provider does exactly one
thing: take the neutral conversation (a list of Messages) plus the available tool
schemas, call its backend, and return a single neutral assistant Message — which
may contain text, tool calls, or both.

Everything provider-specific (auth, URLs, message reshaping, tool-format quirks)
lives behind this interface. The agent loop never imports anthropic or talks HTTP;
it only ever calls `provider.complete(...)`. That discipline is what lets the exact
same agent, tools, and permissions run on Claude or a local Qwen with no changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from cornac.core.messages import Message


class Provider(ABC):
    @abstractmethod
    def complete(self, messages: list[Message], tools: list[dict]) -> Message:
        """Run one model turn.

        Args:
            messages: the full conversation so far, in neutral form. A system
                message may appear first; providers that take the system prompt as
                a separate parameter should hoist it out themselves.
            tools: neutral tool schemas (from ToolRegistry.schemas()) — each is
                {"name", "description", "input_schema"}. The provider reshapes these
                into its own tool format.

        Returns:
            A single assistant Message. If the model asked to use tools, its
            `tool_calls` is populated; otherwise `text` holds the final answer.
        """
        ...

    @property
    def name(self) -> str:
        """Human-readable provider name, for logs and the benchmark report."""
        return self.__class__.__name__
