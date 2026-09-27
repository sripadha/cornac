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
    # The model's context window in tokens, or None if the provider does not know
    # (Week 4c). The agent loop uses it to decide when old tool results must be
    # cleared to make room; a subclass sets it from its configuration (Ollama's
    # num_ctx, a model's documented limit) or leaves it None to disable clearing.
    context_window: int | None = None

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

    def complete_without_tools(self, messages: list[Message], tools: list[dict]) -> Message:
        """Run one model turn in which the model may NOT call a tool (Week 4c).

        The agent loop uses this for the wrap-up: a run that gave up is asked, once,
        to account for itself, and that call must not be able to start another loop.

        `tools` are the schemas the conversation was held with, NOT the tools to
        offer. The default simply drops them and asks for a plain text turn, which
        is what Ollama and the OpenAI-compatible servers accept: a history that
        contains tool calls and tool results with no tool definitions attached is
        fine with them. It is not fine with every backend. Anthropic's API rejects
        a request whose messages carry tool_use/tool_result blocks unless `tools`
        is defined ("Requests which include tool_use or tool_result blocks must
        define tools"), so AnthropicProvider overrides this to keep sending the
        definitions and forbid their use with tool_choice instead. The seam exists
        so the loop can say "no tools this turn" without knowing which of those two
        the backend needs.
        """
        return self.complete(messages, [])

    @property
    def name(self) -> str:
        """Human-readable provider name, for logs and the benchmark report."""
        return self.__class__.__name__
