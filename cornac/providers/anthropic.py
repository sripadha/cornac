"""Anthropic (Claude) provider.

Translates the neutral conversation into Anthropic's Messages API shape and parses
the response back into a neutral assistant Message. Two Anthropic-specific quirks
are handled here so the rest of cornac never has to know about them:

  1. The system prompt is a separate top-level parameter, not a message in the list.
  2. Tool results must be delivered inside a *user* message as `tool_result` content
     blocks — so our neutral role="tool" messages get folded into user turns here.
"""

from __future__ import annotations

import os

from cornac.core.messages import Message, ToolCall
from cornac.providers.base import Provider

DEFAULT_MODEL = "claude-haiku-4-5-20251001"


class AnthropicProvider(Provider):
    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 2048, api_key: str | None = None):
        # Imported lazily so cornac is usable with only the Ollama provider installed.
        import anthropic

        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    @property
    def name(self) -> str:
        return f"anthropic:{self.model}"

    def complete(self, messages: list[Message], tools: list[dict]) -> Message:
        system, api_messages = self._to_anthropic(messages)
        api_tools = [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["input_schema"],
            }
            for t in tools
        ]

        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": api_messages,
        }
        if system:
            kwargs["system"] = system
        if api_tools:
            kwargs["tools"] = api_tools

        response = self._client.messages.create(**kwargs)
        return self._from_anthropic(response)

    # --- neutral -> Anthropic ------------------------------------------------

    def _to_anthropic(self, messages: list[Message]) -> tuple[str, list[dict]]:
        system_parts: list[str] = []
        api_messages: list[dict] = []

        for msg in messages:
            if msg.role == "system":
                system_parts.append(msg.text)

            elif msg.role == "user":
                api_messages.append({"role": "user", "content": msg.text})

            elif msg.role == "assistant":
                content: list[dict] = []
                if msg.text:
                    content.append({"type": "text", "text": msg.text})
                for call in msg.tool_calls:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.name,
                            "input": call.arguments,
                        }
                    )
                api_messages.append({"role": "assistant", "content": content})

            elif msg.role == "tool":
                # Anthropic wants tool results inside a user message. Merge into the
                # previous user message if it's already a tool-result turn, so that
                # all results for one assistant turn arrive together.
                block = {
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": msg.text,
                }
                if msg.is_error:
                    block["is_error"] = True

                if (
                    api_messages
                    and api_messages[-1]["role"] == "user"
                    and isinstance(api_messages[-1]["content"], list)
                ):
                    api_messages[-1]["content"].append(block)
                else:
                    api_messages.append({"role": "user", "content": [block]})

        return "\n\n".join(system_parts), api_messages

    # --- Anthropic -> neutral ------------------------------------------------

    def _from_anthropic(self, response) -> Message:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))

        return Message(role="assistant", text="".join(text_parts), tool_calls=tool_calls)
