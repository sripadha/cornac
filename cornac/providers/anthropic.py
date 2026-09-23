"""Anthropic (Claude) provider.

Translates the neutral conversation into Anthropic's Messages API shape and parses
the response back into a neutral assistant Message. Two Anthropic-specific quirks
are handled here so the rest of cornac never has to know about them:

  1. The system prompt is a separate top-level parameter, not a message in the list.
  2. Tool results must be delivered inside a *user* message as `tool_result` content
     blocks — so our neutral role="tool" messages get folded into user turns here.

Week 4 adds the two bits of metadata the benchmark needs — token usage and the stop
reason — which the API hands back on every response, and turns on the SDK's own
retry logic so a flaky connection does not sink a long benchmark run.
"""

from __future__ import annotations

import os

from cornac.core.messages import Message, ToolCall, Usage
from cornac.providers.base import Provider

DEFAULT_MODEL = "claude-haiku-4-5-20251001"


class AnthropicProvider(Provider):
    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 2048, api_key: str | None = None):
        # Imported lazily so cornac is usable with only the Ollama provider installed.
        import anthropic

        self.model = model
        self.max_tokens = max_tokens
        # Unlike the Ollama provider, we don't write a retry loop here: the Anthropic
        # SDK has one built in. With max_retries=3 it re-sends on connection errors,
        # 429 rate limits and 5xx server errors, with exponential backoff between
        # attempts, and gives up after the third retry. Constructing the client does
        # not open a connection — nothing touches the network until complete() runs.
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            max_retries=3,
        )

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
                # The API rejects a text block that is empty or whitespace-only, so
                # only real text becomes a block.
                if msg.text and msg.text.strip():
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
                # The API also rejects an assistant message with no content at all
                # ({"role": "assistant", "content": []}). That shape is reachable: a
                # reply of "" or one whose only blocks were of a kind _from_anthropic
                # ignores. It bites on the NEXT complete() — when the whole history
                # is re-sent — so an Agent reused for a second run() would fail. An
                # empty turn carries no information, so it is simply left out.
                # (Consecutive same-role turns are fine: the API merges them.)
                if content:
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

        # A real API response always carries `usage`, but a hand-built one in a test
        # (or a future response type) may not — so read it defensively, and leave
        # Message.usage as None when there is genuinely nothing to report.
        usage = None
        api_usage = getattr(response, "usage", None)
        if api_usage is not None:
            usage = Usage(
                input_tokens=getattr(api_usage, "input_tokens", 0) or 0,
                output_tokens=getattr(api_usage, "output_tokens", 0) or 0,
            )

        return Message(
            role="assistant",
            text="".join(text_parts),
            tool_calls=tool_calls,
            usage=usage,
            # Anthropic's own word for why it stopped: "end_turn", "tool_use", or
            # "max_tokens" (cut off — worth knowing when a run looks truncated).
            stop_reason=getattr(response, "stop_reason", None),
        )
