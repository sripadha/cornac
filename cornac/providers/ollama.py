"""Ollama (local model) provider.

Talks to a local Ollama daemon's /api/chat endpoint over HTTP. Ollama uses the
OpenAI-flavored message shape, which differs from Anthropic's in a few ways that we
absorb here so the agent loop stays identical across providers:

  1. The system prompt is just a message with role="system" in the list (no hoisting).
  2. Tool results are their own messages with role="tool" (no folding into user turns).
  3. Tools are wrapped as {"type": "function", "function": {...}} and the argument
     schema is called "parameters", not "input_schema".
  4. Ollama does not always assign ids to tool calls, so we synthesize stable ones.

The fact that supporting a local 7B model takes one small file like this — and zero
changes to the agent, tools, or permissions — is the whole point of the Provider seam.
"""

from __future__ import annotations

import httpx

from cornac.core.messages import Message, ToolCall
from cornac.providers.base import Provider

DEFAULT_MODEL = "qwen2.5:7b-instruct-q4_K_M"
DEFAULT_HOST = "http://localhost:11434"


class OllamaProvider(Provider):
    def __init__(self, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST, timeout: float = 300.0):
        self.model = model
        self.host = host.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    def complete(self, messages: list[Message], tools: list[dict]) -> Message:
        payload: dict = {
            "model": self.model,
            "messages": self._to_ollama(messages),
            "stream": False,
            "options": {"temperature": 0},  # deterministic, for reproducible benchmarks
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["input_schema"],
                    },
                }
                for t in tools
            ]

        resp = self._client.post(f"{self.host}/api/chat", json=payload)
        resp.raise_for_status()
        return self._from_ollama(resp.json())

    # --- neutral -> Ollama ---------------------------------------------------

    def _to_ollama(self, messages: list[Message]) -> list[dict]:
        out: list[dict] = []
        for msg in messages:
            if msg.role in ("system", "user"):
                out.append({"role": msg.role, "content": msg.text})

            elif msg.role == "assistant":
                m: dict = {"role": "assistant", "content": msg.text or ""}
                if msg.tool_calls:
                    m["tool_calls"] = [
                        {"function": {"name": c.name, "arguments": c.arguments}}
                        for c in msg.tool_calls
                    ]
                out.append(m)

            elif msg.role == "tool":
                # Ollama matches tool results to calls by order/name, not by id.
                out.append({"role": "tool", "content": msg.text, "tool_name": msg.name})

        return out

    # --- Ollama -> neutral ---------------------------------------------------

    def _from_ollama(self, data: dict) -> Message:
        message = data.get("message", {})
        text = message.get("content", "") or ""

        tool_calls: list[ToolCall] = []
        for i, call in enumerate(message.get("tool_calls", []) or []):
            fn = call.get("function", {})
            # Ollama returns arguments already parsed into a dict (unlike OpenAI's
            # JSON-string). It usually omits an id, so we synthesize a stable one.
            call_id = call.get("id") or f"call_{i}_{fn.get('name', 'tool')}"
            tool_calls.append(
                ToolCall(id=call_id, name=fn.get("name", ""), arguments=dict(fn.get("arguments", {})))
            )

        return Message(role="assistant", text=text, tool_calls=tool_calls)
