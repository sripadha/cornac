"""Provider-neutral conversation types.

These are the *only* shapes the agent loop and tools ever touch. Each provider
(Anthropic, Ollama, ...) is responsible for translating between these neutral
types and its own wire format. Keeping this layer tiny and provider-free is what
makes cornac genuinely provider-agnostic instead of an Anthropic SDK wrapper with
an `if provider == "..."` branch bolted on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# The four roles in a conversation. We use plain strings (not an enum) because
# every provider's wire format also uses these exact strings, so there is nothing
# to translate — one less moving part.
Role = str  # one of: "system" | "user" | "assistant" | "tool"


@dataclass
class ToolCall:
    """A request *from the model* to run a tool.

    The model never runs anything itself — it emits one of these, and the harness
    decides whether and how to execute it. `id` ties the eventual result back to
    this specific call (a single assistant turn may emit several).
    """

    id: str
    name: str
    arguments: dict


@dataclass
class ToolResult:
    """The outcome of running a tool, to be fed back to the model.

    `is_error` lets the model see that something failed (bad args, denied by
    policy, exception) and adapt, rather than the harness silently swallowing it.
    """

    tool_call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass
class Message:
    """One turn in the conversation, in neutral form.

    - role="system"     -> instructions (providers may hoist this out of the list)
    - role="user"       -> human input; `text` holds it
    - role="assistant"  -> model output; `text` and/or `tool_calls` are populated
    - role="tool"       -> a tool result being returned to the model; built from a
                           ToolResult via `Message.from_tool_result`
    """

    role: Role
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)

    # Only set when role == "tool":
    tool_call_id: str | None = None
    name: str | None = None  # the tool's name; some providers want it echoed back
    is_error: bool = False

    @classmethod
    def user(cls, text: str) -> Message:
        return cls(role="user", text=text)

    @classmethod
    def system(cls, text: str) -> Message:
        return cls(role="system", text=text)

    @classmethod
    def from_tool_result(cls, result: ToolResult) -> Message:
        return cls(
            role="tool",
            text=result.content,
            tool_call_id=result.tool_call_id,
            name=result.name,
            is_error=result.is_error,
        )

    @property
    def wants_tools(self) -> bool:
        """True if this (assistant) message is asking to run one or more tools."""
        return bool(self.tool_calls)
