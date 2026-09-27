"""Provider-neutral conversation types.

These are the *only* shapes the agent loop and tools ever touch. Each provider
(Anthropic, Ollama, ...) is responsible for translating between these neutral
types and its own wire format. Keeping this layer tiny and provider-free is what
makes cornac genuinely provider-agnostic instead of an Anthropic SDK wrapper with
an `if provider == "..."` branch bolted on.

Week 4 adds two small pieces of *metadata* to an assistant Message — how many tokens
the call cost (`usage`) and why the model stopped (`stop_reason`). They live here, in
neutral form, for the same reason everything else does: the benchmark and the agent
need to read them without caring which backend produced them.
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
class Usage:
    """Token counts for one model call (or, summed, for a whole run).

    Tokens are the unit of cost and latency in LLM land, so every serious harness
    reports them. Providers fill this in on the assistant Message they return; the
    agent adds them up across the run. `a + b` returns a NEW Usage — neither operand
    is modified, which keeps per-call and per-run totals from leaking into each other.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    # Prompt-cache accounting (Week 4c). Anthropic bills a cached prefix separately
    # from `input_tokens`: tokens read from the cache come back as cache_read_tokens
    # and tokens written into it as cache_write_tokens, and `input_tokens` then holds
    # only the uncached remainder. Providers without a cache leave both at 0, so for
    # them prompt_tokens == input_tokens and nothing changes.
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )

    @property
    def total_tokens(self) -> int:
        """Everything the run was billed for, cached or not."""
        return self.prompt_tokens + self.output_tokens

    @property
    def prompt_tokens(self) -> int:
        """How big the prompt actually was, cache included.

        This is the number to compare against a model's context window: a cached
        prefix still occupies the window even though it is billed at a discount.
        """
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens


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

    # Only set (by the provider) when role == "assistant". Both are optional so a
    # hand-built or scripted Message needs neither.
    usage: Usage | None = None
    # The backend's own word for why generation ended — Anthropic says "end_turn" /
    # "tool_use" / "max_tokens"; Ollama's done_reason says "stop" / "length". We keep
    # it verbatim rather than inventing a neutral enum: it's for humans reading logs
    # and for the benchmark, and a lossy translation would hide exactly the detail
    # (e.g. "the model got cut off") you want when debugging.
    stop_reason: str | None = None

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
