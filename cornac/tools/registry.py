"""The tool registry: name -> Tool dispatch.

The registry is the single place that (a) holds every tool the agent can use,
(b) hands the provider the tool schemas to advertise to the model, and (c) executes
a ToolCall by name, turning exceptions into error results instead of crashing the
loop. Permission checks live one layer up (in the agent), not here — the registry's
job is dispatch, not policy.
"""

from __future__ import annotations

from cornac.core.messages import ToolCall, ToolResult
from cornac.tools.base import Tool


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.add(t)

    def add(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name!r}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    # --- sub-agents (Week 4b-B) --------------------------------------------------
    # spawn_agent builds a child registry from a subset of its parent's tools, and
    # the agent introduces itself to every tool that wants it (see Agent.__init__).
    # Both need to walk the registry; these keep them from reaching into _tools.
    def names(self) -> list[str]:
        """Tool names in registration order — the order the model sees them in."""
        return list(self._tools)

    def tools(self) -> list[Tool]:
        """The Tool objects themselves, in registration order."""
        return list(self._tools.values())
    # --- end sub-agents ------------------------------------------------------------

    def schemas(self) -> list[dict]:
        """Neutral tool descriptions for the provider to serialize.

        We hand out a provider-neutral shape; each provider reshapes it into its
        own tool format (Anthropic's {name, description, input_schema} vs Ollama's
        {type: "function", function: {...}}).
        """
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in self._tools.values()
        ]

    def execute(self, call: ToolCall) -> ToolResult:
        """Run one tool call, never raising — failures come back as error results."""
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=f"Unknown tool: {call.name!r}. Available: {', '.join(self._tools)}",
                is_error=True,
            )
        try:
            content = tool.run(call.arguments)
            return tool.to_result(call.id, content)
        except Exception as exc:  # noqa: BLE001 — surface any failure back to the model
            return tool.to_result(call.id, f"{type(exc).__name__}: {exc}", is_error=True)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
