"""The Tool contract.

A tool is any capability the model can invoke: read a file, run a shell command,
search the web, spawn a sub-agent. Every tool exposes the same four things:

  - name          : how the model refers to it
  - description   : what it does (the model reads this to decide when to use it)
  - input_schema  : a JSON Schema describing its arguments (so the model knows how
                    to call it, and so we can validate before executing)
  - run(input)    : the actual implementation

Providers serialize (name, description, input_schema) into whatever tool format
their API expects. The harness calls run() after permission checks pass.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Callable

from cornac.core.messages import ToolResult


class Tool(ABC):
    name: str
    description: str
    input_schema: dict  # JSON Schema (an object schema describing the arguments)

    @abstractmethod
    def run(self, arguments: dict) -> str:
        """Execute the tool and return its result as a string.

        Raising is fine — the harness catches exceptions and turns them into an
        error ToolResult so the model can see what went wrong and recover.
        """
        ...

    def to_result(self, tool_call_id: str, content: str, is_error: bool = False) -> ToolResult:
        return ToolResult(
            tool_call_id=tool_call_id,
            name=self.name,
            content=content,
            is_error=is_error,
        )


class _FunctionTool(Tool):
    """A Tool backed by a plain Python function (created via the @tool decorator)."""

    def __init__(self, fn: Callable[..., str], name: str, description: str, input_schema: dict):
        self._fn = fn
        self.name = name
        self.description = description
        self.input_schema = input_schema

    def run(self, arguments: dict) -> str:
        result = self._fn(**arguments)
        return result if isinstance(result, str) else str(result)


# Maps Python type annotations to JSON Schema types, for the @tool convenience path.
_PY_TO_JSON = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def tool(name: str | None = None, description: str | None = None) -> Callable[[Callable], Tool]:
    """Decorator that turns a plain function into a Tool.

    It builds the JSON Schema from the function's signature: each parameter becomes
    a property, its type annotation becomes the JSON type, and a parameter without a
    default becomes required. The function's docstring becomes the description unless
    one is given explicitly.

        @tool()
        def get_current_time(timezone: str = "UTC") -> str:
            "Return the current time in the given IANA timezone."
            ...

    For tools with richer argument validation, subclass Tool directly and hand-write
    input_schema instead.
    """

    def decorator(fn: Callable[..., str]) -> Tool:
        sig = inspect.signature(fn)
        properties: dict = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            annotation = param.annotation if param.annotation is not inspect.Parameter.empty else str
            json_type = _PY_TO_JSON.get(annotation, "string")
            properties[param_name] = {"type": json_type}
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        schema = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required

        return _FunctionTool(
            fn=fn,
            name=name or fn.__name__,
            description=description or (inspect.getdoc(fn) or "").strip(),
            input_schema=schema,
        )

    return decorator
