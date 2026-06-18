"""The agent loop — the heart of the harness.

Strip away every feature and an agent is this:

    while True:
        response = provider.complete(messages, tools)
        messages.append(response)
        if not response.wants_tools:
            return response.text          # model is done talking
        for call in response.tool_calls:
            result = registry.execute(call)
            messages.append(result)       # feed the outcome back, loop again

Everything cornac adds later — permissions, hooks, sub-agents — slots into this
loop at well-defined points without changing its shape. This file keeps the loop
honest and readable; the layers live in their own modules.
"""

from __future__ import annotations

from cornac.core.messages import Message
from cornac.providers.base import Provider
from cornac.tools.registry import ToolRegistry


class Agent:
    def __init__(
        self,
        provider: Provider,
        registry: ToolRegistry | None = None,
        system_prompt: str | None = None,
        max_steps: int = 20,
    ):
        self.provider = provider
        self.registry = registry or ToolRegistry()
        self.system_prompt = system_prompt
        self.max_steps = max_steps  # a safety rail against an agent looping forever
        self.messages: list[Message] = []
        if system_prompt:
            self.messages.append(Message.system(system_prompt))

    def run(self, user_message: str) -> str:
        """Run the agent to completion on one user message and return its final text.

        The conversation accumulates on `self.messages`, so calling run() again
        continues the same conversation rather than starting fresh.
        """
        self.messages.append(Message.user(user_message))

        for _ in range(self.max_steps):
            response = self.provider.complete(self.messages, self.registry.schemas())
            self.messages.append(response)

            # No tool calls -> the model has produced its final answer.
            if not response.wants_tools:
                return response.text

            # Otherwise execute every requested tool and feed the results back.
            for call in response.tool_calls:
                result = self.registry.execute(call)
                self.messages.append(Message.from_tool_result(result))

        # Hit the step ceiling without the model settling on a final answer.
        return (
            "[cornac] Stopped after reaching max_steps "
            f"({self.max_steps}) without a final answer."
        )
