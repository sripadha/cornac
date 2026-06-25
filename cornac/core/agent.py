"""The agent loop — the heart of the harness.

Strip away every feature and an agent is still this:

    while True:
        response = provider.complete(messages, tools)
        messages.append(response)
        if not response.wants_tools:
            return response.text          # model is done talking
        for call in response.tool_calls:
            result = registry.execute(call)
            messages.append(result)       # feed the outcome back, loop again

Week 3 slots two cross-cutting concerns into that exact shape WITHOUT changing it:

  - permissions: before each tool runs, ask the policy ALLOW / DENY / ASK. A DENY
    becomes an error result the model can adapt to — the tool never runs.
  - hooks: fire named lifecycle events so observers (loggers, tracers, the demo) can
    watch the loop without copying it.

Both are optional. With no policy and no hooks, run() behaves exactly as in Week 1.
"""

from __future__ import annotations

from typing import Callable

from cornac.core.messages import Message, ToolResult
from cornac.hooks.bus import HookBus
from cornac.permissions.policy import Decision, Policy
from cornac.providers.base import Provider
from cornac.tools.registry import ToolRegistry

# An "approver" is anything that, given a ToolCall the policy was unsure about, returns
# a final Decision (ALLOW/DENY). cli_ask is the default; a UI or test can supply its own.
Approver = Callable[..., Decision]


class Agent:
    def __init__(
        self,
        provider: Provider,
        registry: ToolRegistry | None = None,
        system_prompt: str | None = None,
        max_steps: int = 20,
        policy: Policy | None = None,
        approver: Approver | None = None,
        hooks: HookBus | None = None,
    ):
        self.provider = provider
        self.registry = registry or ToolRegistry()
        self.system_prompt = system_prompt
        self.max_steps = max_steps  # a safety rail against an agent looping forever
        self.policy = policy        # None -> every tool is allowed (Week 1 behavior)
        self.approver = approver    # None -> an ASK is treated as DENY (safe default)
        self.hooks = hooks or HookBus()
        self.messages: list[Message] = []
        if system_prompt:
            self.messages.append(Message.system(system_prompt))

    def run(self, user_message: str) -> str:
        """Run the agent to completion on one user message and return its final text."""
        user = Message.user(user_message)
        self.messages.append(user)
        self.hooks.fire("on_user_message", user)

        for _ in range(self.max_steps):
            response = self.provider.complete(self.messages, self.registry.schemas())
            self.messages.append(response)
            self.hooks.fire("on_assistant_message", response)

            # No tool calls -> the model has produced its final answer.
            if not response.wants_tools:
                self.hooks.fire("on_stop", response.text)
                return response.text

            # Otherwise handle every requested tool, gated by permissions.
            for call in response.tool_calls:
                self.hooks.fire("pre_tool_use", call)
                result = self._handle_call(call)
                self.hooks.fire("post_tool_use", call, result)
                self.messages.append(Message.from_tool_result(result))

        self.hooks.fire("on_stop", None)
        return (
            "[cornac] Stopped after reaching max_steps "
            f"({self.max_steps}) without a final answer."
        )

    def _handle_call(self, call) -> ToolResult:
        """Apply the permission policy, then execute (or refuse) the tool call."""
        decision = self.policy.check(call) if self.policy else Decision.ALLOW

        # ASK means the policy is unsure -> defer to the human/approver. With no
        # approver wired up, the safe default is to deny rather than run blindly.
        if decision == Decision.ASK:
            decision = self.approver(call, self.policy) if self.approver else Decision.DENY

        self.hooks.fire("on_permission_decision", call, decision)

        if decision == Decision.DENY:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=f"Permission denied: the user did not allow {call.name}.",
                is_error=True,
            )
        return self.registry.execute(call)
