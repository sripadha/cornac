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

Week 4 changes two things, again without touching the loop's shape:

  - run() returns a RunResult instead of a bare string. A string could not tell
    "the model finished" apart from "the loop gave up at max_steps" — Week 3 faked it
    with a sentinel string, which a sub-agent would have handed to its parent as if it
    were a real answer. The benchmark and sub-agents also need to know how many steps a
    run took, how long, and how many tokens. RunResult carries all of that; str(result)
    is still the text, so printing it works as before.

  - a pre_tool_use hook can VETO a tool call by returning Decision.DENY. This is the
    hook bus growing from a logger into a policy extension point — the same thing
    Claude Code's PreToolUse hooks do when they block a call. It lets you enforce rules
    the static policy can't express ("no writes after 5pm", "no bash while the tests
    are red") without editing the harness. A veto is checked BEFORE the policy and the
    approver, so a vetoed call never bothers the human with a prompt.

    Because a veto hook is a gatekeeper and not just a logger, it fails CLOSED: a
    pre_tool_use hook that crashes counts as a veto. The alternative — treating a
    crashed hook as "no objection" — would let the model slip past a rule simply by
    sending an argument shape the hook didn't expect. See hooks/bus.py (gate()).

    The permission gate fails closed for the same reason. A tool runs only on an
    explicit Decision.ALLOW; an approver that returns None, ASK, "yes" or anything
    else has not said yes, and "the gate wasn't sure" must mean "refused", never
    "ran". (Decision is a str Enum, so the bare string "deny" also counts as a veto —
    accepted, since it errs on the safe side; see _handle_call.)
"""

from __future__ import annotations

import time
from typing import Callable

from cornac.core.messages import Message, ToolCall, ToolResult, Usage
from cornac.core.result import RunResult
from cornac.hooks.bus import HOOK_FAILED, HookBus
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

    def run(self, user_message: str) -> RunResult:
        """Run the agent to completion on one user message.

        Returns a RunResult: the final text (None if the loop hit max_steps), why it
        stopped, and the run's cost in steps, seconds, and tokens.
        """
        started = time.perf_counter()
        steps = 0          # one per provider.complete() call
        usage = Usage()    # running token total; providers may leave Message.usage None

        user = Message.user(user_message)
        self.messages.append(user)
        self.hooks.fire("on_user_message", user)

        for _ in range(self.max_steps):
            response = self.provider.complete(self.messages, self.registry.schemas())
            steps += 1
            if response.usage is not None:
                usage = usage + response.usage
            self.messages.append(response)
            self.hooks.fire("on_assistant_message", response)

            # No tool calls -> the model has produced its final answer.
            if not response.wants_tools:
                self.hooks.fire("on_stop", response.text)
                return RunResult(
                    text=response.text,
                    stop_reason="done",
                    steps=steps,
                    duration=time.perf_counter() - started,
                    usage=usage,
                )

            # Otherwise handle every requested tool, gated by hooks and permissions.
            for call in response.tool_calls:
                result = self._handle_call(call)
                self.hooks.fire("post_tool_use", call, result)
                self.messages.append(Message.from_tool_result(result))

        # The safety rail tripped. There is no final answer, and we say so honestly
        # (text=None) instead of inventing one the caller might mistake for the model's.
        self.hooks.fire("on_stop", None)
        return RunResult(
            text=None,
            stop_reason="max_steps",
            steps=steps,
            duration=time.perf_counter() - started,
            usage=usage,
        )

    def _handle_call(self, call: ToolCall) -> ToolResult:
        """Run one tool call through its gates — hooks, then policy — and execute or refuse it.

        Both gates fail CLOSED: the tool runs only when a gate says so explicitly, and
        any other answer — including a broken or unexpected one — is a refusal. The two
        ways a gate can be wrong are not symmetric. A wrongly refused call costs one
        round-trip: the model reads the error and rephrases. A wrongly run call cannot
        be taken back. So whenever a gate's answer is unclear, refusing is the only
        mistake the harness can afford to make.
        """
        # Gate 1: hooks. gate() returns every callback's return value; if any of them
        # said DENY — or crashed, which a gatekeeper must treat the same way — the
        # call is vetoed right here. The policy and approver are never consulted, so
        # a vetoed call never prompts the human.
        #
        # Decision is a str Enum, so the bare string "deny" compares equal to
        # Decision.DENY and vetoes too. That is accepted rather than accidental: it
        # errs on the fail-closed side, and lets a hook that never imported Decision
        # (a bridge to a shell script, say) still block a call. No other spelling
        # counts — "DENY", "no", "block" and False are all read as "no objection".
        vetoes = self.hooks.gate("pre_tool_use", call)
        if Decision.DENY in vetoes or HOOK_FAILED in vetoes:
            self.hooks.fire("on_permission_decision", call, Decision.DENY)
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=f"Vetoed by hook: {call.name} was blocked by a pre_tool_use hook.",
                is_error=True,
            )

        # Gate 2: the permission policy (None -> everything is allowed, as in Week 1).
        decision = self.policy.check(call) if self.policy else Decision.ALLOW

        # ASK means the policy is unsure -> defer to the human/approver. With no
        # approver wired up, the safe default is to deny rather than run blindly.
        if decision == Decision.ASK:
            decision = self.approver(call, self.policy) if self.approver else Decision.DENY

        # The tool runs on an explicit ALLOW and on nothing else. The first version
        # checked `== DENY` and executed on everything else, so an approver that
        # forgot its return statement (None), handed ASK back, or answered "yes"
        # silently ran a gated tool. A policy or approver that says anything but
        # ALLOW/DENY has a bug — and the safe way to surface a bug in a gate is to
        # refuse the call, never to run it. Observers are told the effective
        # decision (DENY), not the odd value that produced it.
        allowed = decision == Decision.ALLOW
        self.hooks.fire(
            "on_permission_decision", call, Decision.ALLOW if allowed else Decision.DENY
        )

        if not allowed:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=f"Permission denied: the user did not allow {call.name}.",
                is_error=True,
            )
        return self.registry.execute(call)
