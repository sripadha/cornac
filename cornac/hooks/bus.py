"""The hook bus — observe the agent loop without modifying it.

In Weeks 1-2 the only way to watch the loop was to *copy* it into a demo script
(walking_skeleton_trace.py, multistep_trace.py both re-implement run()). That's a smell:
to observe behavior you had to duplicate it. The hook bus fixes that.

A hook is just a function you register against a named lifecycle event. The agent loop
"fires" events at well-defined points; every registered callback runs. Nothing about the
loop's logic changes — hooks only observe, with one deliberate exception: a
`pre_tool_use` hook can return Decision.DENY to veto the call (see core/agent.py). To
make that possible, fire() collects and returns every callback's return value, in
registration order.

    bus = HookBus()
    bus.on("pre_tool_use", lambda call: print("about to run", call.name))
    bus.on("post_tool_use", lambda call, result: log(result))

Events fired by the agent (see core/agent.py):
    on_user_message(message)
    on_assistant_message(message)
    on_nudge(text, n)                     -> the model described a step and stopped
                                             (text); the loop is sending nudge number n
                                             (Week 4b). The nudge is a harness message,
                                             not the user's, so it does not also fire
                                             on_user_message.
    pre_tool_use(call)                    -> return Decision.DENY to veto the call
    on_permission_decision(call, decision)
    post_tool_use(call, result)
    on_stuck(call, n)                     -> the model made this identical call, for the
                                             identical result, n times (max_repeats); the
                                             run is ending with stop_reason "stuck"
                                             (Week 4c). The note the result carried is
                                             advisory; this is the hard stop.
    on_context_clear(cleared, freed_chars)-> before a model call the loop replaced
                                             `cleared` old tool results with a one-line
                                             note, freeing `freed_chars` characters, to
                                             keep the prompt inside the context window
                                             (Week 4c). Fired only when at least one
                                             result was cleared.
    on_wrap_up(summary)                   -> a run that gave up (max_steps or stuck) was
                                             asked, with the tools off, for a four-line
                                             account of its partial work; `summary` is
                                             that text, or None if it said nothing
                                             (Week 4c). Not fired when the wrap-up is
                                             disabled or its call failed. Like the
                                             nudge, the exchange is the harness's, so it
                                             fires neither on_user_message nor
                                             on_assistant_message.
    on_stop(final_text)                   -> final_text is None if the run gave up
                                             (max_steps or stuck); the wrap-up, if any,
                                             has already happened by then
    on_run_end(result)                    -> fired LAST, for EVERY run — done, max_steps
                                             or stuck — with the RunResult run() is about
                                             to return (Week 4c). One event a transcript
                                             writer can rely on to close a run.

Fired by the spawn_agent tool on the parent's bus (Week 4b-B; see
tools/builtin/spawn.py):
    on_spawn(task, depth)                 -> a sub-agent is about to run
    on_spawn_done(result, depth)          -> it returned; result is its RunResult, or
                                             None if it raised

A sub-agent runs on a bus of its own, and what reaches the parent's bus from it is
decided by what the event is about. The tool-call events — pre_tool_use,
on_permission_decision, post_tool_use — are forwarded, so a veto or an auditor
registered on the root agent applies to, and sees, every tool call in the whole
delegation tree; pre_tool_use is forwarded as a gate, so a DENY (or a crash) on the
parent's bus vetoes the child's call too. The conversation events — on_user_message,
on_assistant_message, on_nudge, on_stop, and the Week 4c on_stuck, on_context_clear,
on_wrap_up and on_run_end — are not forwarded: they describe one agent's own dialogue
and one agent's own run, and a child's on_stop (or on_run_end) is not the parent's.

Two rules govern a callback that raises:

  1. It must never crash the agent. A hook is an observer; its failure is not the
     agent's failure, so the exception is caught and the loop carries on.
  2. It must not be *silent* either. Week 3 swallowed hook errors with `pass`, which
     meant a broken logger simply logged nothing and nobody noticed — a silent
     observer lies by omission. Now the error is printed to stderr (stderr, so it can
     never be mistaken for the agent's own output) and the callback's slot in the
     returned list holds None.

There is one more rule, for the one event whose answers *decide* something. A
pre_tool_use hook is not only an observer — it can be the rule that stops a call
("no bash while the tests are red"). If such a hook crashes, "None" is the wrong
answer: the agent would read it as "no objection" and run the call the hook was
written to block. A crashed gatekeeper must fail CLOSED. So the agent fires
pre_tool_use through gate() instead of fire(): identical, except a callback that
raises contributes HOOK_FAILED — a marker the agent treats as a veto. The error still
goes to stderr, and the loop still never crashes.

This is the standard publish/subscribe (observer) pattern. It's also the seam that real
harnesses expose for logging, metrics, tracing, and policy plugins.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from typing import Callable

# What a raised callback contributes to a gate() result. Deliberately not None, so a
# gate can tell "the hook had nothing to say" apart from "the hook broke".
HOOK_FAILED = object()


class HookBus:
    def __init__(self) -> None:
        # event name -> list of callbacks, in registration order
        self._hooks: dict[str, list[Callable]] = defaultdict(list)

    def on(self, event: str, callback: Callable) -> None:
        """Register `callback` to run whenever `event` fires."""
        self._hooks[event].append(callback)

    def fire(self, event: str, *args, **kwargs) -> list:
        """Run every callback registered for `event`, in order; return what they returned.

        The result has one entry per callback, in registration order. A callback that
        raised contributes None (and its error goes to stderr). No subscribers -> [].
        """
        return self._dispatch(event, args, kwargs, failed=None)

    def gate(self, event: str, *args, **kwargs) -> list:
        """Like fire(), for an event whose return values gate an action (pre_tool_use).

        The one difference: a callback that raised contributes HOOK_FAILED instead of
        None, so the caller can fail closed. Its error still goes to stderr.

        What counts as a veto is decided by the agent, which looks for Decision.DENY
        in the list this returns. Decision is a str Enum, so the bare string "deny"
        compares equal and vetoes as well — accepted on purpose, since it errs on the
        fail-closed side (see core/agent.py). Nothing else does: "DENY", "no", True
        and "allow" are all read as "no objection".
        """
        return self._dispatch(event, args, kwargs, failed=HOOK_FAILED)

    def _dispatch(self, event: str, args: tuple, kwargs: dict, failed) -> list:
        """The shared loop behind fire() and gate(); `failed` fills a raised callback's slot."""
        results: list = []
        for callback in self._hooks.get(event, []):
            try:
                results.append(callback(*args, **kwargs))
            except Exception as exc:  # noqa: BLE001 — an observer must not crash the subject
                # __qualname__ names the function, including its enclosing scope
                # ("MyTracer.on_call"); fall back to repr() for things like partials.
                who = getattr(callback, "__qualname__", repr(callback))
                print(
                    f"[cornac] hook '{event}' callback {who} raised {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                results.append(failed)
        return results

    def __len__(self) -> int:
        return sum(len(cbs) for cbs in self._hooks.values())
