"""The hook bus — observe the agent loop without modifying it.

In Weeks 1-2 the only way to watch the loop was to *copy* it into a demo script
(walking_skeleton_trace.py, multistep_trace.py both re-implement run()). That's a smell:
to observe behavior you had to duplicate it. The hook bus fixes that.

A hook is just a function you register against a named lifecycle event. The agent loop
"fires" events at well-defined points; every registered callback runs. Nothing about the
loop's logic changes — hooks only observe (and, for some events, can veto; see the agent).

    bus = HookBus()
    bus.on("pre_tool_use", lambda call: print("about to run", call.name))
    bus.on("post_tool_use", lambda call, result: log(result))

Events fired by the agent (see core/agent.py):
    on_user_message(message)
    on_assistant_message(message)
    pre_tool_use(call)
    on_permission_decision(call, decision)
    post_tool_use(call, result)
    on_stop(final_text)

This is the standard publish/subscribe (observer) pattern. It's also the seam that real
harnesses expose for logging, metrics, tracing, and policy plugins.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable


class HookBus:
    def __init__(self) -> None:
        # event name -> list of callbacks, in registration order
        self._hooks: dict[str, list[Callable]] = defaultdict(list)

    def on(self, event: str, callback: Callable) -> None:
        """Register `callback` to run whenever `event` fires."""
        self._hooks[event].append(callback)

    def fire(self, event: str, *args, **kwargs) -> None:
        """Run every callback registered for `event`, in order.

        A misbehaving hook must never break the agent, so we swallow exceptions from
        callbacks (a hook is an observer; its failure is not the agent's failure).
        """
        for callback in self._hooks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception:  # noqa: BLE001 — an observer must not crash the subject
                pass

    def __len__(self) -> int:
        return sum(len(cbs) for cbs in self._hooks.values())
