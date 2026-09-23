"""What a finished `Agent.run()` hands back.

Through Week 3, run() returned a bare string. That was fine while the only reader was
a human at a terminal, but a string can't answer the questions the next steps ask:

  - Did the model actually finish, or did the loop give up at max_steps? (Week 3
    encoded that as a magic sentinel string — which a caller could only detect by
    string-matching, and which a sub-agent would happily pass off as a real answer.)
  - How many model calls did it take? How long? How many tokens?

The benchmark needs those numbers to build its success-rate table, and a parent agent
needs `stop_reason` to know whether a sub-agent's reply is trustworthy. So run() now
returns this small record instead. `str(result)` still gives you the text, so
`print(agent.run(...))` keeps working as before.
"""

from __future__ import annotations

from dataclasses import dataclass

from cornac.core.messages import Usage


@dataclass
class RunResult:
    # The model's final answer. None unless stop_reason == "done" — a run that hit
    # max_steps has no final answer, and we refuse to fake one.
    text: str | None
    # Why the loop ended: "done" (the model stopped calling tools) or "max_steps"
    # (the safety rail tripped). Distinct from Message.stop_reason, which is the
    # *provider's* reason for ending one single generation.
    stop_reason: str
    steps: int        # number of provider.complete() calls this run made
    duration: float   # wall-clock seconds, start of run() to return
    usage: Usage      # token counts summed over every model call in this run

    def __str__(self) -> str:
        return self.text or ""
