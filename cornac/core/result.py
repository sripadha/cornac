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

Week 4b adds `nudges`: how many times the loop had to push the model on after it
announced a step and stopped (see core/agent.py). The benchmark wants that number
because a run the harness had to rescue is a different data point from one it didn't.
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
    # Continue nudges the loop sent this run (Week 4b); 0 means the model never
    # stalled. Last, with a default, so code that builds a RunResult by hand still works.
    nudges: int = 0
    # --- sub-agents (Week 4b-B) ---
    # Sub-agents this run spawned directly: every one that ran, whether it finished,
    # gave up or crashed (the same spawns spawn_agent's budget counts). A run that
    # delegated is a different data point from one that did not, and the benchmark
    # wants to see it.
    #
    # How the fields above read for a run that delegated — only `usage` spans the
    # tree, and the asymmetry is deliberate:
    #   steps, nudges  THIS agent's own model calls and nudges. A child's are its own,
    #                  on its own RunResult (the on_spawn_done hook carries it).
    #   duration       wall-clock, so it INCLUDES every child's run: a child runs
    #                  inside one of this agent's tool calls, and the clock does not
    #                  stop for it. steps and duration therefore do not line up for a
    #                  run that delegated, and that is the honest reading of both.
    #   usage          the sum over the whole tree. A child's usage already includes
    #                  its children's, so the parent's total is what the delegation
    #                  cost end to end (see Agent.absorb_child).
    children: int = 0
    # --- end sub-agents ---

    def __str__(self) -> str:
        return self.text or ""
