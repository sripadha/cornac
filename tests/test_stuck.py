"""Tests for the repeat hard stop (Week 4c): identical call, identical result, three times -> "stuck".

The Week 4b note (tests/test_nudge.py) tells the model it is repeating itself and
leaves the decision to it. This is what happens when the model does not listen: after
max_repeats identical calls with identical results, the loop ends the run with
stop_reason "stuck" instead of spending the rest of its steps on the same call. As
with every loop branch, a scripted provider plays the model, entirely offline.

Most tests here switch the wrap-up off (wrap_up=False) so the script is exactly the
model calls the loop makes; the wrap-up itself is tested in tests/test_wrap_up.py, and
one test below checks that a stuck run goes through it.
"""

import pytest

from cornac import Agent, Message, ToolRegistry
from cornac.core.agent import wrap_up_message
from cornac.core.messages import ToolCall
from cornac.hooks.bus import HookBus
from cornac.permissions.policy import Policy
from cornac.providers.base import Provider
from cornac.tools.base import tool

# A side-effect flag: the only trustworthy proof that a tool did or did not run.
ran: list[str] = []

# What `say` answers next, one entry per call — a tool whose output the test controls,
# standing in for "the tests" that may or may not pass after an edit.
outputs: list[str] = []


@tool()
def mark(label: str = "x") -> str:
    """Record that this tool ran; the same label always gets the same answer."""
    ran.append(label)
    return f"marked {label}"


@tool()
def say() -> str:
    """Answer with the next scripted output."""
    return outputs.pop(0)


class ScriptedProvider(Provider):
    """Returns pre-baked assistant messages, one per turn, and records the tools offered."""

    def __init__(self, script):
        self.script = list(script)
        self.seen_tools: list[list[str]] = []

    def complete(self, messages, tools):
        self.seen_tools.append([t["name"] for t in tools])
        return self.script.pop(0)


def _tool_turn(name="mark", call_id="c1", **arguments):
    if name == "mark" and not arguments:
        arguments = {"label": "x"}
    return Message(role="assistant", tool_calls=[ToolCall(call_id, name, arguments)])


def _two_calls(*labels):
    calls = [ToolCall(f"c{i}", "mark", {"label": label}) for i, label in enumerate(labels)]
    return Message(role="assistant", tool_calls=calls)


def _final(text="done"):
    return Message(role="assistant", text=text)


def _agent(script, **kwargs):
    kwargs.setdefault("wrap_up", False)
    return Agent(provider=ScriptedProvider(script), registry=ToolRegistry([mark, say]), **kwargs)


def _tool_messages(agent):
    return [m for m in agent.messages if m.role == "tool"]


def setup_function():
    ran.clear()
    outputs.clear()


# --- The hard stop -----------------------------------------------------------------

def test_three_identical_calls_with_identical_results_end_the_run_as_stuck():
    agent = _agent([_tool_turn(), _tool_turn(), _tool_turn(), _final("never reached")])
    result = agent.run("go")

    assert result.stop_reason == "stuck"
    assert result.text is None                 # no answer, and none invented
    assert result.steps == 3                   # the three calls that led here
    assert ran == ["x", "x", "x"]              # the third call itself still ran...
    assert len(_tool_messages(agent)) == 3     # ...and its result is in the transcript
    assert agent.provider.script == [_final("never reached")]   # the run ended: no 4th call


def test_the_advisory_note_is_still_appended_on_the_way_to_stuck():
    # The Week 4b note is kept: the second and third results carry it, with the count.
    agent = _agent([_tool_turn()] * 3 + [_final()])
    agent.run("go")

    first, second, third = _tool_messages(agent)
    assert first.text == "marked x"
    assert "already made 2 times" in second.text
    assert "already made 3 times" in third.text


def test_on_stuck_fires_with_the_call_and_the_count():
    seen = []
    bus = HookBus()
    bus.on("on_stuck", lambda call, n: seen.append((call.name, call.arguments, n)))

    _agent([_tool_turn()] * 3 + [_final()], hooks=bus).run("go")

    assert seen == [("mark", {"label": "x"}, 3)]


def test_stuck_fires_on_stop_none_then_on_run_end_with_the_result():
    events = []
    bus = HookBus()
    for name in ["on_stuck", "on_stop", "on_run_end"]:
        bus.on(name, lambda *a, _n=name: events.append((_n, a)))

    result = _agent([_tool_turn()] * 3 + [_final()], hooks=bus).run("go")

    names = [n for n, _ in events]
    assert names == ["on_stuck", "on_stop", "on_run_end"]
    assert events[1][1] == (None,)             # on_stop(None): "gave up", as at max_steps
    assert events[2][1][0] is result           # on_run_end gets the very RunResult returned
    assert result.stop_reason == "stuck"


# --- What is NOT stuck -----------------------------------------------------------------

def test_differing_results_never_end_a_run_as_stuck():
    # The same call, five times, but the answer keeps changing (tests after edits).
    # "Stuck" is about a call whose outcome has stopped changing, not about repetition.
    outputs.extend(["1 failed", "1 passed", "1 failed", "1 passed", "1 failed"])
    result = _agent([_tool_turn("say")] * 5 + [_final("fixed")]).run("go")

    assert result.stop_reason == "done"
    assert result.text == "fixed"
    assert result.steps == 6


def test_only_consecutive_identical_results_count_toward_stuck():
    # fail, pass, fail, fail, fail: the streak is 3 only at the very end.
    outputs.extend(["1 failed", "1 passed", "1 failed", "1 failed", "1 failed"])
    result = _agent([_tool_turn("say")] * 5 + [_final()]).run("go")

    assert result.stop_reason == "stuck"
    assert result.steps == 5


def test_different_arguments_are_different_calls():
    # a, b, c, a, b, c: every call is made twice, none three times in a row.
    labels = ["a", "b", "c", "a", "b", "c"]
    result = _agent([_tool_turn(label=lb) for lb in labels] + [_final()]).run("go")

    assert result.stop_reason == "done"
    assert ran == labels


def test_max_repeats_zero_disables_the_hard_stop_but_keeps_the_note():
    agent = _agent([_tool_turn()] * 5 + [_final("gave up on my own")], max_repeats=0)
    result = agent.run("go")

    assert result.stop_reason == "done"
    assert result.text == "gave up on my own"
    assert ran == ["x"] * 5
    assert "already made 5 times" in _tool_messages(agent)[-1].text


def test_max_repeats_is_the_bound_not_a_hardcoded_three():
    two = _agent([_tool_turn()] * 2 + [_final()], max_repeats=2).run("go")
    assert two.stop_reason == "stuck" and two.steps == 2

    five = _agent([_tool_turn()] * 4 + [_final("ok")], max_repeats=5).run("go")
    assert five.stop_reason == "done" and five.steps == 5


def test_repeat_counters_reset_between_runs():
    # Two runs, two identical calls each: never three in a row within one run.
    agent = _agent([_tool_turn(), _tool_turn(), _final(), _tool_turn(), _tool_turn(), _final()])
    assert agent.run("first").stop_reason == "done"
    assert agent.run("second").stop_reason == "done"


# --- The shape of the stop -------------------------------------------------------------

def test_the_rest_of_the_response_is_handled_before_the_run_ends():
    # The third identical call arrives in a response that also asks for another tool.
    # Every call the model made gets its result before the run ends, so the
    # transcript has no tool call without a result.
    agent = _agent([_tool_turn(), _tool_turn(), _two_calls("x", "y"), _final()])
    result = agent.run("go")

    assert result.stop_reason == "stuck"
    assert ran == ["x", "x", "x", "y"]
    assert [m.tool_call_id for m in _tool_messages(agent)] == ["c1", "c1", "c0", "c1"]


def test_a_refused_call_asked_for_three_times_is_stuck_too():
    # Denied, denied, denied: the model never ran anything, and it is still going
    # nowhere. A policy that will not budge is as fixed an outcome as any.
    agent = _agent([_tool_turn()] * 3 + [_final()], policy=Policy({"tools": {"mark": "deny"}}))
    result = agent.run("go")

    assert result.stop_reason == "stuck"
    assert ran == []
    assert all(m.is_error for m in _tool_messages(agent))


def test_stuck_goes_through_the_wrap_up_path():
    # With the wrap-up on, a stuck run gets the same post-mortem a max_steps run
    # gets: the digest, and one more model call with no tools for the summary.
    script = [_tool_turn()] * 3 + [_final("I kept marking x and it never changed.")]
    agent = _agent(script, wrap_up=True)
    result = agent.run("go")

    assert result.stop_reason == "stuck"
    assert result.steps == 3                                   # the wrap-up is not a step
    assert result.summary == "I kept marking x and it never changed."
    assert agent.provider.seen_tools[-1] == []                 # the wrap-up offered no tools
    assert agent.provider.seen_tools[:-1] == [["mark", "say"]] * 3
    assert result.digest is not None
    assert "1. mark(x) -> ok" in result.digest
    assert "3 model calls, 3 tool calls, 0 errors." in result.digest
    # The model is told the true reason — the repetition — not "out of steps".
    assert agent.messages[-2].role == "user"
    assert agent.messages[-2].text == wrap_up_message("stuck", name="mark", n=3)
    assert "you made the same mark call 3 times" in agent.messages[-2].text


def test_max_repeats_defaults_to_three_and_is_exposed():
    assert _agent([]).max_repeats == 3
    assert _agent([], max_repeats=7).max_repeats == 7


@pytest.mark.parametrize("bad", [1, -1])
def test_max_repeats_one_or_negative_is_refused_at_construction(bad):
    # A fresh call already counts as 1, so max_repeats=1 would end every run on its
    # first tool call — a harness that cannot run any tool. Refused up front, with
    # the reason, instead of silently accepted.
    with pytest.raises(ValueError, match="max_repeats must be 0 .* or at least 2"):
        _agent([], max_repeats=bad)


def test_a_fresh_first_call_is_never_stuck_at_the_smallest_allowed_cap():
    result = _agent([_tool_turn(), _final("done")], max_repeats=2).run("go")
    assert result.stop_reason == "done" and result.steps == 2
