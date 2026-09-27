"""Tests for RunResult — what run() returns since Week 4 — and the pre_tool_use veto.

Everything here is offline. A scripted provider plays the model, so we can pin down
exactly how the loop counts steps, sums usage, reports why it stopped, and lets a
hook block a tool call before the policy is ever asked.
"""

from cornac import Agent, Message, RunResult, ToolRegistry, Usage
from cornac.core.messages import ToolCall
from cornac.hooks.bus import HookBus
from cornac.permissions.policy import Decision, Policy
from cornac.providers.base import Provider
from cornac.tools.base import tool

# A side-effect flag: the only trustworthy proof that a tool did or did not run.
ran: list[str] = []


@tool()
def mark(label: str = "x") -> str:
    """Record that this tool ran."""
    ran.append(label)
    return f"marked {label}"


class ScriptedProvider(Provider):
    """Returns pre-baked assistant messages, one per turn."""

    def __init__(self, script):
        self._script = list(script)

    def complete(self, messages, tools):
        return self._script.pop(0)


class NeverStopsProvider(Provider):
    """A 'model' that asks for a tool on every single turn — it never gives a final answer."""

    def complete(self, messages, tools):
        return Message(role="assistant", tool_calls=[ToolCall("c", "mark", {"label": "loop"})])


class RecordingPolicy(Policy):
    """A Policy that remembers which calls it was asked about."""

    def __init__(self, rules):
        super().__init__(rules)
        self.checked: list[str] = []

    def check(self, call):
        self.checked.append(call.name)
        return super().check(call)


def _tool_turn(label="x"):
    return Message(role="assistant", tool_calls=[ToolCall("c1", "mark", {"label": label})])


def _final(text="done"):
    return Message(role="assistant", text=text)


def _agent(provider, **kwargs):
    return Agent(provider=provider, registry=ToolRegistry([mark]), **kwargs)


def setup_function():
    ran.clear()


# --- Usage arithmetic ------------------------------------------------------------

def test_usage_add_returns_a_new_object_and_sums_fieldwise():
    a = Usage(input_tokens=10, output_tokens=5)
    b = Usage(input_tokens=1, output_tokens=2)
    total = a + b
    assert total == Usage(input_tokens=11, output_tokens=7)
    assert total.total_tokens == 18
    assert a == Usage(10, 5) and b == Usage(1, 2)  # operands untouched
    assert Usage().total_tokens == 0


# --- RunResult bookkeeping -------------------------------------------------------

def test_two_turn_run_counts_two_steps_and_finishes_done():
    agent = _agent(ScriptedProvider([_tool_turn(), _final("all done")]))
    result = agent.run("go")

    assert isinstance(result, RunResult)
    assert result.text == "all done"
    assert result.stop_reason == "done"
    assert result.steps == 2  # one complete() call per scripted turn
    assert ran == ["x"]


def test_duration_is_non_negative_wall_clock_seconds():
    result = _agent(ScriptedProvider([_final()])).run("go")
    assert isinstance(result.duration, float)
    assert result.duration >= 0


def test_max_steps_gives_no_text_and_says_so():
    stops = []
    bus = HookBus()
    bus.on("on_stop", stops.append)

    # max_repeats=0: this provider repeats the identical call by nature, and the rail
    # under test here is max_steps, not the Week 4c repeat stop (tests/test_stuck.py).
    agent = _agent(NeverStopsProvider(), max_steps=3, hooks=bus, max_repeats=0)
    result = agent.run("go")

    assert result.stop_reason == "max_steps"
    assert result.text is None          # no final answer, and we don't invent one
    assert result.steps == 3            # exactly max_steps model calls were made
    assert ran == ["loop"] * 3          # the tool ran each time; the rail is on the loop
    assert stops == [None]              # on_stop(None) signals "gave up", not "finished"


def test_usage_is_summed_across_calls_and_tolerates_none():
    script = [
        _tool_turn(),                       # usage None — e.g. a provider that doesn't report it
        _tool_turn("y"),
        _final(),
    ]
    script[1].usage = Usage(input_tokens=100, output_tokens=10)
    script[2].usage = Usage(input_tokens=250, output_tokens=40)

    result = _agent(ScriptedProvider(script)).run("go")

    assert result.usage == Usage(input_tokens=350, output_tokens=50)
    assert result.usage.total_tokens == 400
    assert result.steps == 3


def test_usage_is_zero_when_no_response_reports_it():
    result = _agent(ScriptedProvider([_final()])).run("go")
    assert result.usage == Usage()


def test_nudges_is_zero_for_a_run_that_never_stalled_and_defaults_to_zero():
    # Week 4b added `nudges` as the last field, with a default, so a RunResult built
    # the Week 4 way (five positional fields) still constructs. The nudge itself is
    # tested in tests/test_nudge.py.
    result = _agent(ScriptedProvider([_tool_turn(), _final()])).run("go")
    assert result.nudges == 0
    assert RunResult("t", "done", 1, 0.0, Usage()).nudges == 0


def test_str_of_result_is_its_text():
    done = _agent(ScriptedProvider([_final("the answer")])).run("go")
    assert str(done) == done.text == "the answer"

    gave_up = _agent(NeverStopsProvider(), max_steps=1).run("go")
    assert str(gave_up) == ""  # so print(result) never prints the word "None"


# --- The pre_tool_use veto -------------------------------------------------------

def test_hook_returning_deny_vetoes_the_call_before_policy_or_approver():
    decisions = []
    approver_calls = []
    post = []

    def approver(call, policy=None):
        approver_calls.append(call.name)
        return Decision.ALLOW

    bus = HookBus()
    bus.on("pre_tool_use", lambda call: Decision.DENY)
    bus.on("on_permission_decision", lambda call, d: decisions.append(d))
    bus.on("post_tool_use", lambda call, result: post.append(result))

    # The policy says ASK, and the approver would say ALLOW — neither must be reached.
    policy = RecordingPolicy({"tools": {"mark": "ask"}})
    agent = _agent(ScriptedProvider([_tool_turn(), _final()]),
                   policy=policy, approver=approver, hooks=bus)
    result = agent.run("go")

    assert ran == []                          # the tool did NOT execute
    assert policy.checked == []               # the policy was never consulted
    assert approver_calls == []               # nor was the human/approver
    assert decisions == [Decision.DENY]       # but the decision was still announced
    assert len(post) == 1 and post[0].is_error  # and post_tool_use still fired

    tool_msg = next(m for m in agent.messages if m.role == "tool")
    assert tool_msg.is_error
    assert "Vetoed" in tool_msg.text
    assert "mark" in tool_msg.text
    assert result.text == "done"              # the model saw the error and carried on


def test_one_deny_among_several_hooks_is_enough_to_veto():
    bus = HookBus()
    bus.on("pre_tool_use", lambda call: None)            # a plain logger
    bus.on("pre_tool_use", lambda call: Decision.DENY)   # the veto
    bus.on("pre_tool_use", lambda call: Decision.ALLOW)  # ALLOW does not cancel a DENY

    agent = _agent(ScriptedProvider([_tool_turn(), _final()]), hooks=bus)
    agent.run("go")

    assert ran == []
    assert next(m for m in agent.messages if m.role == "tool").is_error


def test_hooks_returning_none_or_other_values_do_not_veto():
    bus = HookBus()
    # "DENY" (wrong case) and "allow" are on the list on purpose: only the exact
    # string "deny" is a veto (see the next test), and an explicit "allow" from a
    # hook means nothing — hooks can only object, never approve.
    for value in [None, True, False, 0, "no", "block", "DENY", "allow",
                  Decision.ALLOW, Decision.ASK, object()]:
        bus.on("pre_tool_use", lambda call, _v=value: _v)

    agent = _agent(ScriptedProvider([_tool_turn(), _final()]), hooks=bus)
    agent.run("go")

    assert ran == ["x"]  # the tool ran normally
    tool_msg = next(m for m in agent.messages if m.role == "tool")
    assert not tool_msg.is_error
    assert tool_msg.text == "marked x"


def test_hook_returning_the_string_deny_also_vetoes():
    # Decision is a str Enum, so "deny" == Decision.DENY. That is a documented part
    # of the contract, not an accident: it errs on the fail-closed side, and lets a
    # hook that never imported Decision still block a call.
    bus = HookBus()
    bus.on("pre_tool_use", lambda call: "deny")

    agent = _agent(ScriptedProvider([_tool_turn(), _final()]), hooks=bus)
    agent.run("go")

    assert ran == []
    tool_msg = next(m for m in agent.messages if m.role == "tool")
    assert tool_msg.is_error and "Vetoed" in tool_msg.text


def test_without_a_veto_the_policy_is_still_consulted():
    policy = RecordingPolicy({"tools": {"mark": "allow"}})
    bus = HookBus()
    bus.on("pre_tool_use", lambda call: None)

    _agent(ScriptedProvider([_tool_turn(), _final()]), policy=policy, hooks=bus).run("go")

    assert policy.checked == ["mark"]
    assert ran == ["x"]


def test_a_crashing_pre_tool_use_hook_vetoes_the_call(capsys):
    # A veto hook written for run_bash's argument shape, hit with a different one.
    # Failing open would treat the crash as "no objection" and run the tool; a
    # gatekeeper must fail closed instead.
    bus = HookBus()
    bus.on("pre_tool_use",
           lambda call: Decision.DENY if "curl" in call.arguments["command"] else None)

    policy = RecordingPolicy({"tools": {"mark": "allow"}})
    agent = _agent(ScriptedProvider([_tool_turn(), _final()]), policy=policy, hooks=bus)
    result = agent.run("go")

    assert ran == []                          # the tool did NOT execute
    assert policy.checked == []               # vetoed before the policy, like any veto
    tool_msg = next(m for m in agent.messages if m.role == "tool")
    assert tool_msg.is_error and "Vetoed" in tool_msg.text
    assert "raised KeyError" in capsys.readouterr().err   # and the crash was reported
    assert result.text == "done"              # the loop itself never crashed


def test_children_is_zero_for_a_run_that_never_delegated_and_defaults_to_zero():
    # Week 4b-B added `children` after `nudges`, with a default, for the same reason:
    # a RunResult built by hand — even one that gives nudges — still constructs.
    # Delegation itself is tested in tests/test_spawn.py.
    result = _agent(ScriptedProvider([_tool_turn(), _final()])).run("go")
    assert result.children == 0
    assert RunResult("t", "done", 1, 0.0, Usage()).children == 0
    assert RunResult("t", "done", 1, 0.0, Usage(), 2).children == 0
