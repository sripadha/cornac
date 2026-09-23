"""Tests for the Week 3 permission system and hooks.

These are fully offline: a scripted provider drives the loop, a fake approver stands in
for the human at the terminal, and we assert on what ran, what was refused, and which
hook events fired.
"""

import pytest

from cornac import Agent, Message, ToolRegistry
from cornac.core.messages import ToolCall
from cornac.hooks.bus import HookBus
from cornac.permissions.policy import Decision, Policy
from cornac.providers.base import Provider
from cornac.tools.base import tool


# A side-effect flag: the only trustworthy proof that a tool did or did not run.
ran: list[str] = []


@tool()
def echo(text: str = "hi") -> str:
    """Echo the given text back."""
    ran.append(text)
    return f"echoed: {text}"


def setup_function():
    ran.clear()


class ScriptedProvider(Provider):
    """Returns pre-baked assistant messages, one per turn."""

    def __init__(self, script):
        self._script = list(script)

    def complete(self, messages, tools):
        return self._script.pop(0)


def _agent(script, **kwargs):
    return Agent(provider=ScriptedProvider(script), registry=ToolRegistry([echo]), **kwargs)


# --- Policy decision logic (no agent) ------------------------------------------

def test_policy_simple_allow_deny_ask():
    pol = Policy({"default": "ask", "tools": {"echo": "allow", "danger": "deny"}})
    assert pol.check(ToolCall("1", "echo", {})) == Decision.ALLOW
    assert pol.check(ToolCall("2", "danger", {})) == Decision.DENY
    assert pol.check(ToolCall("3", "unlisted", {})) == Decision.ASK  # falls to default


def test_policy_deny_list_beats_allow_list():
    pol = Policy({"tools": {"run_bash": {
        "deny": ["rm -rf"], "allow": ["ls"], "default": "ask"}}})
    assert pol.check(ToolCall("1", "run_bash", {"command": "ls -la"})) == Decision.ALLOW
    assert pol.check(ToolCall("2", "run_bash", {"command": "rm -rf /"})) == Decision.DENY
    assert pol.check(ToolCall("3", "run_bash", {"command": "whoami"})) == Decision.ASK


def test_policy_remember_session_override():
    pol = Policy({"tools": {"echo": "ask"}})
    assert pol.check(ToolCall("1", "echo", {})) == Decision.ASK
    pol.remember("echo", Decision.ALLOW)
    assert pol.check(ToolCall("2", "echo", {})) == Decision.ALLOW  # remembered


# --- Permissions inside the agent loop -----------------------------------------

def test_denied_tool_does_not_run_and_returns_error():
    script = [
        Message(role="assistant", tool_calls=[ToolCall("c1", "echo", {"text": "hi"})]),
        Message(role="assistant", text="done"),
    ]
    agent = _agent(script, policy=Policy({"tools": {"echo": "deny"}}))
    result = agent.run("go")
    assert result.text == "done"
    tool_msg = next(m for m in agent.messages if m.role == "tool")
    assert tool_msg.is_error
    assert "Permission denied" in tool_msg.text


def test_allowed_tool_runs():
    script = [
        Message(role="assistant", tool_calls=[ToolCall("c1", "echo", {"text": "yo"})]),
        Message(role="assistant", text="done"),
    ]
    agent = _agent(script, policy=Policy({"tools": {"echo": "allow"}}))
    agent.run("go")
    tool_msg = next(m for m in agent.messages if m.role == "tool")
    assert not tool_msg.is_error
    assert tool_msg.text == "echoed: yo"


def test_ask_with_no_approver_defaults_to_deny():
    script = [
        Message(role="assistant", tool_calls=[ToolCall("c1", "echo", {})]),
        Message(role="assistant", text="done"),
    ]
    agent = _agent(script, policy=Policy({"tools": {"echo": "ask"}}))  # no approver
    agent.run("go")
    tool_msg = next(m for m in agent.messages if m.role == "tool")
    assert tool_msg.is_error  # ASK + no approver -> denied


def test_ask_routes_to_approver_which_can_allow():
    calls_seen = []

    def approver(call, policy=None):
        calls_seen.append(call.name)
        return Decision.ALLOW

    script = [
        Message(role="assistant", tool_calls=[ToolCall("c1", "echo", {"text": "ok"})]),
        Message(role="assistant", text="done"),
    ]
    agent = _agent(script, policy=Policy({"tools": {"echo": "ask"}}), approver=approver)
    agent.run("go")
    assert calls_seen == ["echo"]  # approver was consulted
    tool_msg = next(m for m in agent.messages if m.role == "tool")
    assert tool_msg.text == "echoed: ok"  # and it allowed the run


# --- The permission gate fails closed ------------------------------------------
#
# The tool runs on an explicit ALLOW and on nothing else. An approver (or a policy)
# that answers anything else — None from a forgotten return statement, a leftover
# ASK, a truthy "yes" — has a bug, and a bug in a gate must refuse, never run. The
# first version checked `== DENY` and ran the tool on every other value.

@pytest.mark.parametrize("answer", [None, "yes", Decision.ASK, True, object()])
def test_approver_returning_anything_but_allow_is_a_denial(answer):
    decisions = []
    bus = HookBus()
    bus.on("on_permission_decision", lambda call, d: decisions.append(d))

    script = [
        Message(role="assistant", tool_calls=[ToolCall("c1", "echo", {"text": "hi"})]),
        Message(role="assistant", text="done"),
    ]
    agent = _agent(script, policy=Policy({"tools": {"echo": "ask"}}),
                   approver=lambda call, policy=None: answer, hooks=bus)
    result = agent.run("go")

    assert ran == []                                # the tool did NOT execute
    tool_msg = next(m for m in agent.messages if m.role == "tool")
    assert tool_msg.is_error
    assert "Permission denied" in tool_msg.text
    assert decisions == [Decision.DENY]             # observers see the effective decision
    assert result.text == "done"                    # the model saw the error and carried on


@pytest.mark.parametrize("answer", [None, "yes", True, object()])
def test_policy_returning_anything_but_a_decision_is_a_denial(answer):
    # Same rule, one gate earlier: a policy that answers with a non-Decision. It is
    # not ASK, so the approver must not be consulted either — even one that would
    # have said ALLOW.
    class OddPolicy(Policy):
        def check(self, call):
            return answer

    approver_calls = []

    def approver(call, policy=None):
        approver_calls.append(call.name)
        return Decision.ALLOW

    script = [
        Message(role="assistant", tool_calls=[ToolCall("c1", "echo", {"text": "hi"})]),
        Message(role="assistant", text="done"),
    ]
    agent = _agent(script, policy=OddPolicy({}), approver=approver)
    agent.run("go")

    assert ran == []
    assert approver_calls == []
    tool_msg = next(m for m in agent.messages if m.role == "tool")
    assert tool_msg.is_error and "Permission denied" in tool_msg.text


def test_an_explicit_allow_from_the_approver_still_runs_the_tool():
    # The positive side of the same contract, so the gate can't "pass" by refusing everything.
    script = [
        Message(role="assistant", tool_calls=[ToolCall("c1", "echo", {"text": "hi"})]),
        Message(role="assistant", text="done"),
    ]
    agent = _agent(script, policy=Policy({"tools": {"echo": "ask"}}),
                   approver=lambda call, policy=None: Decision.ALLOW)
    agent.run("go")
    assert ran == ["hi"]


# --- Hooks ---------------------------------------------------------------------

def test_hooks_fire_expected_events():
    events = []
    bus = HookBus()
    for name in ["on_user_message", "on_assistant_message", "pre_tool_use",
                 "on_permission_decision", "post_tool_use", "on_stop"]:
        bus.on(name, lambda *a, _n=name: events.append(_n))

    script = [
        Message(role="assistant", tool_calls=[ToolCall("c1", "echo", {})]),
        Message(role="assistant", text="done"),
    ]
    agent = _agent(script, policy=Policy({"tools": {"echo": "allow"}}), hooks=bus)
    agent.run("go")

    # The exact sequence, not just membership: within one tool call the gate fires
    # first, then the decision is announced, then the result is reported.
    assert events == [
        "on_user_message",
        "on_assistant_message",       # turn 1: the model asks for a tool
        "pre_tool_use",
        "on_permission_decision",
        "post_tool_use",
        "on_assistant_message",       # turn 2: the final answer
        "on_stop",
    ]


def test_misbehaving_hook_does_not_crash_the_agent():
    bus = HookBus()
    bus.on("pre_tool_use", lambda call: 1 / 0)  # raises every time

    script = [
        Message(role="assistant", tool_calls=[ToolCall("c1", "echo", {})]),
        Message(role="assistant", text="survived"),
    ]
    agent = _agent(script, policy=Policy({"tools": {"echo": "allow"}}), hooks=bus)
    # The broken hook is reported on stderr (see test_hooks_hardening.py) but never
    # raised into the loop, so the run still completes normally...
    assert agent.run("go").text == "survived"
    # ...and because pre_tool_use is a gate, its crash counts as a veto: the tool is
    # refused rather than run on a hook's say-nothing (see test_run_result.py).
    tool_msg = next(m for m in agent.messages if m.role == "tool")
    assert tool_msg.is_error and "Vetoed" in tool_msg.text
