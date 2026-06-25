"""Tests for the Week 3 permission system and hooks.

These are fully offline: a scripted provider drives the loop, a fake approver stands in
for the human at the terminal, and we assert on what ran, what was refused, and which
hook events fired.
"""

from cornac import Agent, Message, ToolRegistry
from cornac.core.messages import ToolCall
from cornac.hooks.bus import HookBus
from cornac.permissions.policy import Decision, Policy
from cornac.providers.base import Provider
from cornac.tools.base import tool


@tool()
def echo(text: str = "hi") -> str:
    """Echo the given text back."""
    return f"echoed: {text}"


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
    answer = agent.run("go")
    assert answer == "done"
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

    assert events[0] == "on_user_message"
    assert "pre_tool_use" in events
    assert "on_permission_decision" in events
    assert "post_tool_use" in events
    assert events[-1] == "on_stop"


def test_misbehaving_hook_does_not_crash_the_agent():
    bus = HookBus()
    bus.on("pre_tool_use", lambda call: 1 / 0)  # raises every time

    script = [
        Message(role="assistant", tool_calls=[ToolCall("c1", "echo", {})]),
        Message(role="assistant", text="survived"),
    ]
    agent = _agent(script, policy=Policy({"tools": {"echo": "allow"}}), hooks=bus)
    assert agent.run("go") == "survived"  # the broken hook was swallowed
