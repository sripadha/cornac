"""Offline tests for the Week 1 skeleton.

These use a fake provider so the whole loop — tool dispatch, result feedback, the
neutral message round-trip — is exercised with no network, no API key, and no Ollama.
A real LLM is non-deterministic; the loop's plumbing should be deterministic and is
what we pin down here. (Since Week 4, run() returns a RunResult; the final text is
`result.text` — see tests/test_run_result.py for the rest of that record.)
"""

from cornac import Agent, Message, ToolRegistry
from cornac.core.messages import ToolCall
from cornac.providers.base import Provider
from cornac.tools.builtin.clock import get_current_time


class ScriptedProvider(Provider):
    """Returns a pre-baked sequence of assistant messages, one per call.

    Stands in for a real model so we can assert on exactly how the loop behaves.
    """

    def __init__(self, script: list[Message]):
        self._script = script
        self.calls: list[tuple] = []  # records (messages, tools) each turn, for assertions

    def complete(self, messages, tools):
        self.calls.append((list(messages), tools))
        return self._script.pop(0)


def test_loop_executes_tool_then_returns_final_text():
    provider = ScriptedProvider(
        [
            # Turn 1: model asks for the time.
            Message(
                role="assistant",
                tool_calls=[ToolCall(id="c1", name="get_current_time", arguments={"timezone": "UTC"})],
            ),
            # Turn 2: model produces a final answer (no tool calls).
            Message(role="assistant", text="It is currently the time I just looked up."),
        ]
    )
    agent = Agent(provider=provider, registry=ToolRegistry([get_current_time]))

    result = agent.run("What time is it?")

    assert result.text == "It is currently the time I just looked up."
    assert result.stop_reason == "done"
    # system?(none) + user + assistant(toolcall) + tool result + assistant(final) = 4
    roles = [m.role for m in agent.messages]
    assert roles == ["user", "assistant", "tool", "assistant"]
    # The tool result that was fed back must reference the original call id.
    tool_msg = agent.messages[2]
    assert tool_msg.tool_call_id == "c1"
    assert not tool_msg.is_error


def test_unknown_tool_comes_back_as_error_not_crash():
    provider = ScriptedProvider(
        [
            Message(role="assistant", tool_calls=[ToolCall(id="x", name="does_not_exist", arguments={})]),
            Message(role="assistant", text="ok, recovered"),
        ]
    )
    agent = Agent(provider=provider, registry=ToolRegistry([get_current_time]))

    result = agent.run("do something")

    assert result.text == "ok, recovered"
    tool_msg = agent.messages[2]
    assert tool_msg.is_error
    assert "Unknown tool" in tool_msg.text


def test_schemas_are_advertised_to_the_provider():
    provider = ScriptedProvider([Message(role="assistant", text="hi")])
    agent = Agent(provider=provider, registry=ToolRegistry([get_current_time]))

    agent.run("hello")

    _, tools = provider.calls[0]
    assert len(tools) == 1
    assert tools[0]["name"] == "get_current_time"
    assert tools[0]["input_schema"]["type"] == "object"
    assert "timezone" in tools[0]["input_schema"]["properties"]


def test_get_current_time_tool_runs():
    out = get_current_time.run({"timezone": "UTC"})
    assert "UTC" in out
