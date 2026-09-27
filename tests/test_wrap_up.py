"""Tests for the wrap-up (Week 4c): a run that gave up accounts for itself.

When the loop ends at max_steps or stuck, run() now does three things before it
returns: it writes a digest of the run (tests/test_digest.py covers its content), it
makes ONE extra model call with the tools switched off asking for a four-line
summary, and it fires on_run_end. These tests pin down the shape of that exchange —
what is sent, what is not counted, what happens when the call fails — with a
scripted provider that records which tools it was offered on every call.
"""

from cornac import Agent, Message, ToolRegistry, Usage
from cornac.core.agent import WRAP_UP_MESSAGE, WRAP_UP_REASONS, wrap_up_message
from cornac.core.messages import ToolCall
from cornac.hooks.bus import HookBus
from cornac.providers.base import Provider
from cornac.tools.base import tool

ran: list[str] = []


@tool()
def mark(label: str = "x") -> str:
    """Record that this tool ran."""
    ran.append(label)
    return f"marked {label}"


class ScriptedProvider(Provider):
    """Returns pre-baked assistant messages, one per turn, and records the tools offered."""

    def __init__(self, script):
        self.script = list(script)
        self.seen_tools: list[list[str]] = []
        self.calls = 0

    def complete(self, messages, tools):
        self.calls += 1
        self.seen_tools.append([t["name"] for t in tools])
        return self.script.pop(0)


class NeverStopsProvider(ScriptedProvider):
    """A 'model' that asks for a tool on every turn — even when offered none."""

    def __init__(self):
        super().__init__([])

    def complete(self, messages, tools):
        self.calls += 1
        self.seen_tools.append([t["name"] for t in tools])
        return Message(role="assistant", tool_calls=[ToolCall("c", "mark", {"label": "loop"})])


class FailsOnWrapUpProvider(ScriptedProvider):
    """Plays the script, then raises: the wrap-up call is the one that fails."""

    def complete(self, messages, tools):
        if not self.script:
            raise ConnectionError("ollama is down")
        return super().complete(messages, tools)


def _tool_turn(label="x", usage=None):
    # Distinct labels keep the repeat counter quiet: these tests are about max_steps.
    return Message(role="assistant", tool_calls=[ToolCall("c1", "mark", {"label": label})], usage=usage)


def _final(text="done", usage=None):
    return Message(role="assistant", text=text, usage=usage)


def _agent(provider, **kwargs):
    return Agent(provider=provider, registry=ToolRegistry([mark]), **kwargs)


def _labels(n):
    return [chr(ord("a") + i) for i in range(n)]


def setup_function():
    ran.clear()


# --- The wrap-up call ---------------------------------------------------------------------

def test_max_steps_makes_one_extra_call_with_no_tools_and_keeps_its_text_as_summary():
    provider = ScriptedProvider([_tool_turn("a"), _tool_turn("b"),
                                 _final("Marked a and b. Nothing verified. Unfinished. Would run the tests next.")])
    result = _agent(provider, max_steps=2).run("go")

    assert result.stop_reason == "max_steps"
    assert result.text is None                                  # still no answer
    assert result.summary == "Marked a and b. Nothing verified. Unfinished. Would run the tests next."
    assert provider.seen_tools == [["mark"], ["mark"], []]      # tools off for the wrap-up
    assert result.steps == 2                                    # the wrap-up is not a step
    assert provider.calls == 3                                  # but it was a model call


def test_the_wrap_up_is_a_user_message_followed_by_the_reply():
    provider = ScriptedProvider([_tool_turn("a"), _final("summary")])
    agent = _agent(provider, max_steps=1)
    agent.run("go")

    roles = [m.role for m in agent.messages]
    assert roles == ["user", "assistant", "tool", "user", "assistant"]
    assert agent.messages[3].text == wrap_up_message("max_steps", max_steps=1)
    assert agent.messages[4].text == "summary"


def test_the_wrap_up_exchange_fires_on_wrap_up_and_no_dialogue_events():
    events = []
    bus = HookBus()
    for name in ["on_user_message", "on_assistant_message", "on_wrap_up", "on_stop", "on_run_end"]:
        bus.on(name, lambda *a, _n=name: events.append((_n, a)))

    result = _agent(ScriptedProvider([_tool_turn("a"), _final("what I did")]), max_steps=1, hooks=bus).run("go")

    names = [n for n, _ in events]
    # One user message (the task), one assistant message (the loop step); the wrap-up
    # question and answer are the harness's, announced as on_wrap_up only.
    assert names == ["on_user_message", "on_assistant_message", "on_wrap_up", "on_stop", "on_run_end"]
    assert events[2][1] == ("what I did",)
    assert events[3][1] == (None,)
    assert events[4][1][0] is result


def test_wrap_up_usage_counts_toward_the_run_total():
    script = [
        _tool_turn("a", usage=Usage(input_tokens=100, output_tokens=10)),
        _final("summary", usage=Usage(input_tokens=200, output_tokens=20)),
    ]
    result = _agent(ScriptedProvider(script), max_steps=1).run("go")

    assert result.steps == 1
    assert result.usage == Usage(input_tokens=300, output_tokens=30)


def test_wrap_up_false_makes_no_extra_call_and_summary_is_none():
    provider = ScriptedProvider([_tool_turn("a"), _tool_turn("b")])   # exactly max_steps long
    result = _agent(provider, max_steps=2, wrap_up=False).run("go")

    assert result.stop_reason == "max_steps"
    assert result.summary is None
    assert provider.calls == 2
    assert result.digest is not None                # the free digest is always there
    assert "2 model calls, 2 tool calls, 0 errors." in result.digest


def test_a_failing_wrap_up_call_leaves_summary_none_and_reports_to_stderr(capsys):
    stops = []
    ended = []
    bus = HookBus()
    bus.on("on_stop", stops.append)
    bus.on("on_run_end", ended.append)
    bus.on("on_wrap_up", lambda s: ended.append("wrap_up_fired"))

    agent = _agent(FailsOnWrapUpProvider([_tool_turn("a")]), max_steps=1, hooks=bus)
    result = agent.run("go")

    assert result.stop_reason == "max_steps"        # the run still returned normally
    assert result.summary is None
    assert result.digest is not None
    err = capsys.readouterr().err
    assert "[cornac] wrap-up call failed" in err and "ConnectionError: ollama is down" in err
    assert stops == [None]
    assert ended == [result]                        # on_run_end fired; on_wrap_up did not
    # The unanswered request was taken back out: the transcript ends where the run did.
    assert [m.role for m in agent.messages] == ["user", "assistant", "tool"]


def test_a_failed_wrap_up_does_not_leave_its_request_in_front_of_the_next_task(capsys):
    # A reused agent (a notebook, a REPL) must not open its next run with a stale
    # "do not continue the task" sitting right before the new task.
    provider = FailsOnWrapUpProvider([_tool_turn("a")])
    agent = _agent(provider, max_steps=1)
    agent.run("first")
    provider.script = [_final("second answer")]

    second = agent.run("second")

    assert second.stop_reason == "done" and second.text == "second answer"
    roles = [m.role for m in agent.messages]
    assert roles == ["user", "assistant", "tool", "user", "assistant"]
    assert agent.messages[3].text == "second"       # the new task follows the tool result directly


def test_the_wrap_up_hands_the_schemas_to_the_providers_tools_off_seam():
    # The loop does not decide HOW tools are switched off — Anthropic needs the
    # definitions kept and forbidden, the local servers need them dropped — so it
    # calls Provider.complete_without_tools with the registry's schemas and lets the
    # provider choose. tests/test_anthropic_cache.py covers that provider's choice.
    class SeamProvider(ScriptedProvider):
        def __init__(self, script):
            super().__init__(script)
            self.tools_off_calls: list[list[str]] = []

        def complete_without_tools(self, messages, tools):
            self.tools_off_calls.append([t["name"] for t in tools])
            return self.script.pop(0)

    provider = SeamProvider([_tool_turn("a"), _final("Unfinished: marked a.")])
    result = _agent(provider, max_steps=1).run("go")

    assert result.summary == "Unfinished: marked a."
    assert provider.tools_off_calls == [["mark"]]    # the schemas, for the provider to keep or drop
    assert provider.seen_tools == [["mark"]]         # complete() was the loop step only


def test_a_provider_without_the_seam_still_gets_a_plain_tools_off_call():
    # Duck-typed, like every other seam here: a provider from before the method
    # existed is asked through complete(messages, []) as it always was.
    class Bare:
        def __init__(self, script):
            self.script = list(script)
            self.seen_tools = []

        def complete(self, messages, tools):
            self.seen_tools.append([t["name"] for t in tools])
            return self.script.pop(0)

    provider = Bare([_tool_turn("a"), _final("summary")])
    result = Agent(provider=provider, registry=ToolRegistry([mark]), max_steps=1).run("go")

    assert result.summary == "summary"
    assert provider.seen_tools == [["mark"], []]


def test_a_wrap_up_reply_that_still_asks_for_a_tool_runs_nothing():
    provider = NeverStopsProvider()
    agent = _agent(provider, max_steps=2)
    result = agent.run("go")

    assert result.stop_reason == "max_steps"
    assert ran == ["loop", "loop"]                  # the wrap-up's tool call did not run
    assert result.summary is None                   # it said nothing in words
    assert provider.seen_tools[-1] == []
    # The stored reply carries no tool call: there is no result to answer it, and a
    # dangling call would make the next run() on this agent an invalid transcript.
    assert agent.messages[-1].role == "assistant"
    assert agent.messages[-1].tool_calls == []


def test_an_empty_or_blank_wrap_up_reply_gives_summary_none():
    assert _agent(ScriptedProvider([_tool_turn("a"), _final("")]), max_steps=1).run("go").summary is None
    assert _agent(ScriptedProvider([_tool_turn("a"), _final("  \n ")]), max_steps=1).run("go").summary is None


def test_the_digest_is_built_before_the_wrap_up_and_does_not_include_it():
    result = _agent(ScriptedProvider([_tool_turn("a"), _final("the summary text")]), max_steps=1).run("go")

    assert "the summary text" not in result.digest
    assert "1 model calls, 1 tool calls, 0 errors." in result.digest


def test_the_wrap_up_message_refuses_the_model_a_claim_of_success():
    # The one risk of asking a stuck model for a summary is a summary that sounds like
    # success. The message closes on the rule that guards against it — and the rule
    # is unconditional: the harness declares the task unfinished, the model is not
    # asked to judge whether "every step succeeded" (a stuck run's steps did).
    assert "cannot call tools" in WRAP_UP_MESSAGE
    assert "Do not claim the task is complete" in WRAP_UP_MESSAGE
    assert "The task is NOT complete" in WRAP_UP_MESSAGE
    assert "Begin your reply with 'Unfinished:'" in WRAP_UP_MESSAGE
    assert "unless every step" not in WRAP_UP_MESSAGE


def test_the_wrap_up_message_names_why_the_run_stopped():
    # "You are out of steps" is false for a stuck run. Each stop reason gets its own
    # first line, with the facts the model needs to anchor "what is still wrong" on.
    out_of_steps = wrap_up_message("max_steps", max_steps=20)
    stuck = wrap_up_message("stuck", name="write_file", n=3)

    assert out_of_steps.startswith("The run was stopped: you used all 20 of your steps.")
    assert stuck.startswith(
        "The run was stopped: you made the same write_file call 3 times and got the same result every time."
    )
    assert set(WRAP_UP_REASONS) == {"max_steps", "stuck"}
    # Both carry the same four questions and the same unconditional closing rule.
    for text in (out_of_steps, stuck):
        assert "what you did, what you found, what is still wrong or unfinished, and what you would try next" in text
        assert "The task is NOT complete" in text
    # An unknown reason still produces a request rather than a KeyError mid-failure.
    assert wrap_up_message("something_new").startswith("The run was stopped: the run ended without an answer.")


def test_wrap_up_is_on_by_default_and_exposed():
    assert _agent(ScriptedProvider([])).wrap_up is True
    assert _agent(ScriptedProvider([]), wrap_up=False).wrap_up is False


# --- A finished run is untouched -------------------------------------------------------

def test_a_done_run_has_no_digest_no_summary_and_no_extra_call():
    provider = ScriptedProvider([_tool_turn("a"), _final("the answer")])
    result = _agent(provider).run("go")

    assert result.stop_reason == "done"
    assert result.text == "the answer"
    assert result.digest is None and result.summary is None
    assert provider.calls == 2


def test_on_run_end_fires_last_for_a_done_run_too():
    events = []
    bus = HookBus()
    for name in ["on_user_message", "on_assistant_message", "post_tool_use", "on_stop", "on_run_end"]:
        bus.on(name, lambda *a, _n=name: events.append((_n, a)))

    result = _agent(ScriptedProvider([_tool_turn("a"), _final("ok")]), hooks=bus).run("go")

    names = [n for n, _ in events]
    assert names == ["on_user_message", "on_assistant_message", "post_tool_use",
                     "on_assistant_message", "on_stop", "on_run_end"]
    assert events[-2][1] == ("ok",)
    assert events[-1][1][0] is result


def test_the_agent_can_run_again_after_a_wrapped_up_run():
    # The wrap-up exchange stays in the message list, as everything does; a second
    # run() on the same agent starts after it and is not confused by it.
    provider = ScriptedProvider([_tool_turn("a"), _final("summary one"), _final("second answer")])
    agent = _agent(provider, max_steps=1)
    first = agent.run("first")
    second = agent.run("second")

    assert first.stop_reason == "max_steps" and first.summary == "summary one"
    assert second.stop_reason == "done" and second.text == "second answer"
    assert second.digest is None and second.summary is None
