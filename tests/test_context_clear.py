"""Tests for context clearing (Week 4c): old tool results make room before the window fills.

Before every model call the loop estimates the prompt size — max(what the model last
reported, characters / 4) — and once it passes 75% of the context window, replaces
the oldest large tool results with a one-line note until the estimate is under 60%.
The numbers here are chosen so the arithmetic can be followed by hand: a window of
1000 tokens, tool results of 900 characters (= 225 tokens on the character floor), a
user message of 2 characters ("go") and assistant turns with no text at all. Three
results make 2 + 2700 = 2702 chars = 675 tokens, under 75%; a fourth makes 900 tokens,
over it. The note that replaces a cleared result is about 120 characters, so each
clearing frees about 780 characters, or 195 tokens.
"""

from cornac import Agent, Message, ToolRegistry, Usage
from cornac.core.agent import CLEARED_NOTE, MARKER
from cornac.core.messages import ToolCall
from cornac.hooks.bus import HookBus
from cornac.providers.base import Provider
from cornac.tools.base import tool

outputs: list[str] = []


@tool()
def say() -> str:
    """Answer with the next scripted output — a stand-in for a big file read."""
    return outputs.pop(0)


class ScriptedProvider(Provider):
    """Returns pre-baked assistant messages, one per turn; context_window unknown."""

    def __init__(self, script):
        self.script = list(script)

    def complete(self, messages, tools):
        return self.script.pop(0)


class SizedProvider(ScriptedProvider):
    """A provider that knows its window, like OllamaProvider knows num_ctx."""

    context_window = 1000


def _tool_turn(usage=None):
    return Message(role="assistant", tool_calls=[ToolCall("c1", "say", {})], usage=usage)


def _final(text="done"):
    return Message(role="assistant", text=text)


def _agent(script, provider_cls=ScriptedProvider, **kwargs):
    kwargs.setdefault("wrap_up", False)
    return Agent(provider=provider_cls(script), registry=ToolRegistry([say]), **kwargs)


def _tool_messages(agent):
    return [m for m in agent.messages if m.role == "tool"]


def _big(letter, n=900):
    # Distinct letters keep the repeat counter (and the Week 4c hard stop) quiet.
    return letter * n


def _note(n):
    return CLEARED_NOTE.format(name="say", n=n)


def _events(bus):
    seen = []
    bus.on("on_context_clear", lambda cleared, freed: seen.append((cleared, freed)))
    return seen


def setup_function():
    outputs.clear()


# --- Configuration --------------------------------------------------------------------------

def test_context_window_falls_back_to_the_provider_and_can_be_overridden():
    assert _agent([], SizedProvider).context_window == 1000
    assert _agent([], SizedProvider, context_window=4000).context_window == 4000
    assert _agent([]).context_window is None        # nobody knows -> clearing is off


def test_no_context_window_means_nothing_is_ever_cleared():
    outputs.extend([_big("A"), _big("B"), _big("C"), _big("D"), _big("E")])
    bus = HookBus()
    seen = _events(bus)
    agent = _agent([_tool_turn()] * 5 + [_final()], hooks=bus)
    agent.run("go")

    assert all(len(m.text) == 900 for m in _tool_messages(agent))
    assert seen == []


# --- The clearing -----------------------------------------------------------------------------

def test_the_oldest_large_results_are_cleared_until_the_estimate_is_under_sixty_percent():
    # Before model call 5 the prompt is 2 + 4 x 900 chars = 900 tokens > 750 (75% of
    # 1000). The last two results are kept; A is cleared (-> ~705 tokens, still >= 600),
    # then B (-> ~511, under 600): two cleared, and C and D untouched.
    outputs.extend([_big("A"), _big("B"), _big("C"), _big("D")])
    bus = HookBus()
    seen = _events(bus)
    agent = _agent([_tool_turn()] * 4 + [_final()], hooks=bus, context_window=1000)
    result = agent.run("go")

    a, b, c, d = _tool_messages(agent)
    assert a.text == _note(900)
    assert b.text == _note(900)
    assert c.text == _big("C") and d.text == _big("D")
    assert a.text.startswith("[cornac] cleared: this earlier say result (900 chars) was removed")
    assert seen == [(2, 2 * (900 - len(_note(900))))]
    assert result.stop_reason == "done"


def test_nothing_is_cleared_while_the_estimate_is_under_seventy_five_percent():
    # Three results: 2 + 2700 chars = 675 tokens, under 750. No clearing at all.
    outputs.extend([_big("A"), _big("B"), _big("C")])
    bus = HookBus()
    seen = _events(bus)
    agent = _agent([_tool_turn()] * 3 + [_final()], hooks=bus, context_window=1000)
    agent.run("go")

    assert all(len(m.text) == 900 for m in _tool_messages(agent))
    assert seen == []


def test_the_two_most_recent_results_are_never_cleared_even_if_that_is_not_enough():
    # A window of 500 is over 75% full from the third result on, so clearing runs
    # before calls 4 and 5. Each time only the results outside the last two may go —
    # A, then B — and neither pass gets near 60% of 500. C and D are what the model
    # is about to act on, and stay whatever the estimate says.
    outputs.extend([_big("A"), _big("B"), _big("C"), _big("D")])
    bus = HookBus()
    seen = _events(bus)
    agent = _agent([_tool_turn()] * 4 + [_final()], hooks=bus, context_window=500)
    agent.run("go")

    a, b, c, d = _tool_messages(agent)
    assert a.text == _note(900) and b.text == _note(900)
    assert c.text == _big("C") and d.text == _big("D")
    assert [n for n, _ in seen] == [1, 1]


def test_results_under_two_hundred_chars_are_skipped():
    # A, "ok", B, C, D: before call 6 the prompt is ~901 tokens. Candidates are A,
    # "ok" and B (C and D are the last two). A is cleared, "ok" is too small to be
    # worth a note, B is cleared, and the estimate is under 600.
    outputs.extend([_big("A"), "ok", _big("B"), _big("C"), _big("D")])
    bus = HookBus()
    seen = _events(bus)
    agent = _agent([_tool_turn()] * 5 + [_final()], hooks=bus, context_window=1000)
    agent.run("go")

    a, ok, b, c, d = _tool_messages(agent)
    assert a.text == _note(900)
    assert ok.text == "ok"
    assert b.text == _note(900)
    assert c.text == _big("C") and d.text == _big("D")
    assert [n for n, _ in seen] == [2]


def test_already_cleared_results_are_skipped_and_clearing_repeats_as_the_run_grows():
    # Six results. Before call 5: A and B are cleared (2), which brings the prompt to
    # ~511 tokens; E arrives and it is ~736, still under 75%. F arrives: ~961. Before
    # call 7 the notes for A and B are under 200 chars and skipped, and C and D are
    # cleared (2); E and F are the last two.
    outputs.extend([_big(ch) for ch in "ABCDEF"])
    bus = HookBus()
    seen = _events(bus)
    agent = _agent([_tool_turn()] * 6 + [_final()], hooks=bus, context_window=1000)
    agent.run("go")

    texts = [m.text for m in _tool_messages(agent)]
    assert texts[:4] == [_note(900)] * 4
    assert texts[4:] == [_big("E"), _big("F")]
    assert [n for n, _ in seen] == [2, 2]
    saved = 900 - len(_note(900))
    assert [f for _, f in seen] == [2 * saved, 2 * saved]


# --- What clearing must never do --------------------------------------------------------------

def test_system_user_and_assistant_messages_are_never_touched():
    system = "S" * 3000
    task = "T" * 1500
    outputs.extend([_big("A"), _big("B"), _big("C")])
    # 4500 chars before any tool result: already 1125 tokens > 750, and nothing to
    # clear; the loop must not reach for the system prompt or the task instead.
    script = [Message(role="assistant", text="Reading.", tool_calls=[ToolCall("c1", "say", {})])]
    script += [_tool_turn()] * 2 + [_final("end")]
    bus = HookBus()
    seen = _events(bus)
    agent = _agent(script, hooks=bus, system_prompt=system, context_window=1000)
    agent.run(task)

    assert agent.messages[0].role == "system" and agent.messages[0].text == system
    assert agent.messages[1].role == "user" and agent.messages[1].text == task
    assistants = [m for m in agent.messages if m.role == "assistant"]
    assert assistants[0].text == "Reading." and assistants[-1].text == "end"
    # The one clearable result (A; B and C are the last two) was cleared, once.
    a, b, c = _tool_messages(agent)
    assert a.text == _note(900) and b.text == _big("B") and c.text == _big("C")
    assert [n for n, _ in seen] == [1]


def test_a_cleared_message_keeps_its_wire_fields():
    # Providers need role, tool_call_id and name to keep the transcript valid on the
    # wire (a tool result must still answer its call); is_error is left as it was.
    outputs.extend([_big("A"), _big("B"), _big("C"), _big("D")])
    script = [
        Message(role="assistant", tool_calls=[ToolCall("first-call", "say", {})]),
        _tool_turn(), _tool_turn(), _tool_turn(), _final(),
    ]
    agent = _agent(script, context_window=1000)
    agent.run("go")

    a = _tool_messages(agent)[0]
    assert a.role == "tool"
    assert a.tool_call_id == "first-call"
    assert a.name == "say"
    assert a.is_error is False
    assert a.text == _note(900)


# --- The estimate -----------------------------------------------------------------------------

def test_the_reported_prompt_size_is_used_when_it_is_above_the_character_floor():
    # Four results of 300 chars: 1202 chars is only ~300 tokens on the floor, but the
    # model's last report says the prompt was 900 tokens (code is denser than 4
    # chars/token). 900 > 750 triggers clearing; both candidates go (the estimate
    # shrinks in proportion to the characters removed and never gets under 600 with
    # only two clearable results), and the last two are kept.
    outputs.extend([_big("A", 300), _big("B", 300), _big("C", 300), _big("D", 300)])
    script = [_tool_turn()] * 3 + [_tool_turn(usage=Usage(input_tokens=900))] + [_final()]
    bus = HookBus()
    seen = _events(bus)
    agent = _agent(script, hooks=bus, context_window=1000)
    agent.run("go")

    a, b, c, d = _tool_messages(agent)
    assert a.text == _note(300) and b.text == _note(300)
    assert c.text == _big("C", 300) and d.text == _big("D", 300)
    assert [n for n, _ in seen] == [2]


def test_a_low_or_missing_report_does_not_hide_a_big_prompt():
    # Ollama omits prompt_eval_count when the prompt was cached, so the report can
    # say 0 while the prompt is huge. The character floor catches it.
    outputs.extend([_big("A"), _big("B"), _big("C"), _big("D")])
    script = [_tool_turn(usage=Usage(input_tokens=0, output_tokens=5))] * 4 + [_final()]
    bus = HookBus()
    seen = _events(bus)
    agent = _agent(script, hooks=bus, context_window=1000)
    agent.run("go")

    assert [n for n, _ in seen] == [2]


def test_cache_read_tokens_count_toward_the_prompt_size():
    # A cached prefix still occupies the window: Usage.prompt_tokens includes it.
    outputs.extend([_big("A", 300), _big("B", 300), _big("C", 300), _big("D", 300)])
    reported = Usage(input_tokens=100, cache_read_tokens=800)      # prompt_tokens == 900
    script = [_tool_turn()] * 3 + [_tool_turn(usage=reported)] + [_final()]
    bus = HookBus()
    seen = _events(bus)
    _agent(script, hooks=bus, context_window=1000).run("go")

    assert [n for n, _ in seen] == [2]


# --- Where the check runs ------------------------------------------------------------------------

def test_clearing_also_runs_before_the_wrap_up_call():
    # Four results fill the window right as the loop hits max_steps. The wrap-up is a
    # model call like any other, so the same check runs before it.
    outputs.extend([_big("A"), _big("B"), _big("C"), _big("D")])
    bus = HookBus()
    seen = _events(bus)
    agent = _agent([_tool_turn()] * 4 + [_final("summary")], hooks=bus,
                   context_window=1000, wrap_up=True, max_steps=4)
    result = agent.run("go")

    assert result.stop_reason == "max_steps"
    assert result.summary == "summary"
    assert [n for n, _ in seen] == [2]
    assert _tool_messages(agent)[0].text == _note(900)


# --- Clearing and the repeat counter ------------------------------------------------------------
#
# The note that replaces a cleared result says "call the tool again if you still need
# it". The repeat counter keys on (tool, arguments) and compares content, so a model
# doing exactly that would — before the fix — be told "repeating it will not change the
# outcome" and, on the second cleared re-read, have its run ended as "stuck". These
# use a read(name) tool whose output is the same 900 characters every time, like a file.


@tool()
def read(name: str) -> str:
    """A stand-in for read_file: the same big content for the same name, every time."""
    return name * 900


def _read(name):
    return Message(role="assistant", tool_calls=[ToolCall("c1", "read", {"name": name})])


def _reader(script, **kwargs):
    kwargs.setdefault("wrap_up", False)
    return Agent(provider=ScriptedProvider(script), registry=ToolRegistry([read]), **kwargs)


def test_re_reading_a_result_the_loop_cleared_is_a_fresh_call_not_a_repeat():
    # a,b,c,d,a,e,f,a at a 1000-token window: every earlier `a` is cleared before
    # `a` is read again, so no re-read is a repeat, none carries the repeat note,
    # and the run ends "done" — not "stuck" at the eighth step, as it used to.
    stuck = []
    bus = HookBus()
    bus.on("on_stuck", lambda call, n: stuck.append(n))
    clears = _events(bus)
    agent = _reader([_read(n) for n in "abcdaefa"] + [_final("done")], hooks=bus, context_window=1000)

    result = agent.run("go")

    assert result.stop_reason == "done" and result.text == "done"
    assert stuck == []
    assert clears, "the scenario relies on clearing having happened"
    for m in _tool_messages(agent):
        assert "already made" not in m.text            # no repeat note anywhere
    # Every cleared message is the plain cleared note, nothing appended to it.
    for m in _tool_messages(agent):
        if m.text.startswith(MARKER):
            assert m.text == CLEARED_NOTE.format(name="read", n=900)


def test_clearing_an_older_copy_does_not_forget_a_streak_that_is_still_in_view():
    # a, a (repeat 2, with the note), b, c, then a again. The window is chosen so the
    # check before call 5 clears exactly the FIRST `a`: the prompt is then 2 + 900 +
    # 1116 (900 plus the repeat note) + 900 + 900 = 3818 chars = 954.5 tokens, over
    # 75% of 1270 (952.5); clearing a1 frees 780 chars, to 759.5 tokens, under 60%
    # (762), so clearing stops. The second `a`, with its "already made 2 times" note,
    # is still in front of the model, and reading `a` again is the stuck case the cap
    # exists for: the model can see the result and its warning, and repeats anyway.
    stuck = []
    bus = HookBus()
    bus.on("on_stuck", lambda call, n: stuck.append((call.arguments["name"], n)))
    clears = _events(bus)
    agent = _reader([_read(n) for n in "aabca"] + [_final("never")], hooks=bus, context_window=1270)

    result = agent.run("go")

    first_a, second_a = _tool_messages(agent)[:2]
    assert first_a.text == CLEARED_NOTE.format(name="read", n=900)
    assert second_a.text.startswith("a" * 900) and "already made 2 times" in second_a.text
    assert [n for n, _ in clears] == [1]
    assert result.stop_reason == "stuck"
    assert stuck == [("a", 3)]
