"""Tests for the Week 4 HookBus hardening: fire() returns values and reports errors.

Two things changed in bus.py. fire() now returns every callback's return value (that
is what makes the pre_tool_use veto possible), and a callback that raises is reported
on stderr instead of being silently swallowed. Both still hold the original promise: a
broken hook never crashes the agent.
"""

from cornac.hooks.bus import HOOK_FAILED, HookBus


def test_fire_returns_return_values_in_registration_order():
    bus = HookBus()
    bus.on("ev", lambda x: x + 1)
    bus.on("ev", lambda x: None)
    bus.on("ev", lambda x: f"got {x}")

    assert bus.fire("ev", 1) == [2, None, "got 1"]


def test_fire_passes_positional_and_keyword_arguments():
    bus = HookBus()
    bus.on("ev", lambda a, b, k=None: (a, b, k))
    assert bus.fire("ev", 1, 2, k="v") == [(1, 2, "v")]


def test_firing_an_event_with_no_subscribers_returns_empty_list():
    bus = HookBus()
    assert bus.fire("nobody_listens") == []
    assert bus.fire("nobody_listens", 1, 2, key="value") == []
    assert len(bus) == 0  # asking about an unknown event must not register anything


def test_raising_callback_yields_none_and_later_callbacks_still_run(capsys):
    seen = []
    bus = HookBus()

    def boom(call):
        raise ZeroDivisionError("division by zero")

    bus.on("pre_tool_use", lambda call: "first")
    bus.on("pre_tool_use", boom)
    bus.on("pre_tool_use", lambda call: seen.append(call) or "third")

    results = bus.fire("pre_tool_use", "the-call")

    # The bad callback's slot is None; its neighbours are unaffected.
    assert results == ["first", None, "third"]
    assert seen == ["the-call"]

    # The failure is reported on stderr, never stdout, in the documented shape.
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.strip().splitlines()
    assert len(lines) == 1
    line = lines[0]
    assert line.startswith("[cornac] hook 'pre_tool_use' callback ")
    assert "boom" in line  # the callback's __qualname__ names the offender
    assert line.endswith("raised ZeroDivisionError: division by zero")


def test_each_raising_callback_gets_its_own_stderr_line(capsys):
    bus = HookBus()

    def first():
        raise ValueError("one")

    def second():
        raise KeyError("two")

    bus.on("ev", first)
    bus.on("ev", second)

    assert bus.fire("ev") == [None, None]
    err = capsys.readouterr().err
    assert "callback" in err and "first" in err and "raised ValueError: one" in err
    assert "second" in err and "raised KeyError: 'two'" in err


def test_a_well_behaved_callback_writes_nothing_to_stderr(capsys):
    bus = HookBus()
    bus.on("ev", lambda: "fine")
    bus.fire("ev")
    assert capsys.readouterr().err == ""


# --- gate(): the fail-closed variant for events that decide something -----------
#
# fire() is for observers: a broken logger contributes None and life goes on. gate()
# is for pre_tool_use, whose answers can veto a call: a broken veto hook contributes
# HOOK_FAILED instead, so the agent can refuse the call rather than run it.

def test_gate_matches_fire_when_nothing_raises():
    bus = HookBus()
    bus.on("ev", lambda x: x + 1)
    bus.on("ev", lambda x: None)
    assert bus.gate("ev", 1) == bus.fire("ev", 1) == [2, None]
    assert bus.gate("nobody_listens") == []


def test_gate_marks_a_raising_callback_with_hook_failed(capsys):
    bus = HookBus()

    def boom(call):
        raise KeyError("command")

    bus.on("pre_tool_use", lambda call: "first")
    bus.on("pre_tool_use", boom)
    bus.on("pre_tool_use", lambda call: "third")

    results = bus.gate("pre_tool_use", "the-call")

    assert results[0] == "first" and results[2] == "third"
    assert results[1] is HOOK_FAILED          # not None: "broke", not "nothing to say"
    assert results == ["first", HOOK_FAILED, "third"]

    # Same stderr report as fire() — failing closed doesn't mean failing quietly.
    err = capsys.readouterr().err
    assert "[cornac] hook 'pre_tool_use' callback" in err
    assert "boom" in err and "raised KeyError: 'command'" in err


def test_hook_failed_is_distinct_from_every_ordinary_return_value():
    assert HOOK_FAILED is not None
    for value in [None, False, 0, "", [], object()]:
        assert HOOK_FAILED != value
