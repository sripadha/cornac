"""Tests for the transcript writer — the hook trace made durable (Week 4c).

Almost everything here fires events on a bare HookBus: the writer is an observer, and
an observer is tested by showing it what it would see and reading back what it wrote.
Two tests go through a real Agent (one with a real built-in tool, one with a real
spawn_agent) to prove the wiring — attach(), the order of events, the depth bracket —
holds when the loop, not the test, is firing.

Four events (on_stuck, on_context_clear, on_wrap_up, on_run_end) belong to the
Week 4c agent and may not be fired by the loop yet; they are tested by firing them by
hand, which is all a recorder can be held to.
"""

from __future__ import annotations

import json
import re

import pytest

from cornac import Agent, Message, RunResult, ToolRegistry, Usage
from cornac.core.messages import ToolCall, ToolResult
from cornac.hooks.bus import HookBus
from cornac.permissions.policy import Decision
from cornac.providers.base import Provider
from cornac.tools.base import tool
from cornac.tools.builtin.clock import get_current_time
from cornac.tools.builtin.spawn import SpawnAgent
from cornac.transcript import (
    HEADER_KEYS,
    OBSERVED_EVENTS,
    TranscriptWriter,
    format_record,
    main,
    read_transcript,
    to_jsonable,
)

# --- fixtures and helpers -------------------------------------------------------------


class ScriptedProvider(Provider):
    """Plays the model from an ordered script; has a `model` so run_config records one."""

    model = "scripted-1"

    def __init__(self, script):
        self._script = list(script)

    def complete(self, messages, tools):
        if not self._script:
            raise AssertionError("the script ran out: more model calls than expected")
        return self._script.pop(0)


@tool()
def mark(label: str = "x") -> str:
    """A tool whose only effect is its return value."""
    return f"marked {label}"


def _call(name="read_file", call_id="c1", **arguments):
    return ToolCall(call_id, name, arguments or {"path": "a.py"})


def _result(call, content="def foo(): ...", is_error=False):
    return ToolResult(call.id, call.name, content, is_error)


def _bus_writer(tmp_path, name="t.jsonl"):
    """A writer subscribed to a fresh bus, and the bus; no Agent involved."""
    bus = HookBus()
    writer = TranscriptWriter(tmp_path / name).subscribe(bus)
    return bus, writer


def _events(records):
    return [r["event"] for r in records]


# --- the record shape ------------------------------------------------------------------


def test_every_record_starts_with_the_header_keys_and_seq_is_a_total_order(tmp_path):
    bus, writer = _bus_writer(tmp_path)
    call = _call()
    bus.fire("on_user_message", Message.user("fix it"))
    bus.fire("on_assistant_message", Message(role="assistant", tool_calls=[call]))
    bus.fire("on_permission_decision", call, Decision.ALLOW)
    bus.fire("post_tool_use", call, _result(call))
    bus.fire("on_stop", "done")

    raw = writer.path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    assert raw.endswith("\n") and len(lines) == 5          # one object per line
    records = [json.loads(line) for line in lines]          # every line is valid JSON
    assert records == read_transcript(writer.path)          # and the reader agrees

    assert [r["seq"] for r in records] == [0, 1, 2, 3, 4]
    assert all(list(r)[:len(HEADER_KEYS)] == list(HEADER_KEYS) for r in records)
    assert all(a["t"] <= b["t"] for a, b in zip(records, records[1:]))
    assert all(r["depth"] == 0 for r in records)             # nobody spawned
    assert _events(records) == [
        "on_user_message", "on_assistant_message", "on_permission_decision",
        "post_tool_use", "on_stop",
    ]


def test_user_message_and_stop_record_their_text_and_none_stays_null(tmp_path):
    bus, writer = _bus_writer(tmp_path)
    bus.fire("on_user_message", Message.user("what time is it?"))
    bus.fire("on_stop", None)

    user, stop = read_transcript(writer.path)
    assert user["text"] == "what time is it?"
    assert stop["text"] is None   # "gave up", distinguishable from an empty answer


def test_assistant_message_records_text_tool_calls_usage_and_stop_reason(tmp_path):
    bus, writer = _bus_writer(tmp_path)
    reply = Message(
        role="assistant",
        text="Let me read it.",
        tool_calls=[ToolCall("c1", "read_file", {"path": "a.py"}),
                    ToolCall("c2", "grep", {"pattern": "foo", "path": "."})],
        usage=Usage(input_tokens=100, output_tokens=20, cache_read_tokens=50),
        stop_reason="tool_use",
    )
    bus.fire("on_assistant_message", reply)
    bus.fire("on_assistant_message", Message(role="assistant", text="done"))

    with_tools, plain = read_transcript(writer.path)
    assert with_tools["text"] == "Let me read it."
    assert with_tools["tool_calls"] == [
        {"id": "c1", "name": "read_file", "arguments": {"path": "a.py"}},
        {"id": "c2", "name": "grep", "arguments": {"pattern": "foo", "path": "."}},
    ]
    assert with_tools["usage"] == {
        "input_tokens": 100, "output_tokens": 20,
        "cache_read_tokens": 50, "cache_write_tokens": 0,
    }
    assert with_tools["stop_reason"] == "tool_use"
    # A scripted or usage-less reply: empty list, null, null — not missing keys.
    assert plain["tool_calls"] == [] and plain["usage"] is None and plain["stop_reason"] is None


def test_post_tool_use_records_name_args_full_content_and_is_error(tmp_path):
    bus, writer = _bus_writer(tmp_path)
    call = _call("run_bash", command="pytest -q")
    long_output = "F" * 5000 + "\nFAILED tests/test_a.py::test_x\n"
    bus.fire("post_tool_use", call, _result(call, long_output, is_error=True))

    [rec] = read_transcript(writer.path)
    assert rec["name"] == "run_bash"
    assert rec["call_id"] == "c1"
    assert rec["args"] == {"command": "pytest -q"}
    assert rec["content"] == long_output      # untruncated: the file is the full record
    assert rec["is_error"] is True


def test_permission_decision_is_recorded_as_its_plain_value(tmp_path):
    bus, writer = _bus_writer(tmp_path)
    call = _call("write_file", path="b.py", content="x")
    bus.fire("on_permission_decision", call, Decision.ALLOW)
    bus.fire("on_permission_decision", call, Decision.DENY)

    allow, deny = read_transcript(writer.path)
    assert allow["decision"] == "allow" and deny["decision"] == "deny"
    assert allow["name"] == "write_file" and allow["args"] == {"path": "b.py", "content": "x"}


def test_nudge_records_the_stalled_text_and_which_nudge_it_is(tmp_path):
    bus, writer = _bus_writer(tmp_path)
    bus.fire("on_nudge", "Let me fix that and re-run the tests.", 1)

    [rec] = read_transcript(writer.path)
    assert rec["text"] == "Let me fix that and re-run the tests." and rec["n"] == 1


# --- the one thing a recorder must never do -------------------------------------------


def test_pre_tool_use_is_never_registered_so_the_writer_can_never_veto(tmp_path):
    bus, _writer = _bus_writer(tmp_path)

    assert "pre_tool_use" not in OBSERVED_EVENTS
    assert len(bus) == len(OBSERVED_EVENTS)                 # exactly one handler per event
    # gate() with no subscribers returns []: nothing the agent could read as a veto.
    assert bus.gate("pre_tool_use", _call()) == []


# --- depth: the on_spawn / on_spawn_done bracket -----------------------------------------


def test_spawn_bracket_raises_depth_for_the_events_between_and_lowers_it_after(tmp_path):
    bus, writer = _bus_writer(tmp_path)
    call = _call("mark", label="child")
    child_result = RunResult("child answer", "done", 2, 0.5, Usage(10, 5))

    bus.fire("on_spawn", "count the files", 1)               # root spawns a child
    bus.fire("post_tool_use", call, _result(call, "marked child"))   # forwarded from the child
    bus.fire("on_spawn", "deeper", 2)                        # the child spawns a grandchild
    bus.fire("post_tool_use", call, _result(call))           # the grandchild's call
    bus.fire("on_spawn_done", None, 2)                       # the grandchild raised
    bus.fire("on_spawn_done", child_result, 1)
    bus.fire("post_tool_use", call, _result(call))           # the root again

    records = read_transcript(writer.path)
    assert [(r["event"], r["depth"]) for r in records] == [
        ("on_spawn", 0),          # the spawn is the parent's act: parent's depth
        ("post_tool_use", 1),
        ("on_spawn", 1),
        ("post_tool_use", 2),
        ("on_spawn_done", 1),     # aligned with its on_spawn
        ("on_spawn_done", 0),
        ("post_tool_use", 0),
    ]
    assert records[0]["task"] == "count the files" and records[0]["child_depth"] == 1
    assert records[4]["result"] is None                      # a child that raised
    done = records[5]["result"]
    assert done["text"] == "child answer" and done["stop_reason"] == "done"
    assert done["usage"] == {"input_tokens": 10, "output_tokens": 5,
                             "cache_read_tokens": 0, "cache_write_tokens": 0}
    assert writer.depth == 0


def test_depth_never_goes_negative_on_a_stray_spawn_done(tmp_path):
    bus, writer = _bus_writer(tmp_path)
    bus.fire("on_spawn_done", None, 1)
    bus.fire("on_user_message", Message.user("hi"))

    assert [r["depth"] for r in read_transcript(writer.path)] == [0, 0]


# --- the Week 4c events, fired by hand ---------------------------------------------------


def test_week_4c_events_are_recorded_when_fired(tmp_path):
    bus, writer = _bus_writer(tmp_path)
    call = _call("write_file", path="a.py", content="broken")
    final = RunResult(
        text=None, stop_reason="stuck", steps=7, duration=3.25, usage=Usage(700, 90),
        nudges=1, children=0, digest="1. write_file(a.py) -> ERROR: ...", summary="I tried...",
    )

    bus.fire("on_stuck", call, 3)
    bus.fire("on_context_clear", 4, 12345)
    bus.fire("on_wrap_up", "I tried...")
    bus.fire("on_run_end", final)

    stuck, clear, wrap, end = read_transcript(writer.path)
    assert stuck["name"] == "write_file" and stuck["n"] == 3
    assert stuck["args"] == {"path": "a.py", "content": "broken"}
    assert clear["cleared"] == 4 and clear["freed_chars"] == 12345
    assert wrap["summary"] == "I tried..."
    assert end["result"] == {
        "text": None, "stop_reason": "stuck", "steps": 7, "duration": 3.25,
        "usage": {"input_tokens": 700, "output_tokens": 90,
                  "cache_read_tokens": 0, "cache_write_tokens": 0},
        "nudges": 1, "children": 0,
        "digest": "1. write_file(a.py) -> ERROR: ...", "summary": "I tried...",
    }


# --- serialisation never escapes -----------------------------------------------------------


class Nasty:
    """An argument value that fights every conversion: no str(), no deepcopy."""

    def __str__(self):
        raise RuntimeError("no str for you")

    def __deepcopy__(self, memo):
        raise TypeError("cannot copy")


def test_unserialisable_values_fall_back_to_strings_and_the_line_is_still_valid_json(tmp_path):
    bus, writer = _bus_writer(tmp_path)
    call = ToolCall("c9", "weird", {
        "obj": Nasty(),                 # even str() raises
        "s": {2, 1},                    # JSON has no set
        "b": b"bytes",                  # nor bytes
        7: "int key",                   # nor non-string keys
        "nested": [{"deep": Nasty()}],
    })
    bus.fire("post_tool_use", call, _result(call, "ok"))
    # asdict() deep-copies leaves, so a Message holding this call would make it raise.
    bus.fire("on_assistant_message", Message(role="assistant", tool_calls=[call]))

    [tool_rec, assistant_rec] = read_transcript(writer.path)   # both lines parsed
    args = tool_rec["args"]
    assert "Nasty" in args["obj"] and isinstance(args["obj"], str)
    assert sorted(args["s"]) == [1, 2]
    assert args["b"] == "b'bytes'"
    assert args["7"] == "int key"
    assert "Nasty" in args["nested"][0]["deep"]
    # The dataclass path fell back from asdict() to vars(): fields still by name.
    assert assistant_rec["tool_calls"][0]["name"] == "weird"
    assert "Nasty" in assistant_rec["tool_calls"][0]["arguments"]["obj"]


def test_to_jsonable_handles_enums_dataclasses_and_plain_values():
    assert to_jsonable(Decision.DENY) == "deny"
    assert to_jsonable(Usage(1, 2)) == {"input_tokens": 1, "output_tokens": 2,
                                        "cache_read_tokens": 0, "cache_write_tokens": 0}
    assert to_jsonable({"a": (1, 2.5, None, True)}) == {"a": [1, 2.5, None, True]}
    assert to_jsonable(Usage) == str(Usage)      # the class itself is not a dataclass instance


# --- write failure: reported once, never propagated ------------------------------------------


def test_write_failure_is_reported_to_stderr_once_and_never_raises(tmp_path, capsys):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("I am a file, so nothing can be created under me")
    bus = HookBus()
    TranscriptWriter(blocker / "run.jsonl").subscribe(bus)

    for _ in range(5):
        bus.fire("on_user_message", Message.user("hi"))    # would raise inside the writer

    err = capsys.readouterr().err
    lines = [line for line in err.splitlines() if line.strip()]
    assert len(lines) == 1                                # once, not five times
    assert lines[0].startswith("[cornac] transcript: cannot write ")
    assert "run.jsonl" in lines[0] and "run continues" in lines[0]
    # And it was the writer that reported, not the bus catching a raised handler.
    assert "hook 'on_user_message' callback" not in err


# --- lifecycle -------------------------------------------------------------------------------


def test_file_is_opened_lazily_and_appended_to(tmp_path):
    path = tmp_path / "logs" / "run.jsonl"             # the parent directory does not exist
    bus = HookBus()
    writer = TranscriptWriter(path).subscribe(bus)
    assert not path.exists()                           # nothing until the first record

    bus.fire("on_user_message", Message.user("one"))
    assert path.exists()
    writer.close()

    # A second writer on the same path appends: the first run's line is not lost.
    second = TranscriptWriter(path)
    second.record("run_config", provider="x")
    second.record("custom", note="mine")
    second.close()

    records = read_transcript(path)
    assert _events(records) == ["on_user_message", "run_config", "custom"]
    assert [r["seq"] for r in records] == [0, 0, 1]     # seq restarts per writer
    assert records[2]["note"] == "mine"


def test_context_manager_closes_and_a_record_after_close_reopens(tmp_path):
    path = tmp_path / "t.jsonl"
    with TranscriptWriter(path) as writer:
        writer.record("a")
    writer.record("b")                                 # reopened lazily, appended
    assert _events(read_transcript(path)) == ["a", "b"]


# --- the reader is strict ---------------------------------------------------------------


def test_read_transcript_names_a_bad_line_and_skips_blank_ones(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text('{"seq": 0, "event": "a"}\n\n{"seq": 1, "event": "b"}\n', encoding="utf-8")
    assert _events(read_transcript(path)) == ["a", "b"]

    path.write_text('{"seq": 0, "event": "a"}\n{"seq": 1, "event": \n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2 is not valid JSON"):
        read_transcript(path)

    path.write_text('[1, 2]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 1 is not a JSON object"):
        read_transcript(path)


# --- the printer: python -m cornac.transcript FILE ------------------------------------------


def test_main_prints_one_line_per_record_indented_by_depth(tmp_path, capsys):
    bus, writer = _bus_writer(tmp_path)
    call = _call("mark", label="child")
    bus.fire("on_user_message", Message.user("go"))
    bus.fire("on_assistant_message", Message(role="assistant", tool_calls=[
        ToolCall("s1", "spawn_agent", {"task": "count"})]))
    bus.fire("on_spawn", "count", 1)
    bus.fire("on_permission_decision", call, Decision.ALLOW)
    bus.fire("post_tool_use", call, _result(call, "marked child\nsecond line"))
    bus.fire("on_spawn_done", RunResult("3", "done", 1, 0.1, Usage()), 1)
    bus.fire("on_stop", None)

    assert main([str(writer.path)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 7                                  # one line per event

    def indent(line):
        # after "seq +t.tts  " the event text starts; count its leading spaces
        body = line.split("s  ", 1)[1]
        return len(body) - len(body.lstrip(" "))

    assert indent(out[0]) == 0 and indent(out[2]) == 0    # user, spawn: root depth
    assert indent(out[3]) == 2 and indent(out[4]) == 2    # the child's call: one level in
    assert indent(out[5]) == 0                            # spawn done: back at the root
    assert "spawn_agent(" in out[1] and "wants" in out[1]
    assert "permission mark -> ALLOW" in out[3]
    assert "mark -> ok: marked child\\nsecond line" in out[4]   # newline made visible
    assert "stop=done steps=1" in out[5]
    assert "no final answer" in out[6]


def test_main_reports_usage_and_unreadable_files_on_stderr(tmp_path, capsys):
    assert main([]) == 2
    assert "usage:" in capsys.readouterr().err

    assert main([str(tmp_path / "missing.jsonl")]) == 1
    assert "missing.jsonl" in capsys.readouterr().err

    bad = tmp_path / "bad.jsonl"
    bad.write_text("not json\n", encoding="utf-8")
    assert main([str(bad)]) == 1
    assert "line 1" in capsys.readouterr().err


def test_format_record_survives_unknown_events_and_missing_keys():
    record = {"seq": 3, "t": 10.5, "depth": 1, "event": "custom", "note": "n"}
    assert format_record(record, t0=10.0).startswith("   3 +   0.50s    custom note=n")
    bare = format_record({"event": "on_run_end"})
    assert bare.endswith("run end: stop=None steps=None tokens=0 time=?")


# --- through a real Agent ---------------------------------------------------------------------


def test_attach_records_run_config_then_the_run_in_order_with_a_real_tool(tmp_path):
    script = [
        Message(role="assistant",
                tool_calls=[ToolCall("c1", "get_current_time", {"timezone": "UTC"})]),
        Message(role="assistant", text="It is now.", usage=Usage(30, 8), stop_reason="stop"),
    ]
    agent = Agent(provider=ScriptedProvider(script),
                  registry=ToolRegistry([get_current_time, mark]),
                  system_prompt="You are a clock.", max_steps=5)
    path = tmp_path / "run.jsonl"
    writer = TranscriptWriter(path).attach(agent)

    result = agent.run("what time is it?")
    writer.close()

    assert result.text == "It is now."
    records = read_transcript(path)
    events = _events(records)
    assert events[:7] == [
        "run_config", "on_user_message", "on_assistant_message",
        "on_permission_decision", "post_tool_use", "on_assistant_message", "on_stop",
    ]
    # Week 4c's agent fires on_run_end last, once, for every run — done included.
    assert events[7:] == ["on_run_end"]
    assert [r["seq"] for r in records] == list(range(len(records)))

    config = records[0]
    assert config["system_prompt"] == "You are a clock."
    assert config["tools"] == ["get_current_time", "mark"]
    assert config["max_steps"] == 5
    assert config["provider"] == "ScriptedProvider"
    assert config["model"] == "scripted-1"

    assert records[1]["text"] == "what time is it?"
    assert records[2]["tool_calls"][0]["name"] == "get_current_time"
    assert records[3]["decision"] == "allow"                 # no policy -> ALLOW
    tool_rec = records[4]
    assert tool_rec["name"] == "get_current_time" and tool_rec["is_error"] is False
    assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC", tool_rec["content"])
    assert records[5]["usage"]["input_tokens"] == 30 and records[5]["stop_reason"] == "stop"
    assert records[6]["text"] == "It is now."
    if len(records) > 7:
        assert records[7]["result"]["stop_reason"] == "done"


def test_a_real_sub_agents_tool_calls_land_one_level_deeper(tmp_path):
    script = [
        Message(role="assistant", tool_calls=[ToolCall("s1", "spawn_agent", {"task": "count"})]),
        Message(role="assistant", tool_calls=[ToolCall("c1", "mark", {"label": "child"})]),
        Message(role="assistant", text="3 files"),           # the child's answer
        Message(role="assistant", text="there are 3"),       # the parent's
    ]
    agent = Agent(provider=ScriptedProvider(script), registry=ToolRegistry([mark, SpawnAgent()]))
    path = tmp_path / "run.jsonl"
    TranscriptWriter(path).attach(agent)

    result = agent.run("how many files?")
    assert result.text == "there are 3" and result.children == 1

    records = read_transcript(path)
    of_interest = [(r["event"], r["depth"], r.get("name")) for r in records
                   if r["event"] in ("on_spawn", "on_spawn_done", "post_tool_use")]
    assert of_interest == [
        ("on_spawn", 0, None),
        ("post_tool_use", 1, "mark"),           # the child's call, forwarded, one level in
        ("on_spawn_done", 0, None),
        ("post_tool_use", 0, "spawn_agent"),    # the parent's own call, back at the root
    ]
    # The child's dialogue is not forwarded, so the only user message is the root's.
    assert [r["text"] for r in records if r["event"] == "on_user_message"] == ["how many files?"]
    done = next(r for r in records if r["event"] == "on_spawn_done")["result"]
    assert done["text"] == "3 files" and done["stop_reason"] == "done"


# --- Week 4c live-run follow-ups -----------------------------------------------------

def test_records_carry_a_monotonic_elapsed_clock_and_the_printer_prefers_it(tmp_path):
    """time.time() ran backwards 1.5 s mid-run on WSL2; `elapsed` cannot, and the trace uses it."""
    from cornac.transcript import TranscriptWriter, format_record, read_transcript

    w = TranscriptWriter(tmp_path / "t.jsonl")
    w.record("a")
    w.record("b")
    recs = read_transcript(tmp_path / "t.jsonl")
    assert all("elapsed" in r for r in recs)
    assert recs[0]["elapsed"] <= recs[1]["elapsed"]
    # a record whose wall clock went backwards still prints in order via elapsed
    recs[1]["t"] = recs[0]["t"] - 5.0
    line0, line1 = format_record(recs[0], recs[0]["t"]), format_record(recs[1], recs[0]["t"])
    assert "-" not in line1.split("s")[0]      # no negative stamp
    assert line0.index("+") == line1.index("+")


def test_the_printer_labels_soft_failures_as_errors_like_the_digest():
    from cornac.transcript import format_record

    rec = {"seq": 1, "elapsed": 0.1, "depth": 0, "event": "post_tool_use", "name": "run_bash",
           "content": "(exit code 127)\n/bin/sh: 1: python: Permission denied", "is_error": False}
    assert "run_bash -> ERROR" in format_record(rec)
    rec2 = dict(rec, name="edit_file", content="Error: `old` not found in calc.py")
    assert "edit_file -> ERROR" in format_record(rec2)
    rec3 = dict(rec, content="(exit code 0)\nall good")
    assert "run_bash -> ok" in format_record(rec3)
