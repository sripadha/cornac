"""The transcript — a JSONL record of everything the harness saw a run do (Week 4c).

Why a transcript at all
-----------------------
The result this project is built around — the same 4B model going from 12/27 to 26/27
benchmark tasks once the harness grew up (docs/observations.md) — was won by READING
RUNS: which tool the model called, with what, what came back, where it began to loop.
The hook bus (hooks/bus.py) is how you watch a run without copying the loop, and
examples/hook_trace.py shows the idea. But a print() to a terminal is gone when the
run is, and 27 benchmark runs cannot be watched live. A transcript is the hook trace
made durable: every event the bus fires, appended to a file as one JSON object per
line, so a run can be replayed, grepped and diffed hours later.

Why JSONL, one line per event, flushed
--------------------------------------
JSON Lines (one complete JSON object per line) is what Claude Code writes its session
logs in, and the reasons hold here:

  * it is append-only, so a crash mid-run loses at most the line being written and
    everything before it stays readable — a single big JSON document is unreadable
    (no closing bracket) exactly when you most need it;
  * each line is flushed as it is written, so `tail -f run.jsonl` follows a run live;
  * every line stands alone, so `grep '"event": "post_tool_use"' run.jsonl` works
    with no special reader, and read_transcript() is a short loop.

What one record looks like
--------------------------
    {"seq": 7, "t": 1758950400.12, "depth": 1, "event": "post_tool_use",
     "name": "read_file", "call_id": "c1", "args": {"path": "a.py"},
     "content": "...", "is_error": false}

`seq` is the record's position in this writer's stream (a total order — `t` alone can
tie), `t` is wall-clock time, `depth` says which agent in a delegation tree was acting,
and `event` is the HOOK'S OWN NAME, so the file and hooks/bus.py's event list share one
vocabulary. The one record that is not a hook is "run_config", written by attach(): the
system prompt, the tool names, the step budget, the provider and the model — the fixed
inputs a run's behaviour can only be judged against.

Depth, and what a child's events look like from here
----------------------------------------------------
The writer attaches to ONE bus, normally the root agent's. A sub-agent runs on a bus of
its own, and spawn_agent forwards only the events about TOOL CALLS to the parent's bus
(on_permission_decision, post_tool_use) plus on_spawn / on_spawn_done; the child's
dialogue (on_user_message, on_assistant_message, on_nudge, on_stop) stays on the
child's bus (tools/builtin/spawn.py says why). So a transcript shows every tool call in
the whole tree but only the root's conversation — and nothing in a forwarded tool event
says which agent made the call. The writer recovers that from the bracket: on_spawn
raises a depth counter, on_spawn_done lowers it, and every record in between carries
the child's depth. The printer (`python -m cornac.transcript FILE`) indents by it.

Two rules a recorder must obey
------------------------------
1. Never gate. pre_tool_use is the one event whose answers DECIDE something: a callback
   that returns Decision.DENY — or crashes — vetoes the tool call, because the bus
   fails closed there on purpose. A logger that crashed on an unexpected argument
   shape would then block the model's tool calls, quietly turning an observer into a
   policy. So the writer never registers on pre_tool_use. on_permission_decision,
   fired right after it, carries the same call plus the verdict — which is what a
   transcript wants anyway.
2. Never take the run down. A transcript is for understanding a run, not a
   precondition of it. If the disk fills or the path is bad, the run must still finish
   and its result must still reach the caller. The writer reports the failure to
   stderr ONCE (hooks/bus.py: a silent observer lies by omission) and then stops
   writing. In the same spirit, nothing that reaches a handler may raise on its way
   into JSON: a tool argument can be any Python object a model, a tool or a test put
   there, so values are converted — dataclasses.asdict() for dataclasses, str() where
   nothing better exists — and never rejected (see to_jsonable).

Nothing in the file is truncated. The transcript is where you go when the digest and
the summary were not enough, so it keeps whole tool results; only the printer shortens.
"""

from __future__ import annotations

import dataclasses
import json
import re
import sys
import time
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, TextIO

if TYPE_CHECKING:  # type hints only — the module must import with no agent in sight
    from cornac.core.agent import Agent
    from cornac.hooks.bus import HookBus

# The hook events the writer records, in the order hooks/bus.py lists them. Four of
# them (on_stuck, on_context_clear, on_wrap_up, on_run_end) arrive with Week 4c's
# agent; registering for an event nobody fires costs nothing, since a HookBus is a
# dict of callbacks. pre_tool_use is deliberately absent: rule 1 above.
OBSERVED_EVENTS: tuple[str, ...] = (
    "on_user_message",
    "on_assistant_message",
    "on_nudge",
    "on_permission_decision",
    "post_tool_use",
    "on_stop",
    "on_spawn",
    "on_spawn_done",
    "on_stuck",
    "on_context_clear",
    "on_wrap_up",
    "on_run_end",
)

# The record written by attach() before any hook fires: what the run was set up with.
RUN_CONFIG_EVENT = "run_config"

# The keys every record starts with, in this order.
HEADER_KEYS = ("seq", "t", "elapsed", "depth", "event")


def _safe_str(value: Any) -> str:
    """str(value), or the object's bare repr if even __str__ is broken.

    object.__repr__ cannot fail: it prints the class name and the address without
    running any code of the object's own. It is the floor under to_jsonable.
    """
    try:
        return str(value)
    except Exception:  # noqa: BLE001 — a recorder converts, it never refuses
        return object.__repr__(value)


def to_jsonable(value: Any) -> Any:
    """Convert anything the bus might hand us into values json.dumps accepts. Never raises.

    The order of the checks matters:
      * Enum before the primitives, because Decision is a str Enum: json would print it
        as its value anyway, but saying so here makes the file's "allow" a decision
        rather than an accident of inheritance.
      * dataclasses via dataclasses.asdict(), which recurses into nested dataclasses
        (a Message holds ToolCalls and a Usage; a RunResult holds a Usage) and gives a
        dict whose keys are the field names — the shape a reader would guess.
        asdict deep-copies whatever it finds in the leaves, and a leaf that refuses to
        be copied (a lock, a file handle) makes it raise; then vars() gives the same
        field->value view without copying, and failing that, str().
      * dicts and sequences are walked, so a leaf three levels down still gets the
        same treatment. Sets become lists (JSON has no set); non-string keys become
        strings (JSON keys must be).
      * everything else — bytes, a Path, a custom object — becomes str(). Lossy, but a
        transcript line that says "<Nasty object at 0x...>" beats a run with no
        transcript because one argument was odd.
    """
    if isinstance(value, Enum):
        return to_jsonable(value.value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        try:
            value = dataclasses.asdict(value)
        except Exception:  # noqa: BLE001 — a leaf that will not deep-copy
            try:
                value = dict(vars(value))
            except Exception:  # noqa: BLE001 — a dataclass with __slots__ and no __dict__
                return _safe_str(value)
    if isinstance(value, dict):
        return {
            (k if isinstance(k, str) else _safe_str(k)): to_jsonable(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in value]
    return _safe_str(value)


class TranscriptWriter:
    """Append one JSON record per hook event to a file; see the module docstring.

        writer = TranscriptWriter("run.jsonl").attach(agent)
        agent.run("fix the failing test")
        writer.close()                       # optional: the file is flushed per line

    The file is opened on the first record, not in __init__, so building a writer for
    a run that never starts (a usage error in the CLI, say) leaves no empty file
    behind. It is opened for APPEND: a log that a re-run of the same command could
    erase is a log you cannot trust, so two runs to one path give one file with two
    "run_config" records marking where each began (seq restarts with each writer).
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._file: TextIO | None = None
        self._seq = 0        # position of the next record; a total order within this writer
        # Two clocks. `t` (wall clock) says WHEN, for a human matching the file to a
        # calendar; `elapsed` (monotonic seconds since this writer was made) says HOW
        # LONG BETWEEN events and is the one the pretty-printer uses. The wall clock
        # alone was not enough: on WSL2 the first live run showed time.time() jump
        # back 1.5 s between two consecutive records (the host re-synced the guest's
        # clock), and a trace whose stamps run backwards is worse than one with none.
        self._t0 = time.perf_counter()
        self._depth = 0      # which agent is acting: 0 = root, raised/lowered by the spawn bracket
        self._failed = False  # set after the one stderr report; nothing is written after that

    # --- wiring ---------------------------------------------------------------------------

    def attach(self, agent: Agent) -> TranscriptWriter:
        """Record the run's configuration, then observe the agent's bus. Returns self.

        run_config goes first so a reader knows what the run was set up with before
        the first event: the system prompt (the rules the model was given), the tool
        names (what it could do), the step budget, and the provider and model (which
        model's behaviour this is — the whole benchmark story is one model, many
        harnesses). `model` is read with getattr because Provider does not promise
        one; Ollama and Anthropic providers have it, a scripted test provider may not.
        """
        provider = agent.provider
        self.record(
            RUN_CONFIG_EVENT,
            system_prompt=agent.system_prompt,
            tools=agent.registry.names(),
            max_steps=agent.max_steps,
            provider=provider.name,
            model=getattr(provider, "model", None),
        )
        return self.subscribe(agent.hooks)

    def subscribe(self, bus: HookBus) -> TranscriptWriter:
        """Register one handler per OBSERVED_EVENTS on `bus`, and nothing on pre_tool_use.

        Separate from attach() so a bare HookBus can be recorded without an Agent —
        which is also how the tests drive the writer. The handlers are bound methods,
        so if one ever does raise, the bus's stderr report names it
        ("TranscriptWriter._post_tool_use"), not a lambda.
        """
        handlers: dict[str, Callable] = {
            "on_user_message": self._on_user_message,
            "on_assistant_message": self._on_assistant_message,
            "on_nudge": self._on_nudge,
            "on_permission_decision": self._on_permission_decision,
            "post_tool_use": self._post_tool_use,
            "on_stop": self._on_stop,
            "on_spawn": self._on_spawn,
            "on_spawn_done": self._on_spawn_done,
            "on_stuck": self._on_stuck,
            "on_context_clear": self._on_context_clear,
            "on_wrap_up": self._on_wrap_up,
            "on_run_end": self._on_run_end,
        }
        for event in OBSERVED_EVENTS:
            bus.on(event, handlers[event])
        return self

    # --- writing --------------------------------------------------------------------------

    @property
    def depth(self) -> int:
        """The depth the next record will carry (0 = the root agent)."""
        return self._depth

    def record(self, event: str, **fields: Any) -> None:
        """Write one record: the four header keys, then `fields` made JSON-safe.

        Public so a caller (the CLI, a benchmark runner) can add records of its own
        to the same stream — a "run_config" is one such record. Header keys come
        first so a human reading the raw file sees seq/t/depth/event at a glance.
        """
        record: dict[str, Any] = {
            "seq": self._seq,
            "t": time.time(),
            "elapsed": round(time.perf_counter() - self._t0, 3),
            "depth": self._depth,
            "event": event,
        }
        self._seq += 1
        for key, value in fields.items():
            record[key] = to_jsonable(value)
        self._write(record)

    def _write(self, record: dict) -> None:
        """Serialise and append one line; on the first failure, say so once and go quiet.

        Two layers of defence, for the two ways this can fail. Serialisation should
        already be impossible to break after to_jsonable, but json.dumps gets
        default=_safe_str all the same, and if it still raises, a stub record with the
        same header and an "error" field is written instead — the stream keeps its
        seq order and the reader learns that a record was lost, and why. I/O failure
        (bad path, full disk) is the case rule 2 is about: report once, to stderr so
        it can never be mistaken for the agent's output, then stop trying; a hook that
        printed the same error on every one of a hundred events would drown the
        run's real output.
        """
        if self._failed:
            return
        try:
            line = json.dumps(record, ensure_ascii=False, default=_safe_str)
        except Exception as exc:  # noqa: BLE001 — see the docstring
            stub = {key: record.get(key) for key in HEADER_KEYS}
            stub["error"] = f"record could not be serialised: {type(exc).__name__}: {exc}"
            line = json.dumps(stub, default=_safe_str)
        try:
            if self._file is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._file = open(self.path, "a", encoding="utf-8")
            self._file.write(line + "\n")
            self._file.flush()
        except Exception as exc:  # noqa: BLE001 — the run matters more than its log
            self._failed = True
            print(
                f"[cornac] transcript: cannot write {self.path}: {type(exc).__name__}: {exc}. "
                "Transcript recording stopped; the run continues.",
                file=sys.stderr,
            )

    def close(self) -> None:
        """Close the file. Optional — every line is flushed as it is written — but tidy.

        A record after close() reopens the file (lazily, for append) exactly as the
        first one did, so closing early cannot lose events.
        """
        if self._file is not None:
            try:
                self._file.close()
            except Exception:  # noqa: BLE001 — closing is best-effort
                pass
            self._file = None

    def __enter__(self) -> TranscriptWriter:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # --- the handlers, one per observed event -----------------------------------------
    #
    # Each takes exactly the arguments its event is fired with (hooks/bus.py lists them)
    # and records the parts a reader needs. Dataclasses are handed to record() whole;
    # to_jsonable turns them into dicts named by their fields.

    def _on_user_message(self, message) -> None:
        self.record("on_user_message", text=message.text)

    def _on_assistant_message(self, message) -> None:
        # tool_calls become a list of {"id", "name", "arguments"}; usage a dict of
        # token counts or None; stop_reason the provider's own word, verbatim.
        self.record(
            "on_assistant_message",
            text=message.text,
            tool_calls=message.tool_calls,
            usage=message.usage,
            stop_reason=message.stop_reason,
        )

    def _on_nudge(self, text, n) -> None:
        # `text` is what the model said before stalling; `n` is which nudge this is.
        self.record("on_nudge", text=text, n=n)

    def _on_permission_decision(self, call, decision) -> None:
        # The effective decision (ALLOW/DENY), as the loop announces it — a str Enum,
        # so the file holds "allow" / "deny".
        self.record(
            "on_permission_decision",
            name=call.name,
            call_id=call.id,
            args=call.arguments,
            decision=decision,
        )

    def _post_tool_use(self, call, result) -> None:
        # The whole result content, untruncated (module docstring, last paragraph).
        # `call` and `result` name the same tool; the result's name is the one the
        # registry answered under, which is what matters if they ever differed.
        self.record(
            "post_tool_use",
            name=result.name,
            call_id=call.id,
            args=call.arguments,
            content=result.content,
            is_error=result.is_error,
        )

    def _on_stop(self, final_text) -> None:
        # None means the loop gave up (max_steps or stuck) — recorded as null, so a
        # reader can tell "no answer" from "an empty answer".
        self.record("on_stop", text=final_text)

    def _on_spawn(self, task, depth) -> None:
        # Recorded at the PARENT's depth (the spawn is the parent's act), then the
        # counter rises so the child's forwarded tool events sit one level in. The
        # depth spawn_agent reports is kept too, as child_depth, so a writer attached
        # to a non-root bus still shows the tree's absolute depth.
        self.record("on_spawn", task=task, child_depth=depth)
        self._depth += 1

    def _on_spawn_done(self, result, depth) -> None:
        # Lowered BEFORE recording so this record aligns with its on_spawn. Floored
        # at 0: a stray on_spawn_done fired by hand must not push later records to a
        # negative depth. `result` is the child's RunResult (as a dict) or None if
        # the child raised — the same two shapes spawn_agent fires.
        self._depth = max(0, self._depth - 1)
        self.record("on_spawn_done", child_depth=depth, result=result)

    def _on_stuck(self, call, n) -> None:
        # Week 4c: the identical call returned the identical result `n` times and the
        # run is ending as "stuck".
        self.record("on_stuck", name=call.name, call_id=call.id, args=call.arguments, n=n)

    def _on_context_clear(self, cleared, freed_chars) -> None:
        # Week 4c: `cleared` old tool results were replaced by a one-line note, freeing
        # `freed_chars` characters of prompt.
        self.record("on_context_clear", cleared=cleared, freed_chars=freed_chars)

    def _on_wrap_up(self, summary) -> None:
        # Week 4c: the model's own four-line account after running out of steps (or
        # None if that call failed).
        self.record("on_wrap_up", summary=summary)

    def _on_run_end(self, result) -> None:
        # Week 4c: fired last for every run. The RunResult as a dict — text,
        # stop_reason, steps, duration, usage, nudges, children, digest, summary.
        self.record("on_run_end", result=result)


# --- reading ------------------------------------------------------------------------------


def read_transcript(path: str | Path) -> list[dict]:
    """Load a transcript back into a list of records, in file order.

    Blank lines are skipped (a trailing newline is normal). A line that is not a JSON
    object raises ValueError naming the line: the writer flushes each line, so a
    broken one means the file was truncated or edited, and a reader that silently
    dropped it would hide exactly that. The caller decides whether to care; the CLI
    printer below reports it and exits non-zero.
    """
    records: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: line {lineno} is not valid JSON ({exc.msg})") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}: line {lineno} is not a JSON object")
            records.append(record)
    return records


# --- the readable trace ---------------------------------------------------------------------


def _short(text: Any, limit: int = 100) -> str:
    """One printable line of at most `limit` characters; newlines shown as \\n."""
    if text is None:
        return "None"
    flat = str(text).replace("\r", "").replace("\n", "\\n")
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def _call_str(name: Any, args: Any, limit: int = 80) -> str:
    """`name({...args...})` for a tool call, arguments as compact JSON."""
    try:
        rendered = json.dumps(args, ensure_ascii=False, default=_safe_str)
    except Exception:  # noqa: BLE001 — a printer must print
        rendered = _safe_str(args)
    return f"{name}({_short(rendered, limit)})"


def _total_tokens(usage: Any) -> int:
    """Usage.total_tokens recomputed from a usage dict: prompt (incl. cache) + output."""
    if not isinstance(usage, dict):
        return 0
    return sum(
        int(usage.get(key, 0) or 0)
        for key in ("input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens")
    )


def _describe(record: dict) -> str:
    """The event-specific part of a trace line. Tolerant of missing keys: a record
    written by an older or newer writer should still print something useful."""
    event = record.get("event", "?")
    if event == RUN_CONFIG_EVENT:
        tools = ", ".join(record.get("tools") or [])
        return (
            f"run_config provider={record.get('provider')} model={record.get('model')} "
            f"max_steps={record.get('max_steps')} tools=[{tools}]"
        )
    if event == "on_user_message":
        return f"user: {_short(record.get('text'))}"
    if event == "on_assistant_message":
        calls = record.get("tool_calls") or []
        if calls:
            body = "wants " + ", ".join(
                _call_str(c.get("name"), c.get("arguments")) for c in calls if isinstance(c, dict)
            )
        else:
            body = _short(record.get("text"))
        usage = record.get("usage")
        tokens = f" [{_total_tokens(usage)} tok]" if isinstance(usage, dict) else ""
        return f"assistant: {body}{tokens}"
    if event == "on_nudge":
        return f"nudge #{record.get('n')}: model announced \"{_short(record.get('text'), 60)}\""
    if event == "on_permission_decision":
        return f"permission {record.get('name')} -> {str(record.get('decision')).upper()}"
    if event == "post_tool_use":
        content = record.get("content") or ""
        # Same rule as core/digest.py: the loop's is_error flag, or the tool saying so
        # itself ("Error: ..." from edit_file/read_file, "(exit code 127)" from run_bash).
        # The first live run showed "run_bash -> ok: (exit code 127) ..." while the
        # digest of the same run called it an error; one rule, one answer.
        flag = "ERROR" if _looks_failed(record.get("is_error"), content) else "ok"
        return f"{record.get('name')} -> {flag}: {_short(content)} ({len(content)} chars)"
    if event == "on_stop":
        if record.get("text") is None:
            return "stop: no final answer (the loop gave up)"
        return f"stop: final answer ready: {_short(record.get('text'), 60)}"
    if event == "on_spawn":
        return f"spawn -> depth {record.get('child_depth')}: {_short(record.get('task'))}"
    if event == "on_spawn_done":
        result = record.get("result")
        if not isinstance(result, dict):
            return f"spawn done <- depth {record.get('child_depth')}: child raised"
        return (
            f"spawn done <- depth {record.get('child_depth')}: "
            f"stop={result.get('stop_reason')} steps={result.get('steps')}"
        )
    if event == "on_stuck":
        return (
            f"stuck: {_call_str(record.get('name'), record.get('args'))} "
            f"returned the same result {record.get('n')} times; run ends"
        )
    if event == "on_context_clear":
        return (
            f"context clear: {record.get('cleared')} old tool results replaced, "
            f"{record.get('freed_chars')} chars freed"
        )
    if event == "on_wrap_up":
        return f"wrap-up: {_short(record.get('summary'), 160)}"
    if event == "on_run_end":
        result = record.get("result") if isinstance(record.get("result"), dict) else {}
        duration = result.get("duration")
        timing = f"{duration:.1f}s" if isinstance(duration, (int, float)) else "?"
        return (
            f"run end: stop={result.get('stop_reason')} steps={result.get('steps')} "
            f"tokens={_total_tokens(result.get('usage'))} time={timing}"
        )
    # An event this printer does not know (a record() someone added): key=value pairs.
    extras = " ".join(
        f"{key}={_short(value, 40)}" for key, value in record.items() if key not in HEADER_KEYS
    )
    return f"{event} {extras}".rstrip()


def _looks_failed(is_error, content: str) -> bool:
    """The digest's failure rule, applied to a transcript record (see core/digest.py)."""
    if is_error:
        return True
    text = content.lstrip()
    if text.startswith("Error:"):
        return True
    m = re.match(r"\(exit code (\d+)\)", text)
    return m is not None and m.group(1) != "0"


def format_record(record: dict, t0: float | None = None) -> str:
    """One line for one record: seq, seconds since `t0`, then the event indented by depth.

    Two spaces per depth level, so a sub-agent's tool calls sit visibly inside the
    on_spawn / on_spawn_done bracket that produced them.
    """
    depth = record.get("depth") or 0
    seq = record.get("seq", "?")
    # Prefer the monotonic `elapsed` (cannot run backwards); fall back to the wall
    # clock for a file written before `elapsed` existed.
    elapsed = record.get("elapsed")
    t = record.get("t")
    if isinstance(elapsed, (int, float)):
        stamp = f"+{elapsed:7.2f}s"
    elif isinstance(t, (int, float)) and t0 is not None:
        stamp = f"+{t - t0:7.2f}s"
    else:
        stamp = " " * 9
    return f"{seq:>4} {stamp}  {'  ' * int(depth)}{_describe(record)}"


def main(argv: list[str] | None = None) -> int:
    """`python -m cornac.transcript FILE.jsonl`: print the trace, one line per event.

    Exit codes: 0 printed, 1 the file could not be read or holds a bad line (the
    reason goes to stderr), 2 usage.
    """
    args = sys.argv[1:] if argv is None else list(argv)
    if len(args) != 1 or args[0] in ("-h", "--help"):
        print("usage: python -m cornac.transcript FILE.jsonl", file=sys.stderr)
        return 2
    try:
        records = read_transcript(args[0])
    except (OSError, ValueError) as exc:
        print(f"[cornac] transcript: {exc}", file=sys.stderr)
        return 1
    t0 = next((r["t"] for r in records if isinstance(r.get("t"), (int, float))), None)
    for record in records:
        print(format_record(record, t0))
    return 0


__all__ = [
    "OBSERVED_EVENTS",
    "RUN_CONFIG_EVENT",
    "TranscriptWriter",
    "format_record",
    "main",
    "read_transcript",
    "to_jsonable",
]


if __name__ == "__main__":
    sys.exit(main())
