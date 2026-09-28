#!/usr/bin/env python
"""Ladder level 4 - the single-agent harness: better tools plus the loop guards.

Same tools as level 3, same prompt, but now the harness is switched on. The tools get
smarter: edit_file replaces one exact snippet instead of rewriting a whole file,
write_file refuses to save a .py that does not compile (the syntax gate) and reports
what its rewrite removed (the change report). The loop gets guards: a bounded nudge
when the model announces a step and stops instead of taking it, a note when it
repeats the same failed call, a hard stop after three identical repeats, a wrap-up
call that asks an unfinished run what it did, and context clearing inside the
8192-token window. Round three of the model spike measured the tool half of this on
three models and went from 12/27 to 26/27 (docs/observations.md); this is the
closest rung to that result, with the Week 4c loop guards on top. Levels 3 and 4
differ by these pieces and nothing else, so their gap is what the harness is worth.

The run prints as it happens: Level.run takes a HookBus (`hooks=`) with this script's
printers on it and makes it the run's bus, next to the level's own transcript writer
(cornac/transcript.py), so the full record still lands in the transcript file. The
guard events (a nudge, a stuck stop, a context clear, a wrap-up) print too: this is the
level they belong to.

    python examples/stage_4_harness.py              # task swe (default)
    python examples/stage_4_harness.py --task swe2  # or swe3

Needs Ollama running with the frozen model pulled: `ollama pull qwen3.5:4b`.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path

# `python examples/stage_4_harness.py` puts examples/ on sys.path, not the repo root,
# and neither cornac nor benchmark/ is pip-installed: the same bootstrap as run_spike.py.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmark.ladder.levels import LEVELS  # noqa: E402
from cornac.hooks.bus import HookBus  # noqa: E402
from cornac.tools.workspace import Workspace  # noqa: E402

LEVEL = 4
MODEL, SEED, TEMPERATURE = "qwen3.5:4b", 42, 0.3  # round three's frozen settings, run 0's seed


def _short(text, limit: int = 100) -> str:
    """One line of a message or a result, cut for the terminal; the transcript has it all."""
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _call(call) -> str:
    args = ", ".join(f"{k}={_short(v, 60)!r}" for k, v in (call.arguments or {}).items())
    return f"{call.name}({args})"


def watch(bus: HookBus) -> None:
    """Print each model turn and each tool result the moment it happens.

    Level.run takes this bus as the run's own (it adds its counter and the transcript
    writer next to these handlers), so the reader watches the agent work instead of
    staring at a blank terminal for a twelve-step run and reading a dump afterwards.
    """
    bus.on("on_assistant_message", lambda m: print(
        "model: " + (", ".join("wants " + _call(c) for c in m.tool_calls) if m.tool_calls else _short(m.text))
    ))
    bus.on("post_tool_use", lambda call, result: print(
        f"  {call.name} -> {'ERROR' if result.is_error or result.content.lstrip().startswith('Error:') else 'ok'}: "
        f"{_short(result.content)} ({len(result.content)} chars)"
    ))
    # The loop guards this level switches on, as they fire.
    bus.on("on_nudge", lambda text, n: print(f"  nudge #{n}: the model announced \"{_short(text, 60)}\" and stopped"))
    bus.on("on_stuck", lambda call, n: print(f"  stuck: {_call(call)} gave the same result {n} times; run ends"))
    bus.on("on_context_clear", lambda cleared, freed: print(f"  context clear: {cleared} old results replaced, {freed} chars freed"))
    bus.on("on_wrap_up", lambda summary: print(f"  wrap-up: {_short(summary, 160)}"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default="swe", help="swe, swe2 or swe3 (benchmark/spike/tasks)")
    task_name = parser.parse_args().task

    # run_spike.py is a script, not a package, so it is loaded by path (as tests/conftest.py
    # does). It owns load_task and the frozen provider; this script must not copy them.
    spec = importlib.util.spec_from_file_location(
        "run_spike", REPO_ROOT / "benchmark" / "spike" / "run_spike.py")
    spike = importlib.util.module_from_spec(spec)
    sys.modules["run_spike"] = spike  # its @dataclass resolves annotations through here
    spec.loader.exec_module(spike)
    task = spike.load_task(spike.TASKS_DIR / task_name)
    level = LEVELS[LEVEL]
    assert level.number == LEVEL, level

    # A fresh copy of the fixture: the tools work in the copy, never the original.
    tmp = Path(tempfile.mkdtemp(prefix=f"ladder-L{LEVEL}-{task_name}-"))
    workspace_dir = tmp / "workspace"
    shutil.copytree(task.fixture, workspace_dir, ignore=shutil.ignore_patterns("__pycache__"))
    transcript = tmp / "transcript.jsonl"

    print(f"=== level {level.number} ({level.key}): {level.title}\n    adds: {level.adds}")
    print(f"    task {task_name} | {MODEL} | seed {SEED} | temperature {TEMPERATURE}")
    print(f"    transcript: {transcript}\n")
    # As run_ladder does: run_bash inherits this process's PATH, and `python -m pytest`
    # must be THIS interpreter's even when the venv was not activated (else level 2+
    # fails its first test run for a reason that has nothing to do with the model).
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    provider = spike.make_provider(MODEL, seed=SEED, temperature=TEMPERATURE)
    bus = HookBus()
    watch(bus)
    # Look for edit_file in place of write_file, and for a write_file result that says
    # "REMOVED def ...": the change report telling the model what its rewrite lost.
    print("--- the run, as it happens (model turns, tool results, loop guards) ---")
    outcome = level.run(provider, Workspace(workspace_dir), task.prompt, transcript, hooks=bus)

    print(f"\nfinal message: {outcome.final_text or f'(none: the run ended by {outcome.stop_reason})'}")
    if outcome.summary:  # the wrap-up's own account of an unfinished run (Week 4c)
        print(f"wrap-up summary: {outcome.summary}")

    passed, note = task.verify(str(workspace_dir), outcome.final_text or "")
    print(f"\nverdict: {'PASS' if passed else 'FAIL'} - {note}")
    print(f"cost: {outcome.steps} steps, {outcome.tool_calls} tool calls ({outcome.tool_errors} errors), "
          f"{outcome.nudges} nudge(s), {outcome.usage.total_tokens:,} tokens, {outcome.duration:.1f} s, "
          f"stop: {outcome.stop_reason}")
    print(f"kept for inspection: {tmp}  (python -m cornac.transcript {transcript})")


if __name__ == "__main__":
    main()
