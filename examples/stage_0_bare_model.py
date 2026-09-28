#!/usr/bin/env python
"""Ladder level 0 - the bare model: no system prompt, no tools, one call.

This is the control the whole experiment is measured against. The frozen 4B model
gets the task prompt with the workspace files pasted in after it, and must reply
with corrected files as fenced blocks that start `# file: <path>`. There is nothing
else: no persona, no word about the workspace, no way to run the tests. The level's
apply_answer() writes each block to its path, and verify.py grades the WORKSPACE
(does pytest pass? is the test file untouched?), never the prose. So the number this
prints is what the model can do on its own; every level above adds one thing, and
the gap between two levels is what that one thing was worth.

    python examples/stage_0_bare_model.py              # task swe (default)
    python examples/stage_0_bare_model.py --task swe2  # or swe3

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

# `python examples/stage_0_bare_model.py` puts examples/ on sys.path, not the repo root,
# and neither cornac nor benchmark/ is pip-installed: the same bootstrap as run_spike.py.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmark.ladder.levels import LEVELS  # noqa: E402
from cornac.tools.workspace import Workspace  # noqa: E402

LEVEL = 0
MODEL, SEED, TEMPERATURE = "qwen3.5:4b", 42, 0.3  # round three's frozen settings, run 0's seed


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

    # A fresh copy of the fixture: the model's answer lands in the copy, never the original.
    tmp = Path(tempfile.mkdtemp(prefix=f"ladder-L{LEVEL}-{task_name}-"))
    workspace_dir = tmp / "workspace"
    shutil.copytree(task.fixture, workspace_dir, ignore=shutil.ignore_patterns("__pycache__"))

    print(f"=== level {level.number} ({level.key}): {level.title}\n    adds: {level.adds}")
    print(f"    task {task_name} | {MODEL} | seed {SEED} | temperature {TEMPERATURE}\n")
    # As run_ladder does: run_bash inherits this process's PATH, and `python -m pytest`
    # must be THIS interpreter's even when the venv was not activated (else level 2+
    # fails its first test run for a reason that has nothing to do with the model).
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    provider = spike.make_provider(MODEL, seed=SEED, temperature=TEMPERATURE)
    outcome = level.run(provider, Workspace(workspace_dir), task.prompt, tmp / "transcript.jsonl")

    # One call, so the raw reply IS the whole run. Whatever `# file:` blocks it held were
    # already written into the workspace by apply_answer; everything else is prose.
    print("--- the model's reply, verbatim ---")
    print(outcome.final_text or "(empty reply)")
    print(f"--- files written from the reply: {', '.join(outcome.files_applied) or 'none'}\n")

    passed, note = task.verify(str(workspace_dir), outcome.final_text or "")
    print(f"verdict: {'PASS' if passed else 'FAIL'} - {note}")
    usage = outcome.usage
    print(f"cost: {outcome.steps} model call, {usage.total_tokens:,} tokens "
          f"(in {usage.input_tokens:,} / out {usage.output_tokens:,}), {outcome.duration:.1f} s")
    print(f"kept for inspection: {tmp}")


if __name__ == "__main__":
    main()
