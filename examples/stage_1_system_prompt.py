#!/usr/bin/env python
"""Ladder level 1 - a system prompt: the model is told its job. Still no tools.

Level 0 threw the task at a bare model. This level adds exactly one thing: a system
prompt with a persona ("you are a careful software engineer, change as little as
possible, never the test file, reply in the fenced-block format") and the list of
files in the workspace. The USER message is byte-for-byte the same as level 0's -
the prompt plus the pasted files - and there is still one model call and no tool,
so the difference between the two levels' pass rates is what being told the
situation is worth. Cheap to add and easy to overrate: this is where you find out.
apply_answer() writes the `# file:` blocks it gets back; verify.py grades the
workspace, not the prose.

    python examples/stage_1_system_prompt.py              # task swe (default)
    python examples/stage_1_system_prompt.py --task swe2  # or swe3

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

# `python examples/stage_1_system_prompt.py` puts examples/ on sys.path, not the repo root,
# and neither cornac nor benchmark/ is pip-installed: the same bootstrap as run_spike.py.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmark.ladder.levels import LEVELS  # noqa: E402
from cornac.tools.workspace import Workspace  # noqa: E402

LEVEL = 1
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

    # Still one call: the raw reply is the whole run. Compare it with level 0's - the
    # persona asked for "only the corrected file(s)", so the prose should have shrunk.
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
