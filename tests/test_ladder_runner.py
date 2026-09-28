"""Tests for the ladder runner (benchmark/ladder/run_ladder.py).

The runner is the part of the ladder experiment that must be the same for every
level: the fixture copy, the seeds, the grading, the environment, the record. So the
tests here drive run_matrix — the testable core — with a STUB level and a scripted
provider, and check the plumbing: one runs.jsonl line per (level, task, run) with
every promised key, level-major order, per-run seeds, transcripts where the contract
says, --resume skipping finished triples, a level that raises being recorded rather
than crashing the matrix, the table and the delta table, the --levels flag, and the
one environment fix the runner makes (the interpreter's bin dir first on PATH, so
run_bash can run `python -m pytest`).

Nothing here imports benchmark/ladder/levels.py. The stub Level below has the shape
the contract gives (number, key, title, adds, run(provider, workspace, task_prompt,
transcript_path) -> outcome), so these tests stay green whatever the real levels do
and fail only when the RUNNER breaks. Like the spike, the runner is a script, not a
package module, and is loaded by path (tests/conftest.py does the same for the spike).
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from cornac.core.messages import Message, Usage
from cornac.providers.base import Provider
from cornac.tools.builtin.shell import RunBash
from cornac.tools.workspace import Workspace

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_LADDER = REPO_ROOT / "benchmark" / "ladder" / "run_ladder.py"

BROKEN_CALC = "def sum_to(n):\n    return sum(range(n))\n"
FIXED_CALC = "def sum_to(n):\n    return sum(range(n + 1))\n"


@pytest.fixture(scope="module")
def ladder():
    """The runner as a module, registered in sys.modules like the spike fixture."""
    name = "run_ladder_under_test"
    spec = importlib.util.spec_from_file_location(name, RUN_LADDER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        yield module
    finally:
        sys.modules.pop(name, None)


# --- the stub level, the scripted provider and a toy task -----------------------------


@dataclass
class StubOutcome:
    """The LevelOutcome shape from the contract, with defaults, so a test fills in only
    what it wants to see on the run line."""

    passed_precheck: bool = True
    final_text: str | None = ""
    stop_reason: str = "done"
    steps: int = 1
    usage: Usage = field(default_factory=Usage)
    duration: float = 0.0
    tool_calls: int = 0
    tool_errors: int = 0
    nudges: int = 0
    children: int = 0
    files_applied: list = field(default_factory=list)
    rejected: int = 0
    digest: str | None = None
    summary: str | None = None
    messages: list = field(default_factory=list)


@dataclass
class StubLevel:
    """A Level that makes one model call and, if `fixes`, writes the fix to calc.py."""

    number: int
    key: str
    title: str
    adds: str
    fixes: bool = False
    raises: bool = False
    seen: list = field(default_factory=list)  # (workspace root, task prompt) per call

    def run(self, provider, workspace, task_prompt, transcript_path):
        self.seen.append((workspace.root, task_prompt))
        if self.raises:
            raise RuntimeError("the daemon said no")
        reply = provider.complete([Message.user(task_prompt)], [])
        if self.fixes:
            (workspace.root / "calc.py").write_text(FIXED_CALC, encoding="utf-8")
        Path(transcript_path).write_text(
            json.dumps({"event": "one_shot", "text": reply.text}) + "\n", encoding="utf-8"
        )
        return StubOutcome(
            final_text=reply.text,
            usage=reply.usage,
            tool_calls=2 if self.fixes else 0,
            files_applied=["calc.py"] if self.fixes else [],
        )


class ScriptedProvider(Provider):
    """Answers every call with one line that names its seed; counts its calls."""

    model = "scripted-4b"

    def __init__(self, seed: int):
        self.seed = seed
        self.calls = 0

    def complete(self, messages, tools):
        self.calls += 1
        return Message(
            role="assistant",
            text=f"fixed sum_to (seed {self.seed})",
            usage=Usage(input_tokens=100, output_tokens=25),
            stop_reason="stop",
        )

    def request_options(self):
        return {"temperature": 0.3, "seed": self.seed}


def _factory():
    """A provider_factory that remembers every provider it built."""
    made: list[ScriptedProvider] = []

    def make(seed: int) -> ScriptedProvider:
        provider = ScriptedProvider(seed)
        made.append(provider)
        return provider

    make.made = made  # type: ignore[attr-defined]
    return make


@pytest.fixture
def toy_task(ladder, tmp_path):
    """A task like swe but graded by reading calc.py, so a matrix runs in milliseconds."""
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "calc.py").write_text(BROKEN_CALC, encoding="utf-8")
    (fixture / "__pycache__").mkdir()
    (fixture / "__pycache__" / "calc.cpython-312.pyc").write_bytes(b"\x00")

    def verify(workspace_dir: str, final_text: str) -> tuple[bool, str]:
        text = Path(workspace_dir, "calc.py").read_text(encoding="utf-8")
        return (text == FIXED_CALC, "sum_to inclusive" if text == FIXED_CALC else "sum_to still excludes n")

    return ladder.spike.Task(name="toy", prompt="Fix sum_to.", fixture=fixture, verify=verify)


def _two_levels():
    return [
        StubLevel(0, "bare", "Bare model", "nothing: the question and the code"),
        StubLevel(1, "prompted", "System prompt", "a system prompt", fixes=True),
    ]


# --- run_matrix: the run lines -------------------------------------------------------


def test_run_matrix_writes_one_line_per_run_with_every_key(ladder, toy_task, tmp_path):
    out = tmp_path / "out"
    levels = _two_levels()
    factory = _factory()
    lines: list[str] = []

    runs = ladder.run_matrix(levels, [toy_task], 2, factory, out, echo=lines.append)

    on_disk = [json.loads(l) for l in (out / "runs.jsonl").read_text().splitlines() if l.strip()]
    assert runs == on_disk and len(runs) == 4
    for r in runs:
        assert set(ladder.RUN_KEYS) <= set(r), sorted(set(ladder.RUN_KEYS) - set(r))
    # Level-major, then task, then run — and run i uses seed_base + i at every level.
    assert [(r["level"], r["task"], r["run"]) for r in runs] == [
        (0, "toy", 0), (0, "toy", 1), (1, "toy", 0), (1, "toy", 1)
    ]
    assert [r["seed"] for r in runs] == [42, 43, 42, 43]
    assert [p.seed for p in factory.made] == [42, 43, 42, 43]
    # The level that wrote the fix passes, the one that only talked does not.
    assert [r["passed"] for r in runs] == [False, False, True, True]
    assert runs[0]["verify_note"] == "sum_to still excludes n"
    assert runs[2]["files_applied"] == ["calc.py"] and runs[2]["tool_calls"] == 2
    # What the level and the provider said lands on the line.
    r = runs[2]
    assert (r["level_key"], r["adds"]) == ("prompted", "a system prompt")
    assert (r["model"], r["provider"]) == ("scripted-4b", "ScriptedProvider")
    assert (r["prompt_tokens"], r["output_tokens"], r["total_tokens"]) == (100, 25, 125)
    assert r["final_text"] == "fixed sum_to (seed 42)"
    assert r["stop_reason"] == "done" and r["steps"] == 1 and r["error"] is None
    assert r["temperature"] == 0.3 and r["options"] == {"temperature": 0.3, "seed": 42}
    assert r["started_at"].startswith("20")
    assert len(lines) == 4


def test_each_run_gets_a_fresh_fixture_copy_without_caches(ladder, toy_task, tmp_path):
    levels = [StubLevel(1, "prompted", "t", "a", fixes=True)]
    ladder.run_matrix(levels, [toy_task], 2, _factory(), tmp_path / "out", echo=lambda s: None)

    roots = [root for root, _ in levels[0].seen]
    assert len(set(roots)) == 2                                # two runs, two directories
    assert all(not root.exists() for root in roots)            # cleaned up afterwards
    assert (toy_task.fixture / "calc.py").read_text() == BROKEN_CALC  # the fixture is pristine
    assert all(prompt == "Fix sum_to." for _, prompt in levels[0].seen)


def test_transcripts_land_where_the_contract_says(ladder, toy_task, tmp_path):
    out = tmp_path / "out"
    runs = ladder.run_matrix(_two_levels(), [toy_task], 1, _factory(), out, echo=lambda s: None)

    assert [r["transcript"] for r in runs] == [
        "transcripts/L0-toy-r0.jsonl", "transcripts/L1-toy-r0.jsonl"
    ]
    for r in runs:
        record = json.loads((out / r["transcript"]).read_text().splitlines()[0])
        assert record["event"] == "one_shot"


def test_a_level_that_raises_is_a_failed_run_not_a_crash(ladder, toy_task, tmp_path):
    levels = [
        StubLevel(2, "one_tool", "t", "one tool", raises=True),
        StubLevel(3, "tools", "t", "the tools", fixes=True),
    ]
    runs = ladder.run_matrix(levels, [toy_task], 1, _factory(), tmp_path / "out", echo=lambda s: None)

    assert len(runs) == 2                                      # the matrix went on
    bad, good = runs
    assert bad["stop_reason"] == "error" and not bad["passed"]
    assert bad["error"] == "RuntimeError: the daemon said no"
    assert bad["verify_note"] == "sum_to still excludes n"     # the workspace was still graded
    assert good["passed"]


def test_final_text_is_capped_at_2000_chars_on_the_line(ladder, toy_task, tmp_path):
    class Chatty(ScriptedProvider):
        def complete(self, messages, tools):
            return Message(role="assistant", text="x" * 5000, usage=Usage(1, 1))

    runs = ladder.run_matrix(
        [StubLevel(0, "bare", "t", "a")], [toy_task], 1, Chatty, tmp_path / "out", echo=lambda s: None
    )
    assert len(runs[0]["final_text"]) == ladder.FINAL_TEXT_CHARS == 2000


# --- --resume -------------------------------------------------------------------------


def test_resume_skips_the_triples_already_in_runs_jsonl(ladder, toy_task, tmp_path):
    out = tmp_path / "out"
    levels = _two_levels()
    first = ladder.run_matrix(levels, [toy_task], 2, _factory(), out, echo=lambda s: None)

    factory = _factory()
    again = ladder.run_matrix(levels, [toy_task], 2, factory, out, resume=True, echo=lambda s: None)
    assert again == first and len(again) == 4
    assert factory.made == []                                  # nothing was re-run

    # A level added later, or a bigger k, runs only the missing triples.
    more = ladder.run_matrix(
        levels + [StubLevel(2, "one_tool", "t", "one tool")], [toy_task], 3, factory, out,
        resume=True, echo=lambda s: None,
    )
    assert len(more) == 9 and len(factory.made) == 5           # 2 new runs of L0/L1 + 3 of L2
    assert [(r["level"], r["run"]) for r in more[4:]] == [(0, 2), (1, 2), (2, 0), (2, 1), (2, 2)]


def test_without_resume_an_existing_runs_jsonl_is_refused(ladder, toy_task, tmp_path):
    out = tmp_path / "out"
    ladder.run_matrix(_two_levels(), [toy_task], 1, _factory(), out, echo=lambda s: None)
    with pytest.raises(FileExistsError, match="--resume"):
        ladder.run_matrix(_two_levels(), [toy_task], 1, _factory(), out, echo=lambda s: None)


def test_resume_removes_an_orphaned_transcript_before_rerunning(ladder, toy_task, tmp_path):
    # A run interrupted after its transcript began but before its line was written:
    # the transcript must be that run's alone on the retry, not two runs appended.
    out = tmp_path / "out"
    (out / "transcripts").mkdir(parents=True)
    (out / "transcripts" / "L0-toy-r0.jsonl").write_text('{"event": "stale"}\n')
    (out / "runs.jsonl").write_text("")
    ladder.run_matrix([StubLevel(0, "bare", "t", "a")], [toy_task], 1, _factory(), out,
                      resume=True, echo=lambda s: None)
    lines = (out / "transcripts" / "L0-toy-r0.jsonl").read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["event"] == "one_shot"


# --- the stdout line ------------------------------------------------------------------


def test_format_line_has_the_contract_shape(ladder):
    r = {"level": 2, "level_key": "one_tool", "task": "swe", "run": 1, "passed": True,
         "steps": 6, "total_tokens": 14231, "seconds": 21.34, "error": None, "verify_note": "ok"}
    assert ladder.format_line(r) == "L2 one_tool  swe   r1  PASS  6 steps  14,231 tok  21.3 s"
    r.update(passed=False, verify_note="pytest exit 1")
    assert ladder.format_line(r).endswith("FAIL  6 steps  14,231 tok  21.3 s  (pytest exit 1)")
    r.update(error="ReadTimeout: 300s")
    assert ladder.format_line(r).endswith("error=ReadTimeout: 300s")


def test_the_default_echo_flushes_every_run_line(ladder, toy_task, tmp_path, monkeypatch):
    # A matrix is run as `run_ladder.py > log` and tailed; a redirected stdout is
    # block-buffered, and the first smoke run showed all six lines only at exit. So the
    # default echo flushes after each line, the way the run record itself is flushed.
    class Recorder:
        def __init__(self):
            self.events: list[str] = []

        def write(self, text):
            self.events.append(text)

        def flush(self):
            self.events.append("<flush>")

    recorder = Recorder()
    monkeypatch.setattr(sys, "stdout", recorder)
    ladder.run_matrix(_two_levels(), [toy_task], 2, _factory(), tmp_path / "out")

    written = "".join(e for e in recorder.events if e != "<flush>")
    assert written.count("\n") == 4 and written.startswith("L0 bare      toy   r0  FAIL")
    # Every run line is followed by a flush before the next one is written.
    flushes = [i for i, e in enumerate(recorder.events) if e == "<flush>"]
    newlines = [i for i, e in enumerate(recorder.events) if e == "\n"]
    assert len(flushes) >= 4
    assert all(any(nl < f for f in flushes) for nl in newlines)


# --- the report -------------------------------------------------------------------------


def test_build_report_has_the_level_rows_the_overall_column_and_the_deltas(ladder, toy_task, tmp_path):
    runs = ladder.run_matrix(_two_levels(), [toy_task], 2, _factory(), tmp_path / "out",
                             echo=lambda s: None)
    report = ladder.build_report(runs, policy={"default": "deny"}, max_steps=12, k=2,
                                 out_arg="benchmark/ladder/results/x")

    table = ladder.build_table(runs)
    assert table in report and "overall" in table.splitlines()[0]
    assert "| 0 | `bare` | nothing: the question and the code | 0/2 | **0/2 (0%)** |" in table
    assert "| 1 | `prompted` | a system prompt | 2/2 | **2/2 (100%)** |" in table
    assert "answered 2" in table and "done" not in table       # the stop-reasons summary, in the table's words
    delta = ladder.build_delta_table(runs)
    assert delta in report
    assert "| 0 | `bare` | nothing: the question and the code | 0/2 (0%) | baseline |" in delta
    assert "| 1 | `prompted` | a system prompt | 2/2 (100%) | vs L0: +2 passes (+100 pp) |" in delta
    # The header states the fixed conditions and the reproduce command names the run,
    # the step budget the rows were run with included.
    assert "`scripted-4b`" in report and "max_steps 12" in report and "k = 2" in report
    assert '"default": "deny"' in report
    assert "names only the tools that level has" in report
    assert ("python benchmark/ladder/run_ladder.py --levels 0-1 --tasks toy --k 2 "
            "--temperature 0.3 --seed-base 42 --provider ScriptedProvider --model scripted-4b "
            "--max-steps 12 --out benchmark/ladder/results/x") in report
    # The legend defines the columns in the table's own words.
    assert "`answered` (the model stopped on" in report and "the loop's stop_reason `done`" in report
    assert "loop turns per run" in report and "wrap-up call after an" in report


def _rows(level: int, key: str, adds: str, passed: list[bool], max_steps: int = 12) -> list[dict]:
    """Synthetic run lines for the report builders: one per entry of `passed`."""
    return [
        {"level": level, "level_key": key, "adds": adds, "task": "swe", "run": i, "seed": 42 + i,
         "temperature": 0.3, "model": "m", "provider": "ollama", "passed": ok, "stop_reason": "done",
         "steps": 3, "total_tokens": 100, "seconds": 1.0, "max_steps": max_steps}
        for i, ok in enumerate(passed)
    ]


def test_delta_table_with_unequal_run_counts_shows_both_ratios_and_only_percentage_points(ladder):
    # Level 5 added later with --resume --k 1 (3/3) under a level 4 at k=3 (9/9):
    # "-6 passes" would be the eye-catcher for two levels with the same pass rate.
    runs = _rows(4, "harness", "the harness", [True] * 9) + _rows(5, "agents", "sub-agents", [True] * 3)
    delta = ladder.build_delta_table(runs)
    assert "| 5 | `agents` | sub-agents | 3/3 (100%) | vs L4: +0 pp (3/3 against 9/9) |" in delta
    assert "passes" not in delta.splitlines()[-1]
    # Equal counts keep the passes delta, with the compared level named.
    runs = _rows(3, "tools", "tools", [True, False, False]) + _rows(4, "harness", "harness", [True] * 3)
    assert "| 4 | `harness` | harness | 3/3 (100%) | vs L3: +2 passes (+67 pp) |" in ladder.build_delta_table(runs)


def test_delta_table_names_the_row_above_so_a_skipped_level_is_visible(ladder):
    # --levels 0,3,4: level 3's delta is against level 0, and the cell says so.
    runs = _rows(0, "bare", "nothing", [False] * 3) + _rows(3, "tools", "tools", [True] * 3) + _rows(4, "harness", "h", [True] * 3)
    lines = ladder.build_delta_table(runs).splitlines()
    assert lines[0] == "| # | level | adds | overall | vs the row above |"
    assert lines[3].endswith("| vs L0: +3 passes (+100 pp) |")
    assert lines[4].endswith("| vs L3: +0 passes (+0 pp) |")


def test_reproduce_command_carries_the_step_budget_only_when_the_rows_agree(ladder):
    runs = _rows(0, "bare", "n", [False], max_steps=8)
    assert ladder.reproduce_command(runs, 1, "out").endswith("--model m --max-steps 8 --out out")
    mixed = _rows(0, "bare", "n", [False], max_steps=8) + _rows(1, "prompted", "p", [True], max_steps=12)
    assert "--max-steps" not in ladder.reproduce_command(mixed, 1, "out")
    legacy = [{**r, "max_steps": None} for r in _rows(0, "bare", "n", [False])]
    assert "--max-steps" not in ladder.reproduce_command(legacy, 1, "out")


def test_build_table_escapes_pipes_and_caps_a_long_adds_cell(ladder):
    long_adds = "a" * 120
    runs = _rows(0, "bare", "nothing | at all", [False]) + _rows(1, "prompted", long_adds, [True])
    table = ladder.build_table(runs)
    rows = table.splitlines()[2:]
    assert "nothing \\| at all" in rows[0]                         # a bare pipe would break the table
    columns = len(table.splitlines()[0].split(" | "))
    assert all(len(row.split(" | ")) == columns for row in rows)  # the escaped pipe is not a separator
    cell = rows[1].split(" | ")[2]
    assert len(cell) == 90 and cell.endswith("…") and cell.startswith("a" * 89)


def test_build_report_refuses_an_empty_run_list(ladder):
    with pytest.raises(ValueError, match="no runs"):
        ladder.build_report([])


def test_stop_reasons_labels_done_as_answered_and_keeps_the_loops_word_on_the_line(ladder):
    assert ladder.STOP_REASON_LABELS == {"done": "answered"}
    assert ladder.stop_reasons([{"stop_reason": "done"}]) == "answered 1"


def test_stop_reasons_summary_lists_known_reasons_first(ladder):
    rows = [{"stop_reason": s} for s in ("max_steps", "done", "done", "stuck", "odd", None)]
    assert ladder.stop_reasons(rows) == "answered 2 · max_steps 1 · stuck 1 · none 1 · odd 1"


def test_report_only_rebuilds_results_md_from_runs_jsonl(ladder, toy_task, tmp_path, capsys):
    out = tmp_path / "out"
    ladder.run_matrix(_two_levels(), [toy_task], 1, _factory(), out, echo=lambda s: None)
    assert not (out / "results.md").exists()                   # run_matrix writes no report

    ladder.main(["--report-only", str(out)])

    report = (out / "results.md").read_text()
    assert "# The capability ladder" in report and "| 1 | `prompted` |" in report
    assert "wrote" in capsys.readouterr().out


def test_report_only_on_an_empty_runs_jsonl_exits_with_a_message(ladder, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "runs.jsonl").write_text("")
    with pytest.raises(SystemExit, match="no runs in"):
        ladder.main(["--report-only", str(out)])
    assert not (out / "results.md").exists()


# --- a torn last line in runs.jsonl ---------------------------------------------------------


def _tear_last_line(runs_path: Path) -> int:
    """Cut the last record mid-string, as a kill mid-write would; return the rows left whole."""
    text = runs_path.read_text(encoding="utf-8")
    whole = text.splitlines(keepends=True)
    torn = whole[-1][: len(whole[-1]) // 2]
    assert '"' in torn and not torn.endswith("\n")
    runs_path.write_text("".join(whole[:-1]) + torn, encoding="utf-8")
    return len(whole) - 1


def test_resume_drops_a_torn_last_line_reruns_that_triple_and_appends_on_a_fresh_line(ladder, toy_task, tmp_path, capsys):
    out = tmp_path / "out"
    levels = _two_levels()
    ladder.run_matrix(levels, [toy_task], 2, _factory(), out, echo=lambda s: None)
    whole = _tear_last_line(out / "runs.jsonl")
    assert whole == 3

    factory = _factory()
    runs = ladder.run_matrix(levels, [toy_task], 2, factory, out, resume=True, echo=lambda s: None)

    assert [p.seed for p in factory.made] == [43]                      # only (1, toy, 1) was re-run
    assert [(r["level"], r["run"]) for r in runs] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    on_disk = (out / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(on_disk) == 4 and all(json.loads(line) for line in on_disk)  # no glued line, no torn one
    assert "dropping the incomplete last line (4)" in capsys.readouterr().err


def test_report_only_survives_a_torn_last_line_and_reports_the_whole_rows(ladder, toy_task, tmp_path, capsys):
    out = tmp_path / "out"
    ladder.run_matrix(_two_levels(), [toy_task], 2, _factory(), out, echo=lambda s: None)
    _tear_last_line(out / "runs.jsonl")

    ladder.main(["--report-only", str(out)])

    report = (out / "results.md").read_text()
    assert "3 runs in `runs.jsonl`" in report and "| 1 | `prompted` | a system prompt | 1/1 |" in report
    err = capsys.readouterr().err
    assert "warning" in err and "incomplete last line" in err
    # --report-only does not rewrite the file: that is --resume's job, when it re-runs.
    assert not (out / "runs.jsonl").read_text(encoding="utf-8").endswith("\n")


def test_a_malformed_line_anywhere_else_is_a_hard_error_naming_the_line(ladder, toy_task, tmp_path):
    out = tmp_path / "out"
    ladder.run_matrix(_two_levels(), [toy_task], 2, _factory(), out, echo=lambda s: None)
    lines = (out / "runs.jsonl").read_text(encoding="utf-8").splitlines(keepends=True)
    lines[1] = lines[1][:40] + "\n"                                   # a damaged SECOND line
    (out / "runs.jsonl").write_text("".join(lines), encoding="utf-8")

    with pytest.raises(ValueError, match=r"line 2 is not valid JSON"):
        ladder.read_runs(out / "runs.jsonl")
    with pytest.raises(ValueError, match=r"line 2"):
        ladder.run_matrix(_two_levels(), [toy_task], 2, _factory(), out, resume=True, echo=lambda s: None)


def test_read_runs_of_a_missing_file_is_empty_and_blank_lines_are_skipped(ladder, tmp_path):
    assert ladder.read_runs(tmp_path / "nothing.jsonl") == []
    path = tmp_path / "runs.jsonl"
    path.write_text('{"a": 1}\n\n   \n{"b": 2}\n', encoding="utf-8")
    assert ladder.read_runs(path) == [{"a": 1}, {"b": 2}]


# --- grading and warm-up failures ------------------------------------------------------------


def test_a_verifier_that_raises_is_a_failed_run_with_the_crash_on_the_line(ladder, toy_task, tmp_path):
    def exploding_verify(workspace_dir: str, final_text: str):
        raise ZeroDivisionError("verify.py bug")

    task = ladder.spike.Task(name="toy", prompt="Fix sum_to.", fixture=toy_task.fixture, verify=exploding_verify)
    runs = ladder.run_matrix([StubLevel(1, "prompted", "t", "a", fixes=True)], [task], 1, _factory(),
                             tmp_path / "out", echo=lambda s: None)
    (r,) = runs
    assert r["passed"] is False
    assert r["verify_note"] == "verify.py crashed: ZeroDivisionError: verify.py bug"
    assert r["stop_reason"] == "done" and r["error"] is None       # the level itself was fine


def test_warm_up_returns_the_error_text_instead_of_raising(ladder):
    class Down(Provider):
        def complete(self, messages, tools):
            raise ConnectionError("ollama is down")

    assert ladder.warm_up(Down()) == "ConnectionError: ollama is down"
    assert ladder.warm_up(ScriptedProvider(42)) is None


# --- --levels -------------------------------------------------------------------------


@pytest.mark.parametrize("text, expected", [
    ("0-5", [0, 1, 2, 3, 4, 5]),
    ("0,2,5", [0, 2, 5]),
    ("3", [3]),
    ("0-1,4", [0, 1, 4]),
    ("5,0", [0, 5]),
])
def test_parse_levels(ladder, text, expected):
    assert ladder.parse_levels(text) == expected


@pytest.mark.parametrize("text", ["", "a", "3-1", "1-"])
def test_parse_levels_rejects_nonsense(ladder, text):
    with pytest.raises(SystemExit):
        ladder.parse_levels(text)


def test_select_levels_keeps_ladder_order_and_rejects_unknown_numbers(ladder):
    levels = [StubLevel(n, k, "t", "a") for n, k in enumerate(["bare", "prompted", "one_tool"])]
    assert [l.key for l in ladder.select_levels(levels, [2, 0])] == ["bare", "one_tool"]
    with pytest.raises(SystemExit, match=r"unknown level\(s\) \[7\]"):
        ladder.select_levels(levels, [0, 7])


def test_levels_arg_round_trips(ladder):
    assert ladder.levels_arg([0, 1, 2, 3, 4, 5]) == "0-5"
    assert ladder.levels_arg([0, 2, 5]) == "0,2,5"
    assert ladder.levels_arg([3]) == "3"


# --- providers --------------------------------------------------------------------------


def test_the_ollama_factory_is_the_spikes_frozen_provider(ladder):
    provider = ladder.make_provider_factory("ollama", "qwen3.5:4b", 0.3)(43)
    assert isinstance(provider, ladder.spike.SpikeProvider)
    options = provider.request_options()
    assert options["seed"] == 43 and options["temperature"] == 0.3
    assert options["num_ctx"] == ladder.spike.NUM_CTX and options["num_predict"] == ladder.spike.NUM_PREDICT
    assert options["presence_penalty"] == 0.0 and provider.max_retries == 0


def test_the_openai_factory_carries_the_frozen_settings_in_the_servers_words(ladder):
    from cornac.providers.openai_compat import OpenAICompatibleProvider

    provider = ladder.make_provider_factory("openai", "Qwen/Qwen3-27B", 0.3, "http://h:8000/v1/")(44)
    assert isinstance(provider, OpenAICompatibleProvider)
    # temperature and seed as the contract says, plus the per-step output cap (the
    # spike's num_predict) and thinking off for a Qwen 3 family model (vLLM's switch).
    assert provider.request_options() == {
        "temperature": 0.3, "seed": 44, "max_tokens": ladder.spike.NUM_PREDICT,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    assert provider.base_url == "http://h:8000/v1" and provider.model == "Qwen/Qwen3-27B"
    assert provider.context_window == ladder.spike.NUM_CTX and provider.retries == 0
    # A model outside the Qwen 3 family gets no template switch (a server whose template
    # has no thinking mode may reject the field), and the same cap.
    other = ladder.make_provider_factory("openai", "meta-llama/Llama-3.3-70B", 0.3, "http://h:8000/v1")(42)
    assert other.request_options() == {"temperature": 0.3, "seed": 42, "max_tokens": ladder.spike.NUM_PREDICT}
    # The same rule the Ollama provider applies: "qwen3" anywhere in the lowercase name.
    assert "chat_template_kwargs" in ladder.openai_options("unsloth/Qwen3.5-27B-GGUF", 42, 0.3)
    assert "chat_template_kwargs" not in ladder.openai_options("Qwen/Qwen2.5-32B", 42, 0.3)


def test_the_openai_payload_actually_carries_max_tokens_and_the_thinking_switch(ladder, monkeypatch):
    provider = ladder.make_provider_factory("openai", "Qwen/Qwen3-27B", 0.3, "http://h:8000/v1")(44)
    sent: list[dict] = []

    def fake_post(url, payload):
        sent.append(payload)
        return {"choices": [{"message": {"role": "assistant", "content": "ready"}, "finish_reason": "stop"}]}

    monkeypatch.setattr(provider, "_post_with_retries", fake_post)
    provider.complete([Message.user("hi")], [])
    (payload,) = sent
    assert payload["max_tokens"] == ladder.spike.NUM_PREDICT == 2048
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["seed"] == 44 and payload["temperature"] == 0.3


# --- the environment: run_bash must be able to run pytest -----------------------------


def test_run_matrix_puts_the_interpreter_first_on_path(ladder, toy_task, tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin:/bin")               # a bare PATH, restored afterwards
    ladder.run_matrix([StubLevel(0, "bare", "t", "a")], [toy_task], 1, _factory(), tmp_path / "out",
                      echo=lambda s: None)
    bin_dir = str(Path(sys.executable).parent)
    assert os.environ["PATH"].split(os.pathsep) == [bin_dir, "/usr/bin", "/bin"]
    # Idempotent: a second matrix in the same process does not grow PATH.
    ladder.put_interpreter_first_on_path()
    assert os.environ["PATH"].split(os.pathsep) == [bin_dir, "/usr/bin", "/bin"]


def test_run_bash_can_run_python_m_pytest_in_a_fixture_copy(ladder, tmp_path, monkeypatch):
    # The smoke test the contract asks for: with the fix applied, `python -m pytest -q`
    # through run_bash — what the model types from level 2 up — runs the venv's pytest
    # inside a copy of the real swe fixture and reports the one failing test.
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))
    bin_dir = ladder.put_interpreter_first_on_path()
    assert shutil.which("python").startswith(bin_dir)
    assert shutil.which("pytest").startswith(bin_dir)

    task = ladder.spike.discover_tasks(["swe"])[0]
    workspace_dir = tmp_path / "workspace"
    shutil.copytree(task.fixture, workspace_dir, ignore=ladder._ignore_caches)
    out = RunBash(Workspace(workspace_dir)).run({"command": "python -m pytest -q -p no:cacheprovider"})

    assert out.startswith("(exit code 1)"), out
    assert "1 failed, 2 passed" in out and "test_sum_to" in out


# --- main(), end to end, with a stub levels.py and the real swe task -------------------

STUB_LEVELS_PY = '''
"""A stand-in for benchmark/ladder/levels.py: one level that writes the swe fix."""
from dataclasses import dataclass
from pathlib import Path
import json

MAX_STEPS = 12
POLICY_RULES = {"default": "deny", "tools": {"run_bash": "allow"}}

FIXED = (
    'def sum_to(n: int) -> int:\\n    return sum(range(n + 1))\\n\\n\\n'
    'def mean(values) -> float:\\n    return sum(values) / len(values)\\n\\n\\n'
    'def clamp(x, lo, hi):\\n    return max(lo, min(x, hi))\\n'
)


@dataclass
class Outcome:
    passed_precheck: bool = True
    final_text: str = "fixed sum_to"
    stop_reason: str = "done"
    steps: int = 1
    usage: object = None
    duration: float = 0.0
    tool_calls: int = 0
    tool_errors: int = 0
    nudges: int = 0
    children: int = 0
    files_applied: list = None
    digest: str = None
    summary: str = None
    messages: list = None


@dataclass
class Level:
    number: int
    key: str
    title: str
    adds: str

    def run(self, provider, workspace, task_prompt, transcript_path):
        reply = provider.complete([], [])
        (workspace.root / "calc.py").write_text(FIXED, encoding="utf-8")
        Path(transcript_path).write_text(json.dumps({"event": "one_shot", "max_steps": MAX_STEPS}) + "\\n")
        return Outcome(usage=reply.usage, files_applied=["calc.py"])


LEVELS = [Level(0, "bare", "Bare", "nothing"), Level(1, "prompted", "Prompted", "a system prompt")]
'''


def test_main_runs_the_matrix_end_to_end_on_the_real_swe_task(
    ladder, tmp_path, monkeypatch, capsys, request
):
    # main() with every flag it owns, a stub levels.py in place of the real one and a
    # scripted provider in place of Ollama; the task, the fixture copy, the grading
    # (verify.py running the venv's pytest) and the report are all real.
    #
    # load_levels registers the module as "ladder_levels"; the stub must not be left
    # there for the real levels' tests to find.
    request.addfinalizer(lambda: sys.modules.pop("ladder_levels", None))
    stub = tmp_path / "levels_stub.py"
    stub.write_text(STUB_LEVELS_PY, encoding="utf-8")
    monkeypatch.setattr(ladder, "LEVELS_PY", stub)
    factory = _factory()
    monkeypatch.setattr(ladder, "make_provider_factory", lambda *a, **k: factory)
    out = tmp_path / "results"

    ladder.main(["--levels", "1", "--tasks", "swe", "--k", "1", "--max-steps", "5",
                 "--out", str(out), "--model", "qwen3.5:4b"])

    runs = [json.loads(l) for l in (out / "runs.jsonl").read_text().splitlines() if l.strip()]
    assert len(runs) == 1 and runs[0]["passed"], runs
    r = runs[0]
    assert (r["level"], r["level_key"], r["task"], r["seed"]) == (1, "prompted", "swe", 42)
    assert r["verify_note"] == "3 passed, sum_to correct on range(50), test_calc.py untouched"
    assert (r["model"], r["provider"], r["max_steps"]) == ("qwen3.5:4b", "ollama", 5)
    assert sys.modules["ladder_levels"].MAX_STEPS == 5        # the flag reached the levels
    assert json.loads((out / r["transcript"]).read_text())["max_steps"] == 5
    assert [p.seed for p in factory.made] == [42, 42]          # the warm-up, then the run
    report = (out / "results.md").read_text()
    assert "| 1 | `prompted` | a system prompt | 1/1 | **1/1 (100%)** |" in report
    assert '"run_bash": "allow"' in report                     # the policy came from levels.py
    assert "--max-steps 5 --out" in report                     # the reproduce command keeps the budget
    printed = capsys.readouterr().out
    assert "1 level(s) x 1 task(s) x k=1 = 1 runs" in printed
    assert "L1 prompted  swe   r0  PASS  1 steps  125 tok" in printed

    # A second main() into the same directory is refused without --resume, and with it
    # nothing is re-run.
    with pytest.raises(SystemExit, match="--resume"):
        ladder.main(["--levels", "1", "--tasks", "swe", "--k", "1", "--out", str(out)])
    ladder.main(["--levels", "1", "--tasks", "swe", "--k", "1", "--out", str(out), "--resume"])
    assert len((out / "runs.jsonl").read_text().splitlines()) == 1


# --- infrastructure retries (the three lost cells of the first full matrix) ----------


@dataclass
class CrashingLevel(StubLevel):
    """Raises the daemon-crash error on its first `crashes` calls, then behaves like StubLevel."""

    crashes: int = 1
    calls: int = 0

    def run(self, provider, workspace, task_prompt, transcript_path):
        self.calls += 1
        if self.calls <= self.crashes:
            raise RuntimeError("Ollama returned 500: model runner has unexpectedly stopped")
        return super().run(provider, workspace, task_prompt, transcript_path)


def test_a_daemon_crash_is_re_run_from_scratch_and_recorded(ladder, toy_task, tmp_path):
    level = CrashingLevel(2, "one_tool", "One tool", "one tool", fixes=True, crashes=1)
    waits: list[float] = []
    lines: list[str] = []
    runs = ladder.run_matrix([level], [toy_task], 1, _factory(), tmp_path / "out",
                             echo=lines.append, sleep=waits.append)
    assert len(runs) == 1 and level.calls == 2          # crashed once, re-run once
    r = runs[0]
    assert r["passed"] and r["error"] is None and r["stop_reason"] == "done"
    assert r["infra_retries"] == 1
    assert r["infra_errors"] == ["RuntimeError: Ollama returned 500: model runner has unexpectedly stopped"]
    assert waits == [ladder.INFRA_WAIT_SECONDS]
    assert any("infrastructure error" in line for line in lines)
    assert "re-run from scratch" in ladder.build_report(runs, k=1, out_arg="out")


def test_a_persistent_crash_is_recorded_after_the_last_retry(ladder, toy_task, tmp_path):
    level = CrashingLevel(2, "one_tool", "One tool", "one tool", crashes=99)
    runs = ladder.run_matrix([level], [toy_task], 1, _factory(), tmp_path / "out",
                             echo=lambda s: None, sleep=lambda s: None)
    r = runs[0]
    assert level.calls == 1 + ladder.INFRA_RETRIES
    assert r["stop_reason"] == "error" and not r["passed"]
    assert r["infra_retries"] == ladder.INFRA_RETRIES and len(r["infra_errors"]) == ladder.INFRA_RETRIES


def test_other_errors_are_not_retried(ladder, toy_task, tmp_path):
    level = StubLevel(2, "one_tool", "One tool", "one tool", raises=True)   # "the daemon said no"
    runs = ladder.run_matrix([level], [toy_task], 1, _factory(), tmp_path / "out",
                             echo=lambda s: None, sleep=lambda s: pytest.fail("must not wait"))
    assert len(level.seen) == 1 and runs[0]["infra_retries"] == 0
    assert not ladder.is_infrastructure_error(runs[0]["error"])
    assert ladder.is_infrastructure_error("HTTPStatusError: Ollama returned 500: model runner has unexpectedly stopped")
    assert not ladder.is_infrastructure_error("HTTPStatusError: Ollama returned 500: context length exceeded")
