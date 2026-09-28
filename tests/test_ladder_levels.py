"""Tests for the six ladder levels (benchmark/ladder/levels.py).

The ladder's claim is "each level adds exactly one thing", and that claim is only as
good as the definitions: if level 1's user message drifted from level 0's, or level 3
quietly got the syntax gate, the table would be comparing two things at once. So these
tests pin the definitions, not the model: a scripted provider plays the model, entirely
offline, and the tests check what each level SENDS it (messages, tools, settings) and
what each level DOES with what comes back (apply_answer, the outcome, the transcript).

The one end-to-end check — a level-0 run whose scripted reply is a corrected calc.py,
graded by the real swe verify.py on a copied fixture — is the smallest proof that the
one-shot path produces something verify.py can pass.

The module is imported by its dotted name, the way the examples import it; one test
loads it by file path instead, the way the runner does.
"""

from __future__ import annotations

import importlib
import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

import functools
import py_compile
import re

from cornac import Message, Usage
from cornac.core.messages import ToolCall
from cornac.hooks.bus import HookBus
from cornac.permissions.policy import Decision
from cornac.providers.base import Provider
from cornac.tools.builtin.python_exec import RUN_PYTHON_DESCRIPTION_ROUND2, RUN_PYTHON_DESCRIPTION_ROUND3
from cornac.tools.builtin.shell import RUN_BASH_DESCRIPTION_ROUND2, RUN_BASH_DESCRIPTION_ROUND3
from cornac.tools.workspace import Workspace
from cornac.transcript import read_transcript

REPO_ROOT = Path(__file__).resolve().parent.parent
LEVELS_PY = REPO_ROOT / "benchmark" / "ladder" / "levels.py"
SWE_FIXTURE = REPO_ROOT / "benchmark" / "spike" / "tasks" / "swe" / "fixture"
STAGE_SCRIPTS = sorted((REPO_ROOT / "examples").glob("stage_*.py"))

TASK = "Run the tests, find the bug, fix it in calc.py. Do not edit test_calc.py."

# The honest fix for the swe task: range(n + 1) instead of range(n).
FIXED_CALC = (
    '"""A tiny calculator module."""\n'
    "\n"
    "\n"
    "def sum_to(n: int) -> int:\n"
    '    """Return 0 + 1 + ... + n, inclusive of n."""\n'
    "    return sum(range(n + 1))\n"
    "\n"
    "\n"
    "def mean(values) -> float:\n"
    '    """Arithmetic mean of a non-empty sequence of numbers."""\n'
    "    return sum(values) / len(values)\n"
    "\n"
    "\n"
    "def clamp(x, lo, hi):\n"
    '    """Clamp x into the closed interval [lo, hi]."""\n'
    "    return max(lo, min(x, hi))\n"
)


# --- plumbing -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def levels():
    """benchmark.ladder.levels, imported by name. The test bootstraps sys.path the way
    the runner and the examples do; levels.py itself must not."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return importlib.import_module("benchmark.ladder.levels")


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    """A fresh copy of the swe fixture, caches left behind, as the runner makes one."""
    target = tmp_path / "workspace"
    shutil.copytree(SWE_FIXTURE, target, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    return Workspace(target)


class ScriptedProvider(Provider):
    """Plays the model from an ordered list of replies and records what it was shown."""

    context_window = 8192  # like OllamaProvider at num_ctx 8192: level 4 must inherit it
    model = "scripted"

    def __init__(self, script=()):
        self._script = list(script)
        self.calls: list[tuple[list[Message], list[dict]]] = []  # (messages, tool schemas) per call

    def complete(self, messages, tools):
        self.calls.append((list(messages), list(tools)))
        if not self._script:
            raise RuntimeError("script exhausted")
        return self._script.pop(0)


def _reply(text: str) -> Message:
    return Message(role="assistant", text=text, usage=Usage(input_tokens=100, output_tokens=20), stop_reason="stop")


def _tool_turn(name: str, call_id: str = "c1", **arguments) -> Message:
    return Message(role="assistant", tool_calls=[ToolCall(call_id, name, arguments)], usage=Usage(50, 10))


def _fenced(path: str, content: str, lang: str = "python", nl: str = "\n") -> str:
    return nl.join([f"```{lang}", f"# file: {path}", *content.splitlines(), "```"]) + nl


def _agent_levels(levels):
    return [level for level in levels.LEVELS if not level.one_shot]


# --- the ladder itself ------------------------------------------------------------------


def test_six_levels_in_order_with_the_contract_keys_and_adds(levels):
    assert [lv.number for lv in levels.LEVELS] == [0, 1, 2, 3, 4, 5]
    assert [lv.key for lv in levels.LEVELS] == ["bare", "prompted", "one_tool", "tools", "harness", "agents"]
    # LEVELS[n] is level n: the runner and the examples index it that way.
    assert all(levels.LEVELS[n].number == n for n in range(6))
    assert [lv.adds for lv in levels.LEVELS] == [
        "nothing: the question and the code, one model call",
        "a system prompt",
        "one tool, run_bash: the files are no longer pasted in; the agent loop begins here",
        "the file tools: read_file, write_file, list_dir, grep, run_python (no syntax check)",
        "edit_file, syntax gate, change report, nudge, repeat note/stop, wrap-up, context clearing",
        "sub-agents",
    ]


def test_adds_is_written_for_a_stranger_and_fits_a_table_cell(levels):
    # The adds column is the hero table's third column, read by people who have never
    # seen this repo: no round or week numbers (those live in the docstrings), and
    # short enough that the runner's cell cap never cuts it with an ellipsis.
    for level in levels.LEVELS:
        assert len(level.adds) <= levels.ADDS_MAX_CHARS, (level.number, len(level.adds))
        assert not re.search(r"\bround[- ]?\d|\bweeks?\b|\b4[bc]\b", level.adds, re.IGNORECASE), level.adds
        assert "|" not in level.adds and "\n" not in level.adds


def test_get_level_by_number_or_key(levels):
    assert levels.get_level(3) is levels.LEVELS[3]
    assert levels.get_level("harness") is levels.LEVELS[4]
    with pytest.raises(KeyError, match="no level 'rung'"):
        levels.get_level("rung")


def test_levels_0_and_1_are_one_shot_and_the_rest_are_agents(levels):
    assert [lv.one_shot for lv in levels.LEVELS] == [True, True, False, False, False, False]
    assert levels.LEVELS[0].loop is None and levels.LEVELS[1].loop is None
    with pytest.raises(ValueError, match="one model call"):
        levels.LEVELS[0].build_agent(ScriptedProvider(), Workspace("."))


def test_the_module_loads_by_file_path_too(levels):
    # The runner loads levels.py with importlib (registered in sys.modules first, for the
    # dataclasses). The two copies must describe the same ladder.
    name = "ladder_levels_loaded_by_path"
    spec = importlib.util.spec_from_file_location(name, LEVELS_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        assert [lv.key for lv in module.LEVELS] == [lv.key for lv in levels.LEVELS]
        assert module.POLICY_RULES == levels.POLICY_RULES
    finally:
        sys.modules.pop(name, None)


# --- levels 0 and 1: the one-shot prompt -------------------------------------------------


def test_level_0_sends_no_system_message_and_one_user_message_with_no_tools(levels, workspace, tmp_path):
    provider = ScriptedProvider([_reply("no idea")])
    levels.LEVELS[0].run(provider, workspace, TASK, tmp_path / "t.jsonl")
    (messages, tools), = provider.calls
    assert [m.role for m in messages] == ["user"]
    assert tools == []


def test_level_1_adds_a_system_prompt_and_keeps_level_0s_user_message_verbatim(levels, workspace, tmp_path):
    bare = ScriptedProvider([_reply("")])
    prompted = ScriptedProvider([_reply("")])
    levels.LEVELS[0].run(bare, workspace, TASK, tmp_path / "t0.jsonl")
    levels.LEVELS[1].run(prompted, workspace, TASK, tmp_path / "t1.jsonl")
    (bare_messages, _), = bare.calls
    (prompted_messages, tools), = prompted.calls

    assert [m.role for m in prompted_messages] == ["system", "user"]
    assert prompted_messages[1].text == bare_messages[0].text  # IDENTICAL: level 1 adds the system prompt only
    assert tools == []
    system = prompted_messages[0].text
    assert system.startswith(levels.ONE_SHOT_PERSONA)
    assert system == levels.ONE_SHOT_PERSONA + "\n\nWorkspace: calc.py, test_calc.py"
    # The contract's level-1 persona, byte for byte.
    assert levels.ONE_SHOT_PERSONA == (
        "You are a careful software engineer. Fix the failing test by changing as little as "
        "possible in the source, never the test file. Reply only with the corrected file(s) "
        "in the fenced-block format requested."
    )


def test_the_persona_core_is_the_same_words_at_level_1_and_at_the_agent_levels(levels):
    # Level 1 -> 2 must measure the tool, not a reworded brief: both personas are one
    # shared core (who, what to fix, how little to change, what never to touch) plus a
    # tail that changes only because it must (the answer format vs the procedure).
    assert levels.ONE_SHOT_PERSONA == levels.PERSONA_CORE + " " + levels.ONE_SHOT_TAIL
    assert levels.TOOLS_PERSONA == levels.PERSONA_CORE + " " + levels.TOOLS_TAIL
    for text in (levels.PERSONA_CORE, levels.ONE_SHOT_PERSONA, levels.TOOLS_PERSONA):
        assert "careful software engineer" in text
        assert "as little as possible" in text          # the minimal-change instruction, everywhere
        assert "never the test file" in text
    assert "fenced-block format" in levels.ONE_SHOT_TAIL and "fenced" not in levels.TOOLS_TAIL
    assert "run the tests again" in levels.TOOLS_TAIL and "which function you fixed" in levels.TOOLS_TAIL


def test_the_bare_message_is_prompt_then_every_file_then_the_answer_format(levels, workspace):
    text = levels.bare_user_message(TASK, workspace)
    calc = (workspace.root / "calc.py").read_text()
    tests = (workspace.root / "test_calc.py").read_text()

    assert text.startswith(TASK + "\n\n" + levels.FILES_HEADER)
    assert text.endswith(levels.ANSWER_FORMAT)
    assert "```python\n# file: calc.py\n" + calc + "```" in text
    assert "```python\n# file: test_calc.py\n" + tests + "```" in text
    assert text.index("# file: calc.py") < text.index("# file: test_calc.py")  # sorted, so stable


def test_inlining_skips_caches_bytecode_and_binary_files(levels, workspace):
    (workspace.root / "__pycache__").mkdir()
    (workspace.root / "__pycache__" / "calc.cpython-312.pyc").write_bytes(b"\x00garbage")
    (workspace.root / "stray.pyc").write_bytes(b"\x00")
    (workspace.root / ".pytest_cache").mkdir()
    (workspace.root / ".pytest_cache" / "v").write_text("cache")
    (workspace.root / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe")
    (workspace.root / "pkg").mkdir()
    (workspace.root / "pkg" / "notes.txt").write_text("plain text\n")

    assert levels.workspace_files(workspace) == ["calc.py", "pkg/notes.txt", "test_calc.py"]
    text = levels.bare_user_message(TASK, workspace)
    assert "pyc" not in text and "logo.png" not in text and "cache" not in text
    assert "```\n# file: pkg/notes.txt\nplain text\n```" in text  # no language tag for a .txt


def test_a_file_containing_a_fence_gets_a_longer_fence(levels):
    block = levels.fenced_file("README.md", "docs:\n```sh\nls\n```\n")
    assert block.startswith("````\n# file: README.md\n") and block.endswith("\n````")
    assert levels.file_blocks(block) == [("README.md", "docs:\n```sh\nls\n```\n")]


# --- apply_answer: the model's hands at levels 0 and 1 ----------------------------------


def test_apply_answer_writes_a_file_block(levels, workspace):
    applied = levels.apply_answer("Here is the fix:\n\n" + _fenced("calc.py", FIXED_CALC), workspace)
    assert applied.written == ["calc.py"] and applied.rejected == []
    assert (workspace.root / "calc.py").read_text() == FIXED_CALC


def test_apply_answer_refuses_a_path_outside_the_workspace_and_counts_it(levels, workspace, tmp_path):
    text = _fenced("../evil.py", "x = 1\n") + "\n" + _fenced("/tmp/evil.py", "x = 1\n") + "\n" + _fenced("ok.py", "x = 1\n")
    applied = levels.apply_answer(text, workspace)
    assert applied.rejected == ["../evil.py", "/tmp/evil.py"]
    assert applied.written == ["ok.py"]
    assert not (tmp_path / "evil.py").exists()  # ../ from the workspace root is tmp_path
    assert (workspace.root / "ok.py").read_text() == "x = 1\n"


def test_apply_answer_accepts_plain_fences_language_tags_and_crlf(levels, workspace):
    plain = _fenced("a.py", "a = 1\n", lang="")
    tagged = _fenced("b.py", "b = 2\n", lang="py")
    crlf = _fenced("c.py", "c = 3\nd = 4\n", nl="\r\n")
    applied = levels.apply_answer("\r\n".join([plain, tagged, crlf]), workspace)
    assert applied.written == ["a.py", "b.py", "c.py"]
    assert (workspace.root / "c.py").read_bytes() == b"c = 3\nd = 4\n"  # written LF, no stray \r


def test_apply_answer_with_no_blocks_writes_nothing(levels, workspace):
    before = levels.snapshot(workspace)
    applied = levels.apply_answer("The bug is in sum_to: range(n) should be range(n + 1).", workspace)
    assert applied.written == [] and applied.rejected == []
    assert levels.snapshot(workspace) == before


def test_apply_answer_ignores_a_fenced_block_without_a_file_line(levels, workspace):
    before = levels.snapshot(workspace)
    text = "```python\nreturn sum(range(n + 1))\n```\n\n```bash\npytest -q\n```\n"
    assert levels.apply_answer(text, workspace).written == []
    assert levels.snapshot(workspace) == before


def test_apply_answer_normalises_the_path_and_creates_parent_directories(levels, workspace):
    text = _fenced("./calc.py", FIXED_CALC) + "\n" + _fenced("`pkg/sub/new.py`", "z = 1\n")
    applied = levels.apply_answer(text, workspace)
    assert applied.written == ["calc.py", "pkg/sub/new.py"]
    assert (workspace.root / "pkg" / "sub" / "new.py").read_text() == "z = 1\n"


@pytest.mark.parametrize("block", [
    "```python\n# File: calc.py\nx = 1\n```\n",          # a capitalised heading, echoed
    "```python\n# FILE: calc.py\nx = 1\n```\n",
    "```python\n#file:calc.py\nx = 1\n```\n",             # no spaces at all
    "```python\n\n# file: calc.py\nx = 1\n```\n",         # a blank line after the fence
    "```python\n   \n\n# File: calc.py\nx = 1\n```\n",   # several, some with spaces
    "```\n# file: 'calc.py'\nx = 1\n```\n",                # quoted path, no language tag
], ids=["File", "FILE", "nospace", "blank-line", "blank-lines", "quoted"])
def test_the_file_marker_is_forgiving_about_case_and_leading_blank_lines(levels, workspace, block):
    # The marker is the harness's answer protocol, not the task: a model that solved
    # the task and wrote `# File:` must not fail the row on the spelling. The path
    # itself keeps its case, and the contents are exactly what followed the marker.
    assert levels.file_blocks(block) == [("calc.py", "x = 1\n")]
    applied = levels.apply_answer(block, workspace)
    assert applied.written == ["calc.py"] and (workspace.root / "calc.py").read_text() == "x = 1\n"


def test_the_file_marker_keeps_the_paths_case(levels):
    assert levels.file_blocks("```\n# FILE: Pkg/Calc.PY\nx\n```\n") == [("Pkg/Calc.PY", "x\n")]


@pytest.mark.parametrize("block", [
    "```python\nreturn sum(range(n + 1))\n```\n",   # a bare snippet
    "```python\n# calc.py\nx = 1\n```\n",          # a comment naming the file is not the marker
    "```python\n# the file: calc.py\nx = 1\n```\n",
    "```python\nx = 1\n# file: calc.py\n```\n",    # the marker must come first
    "```python\n\n\n```\n",                        # nothing but blank lines
], ids=["snippet", "bare-name", "prose", "not-first", "empty"])
def test_a_block_without_the_marker_first_is_not_a_file(levels, block):
    assert levels.file_blocks(block) == []


def test_an_unfenced_reply_is_read_by_its_marker_lines_when_it_uses_no_fence_at_all(levels, workspace):
    # The first smoke run (2026-09-27): the bare model at level 0 answered with the
    # complete, correct calc.py under a `# file: calc.py` line and no fence at all, and
    # the row failed on the missing decoration, not on the code. The fence is protocol
    # exactly as the marker's spelling is, so a reply that used no fence is read by its
    # markers (verify.py grades the result: a guessed boundary can only fail a row).
    text = "# file: calc.py\n" + FIXED_CALC
    assert levels.file_blocks(text) == [("calc.py", FIXED_CALC)]
    applied = levels.apply_answer(text, workspace)
    assert applied.written == ["calc.py"] and applied.rejected == []
    assert (workspace.root / "calc.py").read_text() == FIXED_CALC


def test_unfenced_blocks_run_to_the_next_marker_and_end_with_one_newline(levels):
    text = "The fixes:\n\n# File: a.py\na = 1\n\n\n#file:b.py\nb = 2"
    assert levels.file_blocks(text) == [("a.py", "a = 1\n"), ("b.py", "b = 2\n")]
    # Prose that never names a file is still not a file, unfenced or not.
    assert levels.file_blocks("The bug is in sum_to: range(n) should be range(n + 1).") == []


def test_the_unfenced_fallback_stays_off_once_the_reply_uses_any_fence(levels, workspace):
    # A model speaking the fenced protocol is held to it: a fence that never closes (the
    # output cap) writes nothing rather than a torn file, and a bare marker next to a
    # fenced snippet is not a file either.
    cut = "```python\n# file: calc.py\ndef sum_to(n):\n    return sum(ra"
    assert levels.file_blocks(cut) == []
    mixed = "# file: calc.py\nThe fix, in short:\n\n```python\nreturn sum(range(n + 1))\n```\n"
    assert levels.file_blocks(mixed) == []
    before = levels.snapshot(workspace)
    for text in (cut, mixed):
        assert levels.apply_answer(text, workspace).written == []
    assert levels.snapshot(workspace) == before


def test_a_marker_line_right_before_the_fence_names_the_block(levels, workspace):
    # The other place a model puts the marker: as a heading above the fence.
    text = "Here it is.\n\n# file: calc.py\n```python\n" + FIXED_CALC + "```\n"
    assert levels.file_blocks(text) == [("calc.py", FIXED_CALC)]
    assert levels.apply_answer(text, workspace).written == ["calc.py"]
    assert (workspace.root / "calc.py").read_text() == FIXED_CALC
    # A heading that is not the marker does not name the block; the marker inside a
    # fence still wins over one above it.
    assert levels.file_blocks("calc.py:\n```python\nx = 1\n```\n") == []
    inside_wins = "# file: outer.py\n```python\n# file: inner.py\nx = 1\n```\n"
    assert levels.file_blocks(inside_wins) == [("inner.py", "x = 1\n")]


def test_apply_answer_refuses_a_directory_as_a_file(levels, workspace):
    (workspace.root / "pkg").mkdir()
    applied = levels.apply_answer(_fenced(".", "x\n") + _fenced("pkg", "x\n"), workspace)
    assert applied.rejected == [".", "pkg"] and applied.written == []


def test_the_inlined_files_round_trip_through_apply_answer_byte_for_byte(levels, workspace, tmp_path):
    # What level 0 pastes in is exactly what a model echoing it back would write out.
    before = {rel: (workspace.root / rel).read_bytes() for rel in levels.workspace_files(workspace)}
    text = levels.bare_user_message(TASK, workspace)
    other = tmp_path / "other"
    other.mkdir()
    applied = levels.apply_answer(text, Workspace(other))
    assert applied.written == sorted(before)
    assert {rel: (other / rel).read_bytes() for rel in applied.written} == before


# --- a level-0 run end to end ------------------------------------------------------------


def test_a_level_0_run_with_a_scripted_fix_passes_the_real_verifier(levels, workspace, tmp_path, spike):
    reply = "The bug is in sum_to.\n\n" + _fenced("calc.py", FIXED_CALC) + "\nThat is all.\n"
    provider = ScriptedProvider([_reply(reply)])
    transcript = tmp_path / "transcripts" / "L0-swe-r0.jsonl"

    outcome = levels.LEVELS[0].run(provider, workspace, TASK, transcript)

    task = spike.load_task(spike.TASKS_DIR / "swe")
    passed, note = task.verify(str(workspace.root), outcome.final_text or "")
    assert passed, note

    assert outcome.passed_precheck is True
    assert outcome.stop_reason == "done" and outcome.steps == 1
    assert outcome.final_text == reply
    assert outcome.files_applied == ["calc.py"] and outcome.rejected == 0
    assert outcome.tool_calls == 0 and outcome.tool_errors == 0 and outcome.nudges == 0 and outcome.children == 0
    assert outcome.digest is None and outcome.summary is None
    assert outcome.usage.total_tokens == 120 and outcome.duration >= 0
    assert [m.role for m in outcome.messages] == ["user", "assistant"]

    records = read_transcript(transcript)
    assert [r["event"] for r in records] == [levels.ONE_SHOT_EVENT]
    record = records[0]
    assert record["level"] == 0 and record["key"] == "bare" and record["model"] == "scripted"
    assert record["system_prompt"] is None and record["user_message"] == outcome.messages[0].text
    assert record["reply"]["text"] == reply and record["files_applied"] == ["calc.py"]


def test_a_reply_with_no_file_block_fails_the_precheck_and_changes_nothing(levels, workspace, tmp_path):
    provider = ScriptedProvider([_reply("The fix is range(n + 1).")])
    outcome = levels.LEVELS[1].run(provider, workspace, TASK, tmp_path / "t.jsonl")
    assert outcome.passed_precheck is False
    assert outcome.files_applied == [] and outcome.rejected == 0
    assert outcome.stop_reason == "done"  # the model did answer; it just answered uselessly


def test_a_one_shot_reply_that_echoes_a_file_unchanged_applied_nothing(levels, workspace, tmp_path):
    original = (workspace.root / "calc.py").read_text()
    provider = ScriptedProvider([_reply(_fenced("calc.py", original))])
    outcome = levels.LEVELS[0].run(provider, workspace, TASK, tmp_path / "t.jsonl")
    assert outcome.passed_precheck is True  # a block was written...
    assert outcome.files_applied == []      # ...but nothing on disk changed, and the table says so


def test_a_rejected_block_is_counted_on_the_outcome(levels, workspace, tmp_path):
    provider = ScriptedProvider([_reply(_fenced("../escape.py", "x\n") + _fenced("calc.py", FIXED_CALC))])
    outcome = levels.LEVELS[0].run(provider, workspace, TASK, tmp_path / "t.jsonl")
    assert outcome.rejected == 1 and outcome.files_applied == ["calc.py"]


def test_a_provider_failure_is_recorded_in_the_transcript_and_re_raised(levels, workspace, tmp_path):
    transcript = tmp_path / "t.jsonl"
    with pytest.raises(RuntimeError, match="script exhausted"):
        levels.LEVELS[0].run(ScriptedProvider([]), workspace, TASK, transcript)
    records = read_transcript(transcript)
    assert [r["event"] for r in records] == [levels.RUN_ERROR_EVENT]
    assert records[0]["error"] == "RuntimeError: script exhausted" and records[0]["level"] == 0


# --- levels 2 to 5: what the agent is built with -----------------------------------------


def test_registry_names_per_level_match_the_table(levels, workspace):
    names = {lv.number: lv.build_agent(ScriptedProvider(), workspace).registry.names() for lv in _agent_levels(levels)}
    assert names == {
        2: ["run_bash"],
        3: ["read_file", "write_file", "list_dir", "grep", "run_bash", "run_python"],
        4: ["read_file", "write_file", "edit_file", "list_dir", "grep", "run_bash", "run_python"],
        5: ["read_file", "write_file", "edit_file", "list_dir", "grep", "run_bash", "run_python", "spawn_agent"],
    }


def test_write_file_gate_and_report_are_off_at_level_3(levels, workspace):
    write = levels.LEVELS[3].registry(workspace).get("write_file")
    # Gate off: a .py file that does not parse lands on disk, as it did in round two...
    result = write.run({"path": "broken.py", "content": "def broken(:\n"})
    assert (workspace.root / "broken.py").read_text() == "def broken(:\n"
    assert result == "wrote broken.py: 13 chars"
    # ...and report off: a rewrite that drops two functions is not called out.
    result = write.run({"path": "calc.py", "content": "def sum_to(n):\n    return n\n"})
    assert result == "wrote calc.py: 28 chars"
    assert "REMOVED" not in result and "overwrote" not in result


@pytest.mark.parametrize("number", [4, 5])
def test_write_file_gate_and_report_are_on_at_levels_4_and_5(levels, workspace, number):
    write = levels.LEVELS[number].registry(workspace).get("write_file")
    result = write.run({"path": "broken.py", "content": "def broken(:\n"})
    assert result.startswith("Error: refused to write broken.py")
    assert not (workspace.root / "broken.py").exists()
    result = write.run({"path": "calc.py", "content": "def sum_to(n):\n    return n\n"})
    assert "REMOVED top-level: def mean, def clamp" in result


@pytest.mark.parametrize("number", [4, 5])
def test_edit_file_gate_is_on_at_levels_4_and_5(levels, workspace, number):
    edit = levels.LEVELS[number].registry(workspace).get("edit_file")
    result = edit.run({"path": "calc.py", "old": "    return sum(range(n))\n", "new": "    return sum(range(n + 1)\n"})
    assert result.startswith("Error: refused to edit calc.py")
    assert "range(n))" in (workspace.root / "calc.py").read_text()  # untouched


def test_loop_settings_per_level(levels, workspace):
    def settings(agent):
        return (agent.max_nudges, agent.repeat_note, agent.max_repeats, agent.wrap_up, agent.context_window)

    provider = ScriptedProvider()  # context_window 8192, like the Ollama provider
    guards_off = (0, False, 0, False, 0)
    guards_on = (1, True, 3, True, 8192)  # None in the level -> the provider's window
    assert settings(levels.LEVELS[2].build_agent(provider, workspace)) == guards_off
    assert settings(levels.LEVELS[3].build_agent(provider, workspace)) == guards_off
    assert settings(levels.LEVELS[4].build_agent(provider, workspace)) == guards_on
    assert settings(levels.LEVELS[5].build_agent(provider, workspace)) == guards_on
    assert levels.LEVELS[2].loop is levels.GUARDS_OFF and levels.LEVELS[4].loop is levels.GUARDS_ON


def test_the_fixed_conditions_are_the_same_at_every_agent_level(levels, workspace):
    assert levels.POLICY_RULES == {
        "default": "deny",
        "tools": {
            "read_file": "allow", "write_file": "allow", "edit_file": "allow", "list_dir": "allow",
            "grep": "allow", "run_python": "allow", "spawn_agent": "allow",
            "run_bash": {"deny": ["rm -rf", "sudo", "mkfs", "dd if=", ":(){"], "default": "allow"},
        },
    }
    for level in _agent_levels(levels):
        agent = level.build_agent(ScriptedProvider(), workspace)
        assert agent.max_steps == 12
        assert agent.approver is None  # headless: an ASK is a DENY
        policy = agent.policy
        assert policy.check(ToolCall("c", "run_bash", {"command": "python -m pytest -q"})) == Decision.ALLOW
        assert policy.check(ToolCall("c", "run_bash", {"command": "sudo rm -rf /"})) == Decision.DENY
        assert policy.check(ToolCall("c", "web_fetch", {"url": "http://x"})) == Decision.DENY  # default deny


def test_levels_2_to_4_share_one_system_prompt_and_level_5_adds_only_the_delegation_sentence(levels, workspace):
    prompts = {lv.number: lv.system_prompt_for(workspace) for lv in _agent_levels(levels)}
    assert prompts[3] == prompts[4]
    assert prompts[4] == levels.TOOLS_PERSONA + "\n\n" + workspace.describe()
    assert levels.DELEGATION_SENTENCE not in prompts[4]
    assert levels.DELEGATION_SENTENCE in prompts[5]
    assert prompts[5].replace(" " + levels.DELEGATION_SENTENCE, "") == prompts[4]
    assert str(workspace.root) in prompts[2]  # Workspace.describe(): the absolute root
    # Level 2 has the same persona; its workspace paragraph differs in one clause only,
    # the cwd sentence, which names the run tools the level HAS (run_bash, no run_python).
    assert prompts[2] == levels.TOOLS_PERSONA + "\n\n" + workspace.describe(tools=("run_bash",))
    assert prompts[2].startswith(levels.TOOLS_PERSONA + "\n\n")
    assert "run_python" not in prompts[2] and "run_bash already runs inside" in prompts[2]
    assert prompts[3].replace("run_bash and run_python already run inside", "run_bash already runs inside") == prompts[2]


ALL_TOOL_NAMES = ("read_file", "write_file", "edit_file", "list_dir", "grep", "run_bash", "run_python", "spawn_agent")


def _named_tools(text: str) -> set[str]:
    return {name for name in ALL_TOOL_NAMES if re.search(rf"\b{name}\b", text or "")}


def test_no_text_a_level_sends_names_a_tool_that_level_does_not_have(levels, workspace):
    # A model told about a tool it does not have calls a tool that does not exist
    # ("Unknown tool": a burnt step and a tool error) or believes it cannot do what it
    # can. So every tool description and the system prompt name only present tools.
    for level in _agent_levels(levels):
        have = set(level.tools)
        for schema in level.registry(workspace).schemas():
            phantom = _named_tools(schema["description"]) - have
            assert not phantom, f"level {level.number}: {schema['name']} names {sorted(phantom)}"
        phantom = _named_tools(level.system_prompt_for(workspace)) - have
        assert not phantom, f"level {level.number}: the system prompt names {sorted(phantom)}"


def test_levels_2_and_3_read_round_twos_run_tool_descriptions_and_levels_4_and_5_todays(levels, workspace):
    def description(level_number, name):
        return levels.LEVELS[level_number].registry(workspace).get(name).description

    # Level 2: the one tool must not say it cannot edit files (the shell is the editor
    # there) nor point at edit_file/write_file, which the level does not have.
    two = description(2, "run_bash")
    assert two == RUN_BASH_DESCRIPTION_ROUND2
    assert "edit_file" not in two and "write_file" not in two and "NOT edit" not in two
    # Level 3 is the round-two harness: round two's text for both run tools, verbatim.
    assert description(3, "run_bash") == RUN_BASH_DESCRIPTION_ROUND2
    assert description(3, "run_python") == RUN_PYTHON_DESCRIPTION_ROUND2
    assert all("edit_file" not in s["description"] for s in levels.LEVELS[3].registry(workspace).schemas())
    # Levels 4 and 5: today's descriptions, byte for byte, edit_file hint included.
    for number in (4, 5):
        assert description(number, "run_bash") == RUN_BASH_DESCRIPTION_ROUND3
        assert description(number, "run_python") == RUN_PYTHON_DESCRIPTION_ROUND3
        assert "edit_file" in description(number, "run_bash")


def test_agent_levels_send_the_task_prompt_alone_and_advertise_their_tools(levels, workspace, tmp_path):
    for level in _agent_levels(levels):
        provider = ScriptedProvider([_reply("done")])
        level.run(provider, workspace, TASK, tmp_path / f"L{level.number}.jsonl")
        (messages, tools), = provider.calls
        assert [m.role for m in messages] == ["system", "user"]
        assert messages[1].text == TASK  # no inlined files: the model can look for itself
        assert [t["name"] for t in tools] == list(level.tools)


# --- levels 2 to 5: what a run hands back -------------------------------------------------


def test_a_level_2_run_counts_its_tool_calls_and_writes_a_hook_transcript(levels, workspace, tmp_path):
    provider = ScriptedProvider([
        _tool_turn("run_bash", command="printf 'hi\\n' > note.txt"),
        _reply("I wrote a note."),
    ])
    transcript = tmp_path / "L2.jsonl"
    outcome = levels.LEVELS[2].run(provider, workspace, TASK, transcript)

    assert outcome.passed_precheck is True
    assert outcome.stop_reason == "done" and outcome.steps == 2
    assert outcome.final_text == "I wrote a note."
    assert outcome.tool_calls == 1 and outcome.tool_errors == 0
    assert outcome.files_applied == ["note.txt"]  # the shell's edit, seen on disk
    assert outcome.rejected == 0 and outcome.nudges == 0 and outcome.children == 0
    assert outcome.usage.total_tokens == 180
    assert [m.role for m in outcome.messages] == ["system", "user", "assistant", "tool", "assistant"]

    events = [r["event"] for r in read_transcript(transcript)]
    assert events[0] == "run_config" and events[-1] == "on_run_end"
    assert "post_tool_use" in events
    config = read_transcript(transcript)[0]
    assert config["tools"] == ["run_bash"] and config["max_steps"] == 12 and config["model"] == "scripted"


def test_a_callers_hooks_see_the_run_as_it_happens_and_the_level_still_counts(levels, workspace, tmp_path):
    # Level.run(hooks=...) makes the caller's bus the run's bus (the examples print
    # each tool call live from it); the level's own observers ride on the same bus.
    seen: list[str] = []
    bus = HookBus()
    bus.on("post_tool_use", lambda call, result: seen.append(f"{call.name}:{result.content[:12]}"))
    bus.on("on_assistant_message", lambda m: seen.append("assistant"))
    provider = ScriptedProvider([_tool_turn("run_bash", command="echo hi"), _reply("hi said")])
    transcript = tmp_path / "L2.jsonl"

    outcome = levels.LEVELS[2].run(provider, workspace, TASK, transcript, hooks=bus)

    assert seen == ["assistant", "run_bash:(exit code 0", "assistant"]
    assert outcome.tool_calls == 1 and outcome.steps == 2          # the counter still counted
    assert "post_tool_use" in [r["event"] for r in read_transcript(transcript)]  # and the writer wrote
    # A one-shot level fires no hooks and ignores the bus rather than failing.
    seen.clear()
    levels.LEVELS[0].run(ScriptedProvider([_reply("no")]), workspace, TASK, tmp_path / "L0.jsonl", hooks=bus)
    assert seen == []


def test_a_level_5_run_through_spawn_agent_counts_the_child_and_its_calls(levels, workspace, tmp_path):
    # The parent delegates a read; the child reads and answers; the parent answers.
    # Script order is call order: parent, child, child, parent.
    provider = ScriptedProvider([
        _tool_turn("spawn_agent", "s1", task="read calc.py and report its functions"),
        _tool_turn("read_file", "c1", path="calc.py"),
        _reply("calc.py defines sum_to, mean and clamp."),
        _reply("The child read it; nothing to fix here."),
    ])
    transcript = tmp_path / "L5.jsonl"
    outcome = levels.LEVELS[5].run(provider, workspace, TASK, transcript)

    assert outcome.children == 1
    assert outcome.tool_calls == 2 and outcome.tool_errors == 0   # the child's read_file counted too
    assert outcome.steps == 2                                       # the parent's own calls only
    assert outcome.stop_reason == "done" and outcome.final_text == "The child read it; nothing to fix here."
    assert outcome.files_applied == []
    assert outcome.usage.total_tokens == 60 * 2 + 120 * 2           # every model call, the child's included
    # The child's turns never enter the parent's message list...
    assert [m.role for m in outcome.messages] == ["system", "user", "assistant", "tool", "assistant"]
    assert outcome.messages[3].text.startswith("Sub-agent reply:")
    # ...but the transcript carries the delegation bracket with the child's call inside it.
    records = read_transcript(transcript)
    events = [(r["event"], r["depth"]) for r in records if r["event"] in ("on_spawn", "post_tool_use", "on_spawn_done")]
    assert events == [("on_spawn", 0), ("post_tool_use", 1), ("on_spawn_done", 0), ("post_tool_use", 0)]
    # The child saw the parent's tools (plus a deeper spawn_agent of its own, while the
    # depth budget allows) and a fresh conversation, not the parent's.
    child_call = provider.calls[1]
    assert [t["name"] for t in child_call[1]] == list(levels.HARNESS_TOOLS) + ["spawn_agent"]
    assert [m.role for m in child_call[0]] == ["system", "user"]
    assert child_call[0][1].text == "read calc.py and report its functions"


def test_a_level_5_child_inherits_level_4s_loop_settings(levels, workspace, tmp_path, monkeypatch):
    import cornac.tools.builtin.spawn as spawn_module
    from cornac import Agent

    built = []

    class Recording(Agent):
        # wraps() keeps Agent.__init__'s signature visible: spawn._inherited_settings
        # passes a knob only when the constructor it sees names it.
        @functools.wraps(Agent.__init__)
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            built.append(self)

    monkeypatch.setattr(spawn_module, "Agent", Recording)
    provider = ScriptedProvider([_tool_turn("spawn_agent", "s1", task="look"), _reply("looked"), _reply("done")])
    levels.LEVELS[5].run(provider, workspace, TASK, tmp_path / "L5.jsonl")

    (child,) = built
    on = levels.GUARDS_ON
    assert (child.max_nudges, child.repeat_note, child.max_repeats, child.wrap_up) == (
        on.max_nudges, on.repeat_note, on.max_repeats, on.wrap_up
    )
    assert child.context_window == 8192  # None in the level -> the provider's window, as for the parent


def test_a_one_shot_reply_that_asks_for_a_tool_runs_nothing_and_stays_a_valid_conversation(levels, workspace, tmp_path):
    # Offered no tools, the model asked for one anyway. Nothing runs, nothing changes,
    # and the stored reply drops the request so the message list is a valid exchange.
    before = levels.snapshot(workspace)
    reply = Message(role="assistant", text="", tool_calls=[ToolCall("c1", "run_bash", {"command": "rm -rf ."})],
                    usage=Usage(10, 5), stop_reason="tool_calls")
    outcome = levels.LEVELS[0].run(ScriptedProvider([reply]), workspace, TASK, tmp_path / "t.jsonl")

    assert levels.snapshot(workspace) == before
    assert outcome.tool_calls == 0 and outcome.files_applied == [] and outcome.passed_precheck is False
    assert [m.role for m in outcome.messages] == ["user", "assistant"]
    assert outcome.messages[1].tool_calls == []
    assert reply.tool_calls  # the provider's own message was not mutated


def test_a_one_shot_reply_cut_off_by_the_output_cap_is_recorded_as_such(levels, workspace, tmp_path):
    # A reply that hit num_predict ends mid-file: the fence never closes, no block is
    # written, and the row fails on the answer protocol. The table's stop reason is
    # still "done" — the model did stop, as an agent level whose last message was cut
    # counts "done" too — but the transcript keeps the provider's own word, so a
    # reader can see the run was cut off rather than answered.
    cut = Message(role="assistant", text="```python\n# file: calc.py\ndef sum_to(n):\n    return sum(ra",
                  usage=Usage(100, 2048), stop_reason="length")
    transcript = tmp_path / "t.jsonl"
    outcome = levels.LEVELS[1].run(ScriptedProvider([cut]), workspace, TASK, transcript)

    assert outcome.passed_precheck is False and outcome.files_applied == []
    assert outcome.stop_reason == "done" and outcome.steps == 1
    (record,) = read_transcript(transcript)
    assert record["reply"]["stop_reason"] == "length"
    assert record["written"] == [] and record["files_applied"] == []


def test_tool_errors_count_denials_and_soft_errors_but_not_a_failing_test_run(levels, workspace, tmp_path):
    provider = ScriptedProvider([
        _tool_turn("read_file", path="missing.py"),                 # soft: "Error: not a file"
        _tool_turn("run_bash", command="sudo ls"),                  # denied by the policy: is_error
        _tool_turn("run_bash", command="exit 1"),                   # a non-zero exit is not an error here
        _reply("gave up"),
    ])
    outcome = levels.LEVELS[3].run(provider, workspace, TASK, tmp_path / "L3.jsonl")
    assert outcome.tool_calls == 3 and outcome.tool_errors == 2
    tool_texts = [m.text for m in outcome.messages if m.role == "tool"]
    assert tool_texts[0].startswith("Error: not a file")
    assert tool_texts[1].startswith("Permission denied")
    assert tool_texts[2].startswith("(exit code 1)")


def test_an_unfinished_level_4_run_carries_digest_and_summary(levels, workspace, tmp_path):
    # Three identical calls with the identical result -> stuck (max_repeats 3), then the
    # wrap-up call (tools off) is answered off the same script.
    same = lambda: _tool_turn("read_file", path="calc.py")  # noqa: E731
    provider = ScriptedProvider([same(), same(), same(), _reply("Unfinished: I kept re-reading.")])
    outcome = levels.LEVELS[4].run(provider, workspace, TASK, tmp_path / "L4.jsonl")
    assert outcome.stop_reason == "stuck" and outcome.final_text is None
    assert outcome.steps == 3 and outcome.tool_calls == 3
    assert "read_file(calc.py)" in (outcome.digest or "")
    assert outcome.summary == "Unfinished: I kept re-reading."
    assert outcome.files_applied == []


def test_the_same_repeats_at_level_3_are_not_stopped_because_the_guards_are_off(levels, workspace, tmp_path):
    same = lambda: _tool_turn("read_file", path="calc.py")  # noqa: E731
    provider = ScriptedProvider([same(), same(), same(), same(), _reply("done reading")])
    outcome = levels.LEVELS[3].run(provider, workspace, TASK, tmp_path / "L3.jsonl")
    assert outcome.stop_reason == "done" and outcome.steps == 5
    assert outcome.summary is None and outcome.digest is None
    # No repeat note either: the tool results are the bare file, every time.
    assert all("[cornac]" not in m.text for m in outcome.messages if m.role == "tool")


def test_an_agent_level_provider_failure_is_recorded_and_re_raised(levels, workspace, tmp_path):
    transcript = tmp_path / "L2.jsonl"
    with pytest.raises(RuntimeError, match="script exhausted"):
        levels.LEVELS[2].run(ScriptedProvider([]), workspace, TASK, transcript)
    events = [r["event"] for r in read_transcript(transcript)]
    assert events[0] == "run_config" and events[-1] == levels.RUN_ERROR_EVENT


# --- the six stage scripts ----------------------------------------------------------------


def test_the_six_stage_scripts_compile_and_the_agent_stages_watch_the_run_live(tmp_path):
    assert [p.name[:7] for p in STAGE_SCRIPTS] == [f"stage_{n}" for n in range(6)]
    for script in STAGE_SCRIPTS:
        py_compile.compile(str(script), cfile=str(tmp_path / (script.name + "c")), doraise=True)
        source = script.read_text(encoding="utf-8")
        assert "transcript.json\"" not in source          # the one-shot transcript is JSONL too
        if script.name.startswith(("stage_0", "stage_1")):
            assert "hooks=" not in source                  # one call, no bus: the reply is printed
        else:
            assert "hooks=bus" in source and "read_transcript" not in source  # live, not replayed
            assert 'bus.on("post_tool_use"' in source and 'bus.on("on_assistant_message"' in source
