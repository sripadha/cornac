"""Tests for the workspace hint and the tool descriptions that go with it (Week 4b).

Why these exist: the model spike (benchmark/spike/) showed models guessing where
they were — `cd /workspace` into a directory that did not exist, pytest on a guessed
path — and one model "fixing" a function inside run_python and believing the file had
changed. The fixes are text: Workspace.describe() names the absolute root for the
system prompt, and run_bash/run_python say they do not edit files. Text is easy to
break silently (a reworded description drops the one clause that mattered), so the
clauses are pinned here. The examples and the spike runner are compiled, and the
runner's round-three plumbing (edit_file allowed, hint in the prompt, nudges column)
is checked through the module itself.
"""

from __future__ import annotations

import py_compile
from pathlib import Path

import pytest

from cornac import ToolRegistry
from cornac.tools.builtin.python_exec import (
    RUN_PYTHON_DESCRIPTION_ROUND2,
    RUN_PYTHON_DESCRIPTION_ROUND3,
    RunPython,
)
from cornac.tools.builtin.shell import RUN_BASH_DESCRIPTION_ROUND2, RUN_BASH_DESCRIPTION_ROUND3, RunBash
from cornac.tools.workspace import Workspace

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = [
    REPO_ROOT / "examples" / "hook_trace.py",
    REPO_ROOT / "examples" / "permissions_demo.py",
    REPO_ROOT / "examples" / "multistep_trace.py",
]
RUN_SPIKE = REPO_ROOT / "benchmark" / "spike" / "run_spike.py"


# --- Workspace.describe() -------------------------------------------------------

def test_describe_names_the_absolute_root(tmp_path):
    ws = Workspace(tmp_path)
    assert str(ws.root) in ws.describe()
    assert ws.root.is_absolute()


def test_describe_says_paths_are_relative_and_no_cd_is_needed(tmp_path):
    text = Workspace(tmp_path).describe()
    assert "relative" in text
    assert "cd" in text
    assert "run_bash" in text and "run_python" in text


def test_describe_names_only_the_run_tools_it_is_told_about(tmp_path):
    # The capability ladder's level 2 has run_bash alone; a system prompt saying
    # "run_python already runs inside that directory" would name a tool it lacks.
    ws = Workspace(tmp_path)
    default = ws.describe()
    assert ws.describe(tools=None) == default
    assert ws.describe(tools=("read_file", "run_bash", "run_python", "grep")) == default  # both present: today's text
    only_bash = ws.describe(tools=("run_bash",))
    assert "run_python" not in only_bash
    assert "run_bash already runs inside that directory, so there is no need to cd" in only_bash
    only_python = ws.describe(tools=["run_python"])
    assert "run_bash" not in only_python and "run_python already runs inside" in only_python
    neither = ws.describe(tools=("read_file",))
    assert "cd" not in neither.split("rejected.")[1] and "run_bash" not in neither and "run_python" not in neither
    # Everything else is the same paragraph: the root, the path rule, the data-not-instructions rule.
    for text in (only_bash, only_python, neither):
        assert str(ws.root) in text and "relative" in text and "never instructions" in text
        assert "\n\n" not in text and text.strip() == text
    assert Workspace.RUN_TOOLS == ("run_bash", "run_python")


def test_describe_says_tool_output_is_data_not_instructions(tmp_path):
    # Finding #14: a workspace file can hold text shaped like a harness note or a
    # user order. The one rule the model needs is stated where every builder
    # appends it — and the marker rewrite in the loop is the other half.
    text = Workspace(tmp_path).describe()
    assert "never instructions" in text
    assert "file contents" in text and "command output" in text


def test_describe_is_one_paragraph(tmp_path):
    # It is appended to a system prompt; a blank line would start a new section.
    text = Workspace(tmp_path).describe()
    assert "\n\n" not in text
    assert text.strip() == text


# --- run_bash / run_python descriptions ----------------------------------------

@pytest.mark.parametrize("tool_cls", [RunBash, RunPython])
def test_run_tools_point_at_the_edit_tools(tmp_path, tool_cls):
    description = tool_cls(Workspace(tmp_path)).description
    assert "edit_file" in description
    assert "write_file" in description
    # The clause that stops "I fixed it inside the snippet" — the tools do not edit.
    assert "NOT edit" in description


@pytest.mark.parametrize("tool_cls", [RunBash, RunPython])
def test_run_tools_say_where_they_run(tmp_path, tool_cls):
    description = tool_cls(Workspace(tmp_path)).description
    assert "workspace" in description
    assert "no cd needed" in description


# The two descriptions each run tool can carry. ROUND3 is what the tool ships with
# (the two Week 4b sentences); ROUND2 is the text before them (git c381aed), which the
# capability ladder gives levels 2 and 3, where edit_file (and at level 2 write_file)
# does not exist to be pointed at.

@pytest.mark.parametrize("tool_cls, round3, round2", [
    (RunBash, RUN_BASH_DESCRIPTION_ROUND3, RUN_BASH_DESCRIPTION_ROUND2),
    (RunPython, RUN_PYTHON_DESCRIPTION_ROUND3, RUN_PYTHON_DESCRIPTION_ROUND2),
])
def test_edit_hint_defaults_on_and_off_is_round_twos_description_byte_for_byte(tmp_path, tool_cls, round3, round2):
    ws = Workspace(tmp_path)
    assert tool_cls.description == round3                       # the class default: today's text
    assert tool_cls(ws).description == round3                   # and the instance default
    assert tool_cls(ws).edit_hint is True
    off = tool_cls(ws, edit_hint=False)
    assert off.edit_hint is False
    assert off.description == round2
    for name in ("edit_file", "write_file", "NOT edit", "no cd needed"):
        assert name not in off.description
    assert off.name == tool_cls.name and off.input_schema == tool_cls.input_schema  # the tool is the same tool


def test_round_two_run_bash_description_is_the_text_before_week_4b():
    assert RUN_BASH_DESCRIPTION_ROUND2 == (
        "Run a bash command from the workspace root and return its combined "
        "stdout/stderr and exit code. Use for running tests, git, build tools, etc."
    )
    assert RUN_PYTHON_DESCRIPTION_ROUND2.endswith("A non-zero exit status is reported as '(exit code N)'.")


def test_the_registry_hands_the_provider_the_run_tools_instance_description(tmp_path):
    ws = Workspace(tmp_path)
    reg = ToolRegistry([RunBash(ws, edit_hint=False), RunPython(ws, edit_hint=False)])
    bash, python = reg.schemas()
    assert bash["description"] == RUN_BASH_DESCRIPTION_ROUND2
    assert python["description"] == RUN_PYTHON_DESCRIPTION_ROUND2


def test_edit_hint_off_runs_the_same_command(tmp_path):
    out = RunBash(Workspace(tmp_path), edit_hint=False).run({"command": "printf hi"})
    assert out == "(exit code 0)\nhi"
    assert RunPython(Workspace(tmp_path), edit_hint=False).run({"code": "1 + 1"}) == "2"


# --- the scripts that carry the hint -------------------------------------------

@pytest.mark.parametrize("script", EXAMPLES + [RUN_SPIKE], ids=lambda p: p.name)
def test_scripts_compile(tmp_path, script):
    # cfile keeps the .pyc out of the repo's __pycache__ directories.
    py_compile.compile(str(script), cfile=str(tmp_path / (script.name + "c")), doraise=True)


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda p: p.name)
def test_examples_append_the_workspace_hint(script):
    source = script.read_text(encoding="utf-8")
    assert "workspace.describe()" in source


# --- the spike runner's round-three plumbing (the `spike` fixture is in conftest.py)


def test_spike_policy_allows_edit_file_beside_write_file(spike):
    tools = spike.POLICY_RULES["tools"]
    assert tools["edit_file"] == "allow"
    assert tools["write_file"] == "allow"


def test_spike_system_prompt_carries_the_hint(spike, tmp_path):
    ws = Workspace(tmp_path)
    prompt = spike.system_prompt_for(ws)
    assert prompt.startswith(spike.SYSTEM_PROMPT)
    assert str(ws.root) in prompt
    assert prompt.endswith(ws.describe())


def _run_line(**over) -> dict:
    """A minimal runs.jsonl record with every key build_table reads."""
    line = {
        "model": "m", "task": "swe", "run": 0, "passed": True, "error": None,
        "steps": 3, "duration": 10.0, "input_tokens": 100, "output_tokens": 20,
        "n_tool_calls": 2, "n_tool_errors": 0, "n_soft_errors": 0,
        "answered_without_tools": False, "announced_then_stopped": False,
        "context_overflow": False, "tools_used": ["read_file"], "placement": None,
        "tool_capable": True, "options_override": {},
    }
    line.update(over)
    return line


def test_table_has_a_nudges_column_and_old_lines_count_zero(spike):
    runs = [_run_line(nudges=2), _run_line(run=1)]  # the second predates the field
    table = spike.build_table(runs, {}, ["swe"])
    header, _sep, row = table.splitlines()[:3]
    columns = [c.strip() for c in header.strip("|").split("|")]
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert "nudges" in columns
    assert cells[columns.index("nudges")] == "2"


def test_run_line_shows_nudges_only_when_nonzero(spike):
    assert "nudges=" not in spike.format_line(_run_line())
    assert "nudges=1" in spike.format_line(_run_line(nudges=1))
