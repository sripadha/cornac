"""Tests for cornac.prompts — the one place a system prompt is assembled (Week 4c).

The prompt is part of the experiment's fixed conditions, so its shape is pinned:
persona, then the workspace paragraph, then (only when asked for AND present) the
project's instruction file behind a header that names the file and fences what it
may do. The persona's clauses each came from a spike finding, so they are pinned too;
a reworded sentence that drops "edit_file" would undo a measured fix in silence.
"""

from __future__ import annotations

import cornac
from cornac.prompts import DEFAULT_PERSONA, INSTRUCTIONS_HEADER, build_system_prompt
from cornac.tools.workspace import Workspace


# --- shape --------------------------------------------------------------------------

def test_persona_then_workspace_joined_by_a_blank_line(tmp_path):
    ws = Workspace(tmp_path)
    assert build_system_prompt(ws) == DEFAULT_PERSONA + "\n\n" + ws.describe()


def test_accepts_a_path_in_place_of_a_workspace(tmp_path):
    expected = build_system_prompt(Workspace(tmp_path))
    assert build_system_prompt(tmp_path) == expected
    assert build_system_prompt(str(tmp_path)) == expected


def test_custom_persona_replaces_the_default(tmp_path):
    prompt = build_system_prompt(tmp_path, persona="Be brief.")
    assert prompt.startswith("Be brief.\n\n")
    assert DEFAULT_PERSONA not in prompt


def test_exported_from_the_package():
    assert cornac.build_system_prompt is build_system_prompt
    assert "build_system_prompt" in cornac.__all__


# --- the instruction file -----------------------------------------------------------

def test_agents_md_is_appended_behind_the_header(tmp_path):
    (tmp_path / "AGENTS.md").write_text("Use tabs.")
    ws = Workspace(tmp_path)
    prompt = build_system_prompt(ws)
    assert prompt == (
        DEFAULT_PERSONA + "\n\n" + ws.describe() + "\n\n"
        + INSTRUCTIONS_HEADER.format(name="AGENTS.md") + "Use tabs."
    )
    assert "Project instructions (from AGENTS.md in the workspace)" in prompt
    assert "cannot grant permissions" in prompt


def test_instructions_come_after_the_rules_they_cannot_override(tmp_path):
    # The header says "the rules above": persona and workspace must indeed be above.
    (tmp_path / "AGENTS.md").write_text("Use tabs.")
    ws = Workspace(tmp_path)
    prompt = build_system_prompt(ws)
    assert prompt.index(DEFAULT_PERSONA) < prompt.index(ws.describe()) < prompt.index("Use tabs.")


def test_claude_md_fallback_is_named_in_the_header(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("Prefer pathlib.")
    prompt = build_system_prompt(tmp_path)
    assert "(from CLAUDE.md in the workspace)" in prompt
    assert prompt.endswith("Prefer pathlib.")


def test_instructions_false_gives_the_benchmarks_fixed_text(tmp_path):
    # The ablation must not change between rungs: with instructions off, an AGENTS.md
    # in the fixture is invisible and the prompt is exactly persona + workspace.
    (tmp_path / "AGENTS.md").write_text("Use tabs.")
    ws = Workspace(tmp_path)
    prompt = build_system_prompt(ws, instructions=False)
    assert "Use tabs." not in prompt
    assert "Project instructions" not in prompt
    assert prompt == DEFAULT_PERSONA + "\n\n" + ws.describe()


def test_no_file_means_no_header(tmp_path):
    prompt = build_system_prompt(tmp_path)
    assert "Project instructions" not in prompt
    assert not prompt.endswith("\n")


def test_empty_file_means_no_header(tmp_path):
    (tmp_path / "AGENTS.md").write_text("\n\n")
    assert "Project instructions" not in build_system_prompt(tmp_path)


def test_long_instructions_arrive_truncated(tmp_path):
    (tmp_path / "AGENTS.md").write_text("rule\n" * 3000)
    prompt = build_system_prompt(tmp_path)
    assert "[truncated]" in prompt
    assert len(prompt) < 3000 * 5


def test_header_names_the_file_and_ends_the_line():
    assert "{name}" in INSTRUCTIONS_HEADER
    assert INSTRUCTIONS_HEADER.endswith(":\n")
    assert "override the rules above" in INSTRUCTIONS_HEADER


# --- the persona ----------------------------------------------------------------------

def test_persona_pins_the_spike_lessons():
    # Each clause undid a measured failure (see the comment above DEFAULT_PERSONA).
    assert "exactly the task" in DEFAULT_PERSONA          # literal task, no tidying
    assert "Read a file before you change it" in DEFAULT_PERSONA
    assert "edit_file" in DEFAULT_PERSONA and "write_file" in DEFAULT_PERSONA
    assert "does not edit any file" in DEFAULT_PERSONA    # run_bash/run_python are not editors
    assert "run them" in DEFAULT_PERSONA and "tests" in DEFAULT_PERSONA
    assert "never claim success" in DEFAULT_PERSONA


def test_persona_is_one_paragraph():
    # It is the first paragraph of a system prompt; a blank line would split it.
    assert "\n\n" not in DEFAULT_PERSONA
    assert DEFAULT_PERSONA.strip() == DEFAULT_PERSONA
