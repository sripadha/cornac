"""Tests for Workspace.instructions() — the AGENTS.md / CLAUDE.md reader (Week 4c).

Why these exist: the instruction file is the one piece of a system prompt that comes
from the workspace rather than from cornac, so its rules are the kind that break
silently — a nested file picked up by accident, a fallback that stops falling back,
a cap that forgets its note. Each rule is pinned here: AGENTS.md first, CLAUDE.md as
the fallback, root only, None when there is nothing to read, and a "[truncated]"
note whenever the text was cut.
"""

from __future__ import annotations

from pathlib import Path

from cornac.tools.workspace import Workspace


def test_reads_agents_md_at_the_root(tmp_path):
    (tmp_path / "AGENTS.md").write_text("Run the tests with `make test`.\n")
    ws = Workspace(tmp_path)
    assert ws.instructions() == "Run the tests with `make test`."
    assert ws.instructions_path() == tmp_path / "AGENTS.md"


def test_falls_back_to_claude_md(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("Docstrings explain WHY.")
    ws = Workspace(tmp_path)
    assert ws.instructions() == "Docstrings explain WHY."
    assert ws.instructions_path() == tmp_path / "CLAUDE.md"


def test_agents_md_wins_when_both_exist(tmp_path):
    # The vendor-neutral name is preferred over Claude Code's own.
    (tmp_path / "AGENTS.md").write_text("from agents")
    (tmp_path / "CLAUDE.md").write_text("from claude")
    assert Workspace(tmp_path).instructions() == "from agents"


def test_none_when_neither_exists(tmp_path):
    ws = Workspace(tmp_path)
    assert ws.instructions() is None
    assert ws.instructions_path() is None


def test_nested_instruction_files_are_ignored(tmp_path):
    # Root only: a file in a subdirectory is not the project's instructions.
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "AGENTS.md").write_text("nested")
    assert Workspace(tmp_path).instructions() is None


def test_a_directory_with_the_name_is_not_a_file(tmp_path):
    (tmp_path / "AGENTS.md").mkdir()
    assert Workspace(tmp_path).instructions() is None


def test_empty_or_blank_file_counts_as_none(tmp_path):
    # Nothing to follow, so the prompt should not carry an empty header.
    (tmp_path / "AGENTS.md").write_text("  \n\n")
    assert Workspace(tmp_path).instructions() is None


def test_short_file_is_returned_whole_without_a_note(tmp_path):
    (tmp_path / "AGENTS.md").write_text("short")
    text = Workspace(tmp_path).instructions()
    assert text == "short"
    assert "[truncated]" not in text


def test_long_file_is_capped_with_a_truncated_note(tmp_path):
    body = ("one line of project conventions\n" * 400).strip()  # ~13 KB, over the default cap
    (tmp_path / "AGENTS.md").write_text(body)
    text = Workspace(tmp_path).instructions()
    assert text.startswith(body[:6000])           # the start of the file, untouched
    assert "[truncated]" in text                    # ...and an honest note
    assert "AGENTS.md" in text                      # naming the file it cut
    assert len(text) < len(body)                    # it really is shorter
    assert len(text) <= 6000 + 120                  # the note itself is small


def test_max_chars_is_respected(tmp_path):
    body = "abcdefghij" * 10  # 100 chars
    (tmp_path / "CLAUDE.md").write_text(body)
    text = Workspace(tmp_path).instructions(max_chars=25)
    assert text.startswith(body[:25])
    assert "[truncated]" in text and "CLAUDE.md" in text
    # Exactly at the cap: no cut, no note.
    assert Workspace(tmp_path).instructions(max_chars=100) == body


def test_invalid_utf8_does_not_raise(tmp_path):
    # An instruction file is a convenience; a stray byte must not abort the run.
    (tmp_path / "AGENTS.md").write_bytes(b"caf\xe9 rules apply\n")
    text = Workspace(tmp_path).instructions()
    assert text is not None
    assert "rules apply" in text


def test_unreadable_file_is_none(tmp_path, monkeypatch):
    (tmp_path / "AGENTS.md").write_text("secret")

    def refuse(self, *args, **kwargs):
        raise PermissionError("no")

    monkeypatch.setattr(Path, "open", refuse)
    assert Workspace(tmp_path).instructions() is None


# --- The fence ---------------------------------------------------------------------------
#
# AGENTS.md may be a symlink, and is_file()/read follow a link wherever it points. A
# cloned repo shipping `AGENTS.md -> ~/.ssh/id_rsa` must not get the key into the
# system prompt (and off to the model provider) through the one reader in the harness
# that skipped resolve(). read_file refuses that path; instructions() must too.


def test_a_symlink_that_leaves_the_workspace_is_ignored(tmp_path):
    from cornac.prompts import INSTRUCTIONS_HEADER, build_system_prompt

    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("SECRET_TOKEN=abc123\n")
    root = tmp_path / "repo"
    root.mkdir()
    (root / "AGENTS.md").symlink_to(secret)
    ws = Workspace(root)

    assert ws.instructions_path() is None
    assert ws.instructions() is None
    prompt = build_system_prompt(ws)
    assert "SECRET_TOKEN" not in prompt
    assert INSTRUCTIONS_HEADER[:20] not in prompt


def test_a_symlink_that_stays_inside_the_workspace_is_followed(tmp_path):
    (tmp_path / "docs").mkdir()
    target = tmp_path / "docs" / "agent-notes.md"
    target.write_text("Run `make test`.")
    (tmp_path / "AGENTS.md").symlink_to(target)
    ws = Workspace(tmp_path)

    assert ws.instructions() == "Run `make test`."
    assert ws.instructions_path() == target          # the resolved path: what was checked is what is read


def test_a_link_to_a_secret_falls_back_to_claude_md_when_that_is_real(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_text("nope")
    root = tmp_path / "repo"
    root.mkdir()
    (root / "AGENTS.md").symlink_to(outside / "secret")
    (root / "CLAUDE.md").write_text("from claude")

    assert Workspace(root).instructions() == "from claude"


def test_only_max_chars_plus_one_characters_are_read_from_disk(tmp_path, monkeypatch):
    # A link to a multi-gigabyte file must not be loaded whole to show 6000 chars.
    (tmp_path / "AGENTS.md").write_text("x" * 50_000)
    real_open = Path.open
    sizes: list[int] = []

    class Recording:
        def __init__(self, f):
            self._f = f

        def read(self, n=-1):
            sizes.append(n)
            return self._f.read(n)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return self._f.__exit__(*exc)

    monkeypatch.setattr(Path, "open", lambda self, *a, **k: Recording(real_open(self, *a, **k)))
    text = Workspace(tmp_path).instructions()

    assert sizes == [6001]
    assert text.startswith("x" * 6000) and "[truncated]" in text
    assert "longer than 6000 characters" in text
