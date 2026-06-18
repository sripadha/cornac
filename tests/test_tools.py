"""Tests for the Week 2 built-in tools.

Focus areas:
  - the Workspace sandbox actually blocks path escapes (security-critical)
  - each file tool does the right thing on valid input
  - tool errors come back as readable strings, not crashes (Layer 2 contract)

Web tools (web_search, web_fetch) are not tested here — they hit the live network and
would make the suite flaky. They're exercised manually in the Week 2 demo.
"""

import pytest

from cornac.tools.builtin import default_tools
from cornac.tools.builtin.files import Grep, ListDir, ReadFile, WriteFile
from cornac.tools.builtin.shell import RunBash
from cornac.tools.workspace import Workspace, WorkspaceError


@pytest.fixture
def ws(tmp_path):
    """A throwaway workspace with a couple of files, fresh per test."""
    (tmp_path / "hello.txt").write_text("hello world\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "code.py").write_text("def add(a, b):\n    return a + b\n")
    return Workspace(tmp_path)


# --- sandbox (the important one) -----------------------------------------------

def test_resolve_allows_paths_inside_root(ws):
    p = ws.resolve("hello.txt")
    assert p.name == "hello.txt"
    assert p.is_file()


def test_resolve_blocks_parent_traversal(ws):
    with pytest.raises(WorkspaceError):
        ws.resolve("../../../etc/passwd")


def test_resolve_blocks_absolute_escape(ws):
    with pytest.raises(WorkspaceError):
        ws.resolve("/etc/passwd")


def test_read_file_escape_becomes_error_result_not_crash(ws):
    # Going through registry-style execution, an escape attempt must surface as an
    # error result the model can see — never crash the process.
    from cornac.tools.registry import ToolRegistry
    from cornac.core.messages import ToolCall

    reg = ToolRegistry([ReadFile(ws)])
    result = reg.execute(ToolCall(id="x", name="read_file", arguments={"path": "../../etc/passwd"}))
    assert result.is_error
    assert "escape" in result.content


# --- file tools ----------------------------------------------------------------

def test_read_file(ws):
    assert ReadFile(ws).run({"path": "hello.txt"}) == "hello world\n"


def test_write_then_read_roundtrip(ws):
    out = WriteFile(ws).run({"path": "notes/new.txt", "content": "abc"})
    assert "Wrote 3 chars" in out
    assert ReadFile(ws).run({"path": "notes/new.txt"}) == "abc"


def test_list_dir(ws):
    listing = ListDir(ws).run({"path": "."})
    assert "sub/" in listing
    assert "hello.txt" in listing


def test_grep_finds_definition(ws):
    out = Grep(ws).run({"pattern": r"def add"})
    assert "sub/code.py" in out
    assert ":1:" in out  # found on line 1


def test_grep_no_match(ws):
    out = Grep(ws).run({"pattern": "nonexistent_symbol_xyz"})
    assert "no matches" in out


# --- shell ---------------------------------------------------------------------

def test_run_bash_echo(ws):
    out = RunBash(ws).run({"command": "echo hi"})
    assert "exit code 0" in out
    assert "hi" in out


def test_run_bash_runs_in_workspace(ws):
    # pwd should be the workspace root, proving cwd is pinned to the sandbox.
    out = RunBash(ws).run({"command": "pwd"})
    assert str(ws.root) in out


# --- assembly ------------------------------------------------------------------

def test_default_tools_has_full_set(ws):
    tools = default_tools(ws)
    names = {t.name for t in tools}
    assert names == {
        "read_file", "write_file", "list_dir", "grep",
        "run_bash", "run_python", "web_search", "web_fetch",
        "get_current_time",
    }
