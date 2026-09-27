"""Tests for the Week 2 built-in tools.

Focus areas:
  - the Workspace sandbox actually blocks path escapes (security-critical)
  - each file tool does the right thing on valid input
  - tool errors come back as readable strings, not crashes (Layer 2 contract)

Web tools (web_search, web_fetch) are not tested here — they hit the live network and
would make the suite flaky. They're exercised manually in the Week 2 demo.
"""

import signal
import threading
import time

import pytest

from cornac.tools.builtin import default_tools
from cornac.tools.builtin.files import EditFile, Grep, ListDir, ReadFile, WriteFile
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
    assert out == "created notes/new.txt: 3 chars"
    assert ReadFile(ws).run({"path": "notes/new.txt"}) == "abc"


def test_write_file_overwrite_reports_before_and_after_size(ws):
    out = WriteFile(ws).run({"path": "hello.txt", "content": "hi"})
    assert out == "overwrote hello.txt: 12 -> 2 chars"
    assert ReadFile(ws).run({"path": "hello.txt"}) == "hi"


def test_edit_file_changes_one_line_in_place(ws):
    out = EditFile(ws).run({"path": "sub/code.py", "old": "a + b", "new": "a - b"})
    assert out == "edited sub/code.py: replaced 1 occurrence (lines 2-2)"
    assert ReadFile(ws).run({"path": "sub/code.py"}) == "def add(a, b):\n    return a - b\n"


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


def test_grep_path_may_be_a_single_file(ws):
    # Regression: rglob("*") on a FILE yields nothing, so this used to say "no matches"
    # for a file that plainly contained the pattern — a false answer the model trusted.
    out = Grep(ws).run({"pattern": r"return a \+ b", "path": "sub/code.py"})
    assert "sub/code.py:2:" in out


def test_grep_missing_path_is_an_error_not_no_matches(ws):
    out = Grep(ws).run({"pattern": "def", "path": "does/not/exist.py"})
    assert out.startswith("Error: no such file or directory")
    assert "no matches" not in out


# --- grep and the sandbox: symlinks --------------------------------------------

def test_grep_does_not_read_through_a_symlink_that_leaves_the_workspace(ws, tmp_path_factory):
    # rglob, is_file() and read_text() all follow symlinks, so a link planted in the
    # workspace (by a cloned repo, or an earlier run_bash) used to let grep — which
    # never asks permission — read ~/.ssh/id_rsa, /etc/passwd, anything. Every other
    # tool runs its path through ws.resolve() and refuses; grep must, per file.
    secret = tmp_path_factory.mktemp("outside") / "secret.txt"
    secret.write_text("SECRET_TOKEN=abc123\n")
    (ws.root / "leak.txt").symlink_to(secret)

    out = Grep(ws).run({"pattern": "SECRET_TOKEN", "path": "."})
    assert "abc123" not in out
    assert "no matches" in out
    # Named directly, the link is refused the way read_file refuses it.
    with pytest.raises(WorkspaceError):
        Grep(ws).run({"pattern": "SECRET_TOKEN", "path": "leak.txt"})


def test_grep_still_follows_a_symlink_that_stays_inside_the_workspace(ws):
    (ws.root / "alias.py").symlink_to(ws.root / "sub" / "code.py")
    out = Grep(ws).run({"pattern": r"def add"})
    assert "alias.py:1:" in out       # reported under the name the model can read_file
    assert "sub/code.py:1:" in out


# --- grep and the clock: a pattern that backtracks forever -----------------------

needs_alarm = pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="SIGALRM is POSIX-only")

CATASTROPHIC = r"(a+)+$"           # the textbook exponential regex
STUCK_LINE = "a" * 40 + "b\n"       # 2^40 ways to fail; effectively never returns


@needs_alarm
def test_grep_gives_up_on_a_pattern_that_backtracks_exponentially(ws):
    # Python's re has no timeout, and the pattern is the model's to choose. grep
    # carries its own bound so the loop gets its turn back, with an error it can act on.
    (ws.root / "aaa.txt").write_text(STUCK_LINE)
    started = time.perf_counter()
    out = Grep(ws, timeout=0.3).run({"pattern": CATASTROPHIC, "path": "aaa.txt"})
    assert time.perf_counter() - started < 5
    assert out.startswith("Error: grep stopped after 0.3s")
    assert CATASTROPHIC in out
    assert "exponentially" in out


@needs_alarm
def test_grep_returns_the_hits_it_found_before_running_out_of_time(ws):
    # Files are searched in sorted order: the hit in 0.txt lands before 1.txt sticks.
    (ws.root / "0.txt").write_text("aaa\n")
    (ws.root / "1.txt").write_text(STUCK_LINE)
    out = Grep(ws, timeout=0.3).run({"pattern": CATASTROPHIC, "path": "."})
    lines = out.splitlines()
    assert lines[0] == "0.txt:1:aaa"
    assert lines[-1].startswith("Error: grep stopped")


@needs_alarm
def test_grep_leaves_no_alarm_or_handler_behind(ws):
    # The guard borrows SIGALRM; after grep returns — normally or by timeout — the
    # timer must be disarmed and the previous handler back in place.
    before = signal.getsignal(signal.SIGALRM)
    Grep(ws).run({"pattern": "hello"})
    (ws.root / "aaa.txt").write_text(STUCK_LINE)
    Grep(ws, timeout=0.2).run({"pattern": CATASTROPHIC, "path": "aaa.txt"})
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    assert signal.getsignal(signal.SIGALRM) == before


def test_grep_works_off_the_main_thread_where_the_guard_steps_aside(ws):
    # SIGALRM can only be armed from the main thread. Anywhere else grep must still
    # search — unguarded, but working — rather than crash on signal.signal().
    result = {}
    worker = threading.Thread(
        target=lambda: result.update(out=Grep(ws).run({"pattern": r"def add"}))
    )
    worker.start()
    worker.join()
    assert "sub/code.py:1:" in result["out"]


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
        "read_file", "write_file", "edit_file", "list_dir", "grep",
        "run_bash", "run_python", "web_search", "web_fetch",
        "get_current_time",
        "spawn_agent",  # sub-agents (Week 4b-B)
    }


# --- walking the registry (Week 4b-B) ------------------------------------------

def test_registry_names_and_tools_follow_registration_order(ws):
    # spawn_agent builds a child's registry from names(), and Agent.__init__ binds
    # every tool from tools(); both promise registration order — the order the
    # model sees the tools in, which is what schemas() hands the provider.
    from cornac.tools.registry import ToolRegistry

    a, b, c = Grep(ws), ReadFile(ws), ListDir(ws)
    reg = ToolRegistry([a, b, c])

    assert reg.names() == ["grep", "read_file", "list_dir"]
    assert all(x is y for x, y in zip(reg.tools(), [a, b, c])) and len(reg.tools()) == 3
    assert reg.names() == [s["name"] for s in reg.schemas()]
    assert ToolRegistry().names() == [] and ToolRegistry().tools() == []
