"""Tests that a tool timeout stops the whole command, not just its first process.

The bug this guards against: subprocess.run(timeout=...) kills only the direct child.
For run_bash that child is /bin/sh; the program sh started (a grandchild) kept
running after the tool reported "command timed out". The fix runs the child in its
own process group and kills the group (see cornac/tools/_subprocess.py).

Each test plants a grandchild that writes its own pid to a file and then sleeps for
far longer than the tool's timeout. After the timeout fires, that pid must be gone.
"""

import os
import shlex
import sys
import time
from pathlib import Path

import pytest

from cornac.tools.builtin.python_exec import RunPython
from cornac.tools.builtin.shell import HEAD_CHARS, TAIL_CHARS, RunBash
from cornac.tools.workspace import Workspace

# The grandchild: record my pid, then hang around.
GRANDCHILD = "import os, time; open('grandchild.pid', 'w').write(str(os.getpid())); time.sleep(30)"
GRANDCHILD_CMD = f"{shlex.quote(sys.executable)} -c {shlex.quote(GRANDCHILD)}"


@pytest.fixture
def ws(tmp_path):
    return Workspace(tmp_path)


def _is_dead(pid: int) -> bool:
    """True if no live process has this pid."""
    try:
        os.kill(pid, 0)   # signal 0 sends nothing; it only asks "does this pid exist?"
    except ProcessLookupError:
        return True
    # The pid still exists — but a killed orphan lingers as a zombie ("Z") for a
    # moment until init reaps it. A zombie is dead for our purposes.
    try:
        return "State:\tZ" in Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return False


def _dies_within(pid: int, seconds: float = 3.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if _is_dead(pid):
            return True
        time.sleep(0.05)
    return False


def _grandchild_pid(ws: Workspace) -> int:
    return int((ws.root / "grandchild.pid").read_text())


def test_run_bash_timeout_kills_the_grandchild(ws):
    # "<python> ; true": because another command follows, sh FORKS python instead of
    # exec-ing into it, so python is a grandchild of the tool. Killing only sh would
    # leave it running.
    out = RunBash(ws, timeout=1).run({"command": f"{GRANDCHILD_CMD} ; true"})
    assert out.startswith("Error: command timed out after 1s")
    pid = _grandchild_pid(ws)
    assert _dies_within(pid), f"grandchild {pid} survived the timeout"


def test_run_bash_timeout_kills_a_backgrounded_command(ws):
    # The classic "start the server in the background" shape.
    out = RunBash(ws, timeout=1).run({"command": f"{GRANDCHILD_CMD} & wait"})
    assert out.startswith("Error: command timed out after 1s")
    pid = _grandchild_pid(ws)
    assert _dies_within(pid), f"backgrounded grandchild {pid} survived the timeout"


def test_run_python_timeout_kills_the_grandchild(ws):
    code = f"import subprocess, sys; subprocess.run([sys.executable, '-c', {GRANDCHILD!r}])"
    out = RunPython(ws, timeout=1).run({"code": code})
    assert out.startswith("Error: code timed out after 1s")
    pid = _grandchild_pid(ws)
    assert _dies_within(pid), f"grandchild {pid} survived the timeout"


# --- what the command printed before it hung is kept ---------------------------------
#
# A timeout used to throw away everything the command had already printed, leaving
# the model with "timed out" and nothing to reason from. The partial output is exactly
# the kind of thing head+tail truncation was built to preserve.

def test_run_bash_timeout_keeps_partial_stdout_and_stderr(ws):
    out = RunBash(ws, timeout=1).run(
        {"command": "echo before-hang; echo on-stderr 1>&2; sleep 30"}
    )
    assert out.startswith("Error: command timed out after 1s\n--- partial output ---\n")
    assert "before-hang" in out
    assert "on-stderr" in out


def test_run_python_timeout_keeps_partial_output(ws):
    # flush=True: a SIGKILLed python never flushes its stdout buffer, so an unflushed
    # print would genuinely be lost — that is the snippet's doing, not the tool's.
    code = "import time; print('before-hang', flush=True); time.sleep(30)"
    out = RunPython(ws, timeout=1).run({"code": code})
    assert out.startswith("Error: code timed out after 1s\n--- partial output ---\n")
    assert "before-hang" in out


def test_timeout_with_nothing_printed_has_no_partial_section(ws):
    out = RunBash(ws, timeout=1).run({"command": "sleep 30"})
    assert out == "Error: command timed out after 1s"


def test_partial_output_is_truncated_head_and_tail_like_any_other(ws):
    code = (
        "import sys, time; sys.stdout.write('START' + 'A' * 60000 + 'END'); "
        "sys.stdout.flush(); time.sleep(30)"
    )
    out = RunPython(ws, timeout=1).run({"code": code})
    assert "START" in out and "END" in out
    assert "truncated" in out
    assert len(out) < HEAD_CHARS + TAIL_CHARS + 500


# --- a descendant that escaped the kill must not stall the tool ----------------------
#
# killpg reaches every process still in the group. One that deliberately left it —
# os.setsid(), the double-fork a daemon does, `nohup server &` — survives, and it still
# holds the tool's stdout/stderr pipe. The naive follow-up ("read the pipe to EOF")
# then blocks until that survivor exits: a 1s timeout that returns after 30s. The
# tool must settle for what it captured and come back promptly.

# The escapee: leave the group, record my pid, then outlive the timeout by a lot.
ESCAPED = (
    "import os, time; os.setsid(); "
    "open('escaped.pid', 'w').write(str(os.getpid())); time.sleep(30)"
)
ESCAPED_CMD = f"{shlex.quote(sys.executable)} -c {shlex.quote(ESCAPED)}"


def _kill_escaped(ws: Workspace) -> None:
    """Clean up the escapee ourselves — by design, the tool could not."""
    try:
        os.kill(int((ws.root / "escaped.pid").read_text()), 9)
    except (OSError, ValueError):
        pass   # never started, or already gone


def test_run_bash_timeout_does_not_wait_for_a_descendant_that_left_the_group(ws):
    started = time.monotonic()
    try:
        out = RunBash(ws, timeout=1).run({"command": f"echo before; {ESCAPED_CMD} & sleep 30"})
        elapsed = time.monotonic() - started
    finally:
        _kill_escaped(ws)
    assert out.startswith("Error: command timed out after 1s")
    assert "before" in out, "what was captured before the kill is still shown"
    # 1s timeout + a bounded pipe drain; anything near 30s means we waited for the escapee.
    assert elapsed < 4, f"tool blocked for {elapsed:.1f}s on a process it could not kill"


def test_a_fast_command_is_unaffected(ws):
    # The group-kill path only runs on timeout; normal output and exit codes are intact.
    out = RunBash(ws, timeout=5).run({"command": "echo hi; echo oops 1>&2; exit 3"})
    assert out == "(exit code 3)\nhi\noops"
