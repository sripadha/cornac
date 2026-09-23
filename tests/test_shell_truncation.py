"""Tests for head+tail output truncation in run_bash and run_python.

The bug this guards against: cutting long output with `out[:limit]` throws away the
END of the output — where tracebacks and pytest summaries live. After the fix, a long
output must still show its start AND its end, with a notice in between saying how
much was dropped. Short output must come back untouched.
"""

import shlex
import sys

import pytest

from cornac.tools._truncate import truncate_head_tail
from cornac.tools.builtin.python_exec import RunPython
from cornac.tools.builtin.shell import HEAD_CHARS, TAIL_CHARS, RunBash
from cornac.tools.workspace import Workspace

# A snippet that prints ~80,000 chars with recognizable markers at the start, the
# middle, and the end. Well over HEAD_CHARS + TAIL_CHARS (50,000), so it gets cut.
LONG_PRINT = "print('START-MARKER' + 'A'*60000 + 'MIDDLE-MARKER' + 'Z'*20000 + 'END-MARKER')"


@pytest.fixture
def ws(tmp_path):
    return Workspace(tmp_path)


# --- the helper itself ------------------------------------------------------------

def test_short_text_is_returned_unchanged():
    assert truncate_head_tail("hello", head=10, tail=10) == "hello"


def test_text_exactly_at_limit_is_returned_unchanged():
    text = "a" * 20
    assert truncate_head_tail(text, head=10, tail=10) == text


def test_long_text_keeps_head_and_tail_with_notice():
    text = "H" * 10 + "M" * 30 + "T" * 10  # 50 chars; keep 10 + 10, drop 30
    out = truncate_head_tail(text, head=10, tail=10)
    assert out.startswith("H" * 10)
    assert out.endswith("T" * 10)
    assert "M" not in out
    assert "[output truncated: 30 chars omitted; showing first 10 and last 10]" in out


def test_label_names_what_was_cut():
    out = truncate_head_tail("x" * 100, head=5, tail=5, label="stderr")
    assert "[stderr truncated:" in out


def test_tail_of_zero_keeps_only_the_head():
    # The trap: text[-0:] is text[0:] — the WHOLE string. An unguarded tail=0 returned
    # the notice followed by everything it had just claimed to cut.
    out = truncate_head_tail("abcdefghij", head=3, tail=0)
    assert out.startswith("abc")
    assert out.endswith("showing first 3 and last 0] ...\n")
    assert "defghij" not in out
    assert "7 chars omitted" in out


@pytest.mark.parametrize("head", [0, -5])
def test_head_of_zero_or_less_keeps_only_the_tail(head):
    out = truncate_head_tail("abcdefghij", head=head, tail=3)
    assert out.startswith("\n... [output truncated: 7 chars omitted; showing first 0")
    assert out.endswith("hij")
    assert "abcdefg" not in out


def test_both_limits_zero_keeps_nothing_but_the_notice():
    out = truncate_head_tail("abc", head=0, tail=0)
    assert "abc" not in out
    assert "3 chars omitted" in out


# --- run_bash -----------------------------------------------------------------------

def test_run_bash_long_output_keeps_start_and_end(ws):
    # Run the snippet through the venv's own python so the test doesn't depend on
    # what `python` resolves to on the machine's PATH.
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(LONG_PRINT)}"
    out = RunBash(ws).run({"command": command})

    assert "exit code 0" in out
    assert "START-MARKER" in out
    assert "END-MARKER" in out
    assert "truncated" in out
    assert "MIDDLE-MARKER" not in out
    # The result should be about head+tail in size, not the full 80k.
    assert len(out) < HEAD_CHARS + TAIL_CHARS + 500


def test_run_bash_short_output_is_untouched(ws):
    out = RunBash(ws).run({"command": "echo hi"})
    assert "hi" in out
    assert "truncated" not in out


def test_run_bash_traceback_survives_truncation(ws):
    # The whole point of keeping the tail: an error raised AFTER a flood of output
    # must still be visible to the model.
    code = "print('x'*60000); raise ValueError('boom')"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"
    out = RunBash(ws).run({"command": command})
    assert "exit code 1" in out
    assert "ValueError: boom" in out


# --- run_python ---------------------------------------------------------------------

def test_run_python_long_output_keeps_start_and_end(ws):
    out = RunPython(ws).run({"code": LONG_PRINT})

    assert "START-MARKER" in out
    assert "END-MARKER" in out
    assert "truncated" in out
    assert "MIDDLE-MARKER" not in out
    assert len(out) < HEAD_CHARS + TAIL_CHARS + 500


def test_run_python_short_output_is_untouched(ws):
    out = RunPython(ws).run({"code": "print('hi')"})
    assert out == "hi"


def test_run_python_traceback_survives_truncation(ws):
    out = RunPython(ws).run({"code": "print('x'*60000); raise ValueError('boom')"})
    assert "--- stderr ---" in out
    assert "ValueError: boom" in out
