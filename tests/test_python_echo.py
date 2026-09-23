"""Tests for run_python's notebook-cell echo.

The gap this guards: the weak local model ends its snippet with a bare expression —
`total` rather than `print(total)` — the way a notebook user would. Under a plain
`python -c` that value is computed and thrown away, the tool reports "(no output)",
and the model gives up. run_python now echoes repr() of a trailing bare expression,
exactly as IPython's "last_expr" mode does — and only then: a trailing print,
assignment, def, or None-valued call must not echo anything extra.

The other half of the contract is that the scaffolding leaves everything else alone:
tracebacks still point at the line numbers the model wrote, `__name__` is still
"__main__", the snippet's stdin is untouched (it may call input()), a non-zero exit
is reported rather than swallowed, and the workspace is left exactly as the snippet
expects to find it — readable or not.
"""

import contextlib
import os
import sys
import tempfile
from textwrap import dedent

import pytest

from cornac.tools.builtin.python_exec import HEAD_CHARS, TAIL_CHARS, RunPython
from cornac.tools.workspace import Workspace


@pytest.fixture
def ws(tmp_path):
    # Its own subdirectory, so a test can point the stash somewhere else under
    # tmp_path and still assert that the workspace itself stays empty.
    root = tmp_path / "workspace"
    root.mkdir()
    return Workspace(root)


@pytest.fixture
def stash_dir(tmp_path, monkeypatch):
    """Where run_python parks the snippet while the child reads it.

    The stash lives in the system temp dir (the workspace may be read-only). To
    check nothing is left behind there, point tempfile at a fresh directory of our
    own for the duration of the test.
    """
    d = tmp_path / "stash"
    d.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(d))
    return d


@pytest.fixture
def run(ws):
    """run(code) -> the tool's output for that snippet, in a fresh workspace."""
    return lambda code, **kw: RunPython(ws, **kw).run({"code": dedent(code)})


# --- the echo rule --------------------------------------------------------------------

def test_trailing_name_is_echoed(run):
    assert run("x = 2 + 2\nx\n") == "4"


def test_bare_expression_is_echoed(run):
    assert run("2 + 2") == "4"


def test_echo_uses_repr_not_str(run):
    # A notebook shows 'hi' with quotes; so do we. The model can tell a string from a
    # number, and repr is what it would see in IPython.
    assert run("'hi'") == "'hi'"


def test_assignment_as_last_statement_echoes_nothing(run):
    assert run("x = 5") == "(no output)"


def test_trailing_print_prints_exactly_once(run):
    # print() returns None, so nothing is echoed on top of what it printed.
    assert run("print('a')") == "a"


def test_trailing_none_echoes_nothing(run):
    assert run("None") == "(no output)"


def test_def_as_last_statement_echoes_nothing(run):
    assert run("def f():\n    return 1\n") == "(no output)"


def test_none_returning_call_echoes_nothing(run):
    assert run("items = []\nitems.append(1)\n") == "(no output)"


def test_everything_before_the_expression_still_runs_in_order(run):
    assert run("print('before')\n1 + 1\n") == "before\n2"


def test_csv_style_loop_ending_with_the_bare_total(ws, run):
    # The shape that motivated all this: a data snippet that computes an answer and
    # then just names it, the way a notebook user would.
    (ws.root / "sales.csv").write_text("item,qty,price\nwidget,10,2.5\ngadget,4,9.0\n")
    code = """\
        import csv
        total_revenue = 0.0
        with open('sales.csv') as f:
            for row in csv.DictReader(f):
                total_revenue += int(row['qty']) * float(row['price'])
        total_revenue
    """
    assert run(code) == "61.0"


def test_multi_line_trailing_expression(run):
    assert run("(1 +\n 2)\n") == "3"


def test_semicolon_separated_trailing_expression(run):
    # `x = 1; x + 1` is two statements to the parser; the last one is the expression.
    assert run("x = 1; x + 1") == "2"


def test_trailing_comment_and_blank_lines_still_echo(run):
    assert run("x = 7\nx  # the answer\n\n\n") == "7"


def test_leading_byte_order_mark_is_ignored(run):
    # A real .py file that starts with a BOM runs fine under `python file.py`; the
    # snippet is read the same way, so a BOM-prefixed snippet must not be a
    # SyntaxError on line 1.
    assert run("\ufeffx = 1\nx\n") == "1"


def test_huge_repr_is_truncated_head_and_tail(run):
    # A 50,000-element list has a ~290k-char repr: it goes through the same head+tail
    # trimming as anything else the snippet prints.
    out = run("list(range(50000))")
    assert out.startswith("[0, 1, 2")
    assert out.endswith("49999]")
    assert "truncated" in out
    assert len(out) < HEAD_CHARS + TAIL_CHARS + 500


# --- errors keep the model's own line numbers ------------------------------------------
#
# The source is never rewritten (no injected print, no wrapper function), so a
# traceback points at the line the model wrote. The runner's own frames are stripped,
# so the model sees the same shape a plain `python -c` produces — nothing to misread.

def test_syntax_error_reports_the_original_line(run):
    out = run("a = 1\nb = 2\nc = = 3\n")
    assert "SyntaxError" in out
    assert "line 3" in out
    assert 'File "<string>"' not in out   # no frames from the runner itself


def test_runtime_error_reports_the_original_line(run):
    out = run("a = 1\nb = 1 / 0\nc = 3\n")
    assert "ZeroDivisionError" in out
    assert "line 2" in out
    assert "b = 1 / 0" in out             # the offending line is quoted, as for a real file
    assert 'File "<string>"' not in out


def test_error_inside_the_trailing_expression_keeps_its_line(run):
    # The popped expression is compiled from the same AST node, so its lineno survives.
    out = run("x = 1\n1 / 0\n")
    assert "ZeroDivisionError" in out
    assert "line 2" in out


def test_error_while_echoing_the_value_names_the_line(run):
    # The expression itself evaluates fine; it is repr() of the result that fails, and
    # that call is the runner's, not the snippet's — so there is no snippet frame to
    # show. The model still needs to know which line produced the unprintable value.
    out = run("class C:\n    def __repr__(self): return 5\nC()\n")
    assert "TypeError" in out
    assert "line 3" in out
    assert 'File "<string>"' not in out


def test_keyboard_interrupt_shows_only_the_snippet_frame(run):
    # KeyboardInterrupt is not an Exception. Left to the interpreter's default handler
    # it would print the runner's own `<string>` frame above the snippet's — exactly
    # the leak the frame-stripping exists to prevent.
    out = run("raise KeyboardInterrupt\n")
    assert "KeyboardInterrupt" in out
    assert 'File "<run_python>", line 1' in out
    assert 'File "<string>"' not in out
    assert out.startswith("(exit code 130)")   # 128 + SIGINT, as a shell would report it


# --- a non-zero exit is reported, not swallowed --------------------------------------
#
# run_bash prefixes "(exit code N)"; a snippet that deliberately signals failure with
# sys.exit(3) must not read as success just because it printed something first.

def test_sys_exit_code_is_reported_with_the_output(run):
    assert run("print('x')\nimport sys\nsys.exit(3)\n") == "(exit code 3)\nx"


def test_exit_code_with_no_output_stands_alone(run):
    # os._exit skips every Python-level cleanup, including the runner's own — the
    # exit status is all that comes back, so it is all that is shown.
    assert run("import os\nos._exit(5)\n") == "(exit code 5)"
    assert run("raise SystemExit(7)\n") == "(exit code 7)"


def test_uncaught_exception_exits_one(run):
    assert run("1 / 0\n").startswith("(exit code 1)\n")


def test_successful_run_is_not_annotated(run):
    # Only failure is marked: the happy path stays a clean echo the model can use.
    assert run("print('ok')\n2 + 2\n") == "ok\n4"


# --- parity with a plain `python -c` run ------------------------------------------------

def test_dunder_main_block_runs(run):
    assert run('if __name__ == "__main__":\n    print("main")\n') == "main"


def test_argv_looks_like_a_plain_python_c_run(run):
    # The runner receives the snippet's path as an argument and pops it, so the
    # snippet sees the argv it always did.
    assert run("import sys\nsys.argv\n") == "['-c']"


def test_functions_defined_in_the_snippet_can_be_pickled(run):
    # multiprocessing looks workers up as __main__.<name>; a bare dict namespace
    # would break that, a real __main__ module does not.
    assert run("import pickle\ndef f(): pass\npickle.loads(pickle.dumps(f)) is f\n") == "True"


# --- stdin is the snippet's, not the runner's -------------------------------------------

@contextlib.contextmanager
def stdin_from(path):
    """Point this process's fd 0 at `path` for the duration, then put it back.

    pytest swaps sys.stdin for a dummy object but leaves fd 0 alone — and fd 0 is
    what a child process inherits, so it is the only knob that reaches the snippet.
    """
    try:
        saved = os.dup(0)
    except OSError:
        saved = None                      # the process has no stdin at all
    fd = os.open(path, os.O_RDONLY)
    try:
        os.dup2(fd, 0)
        yield
    finally:
        if saved is None:
            os.close(0)
        else:
            os.dup2(saved, 0)
            os.close(saved)
        os.close(fd)


def test_input_with_no_stdin_fails_inside_the_snippet(run):
    # If the runner had used stdin to deliver the source, input() would have read
    # the tail of the snippet, or the runner would have failed first. Neither: the
    # EOFError is raised from the snippet's own input() call.
    with stdin_from(os.devnull):
        out = run("name = input()\nname\n")
    assert "EOFError" in out
    assert 'File "<run_python>", line 1' in out
    assert 'File "<string>"' not in out


def test_snippet_reads_the_agents_own_stdin(run, tmp_path):
    feed = tmp_path / "feed.txt"
    feed.write_text("hello from stdin\n")
    with stdin_from(feed):
        out = run("input()")
    assert out == "'hello from stdin'"    # read from our stdin, then echoed as repr


# --- the workspace is left as the snippet expects -----------------------------------------
#
# The source travels through a temp file in the system temp dir, never the workspace,
# and the runner deletes it before the snippet runs. So the workspace shows the model
# its own files and nothing else, and the tool works in a workspace it cannot write to.

def test_snippet_does_not_see_its_own_source_file(run):
    assert run("import os\nos.listdir('.')\n") == "[]"


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_runs_in_a_read_only_workspace(ws, run):
    # A dataset folder, a checkout owned by someone else: `python -c` ran there, and
    # so must this. Stashing the snippet inside the workspace would fail before the
    # interpreter even started.
    os.chmod(ws.root, 0o555)
    try:
        assert run("1 + 1") == "2"
    finally:
        os.chmod(ws.root, 0o755)          # let pytest clean tmp_path up


def test_no_temp_file_is_left_behind(ws, stash_dir, run):
    run("x = 1")
    assert list(stash_dir.iterdir()) == []
    assert list(ws.root.iterdir()) == []


def test_temp_file_is_removed_even_if_the_interpreter_never_started(
    ws, stash_dir, run, monkeypatch
):
    # The runner never ran, so it never deleted the file; the tool's own cleanup must.
    monkeypatch.setattr(sys, "executable", str(ws.root / "no-such-python"))
    with pytest.raises(FileNotFoundError):
        run("x = 1")
    assert list(stash_dir.iterdir()) == []


def test_temp_file_is_removed_after_a_timeout(ws, stash_dir, run):
    out = run("import time\ntime.sleep(30)\n", timeout=1)
    assert out.startswith("Error: code timed out after 1s")
    assert list(stash_dir.iterdir()) == []


def test_temp_file_is_removed_when_the_code_cannot_be_written(ws, stash_dir, run):
    # A lone surrogate is legal in a JSON string, so a model can produce one, but
    # UTF-8 cannot encode it. The write fails after the file was created — that must
    # not leave an empty stash behind. The error itself still reaches the model (the
    # registry turns it into an error result), as it did before the stash existed.
    with pytest.raises(UnicodeEncodeError):
        run("s = '\ud83d'\nlen(s)\n")
    assert list(stash_dir.iterdir()) == []
    assert list(ws.root.iterdir()) == []
