"""Tests for edit_file, the Python syntax gate, and write_file's change report.

Background (benchmark/spike/results_round2/): the small models found the right
one-line fix and then broke the file applying it, because the only way to change a
line was to rewrite the whole file from memory. Half of the .py rewrites did not
parse; a dozen more silently dropped functions the task never mentioned; and the tool
said "Wrote 179 chars" either way. These tests pin the three fixes:

    edit_file      replace ONE exact snippet, leave the rest alone, report the lines
    syntax gate    a .py write/edit that would not compile is refused, file untouched
    change report  write_file says created/overwrote, size before -> after, and which
                   top-level def/class names disappeared

Round three (results_round3/, 26/27 with the above) added two edit_file gaps and a
review added the byte-level ones: an `old` that is right except for its indentation
is matched after dedenting both sides; `old == new` is refused as a no-op instead of
reported as an edit; a CRLF file keeps CRLF, a byte-order mark survives, a non-UTF-8
file is refused rather than rewritten with U+FFFD outside the snippet; and the gate
is compile(), so it also catches what only the compiler rejects.

Everything here runs on a tmp_path workspace — no model, no network.
"""

import ast
import time

import pytest

from cornac.core.messages import ToolCall
from cornac.permissions.policy import DEFAULT_RULES, Decision, Policy
from cornac.tools.builtin import default_tools, files
from cornac.tools.builtin.files import (
    HINT_MAX_LINE,
    HINT_MAX_LINES,
    EditFile,
    ReadFile,
    WriteFile,
)
from cornac.tools.registry import ToolRegistry
from cornac.tools.workspace import Workspace, WorkspaceError

# The swe2 fixture from the spike, verbatim: the file the models kept breaking.
UTILS_PY = '''"""Numeric helpers shared by the grading code."""


def within(x, lo, hi):
    """True iff x lies in the closed interval [lo, hi], both ends included."""
    return lo < x < hi


def clamp(x, lo, hi):
    """Clamp x into the closed interval [lo, hi]."""
    return max(lo, min(x, hi))


def mean(values):
    """Arithmetic mean of a non-empty sequence of numbers."""
    return sum(values) / len(values)
'''

# What one model actually sent: the fix is right, `clamp` and `mean` are gone.
UTILS_PY_TRUNCATED = '''"""Numeric helpers shared by the grading code."""


def within(x, lo, hi):
    """True iff x lies in the closed interval [lo, hi], both ends included."""
    return lo <= x <= hi
'''

# What another model sent: closing triple quote glued to the return statement.
UTILS_PY_BROKEN = UTILS_PY.replace(
    '"""Arithmetic mean of a non-empty sequence of numbers."""\n    return',
    '"""Arithmetic mean of a non-empty sequence of numbers."""    return',
)


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "utils.py").write_text(UTILS_PY)
    (tmp_path / "notes.txt").write_text("alpha\nbeta\ngamma\n")
    return Workspace(tmp_path)


def read(ws, rel):
    return (ws.root / rel).read_text()


# --- edit_file: the happy path ---------------------------------------------------

def test_unique_replace_changes_only_that_line_and_reports_it(ws):
    out = EditFile(ws).run({
        "path": "utils.py",
        "old": "    return lo < x < hi\n",
        "new": "    return lo <= x <= hi\n",
    })
    assert out == "edited utils.py: replaced 1 occurrence (lines 6-6)"
    text = read(ws, "utils.py")
    assert "return lo <= x <= hi" in text
    assert "def clamp" in text and "def mean" in text  # the rest of the file survived
    assert text == UTILS_PY.replace("lo < x < hi", "lo <= x <= hi")


def test_multiline_replace_reports_the_span_of_the_new_text(ws):
    out = EditFile(ws).run({
        "path": "utils.py",
        "old": "def mean(values):\n",
        "new": "def mean(values):\n    if not values:\n        raise ValueError('empty')\n",
    })
    assert out.endswith("(lines 14-16)")
    assert "raise ValueError('empty')" in read(ws, "utils.py")


def test_deleting_text_reports_the_lines_it_occupied(ws):
    out = EditFile(ws).run({"path": "notes.txt", "old": "beta\n", "new": ""})
    assert out == "edited notes.txt: replaced 1 occurrence (lines 2-2)"
    assert read(ws, "notes.txt") == "alpha\ngamma\n"


def test_non_python_files_are_edited_as_plain_text(ws):
    out = EditFile(ws).run({"path": "notes.txt", "old": "beta", "new": "BETA"})
    assert out.startswith("edited notes.txt")
    assert read(ws, "notes.txt") == "alpha\nBETA\ngamma\n"


# --- edit_file: the error cases -------------------------------------------------

def test_zero_occurrences_is_an_error_and_file_is_untouched(ws):
    out = EditFile(ws).run({"path": "utils.py", "old": "def median(", "new": "x"})
    assert out.startswith("Error:")
    assert "not found" in out
    assert read(ws, "utils.py") == UTILS_PY


def test_ambiguous_whitespace_mismatch_points_at_the_line(ws):
    # The classic small-model slip: right text, wrong indentation. When the dedented
    # snippet matches ONE block it is applied (see the fallback tests below); when it
    # matches several, the exact-match rule stands and the hint names the first line.
    (ws.root / "two.py").write_text("def f():\n    return 1\n\n\ndef g():\n    if x:\n        return 1\n")
    out = EditFile(ws).run({"path": "two.py", "old": "            return 1\n", "new": "x"})
    assert out.startswith("Error:")
    assert "not found" in out
    assert "line 2" in out.lower()
    assert "indentation" in out
    assert read(ws, "two.py") == "def f():\n    return 1\n\n\ndef g():\n    if x:\n        return 1\n"


def test_zero_occurrences_offers_the_closest_line(ws):
    out = EditFile(ws).run({"path": "utils.py", "old": "    return lo < x <= hi\n", "new": "x"})
    assert out.startswith("Error:")
    assert "not found" in out
    assert "line" in out.lower() and "return lo < x < hi" in out


def test_two_or_more_occurrences_is_an_error_naming_the_count(ws):
    out = EditFile(ws).run({"path": "utils.py", "old": "closed interval", "new": "x"})
    assert out.startswith("Error:")
    assert "2 times" in out
    assert "unique" in out or "exactly once" in out
    assert read(ws, "utils.py") == UTILS_PY


def test_empty_old_is_an_error(ws):
    out = EditFile(ws).run({"path": "utils.py", "old": "", "new": "x"})
    assert out.startswith("Error:")
    assert read(ws, "utils.py") == UTILS_PY


def test_missing_file_is_an_error(ws):
    out = EditFile(ws).run({"path": "nope.py", "old": "a", "new": "b"})
    assert out == "Error: not a file: nope.py"


def test_path_escape_still_raises_workspace_error(ws):
    with pytest.raises(WorkspaceError):
        EditFile(ws).run({"path": "../../etc/passwd", "old": "root", "new": "x"})


def test_path_escape_through_registry_is_an_error_result(ws):
    reg = ToolRegistry([EditFile(ws)])
    result = reg.execute(ToolCall("1", "edit_file",
                                  {"path": "../../etc/passwd", "old": "root", "new": "x"}))
    assert result.is_error
    assert "escape" in result.content


# --- the re-indent fallback (round three, gap A) ---------------------------------

# Round three, verbatim: qwen3:8b sent the right three lines of utils.py with four
# extra spaces on every one, was told "Line 4 matches except for whitespace", and
# re-sent the identical call six times until max_steps.
ROUND3_OLD = (
    '    def within(x, lo, hi):\n'
    '        """True iff x lies in the closed interval [lo, hi], both ends included."""\n'
    '        return lo < x < hi'
)
ROUND3_NEW = ROUND3_OLD.replace("lo < x < hi", "lo <= x <= hi")


def test_over_indented_snippet_is_matched_and_reindented_to_the_file(ws):
    out = EditFile(ws).run({"path": "utils.py", "old": ROUND3_OLD, "new": ROUND3_NEW})
    assert out.startswith("edited utils.py: replaced 1 occurrence (lines 4-6)")
    assert "matched after adjusting indentation" in out
    assert read(ws, "utils.py") == UTILS_PY.replace("lo < x < hi", "lo <= x <= hi")


def test_over_indented_single_line_is_reindented_too(ws):
    # Eight spaces where the file has four (fewer would be a substring and match as is).
    out = EditFile(ws).run({
        "path": "utils.py",
        "old": "        return lo < x < hi\n",
        "new": "        return lo <= x <= hi\n",
    })
    assert out.startswith("edited utils.py: replaced 1 occurrence (lines 6-6)")
    assert "matched after adjusting indentation" in out
    assert read(ws, "utils.py") == UTILS_PY.replace("lo < x < hi", "lo <= x <= hi")


def test_under_indented_nested_block_is_reindented_to_its_margin(ws):
    # The other direction: a nested block sent flush left keeps its relative indent
    # and gets the file's margin (eight spaces) back.
    (ws.root / "nest.py").write_text("class A:\n    def f(self):\n        if x:\n            return 1\n")
    out = EditFile(ws).run({"path": "nest.py", "old": "if x:\n    return 1\n", "new": "if y:\n    return 2\n"})
    assert "matched after adjusting indentation" in out
    assert read(ws, "nest.py") == "class A:\n    def f(self):\n        if y:\n            return 2\n"


def test_over_indented_deletion_removes_the_whole_line(ws):
    out = EditFile(ws).run({"path": "notes.txt", "old": "    beta\n", "new": ""})
    assert out.startswith("edited notes.txt: replaced 1 occurrence (lines 2-2)")
    assert read(ws, "notes.txt") == "alpha\ngamma\n"


def test_exact_match_never_reports_an_adjustment(ws):
    out = EditFile(ws).run({"path": "notes.txt", "old": "beta\n", "new": "BETA\n"})
    assert out == "edited notes.txt: replaced 1 occurrence (lines 2-2)"


def test_reindented_edit_still_goes_through_the_syntax_gate(ws):
    broken = ROUND3_OLD.replace("return lo < x < hi", "return lo < x <")
    out = EditFile(ws).run({"path": "utils.py", "old": ROUND3_OLD, "new": broken})
    assert out.startswith("Error: refused to edit utils.py")
    assert read(ws, "utils.py") == UTILS_PY


def test_fallback_needs_the_relative_indentation_to_agree(ws):
    # Extra spaces on the last line only is not a margin: no block dedents to it.
    old = ROUND3_OLD.replace("        return", "            return")
    out = EditFile(ws).run({"path": "utils.py", "old": old, "new": ROUND3_NEW})
    assert out.startswith("Error:") and "not found" in out
    assert read(ws, "utils.py") == UTILS_PY


# --- old == new is a no-op, not an edit (round three, gap B) ---------------------

def test_identical_old_and_new_is_an_error_not_an_edit(ws):
    # Round three: the model "fixed" grades.py with old == new and was told "replaced
    # 1 occurrence"; it believed the file had changed.
    before = (ws.root / "utils.py").read_bytes()
    out = EditFile(ws).run({
        "path": "utils.py",
        "old": "    return lo < x < hi\n",
        "new": "    return lo < x < hi\n",
    })
    assert out.startswith("Error:")
    assert "old and new are identical; nothing changed" in out
    assert "utils.py already contains that text" in out
    assert (ws.root / "utils.py").read_bytes() == before


def test_reindented_new_that_equals_the_file_is_the_no_op_error_too(ws):
    # Gap B through the fallback: `old` is over-indented, `new` differs from it only
    # in indentation, and re-indenting `new` to the file's margin gives back the
    # block already on disk. The identical-snippet guard cannot see it (old != new),
    # so the final text is compared to the original instead.
    before = (ws.root / "utils.py").read_bytes()
    out = EditFile(ws).run({
        "path": "utils.py",
        "old": "        return lo < x < hi\n",   # eight spaces; the file has four
        "new": "    return lo < x < hi\n",
    })
    assert out.startswith("Error:")
    assert "changes nothing in utils.py" in out
    assert "replaced" not in out
    assert (ws.root / "utils.py").read_bytes() == before


def test_identical_old_and_new_absent_from_the_file_is_still_the_no_op_error(ws):
    out = EditFile(ws).run({"path": "utils.py", "old": "nope", "new": "nope"})
    assert "old and new are identical; nothing changed" in out
    assert "already contains" not in out


def test_identical_after_line_ending_normalisation_is_identical(ws):
    (ws.root / "win.txt").write_bytes(b"a\r\nb\r\n")
    out = EditFile(ws).run({"path": "win.txt", "old": "b\n", "new": "b\r\n"})
    assert "old and new are identical; nothing changed" in out
    assert (ws.root / "win.txt").read_bytes() == b"a\r\nb\r\n"


# --- bytes outside the snippet are never touched ---------------------------------

def test_non_utf8_file_is_refused_and_left_byte_identical(ws):
    # A latin-1 e-acute on line 1: read with errors="replace" it became U+FFFD and was
    # written back when line 2 was edited — data loss with a success message.
    raw = b"# caf\xe9\nx = 1\n"
    (ws.root / "lat.py").write_bytes(raw)
    out = EditFile(ws).run({"path": "lat.py", "old": "x = 1", "new": "x = 2"})
    assert out.startswith("Error:")
    assert "not UTF-8" in out and "edit_file only edits UTF-8 files" in out
    assert (ws.root / "lat.py").read_bytes() == raw


def test_read_file_still_shows_a_non_utf8_file(ws):
    # Display only: nothing read_file returns is written back, so U+FFFD is harmless.
    (ws.root / "lat.txt").write_bytes(b"caf\xe9\n")
    assert ReadFile(ws).run({"path": "lat.txt"}) == "caf\ufffd\n"


def test_crlf_file_keeps_its_line_endings_and_lf_snippets_match(ws):
    raw = b"def f():\r\n    return 1\r\n\r\n\r\ndef g():\r\n    return 2\r\n"
    (ws.root / "win.py").write_bytes(raw)
    out = EditFile(ws).run({"path": "win.py", "old": "    return 1\n", "new": "    return 10\n"})
    assert out == "edited win.py: replaced 1 occurrence (lines 2-2)"
    assert (ws.root / "win.py").read_bytes() == raw.replace(b"return 1\r\n", b"return 10\r\n")


def test_crlf_snippets_match_a_crlf_file(ws):
    (ws.root / "win.txt").write_bytes(b"a\r\nb\r\nc\r\n")
    out = EditFile(ws).run({"path": "win.txt", "old": "b\r\n", "new": "B\r\n"})
    assert out.startswith("edited win.txt")
    assert (ws.root / "win.txt").read_bytes() == b"a\r\nB\r\nc\r\n"


def test_crlf_snippet_on_an_lf_file_is_normalised_to_lf(ws):
    out = EditFile(ws).run({"path": "notes.txt", "old": "beta\r\n", "new": "BETA\r\n"})
    assert out.startswith("edited notes.txt")
    assert (ws.root / "notes.txt").read_bytes() == b"alpha\nBETA\ngamma\n"


def test_write_file_keeps_crlf_content_and_reports_the_size_on_disk(ws):
    (ws.root / "win.py").write_bytes(b"x = 1\r\ny = 2\r\n")  # 14 bytes on disk
    out = WriteFile(ws).run({"path": "win.py", "content": "x = 1\r\n"})
    assert out == "overwrote win.py: 14 -> 7 chars"
    assert (ws.root / "win.py").read_bytes() == b"x = 1\r\n"


def test_bom_file_can_be_edited_and_keeps_its_bom(ws):
    # Python runs a BOM-prefixed file happily; compile() of the str does not, so the
    # gate used to refuse every edit to such a file as "not valid Python".
    (ws.root / "bom.py").write_bytes(b"\xef\xbb\xbfdef f():\n    return 1\n")
    out = EditFile(ws).run({"path": "bom.py", "old": "return 1", "new": "return 2"})
    assert out == "edited bom.py: replaced 1 occurrence (lines 2-2)"
    assert (ws.root / "bom.py").read_bytes() == b"\xef\xbb\xbfdef f():\n    return 2\n"


def test_write_file_accepts_content_that_starts_with_a_bom(ws):
    out = WriteFile(ws).run({"path": "bom.py", "content": "\ufeffx = 1\n"})
    assert out.startswith("created bom.py")
    assert (ws.root / "bom.py").read_bytes() == b"\xef\xbb\xbfx = 1\n"


def test_bom_does_not_hide_removed_names_from_the_report(ws):
    (ws.root / "bom.py").write_bytes(b"\xef\xbb\xbfdef f():\n    pass\n\n\ndef g():\n    pass\n")
    out = WriteFile(ws).run({"path": "bom.py", "content": "def f():\n    pass\n"})
    assert "REMOVED top-level: def g" in out


# --- the syntax gate ------------------------------------------------------------

def test_write_file_refuses_a_broken_py_and_leaves_the_file_byte_identical(ws):
    before = (ws.root / "utils.py").read_bytes()
    out = WriteFile(ws).run({"path": "utils.py", "content": UTILS_PY_BROKEN})
    assert out.startswith("Error:")
    assert "SyntaxError" in out
    assert "line 15" in out                    # the docstring line the return got glued to
    assert "return sum(values)" in out         # the offending line itself
    assert "unchanged" in out
    assert (ws.root / "utils.py").read_bytes() == before


def test_write_file_refuses_a_broken_new_py_and_creates_nothing(ws):
    out = WriteFile(ws).run({"path": "pkg/new.py", "content": "def f(:\n    pass\n"})
    assert out.startswith("Error:")
    assert "line 1" in out
    assert not (ws.root / "pkg").exists()      # not even the parent directory


def test_edit_file_refuses_an_edit_that_breaks_the_py_and_leaves_it_byte_identical(ws):
    before = (ws.root / "utils.py").read_bytes()
    out = EditFile(ws).run({
        "path": "utils.py",
        "old": '"""Arithmetic mean of a non-empty sequence of numbers."""\n    return',
        "new": '"""Arithmetic mean of a non-empty sequence of numbers."""    return',
    })
    assert out.startswith("Error:")
    assert "SyntaxError" in out
    assert "line 15" in out
    assert "unchanged" in out
    assert (ws.root / "utils.py").read_bytes() == before


def test_valid_py_passes_both_tools(ws):
    assert WriteFile(ws).run({"path": "ok.py", "content": "x = 1\n"}).startswith("created ok.py")
    out = EditFile(ws).run({"path": "ok.py", "old": "x = 1", "new": "x = 2"})
    assert out.startswith("edited ok.py")
    assert read(ws, "ok.py") == "x = 2\n"


def test_gate_is_case_insensitive_about_the_suffix(ws):
    # A courtesy check, not a boundary (see _syntax_refusal) — but SHOUT.PY is Python
    # to the interpreter, so it is Python to the gate.
    out = WriteFile(ws).run({"path": "SHOUT.PY", "content": "def f(:\n"})
    assert out.startswith("Error: refused to write SHOUT.PY")
    assert not (ws.root / "SHOUT.PY").exists()


@pytest.mark.parametrize("content, msg", [
    ("def f():\n    pass\n\nreturn 1\n", "'return' outside function"),
    ("x = 1\nfrom __future__ import annotations\n", "__future__"),
    ("break\n", "'break' outside loop"),
], ids=["return-outside", "late-future", "break-outside"])
def test_gate_catches_what_only_the_compiler_rejects(ws, content, msg):
    # ast.parse accepts all three; only compile() refuses them, and a file written
    # with one fails at import — the next pytest run, two steps later.
    ast.parse(content)
    out = WriteFile(ws).run({"path": "late.py", "content": content})
    assert out.startswith("Error: refused to write late.py")
    assert msg in out
    assert not (ws.root / "late.py").exists()


def test_gate_fetches_the_offending_line_when_the_compiler_gives_none(ws):
    # The compiler's own errors carry no source text; the gate looks the line up.
    out = WriteFile(ws).run({"path": "late.py", "content": "x = 1\ny = 2\nreturn x + y\n"})
    assert "at line 3: return x + y" in out


def test_gate_handles_an_error_without_a_line_number(ws):
    # A null byte: SyntaxError with no line on 3.12+, ValueError before. Either way
    # the refusal must not claim a line it does not have.
    out = WriteFile(ws).run({"path": "nul.py", "content": "x = 1\0\n"})
    assert out.startswith("Error: refused to write nul.py")
    assert "null bytes" in out
    assert "at line" not in out
    assert not (ws.root / "nul.py").exists()


def test_gate_reports_a_value_error_as_invalid_source(ws, monkeypatch):
    # Python 3.10 and 3.11 raise ValueError for a null byte, which the gate renders
    # as "invalid source". The suite runs on one interpreter, so that behaviour is
    # played by a stand-in: a module-level `compile` shadows the builtin inside
    # files.py, and monkeypatch removes it afterwards.
    def older_compile(source, filename, mode, **kwargs):
        raise ValueError("source code string cannot contain null bytes")

    monkeypatch.setattr(files, "compile", older_compile, raising=False)
    out = WriteFile(ws).run({"path": "nul.py", "content": "x = 1\n"})
    assert out.startswith("Error: refused to write nul.py")
    assert "invalid source: source code string cannot contain null bytes" in out
    assert not (ws.root / "nul.py").exists()


def test_non_py_files_are_not_parsed_as_python(ws):
    weird = 'def f(:\n    """unterminated\n{{ not python at all ]]\n'
    out = WriteFile(ws).run({"path": "weird.txt", "content": weird})
    assert out.startswith("created weird.txt")
    assert read(ws, "weird.txt") == weird
    out = EditFile(ws).run({"path": "weird.txt", "old": "not python", "new": "still not python"})
    assert out.startswith("edited weird.txt")


# --- the gate on content the parser itself gives up on ---------------------------

# ast.parse does not always answer with a SyntaxError. Thousands of chained unary
# minuses overflow the parser stack (MemoryError); thousands of chained `1+` overflow
# the AST builder (RecursionError). Both are plain Exceptions the registry would
# report raw — and, worse, the report on the OLD file used to run AFTER the write.
TOO_DEEP = "x = " + "-" * 50_000 + "1\n"
TOO_DEEP_TOO = "x = " + "1+" * 20_000 + "1\n"


@pytest.mark.parametrize("content", [TOO_DEEP, TOO_DEEP_TOO], ids=["unary-minus", "chained-plus"])
def test_write_file_refuses_content_the_parser_cannot_handle(ws, content):
    out = WriteFile(ws).run({"path": "deep.py", "content": content})
    assert out.startswith("Error: refused to write deep.py")
    assert "too deeply nested" in out
    assert "unchanged" in out
    assert not (ws.root / "deep.py").exists()


def test_edit_file_refuses_an_edit_the_parser_cannot_handle(ws):
    out = EditFile(ws).run({
        "path": "utils.py",
        "old": "return lo < x < hi",
        "new": "return " + "-" * 50_000 + "1",
    })
    assert out.startswith("Error: refused to edit utils.py")
    assert "too deeply nested" in out
    assert read(ws, "utils.py") == UTILS_PY


def test_overwrite_report_survives_an_old_file_the_parser_cannot_handle(ws):
    # The old file reached the disk some other way (run_bash, a checkout), so the
    # gate never saw it. write_file used to write the new content and THEN choke on
    # parsing the old one for the change report — telling the model the write had
    # failed when the file on disk had already changed.
    (ws.root / "deep.py").write_text(TOO_DEEP)
    out = WriteFile(ws).run({"path": "deep.py", "content": "x = 1\n"})
    assert out == f"overwrote deep.py: {len(TOO_DEEP)} -> 6 chars"
    assert read(ws, "deep.py") == "x = 1\n"


# --- the not-found hint stays cheap ----------------------------------------------

def test_not_found_hint_is_skipped_on_a_huge_file(ws):
    # difflib is quadratic per line; on a big file the "closest line" hint cost seconds
    # per miss, and a small model misses repeatedly. Past HINT_MAX_LINES it is skipped.
    rows = (f"row_{i} = ({i}, {i * 2}, 'label {i}')" for i in range(HINT_MAX_LINES + 1))
    (ws.root / "table.py").write_text("\n".join(rows) + "\n")
    started = time.perf_counter()
    out = EditFile(ws).run({"path": "table.py", "old": "row_7 = (7, 14, 'labell 7')", "new": "x"})
    assert time.perf_counter() - started < 2
    assert out.startswith("Error:") and "not found" in out
    assert "Closest line" not in out


def test_not_found_hint_is_skipped_for_a_very_long_snippet_line(ws):
    long_line = "    return " + " + ".join(f"x{i}" for i in range(80)) + "\n"
    assert len(long_line.strip()) > HINT_MAX_LINE
    (ws.root / "wide.py").write_text("def f():\n" + long_line)
    out = EditFile(ws).run({"path": "wide.py", "old": long_line.replace("x1 ", "x1  "), "new": "x"})
    assert "not found" in out
    assert "Closest line" not in out
    assert "indentation" not in out      # the cheap whitespace pass did not fire either


def test_not_found_hint_still_offered_on_an_ordinary_file(ws):
    # The caps must not take the hint away from the files the benchmark actually has.
    out = EditFile(ws).run({"path": "utils.py", "old": "    return lo < x <= hi\n", "new": "x"})
    assert "Closest line" in out


# --- write_file's change report -------------------------------------------------

def test_overwrite_report_shows_size_delta_and_removed_names(ws):
    out = WriteFile(ws).run({"path": "utils.py", "content": UTILS_PY_TRUNCATED})
    assert out.startswith("overwrote utils.py: 404 -> 179 chars")
    assert "REMOVED top-level: def clamp, def mean" in out
    assert "restore them with edit_file" in out


def test_overwrite_report_without_removed_names_is_just_the_sizes(ws):
    fixed = UTILS_PY.replace("lo < x < hi", "lo <= x <= hi")
    out = WriteFile(ws).run({"path": "utils.py", "content": fixed})
    assert out == f"overwrote utils.py: {len(UTILS_PY)} -> {len(fixed)} chars"
    assert "REMOVED" not in out


def test_overwrite_report_skips_the_comparison_when_the_old_file_does_not_parse(ws):
    # The broken file reached the disk some other way (run_bash, a checkout). A model
    # fixing it gets no comparison rather than a wrong one.
    (ws.root / "utils.py").write_text(UTILS_PY_BROKEN)
    out = WriteFile(ws).run({"path": "utils.py", "content": UTILS_PY_TRUNCATED})
    assert out.startswith("overwrote utils.py:")
    assert "REMOVED" not in out


def test_write_file_refuses_a_directory_and_leaves_it_alone(ws):
    (ws.root / "pkg").mkdir()
    (ws.root / "pkg" / "a.txt").write_text("keep")
    out = WriteFile(ws).run({"path": "pkg", "content": "x"})
    assert out == "Error: is a directory: pkg"
    assert (ws.root / "pkg" / "a.txt").read_text() == "keep"


def test_whitespace_only_old_gets_a_plain_not_found_error(ws):
    out = EditFile(ws).run({"path": "notes.txt", "old": "   \n", "new": "x"})
    assert out == "Error: `old` not found in notes.txt."
    assert read(ws, "notes.txt") == "alpha\nbeta\ngamma\n"


def test_overwrite_report_lists_removed_classes_too(ws):
    (ws.root / "m.py").write_text("class A:\n    pass\n\n\ndef f():\n    pass\n")
    out = WriteFile(ws).run({"path": "m.py", "content": "def f():\n    pass\n"})
    assert "REMOVED top-level: class A" in out


def test_overwrite_of_a_non_py_file_reports_sizes_only(ws):
    out = WriteFile(ws).run({"path": "notes.txt", "content": "x"})
    assert out == "overwrote notes.txt: 17 -> 1 chars"


def test_created_report_is_short(ws):
    out = WriteFile(ws).run({"path": "sub/dir/new.txt", "content": "abc"})
    assert out == "created sub/dir/new.txt: 3 chars"
    assert read(ws, "sub/dir/new.txt") == "abc"


# --- wiring: default_tools, DEFAULT_RULES, example policy -------------------------

def test_default_tools_contains_edit_file_right_after_write_file(ws):
    names = [t.name for t in default_tools(ws)]
    assert "edit_file" in names
    assert names.index("edit_file") == names.index("write_file") + 1


def test_default_rules_ask_before_edit_file():
    assert DEFAULT_RULES["tools"]["edit_file"] == "ask"
    call = ToolCall("1", "edit_file", {"path": "a.py", "old": "x", "new": "y"})
    assert Policy().check(call) == Decision.ASK


def test_example_policy_yaml_loads_and_asks_before_edit_file():
    policy = Policy.from_yaml("examples/cornac.policy.yaml")
    call = ToolCall("1", "edit_file", {"path": "a.py", "old": "x", "new": "y"})
    assert policy.check(call) == Decision.ASK


# --- the ladder flags: syntax_gate and change_report -------------------------------------
#
# benchmark/ladder measures the gate and the report by switching them OFF at level 3
# (the round-two harness) and on at level 4. These tests pin what each flag does and
# does not change, and that the defaults are today's tool byte for byte — including
# the description the model reads, which has to describe the instance it belongs to.

WRITE_FILE_DESCRIPTION_ROUND3 = (
    "Write text to a file in the workspace, creating parent directories as needed. "
    "Overwrites the WHOLE file if it exists, so to change a few lines of an existing "
    "file use edit_file instead. For a .py path the content must parse as Python or "
    "the write is refused and the file left unchanged. The result says whether the "
    "file was created or overwritten, its size before and after, and (for .py) any "
    "top-level functions or classes that disappeared. Paths are relative to the "
    "workspace root."
)

EDIT_FILE_DESCRIPTION_ROUND3 = (
    "Replace ONE exact snippet of text in a file with new text, leaving the rest of "
    "the file untouched. This is the preferred way to change a few lines; use "
    "write_file only to create a file or replace all of it. `old` must match the "
    "file text exactly, including indentation and line breaks, and must occur "
    "exactly once (include a neighbouring line or two to make it unique). For a "
    ".py path the edited file must still parse as Python or the edit is refused "
    "and the file left unchanged. Paths are relative to the workspace root."
)

# Round two's write_file description (git: cornac/tools/builtin/files.py before Week
# 4b), verbatim. Level 3 of the ladder must read this, not a text that points at an
# edit_file the level does not have.
WRITE_FILE_DESCRIPTION_ROUND2 = (
    "Write text to a file in the workspace, creating parent directories as needed. "
    "Overwrites if the file exists. Paths are relative to the workspace root."
)


def test_flags_default_on_and_are_exposed(ws):
    w, e = WriteFile(ws), EditFile(ws)
    assert (w.syntax_gate, w.change_report, e.syntax_gate) == (True, True, True)
    off = WriteFile(ws, syntax_gate=False, change_report=False)
    assert (off.syntax_gate, off.change_report) == (False, False)
    assert EditFile(ws, syntax_gate=False).syntax_gate is False


def test_default_descriptions_are_round_three_byte_for_byte(ws):
    # Class and instance agree, and both are the text round three was measured with.
    assert WriteFile.description == WRITE_FILE_DESCRIPTION_ROUND3
    assert WriteFile(ws).description == WRITE_FILE_DESCRIPTION_ROUND3
    assert EditFile.description == EDIT_FILE_DESCRIPTION_ROUND3
    assert EditFile(ws).description == EDIT_FILE_DESCRIPTION_ROUND3


def test_both_flags_off_is_the_round_two_description(ws):
    # No gate promised, no report promised, and no edit_file to point at.
    off = WriteFile(ws, syntax_gate=False, change_report=False)
    assert off.description == WRITE_FILE_DESCRIPTION_ROUND2
    assert "edit_file" not in off.description


def test_each_flag_removes_only_its_own_sentence(ws):
    no_gate = WriteFile(ws, syntax_gate=False).description
    assert "must parse as Python" not in no_gate
    assert "created or overwritten" in no_gate and "edit_file" in no_gate

    no_report = WriteFile(ws, change_report=False).description
    assert "created or overwritten" not in no_report
    assert "must parse as Python" in no_report and "edit_file" in no_report

    no_edit_gate = EditFile(ws, syntax_gate=False).description
    assert "must still parse as Python" not in no_edit_gate
    assert no_edit_gate.startswith("Replace ONE exact snippet")
    assert no_edit_gate.endswith("Paths are relative to the workspace root.")


def test_the_registry_hands_the_provider_the_instance_description(ws):
    # schemas() is what the provider serialises for the model; it must carry the
    # description of the tool as built, not the class default.
    reg = ToolRegistry([WriteFile(ws, syntax_gate=False, change_report=False), EditFile(ws, syntax_gate=False)])
    write_schema, edit_schema = reg.schemas()
    assert write_schema["description"] == WRITE_FILE_DESCRIPTION_ROUND2
    assert "parse as Python" not in edit_schema["description"]


def test_default_tools_build_the_scaffolded_tools(ws):
    by_name = {t.name: t for t in default_tools(ws)}
    assert (by_name["write_file"].syntax_gate, by_name["write_file"].change_report) == (True, True)
    assert by_name["edit_file"].syntax_gate is True


# write_file with the gate off

def test_gate_off_writes_a_broken_py_as_round_two_did(ws):
    out = WriteFile(ws, syntax_gate=False).run({"path": "utils.py", "content": UTILS_PY_BROKEN})
    assert read(ws, "utils.py") == UTILS_PY_BROKEN          # on disk, broken
    # The report is still on: sizes, but no REMOVED line, because the comparison
    # needs both sides to parse and the new one does not (see _removed_top_level).
    assert out == f"overwrote utils.py: {len(UTILS_PY)} -> {len(UTILS_PY_BROKEN)} chars"
    assert "Error" not in out and "SyntaxError" not in out


def test_gate_off_creates_a_broken_new_py_and_its_parent(ws):
    out = WriteFile(ws, syntax_gate=False).run({"path": "pkg/new.py", "content": "def f(:\n    pass\n"})
    assert out == "created pkg/new.py: 17 chars"
    assert read(ws, "pkg/new.py") == "def f(:\n    pass\n"


def test_gate_off_still_refuses_a_directory(ws):
    # Not scaffolding: writing text over a directory is an error in any harness.
    (ws.root / "pkg").mkdir()
    out = WriteFile(ws, syntax_gate=False, change_report=False).run({"path": "pkg", "content": "x"})
    assert out == "Error: is a directory: pkg"
    assert (ws.root / "pkg").is_dir()


# write_file with the report off

def test_report_off_says_wrote_and_the_size_and_nothing_else(ws):
    # The round-two message over the round-two disaster: two functions gone, and
    # the tool says how many characters landed. That silence is what level 3 pays for.
    out = WriteFile(ws, change_report=False).run({"path": "utils.py", "content": UTILS_PY_TRUNCATED})
    assert out == f"wrote utils.py: {len(UTILS_PY_TRUNCATED)} chars"
    assert "REMOVED" not in out and "overwrote" not in out
    assert read(ws, "utils.py") == UTILS_PY_TRUNCATED


def test_report_off_uses_the_same_verb_for_a_new_file(ws):
    out = WriteFile(ws, change_report=False).run({"path": "sub/dir/new.txt", "content": "abc"})
    assert out == "wrote sub/dir/new.txt: 3 chars"           # not "created"
    assert read(ws, "sub/dir/new.txt") == "abc"


def test_report_off_keeps_the_gate(ws):
    # The flags are independent: no report does not mean no gate.
    before = (ws.root / "utils.py").read_bytes()
    out = WriteFile(ws, change_report=False).run({"path": "utils.py", "content": UTILS_PY_BROKEN})
    assert out.startswith("Error:") and "SyntaxError" in out
    assert (ws.root / "utils.py").read_bytes() == before


def test_report_off_does_not_read_the_old_file(ws):
    # Round two's tool never looked at what it was replacing. With the report off
    # neither does this one — so a file it cannot decode is no obstacle to writing.
    (ws.root / "lat.py").write_bytes(b"# caf\xe9\nx = 1\n")    # latin-1, not UTF-8
    out = WriteFile(ws, change_report=False).run({"path": "lat.py", "content": "x = 2\n"})
    assert out == "wrote lat.py: 6 chars"
    assert (ws.root / "lat.py").read_bytes() == b"x = 2\n"


# both off: level 3

def test_both_off_is_the_round_two_tool(ws):
    tool = WriteFile(ws, syntax_gate=False, change_report=False)
    out = tool.run({"path": "utils.py", "content": UTILS_PY_BROKEN})
    assert out == f"wrote utils.py: {len(UTILS_PY_BROKEN)} chars"
    assert read(ws, "utils.py") == UTILS_PY_BROKEN
    # The valid write is reported the same way: the round-two tool had one message.
    assert tool.run({"path": "utils.py", "content": UTILS_PY}) == f"wrote utils.py: {len(UTILS_PY)} chars"


# edit_file with the gate off

def test_edit_file_gate_off_applies_an_edit_that_breaks_the_file(ws):
    out = EditFile(ws, syntax_gate=False).run({
        "path": "utils.py",
        "old": '"""Arithmetic mean of a non-empty sequence of numbers."""\n    return',
        "new": '"""Arithmetic mean of a non-empty sequence of numbers."""    return',
    })
    assert out == "edited utils.py: replaced 1 occurrence (lines 15-15)"
    assert read(ws, "utils.py") == UTILS_PY_BROKEN


def test_edit_file_gate_off_keeps_every_other_check(ws):
    # Only the compile check is a flag. Not-found, ambiguity and the no-op guard are
    # what makes edit_file an edit tool at all, and stay on.
    tool = EditFile(ws, syntax_gate=False)
    assert tool.run({"path": "utils.py", "old": "def median(", "new": "x"}).startswith("Error: `old` not found")
    assert tool.run({"path": "utils.py", "old": "closed interval", "new": "x"}).startswith("Error: `old` occurs 2 times")
    assert tool.run({"path": "utils.py", "old": "return lo < x < hi", "new": "return lo < x < hi"}).startswith(
        "Error: old and new are identical"
    )
    assert read(ws, "utils.py") == UTILS_PY
