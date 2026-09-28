"""File tools: read_file, write_file, edit_file, list_dir, grep.

These are class-based tools (subclassing Tool directly) rather than @tool functions,
because each one needs a Workspace to confine it to the sandbox root. The constructor
takes the Workspace; run() validates every path through ws.resolve() before touching
disk, so a path that escapes the root becomes a clean error result (Layer 2) the model
can recover from — never an actual read/write outside the sandbox.

Why edit_file, the syntax gate and the change report exist
----------------------------------------------------------
The round-two model spike (benchmark/spike/results_round2/, 45 coding runs) showed
that most failures were the harness's fault, not the model's. The models found the
right one-line fix, but the only way to apply it was write_file — rewrite the whole
file from memory — and that is where things went wrong:

  - Of the 65 write_file calls that targeted a .py file, 32 did not parse. The usual
    shape: the newline after a docstring's closing triple quote went missing, so the
    `return` landed on the docstring's line.
  - 12 more parsed but silently dropped functions the task never mentioned:
    utils.py went from 404 to 179 chars, `clamp` and `mean` gone, and the tool said
    "Wrote 179 chars". The model believed it.
  - The broken file was only discovered by the NEXT pytest run, after which the same
    broken write was re-sent three to six times.

Three pieces of scaffolding close those gaps:

  1. edit_file replaces one exact snippet and leaves the rest of the file alone, so
     "change this one line" no longer means "reproduce the whole file".
  2. A Python syntax gate, in both write_file and edit_file: for a .py path the new
     content is compiled BEFORE it is written; if it does not compile the write is
     refused, the file is untouched, and the model gets the SyntaxError with its line
     number now instead of a test failure two steps later. It is a courtesy check for
     the model, not a security boundary (see _syntax_refusal).
  3. write_file reports what it did: created or overwrote, size before -> after, and
     for .py files the top-level def/class names that disappeared — "Wrote 179 chars"
     becomes "overwrote utils.py: 404 -> 179 chars; REMOVED top-level: def clamp,
     def mean", which a model can act on.

Round three (the same tasks on the frozen model, with the above in place) went from
12/27 to 26/27 and exposed two more edit_file gaps, both closed here: a snippet that
was right except for its indentation was refused six times in a row (the model never
corrected the spaces, so the tool now matches it after dedenting both sides), and an
edit whose `old` equalled its `new` was reported as "replaced 1 occurrence" — a no-op
the model took for a fix. Both tools also read and write the file's bytes as they
are: a CRLF file stays CRLF, a byte-order mark survives, and a file that is not UTF-8
is refused rather than rewritten with replacement characters outside the snippet.

Why write_file and edit_file take flags (the capability ladder)
---------------------------------------------------------------
benchmark/ladder asks what each piece of scaffolding above is actually worth. It runs
the SAME frozen model on the same tasks six times, adding one thing per level, and
level 3 is "the round-two harness": write_file with no gate and no report, and no
edit_file at all. A tool that always gates and always reports cannot be measured
against itself, so the two behaviours are constructor flags —
`WriteFile(workspace, syntax_gate=True, change_report=True)` and
`EditFile(workspace, syntax_gate=True)` — and the level that switches them off gets the
round-two tool back, description included. The defaults keep today's behaviour byte
for byte: default_tools() passes neither flag, so nothing built on it changes. The
flags exist so the ladder can measure the gate and the report as separate rungs, not
so anyone runs a coding agent without them.
"""

from __future__ import annotations

import ast
import difflib
import re
import signal
import textwrap
import threading
from contextlib import contextmanager
from pathlib import Path

from cornac.tools.base import Tool
from cornac.tools.workspace import Workspace, WorkspaceError

MAX_BYTES = 100_000  # cap on how much a single read/grep returns, to protect context
GREP_TIMEOUT = 10.0  # seconds one grep call may spend matching (see _time_limit)

# _not_found_hint runs difflib over every line of the file, and SequenceMatcher is
# quadratic in line length. On a benchmark-sized file that costs nothing; on a
# 20k-line file of 3k-char rows it cost 24 seconds per miss — and a miss is exactly
# when a small model retries. Past these sizes the hint is skipped rather than paid
# for (5000 lines x 200 chars of near-identical rows measures at 0.3 s).
HINT_MAX_LINES = 5000  # lines in the file
HINT_MAX_LINE = 200    # characters in the snippet line being looked for


# --- helpers shared by write_file and edit_file --------------------------------

def _is_python(rel: str) -> bool:
    """Whether the gate applies to `rel`. Case-insensitive, so FOO.PY is gated too."""
    return rel.lower().endswith(".py")


def _syntax_refusal(rel: str, content: str, verb: str) -> str | None:
    """The Python syntax gate.

    Returns None when `content` may be written to `rel`: either the path is not a
    .py file (anything goes), or it is and the content compiles. Otherwise returns an
    error message carrying the error's text, line number and the offending line, for
    the tool to return WITHOUT writing.

    The check is compile(), not ast.parse(). The parser accepts `return` outside a
    function, `break` outside a loop and a late `from __future__` import; only the
    compiler rejects them, and a file written with one fails at import — the next
    pytest run, two steps later, which is exactly the discovery the gate exists to
    prevent. compile() runs nothing; dont_inherit keeps this module's own __future__
    flags out of the model's file. A leading byte-order mark is dropped for the
    check only: Python accepts one on disk, compile() rejects it in a str.

    This is a courtesy check for the model, not a security boundary. It keys on the
    path's suffix, so a rename through run_bash goes around it, and it stops nothing
    a policy allows; the permission policy is the control.
    """
    if not _is_python(rel):
        return None
    source = content.removeprefix("\ufeff")
    try:
        compile(source, rel, "exec", dont_inherit=True)
        return None
    except SyntaxError as exc:
        lineno = exc.lineno
        line = exc.text
        if line is None and lineno is not None:
            # The parser's errors carry the source line; the compiler's own
            # ("'return' outside function") and some tokenizer errors (unexpected
            # EOF) do not. Fetch it ourselves so the model can see it.
            lines = source.splitlines()
            line = lines[lineno - 1] if 0 < lineno <= len(lines) else None
        where = f" at line {lineno}" if lineno is not None else ""
        shown = f": {line.rstrip()}" if line else ""
        detail = f"SyntaxError: {exc.msg}{where}{shown}"
    except ValueError as exc:  # Python <3.12 raises this for a null byte in the source
        detail = f"invalid source: {exc}"
    except (MemoryError, RecursionError):
        # ast.parse gives up on pathological nesting (thousands of chained unary
        # minuses, say) with one of these rather than a SyntaxError. Both are plain
        # Exceptions the registry would otherwise report raw; the gate's own message,
        # with "the file is unchanged", is what the model can act on.
        detail = "the source is too deeply nested to parse"
    return (
        f"Error: refused to {verb} {rel}: the new content is not valid Python, so the "
        f"file is unchanged. {detail}. Fix the content and try again."
    )


def _top_level_names(source: str) -> list[str] | None:
    """['def clamp', 'class Roster', ...] in file order, or None if it does not parse."""
    try:
        tree = ast.parse(source.removeprefix("\ufeff"))  # a BOM is not a syntax error
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return None  # the last two: the parser gave up on pathological nesting
    names = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append(f"def {node.name}")
        elif isinstance(node, ast.ClassDef):
            names.append(f"class {node.name}")
    return names


def _removed_top_level(old: str, new: str) -> list[str]:
    """Top-level def/class names present in `old` but missing from `new`.

    Empty if either side does not parse: a model fixing an already-broken file gets
    no comparison rather than a wrong one.
    """
    before = _top_level_names(old)
    after = _top_level_names(new)
    if before is None or after is None:
        return []
    return [name for name in before if name not in after]


def _read_utf8(path: Path) -> str | None:
    """The file as a str whose bytes round-trip, or None if it is not UTF-8.

    Path.read_text() would translate every line ending to "\\n" and, with
    errors="replace", swap each undecodable byte for U+FFFD — and edit_file writes
    the whole str back, so both changes would land on disk, OUTSIDE the snippet the
    model asked to change (a latin-1 accent on line 1 became U+FFFD when line 2 was
    edited). Decoding the raw bytes keeps CRLF and a byte-order mark as they are, and
    a file that does not decode is refused rather than quietly rewritten.
    """
    try:
        return path.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        return None


def _line_ending(text: str) -> str:
    """The file's line ending: CRLF if it uses any, else LF."""
    return "\r\n" if "\r\n" in text else "\n"


def _with_line_ending(snippet: str, nl: str) -> str:
    """`snippet` with every line break rewritten as `nl`.

    A model writes "\\n"; a Windows checkout has "\\r\\n". Before the snippets were
    converted to the file's ending, an edit to a CRLF file could never match and the
    hint blamed indentation.
    """
    return snippet.replace("\r\n", "\n").replace("\n", nl)


def _margin(lines: list[str]) -> str:
    """The leading whitespace every non-blank line of `lines` shares."""
    indents = [line[: len(line) - len(line.lstrip(" \t"))] for line in lines if line.strip()]
    if not indents:
        return ""
    margin = indents[0]
    for indent in indents[1:]:
        while not indent.startswith(margin):
            margin = margin[:-1]
    return margin


def _match_reindented(text: str, old: str, nl: str) -> tuple[int, int, str] | None:
    """Where `old` sits in `text` once each side's own indentation is ignored.

    Round three, verbatim: the model sent the right three lines of utils.py with four
    extra spaces on every one, was told "line 4 matches except for whitespace", and
    re-sent the identical call six times until max_steps. The RELATIVE indentation
    was right; only the margin was off, and a margin is what textwrap.dedent removes.
    So every block of consecutive file lines as long as `old` is compared with `old`
    after both are dedented. Exactly one block matching returns (start, end, margin):
    the block's character span in `text`, and the indentation the file gives it,
    which is what `new` must be re-indented to. None or several matches return None
    and leave the not-found error to explain. Whole lines only: a snippet that starts
    mid-line has no indentation to adjust.
    """
    trailing = old.endswith(nl)
    body = old[: -len(nl)] if trailing else old
    wanted = textwrap.dedent(body.replace(nl, "\n"))
    if not wanted.strip():
        return None
    lines = text.split(nl)
    n = wanted.count("\n") + 1
    head = wanted.split("\n", 1)[0].strip()  # cheap first-line check before dedenting
    found = [
        i
        for i in range(len(lines) - n + 1)
        if lines[i].strip() == head and textwrap.dedent("\n".join(lines[i : i + n])) == wanted
    ]
    if len(found) != 1:
        return None
    i = found[0]
    start = sum(len(line) + len(nl) for line in lines[:i])
    end = start + len(nl.join(lines[i : i + n]))
    if trailing and text.startswith(nl, end):
        end += len(nl)
    return start, end, _margin(lines[i : i + n])


def _reindent(snippet: str, margin: str, nl: str) -> str:
    """`snippet` with its own common indentation replaced by `margin`; blank lines stay blank."""
    trailing = snippet.endswith(nl)
    body = snippet[: -len(nl)] if trailing else snippet
    text = textwrap.indent(textwrap.dedent(body.replace(nl, "\n")), margin).replace("\n", nl)
    return text + (nl if trailing else "")


def _line_span(text: str, start: int, snippet: str) -> tuple[int, int]:
    """1-based (first, last) line numbers that `snippet` occupies when it sits at
    character offset `start` of `text`. A trailing newline belongs to the last line
    rather than starting a new one."""
    first = text.count("\n", 0, start) + 1
    newlines = snippet.count("\n")
    if snippet.endswith("\n"):
        newlines -= 1
    return first, first + max(newlines, 0)


def _not_found_hint(text: str, old: str) -> str:
    """Why `old` might not have matched: the same line with different indentation, or
    the closest line we can find. The first pass is one cheap scan; the difflib pass
    behind it is bounded by HINT_MAX_LINES / HINT_MAX_LINE. Often enough for the
    model to correct its snippet on the next call. (By the time this runs, the
    re-indent fallback has already failed: the whitespace case it still reports is
    an ambiguous one, or a snippet whose lines disagree on their relative indent.)"""
    wanted = next((ln.strip() for ln in old.splitlines() if ln.strip()), "")
    if not wanted:
        return ""
    lines = text.splitlines()
    for lineno, line in enumerate(lines, start=1):
        if line.strip() == wanted:
            return (
                f" Line {lineno} matches except for whitespace/indentation: {line!r}. "
                "`old` must match the file text exactly, including leading spaces."
            )
    if len(lines) > HINT_MAX_LINES or len(wanted) > HINT_MAX_LINE:
        return ""  # difflib would cost seconds here; see HINT_MAX_LINES
    close = difflib.get_close_matches(wanted, [ln.strip() for ln in lines], n=1, cutoff=0.6)
    if close:
        lineno = next(i for i, ln in enumerate(lines, start=1) if ln.strip() == close[0])
        return f" Closest line is {lineno}: {lines[lineno - 1]!r}."
    return ""


# --- a wall-clock bound for in-process work ---------------------------------------

class _OutOfTime(Exception):
    """Raised inside a _time_limit block when its budget is spent.

    Deliberately NOT the builtin TimeoutError: that one is a subclass of OSError, and
    the `except OSError` that skips unreadable files in grep would swallow it."""


@contextmanager
def _time_limit(seconds: float):
    """Raise _OutOfTime inside the block if it has run longer than `seconds`.

    Why this exists: grep runs a regular expression the MODEL wrote, and Python's re
    has no timeout. A pattern with nested repetition — (a+)+$ — backtracks
    exponentially: 24 seconds on a 29-character line, doubling with every character
    after that, so on a real line it never returns. run_bash and run_python cannot
    hang the loop that way because their subprocess gets killed (see _subprocess.py);
    a tool that runs in-process needs a bound of its own. A line-length cap is not
    one — the 29-character line is already past saving.

    SIGALRM is that bound. The interpreter checks for pending signals even in the
    middle of a running match, so the handler's exception cuts the match short and
    the loop gets its turn back. It comes with two limits, and in both cases the
    block simply runs unguarded rather than failing: only the main thread of a POSIX
    process may arm the alarm, and a caller that already has one pending (a test
    runner's timeout, say) must not have it clobbered.
    """
    usable = (
        hasattr(signal, "SIGALRM")
        and threading.current_thread() is threading.main_thread()
        and signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    )
    if not usable:
        yield
        return

    def out_of_time(signum, frame):
        raise _OutOfTime(f"exceeded {seconds:g}s")

    previous = signal.signal(signal.SIGALRM, out_of_time)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)  # disarm before restoring the handler
        signal.signal(signal.SIGALRM, previous)


# --- what write_file and edit_file tell the model about themselves -----------------

# The model reads a tool's description to decide when and how to call it, so the
# description has to describe THIS instance: a write_file built without the gate must
# not promise one, and one built without the report must not promise "created or
# overwritten". The text is kept as separate sentences and assembled per instance, so
# that the default (every piece on) is today's description byte for byte.
_WRITE_INTRO = "Write text to a file in the workspace, creating parent directories as needed. "
# With the Week 4b scaffolding on, the model is pointed at edit_file for small changes.
_WRITE_OVERWRITES_USE_EDIT = (
    "Overwrites the WHOLE file if it exists, so to change a few lines of an existing "
    "file use edit_file instead. "
)
# With all of it off — level 3 of the ladder, the round-two harness — the round-two
# sentence, verbatim. There was no edit_file to point at then, and a model told about
# a tool it does not have calls a tool that does not exist ("Unknown tool" from the
# registry): a penalty round two never paid, and one the ladder must not charge to the
# level that is meant to reproduce it.
_WRITE_OVERWRITES_PLAIN = "Overwrites if the file exists. "
_WRITE_GATE = (
    "For a .py path the content must parse as Python or the write is refused and the "
    "file left unchanged. "
)
_WRITE_REPORT = (
    "The result says whether the file was created or overwritten, its size before and "
    "after, and (for .py) any top-level functions or classes that disappeared. "
)
_EDIT_INTRO = (
    "Replace ONE exact snippet of text in a file with new text, leaving the rest of "
    "the file untouched. This is the preferred way to change a few lines; use "
    "write_file only to create a file or replace all of it. `old` must match the "
    "file text exactly, including indentation and line breaks, and must occur "
    "exactly once (include a neighbouring line or two to make it unique). "
)
_EDIT_GATE = (
    "For a .py path the edited file must still parse as Python or the edit is refused "
    "and the file left unchanged. "
)
_PATHS_RELATIVE = "Paths are relative to the workspace root."


def _write_file_description(syntax_gate: bool, change_report: bool) -> str:
    """write_file's description for these flags. Both True is today's text exactly.

    The edit_file hint goes with the scaffolding as a whole: with either flag on this
    is the Week 4b tool, which was designed alongside edit_file; with both off it is
    the round-two tool, and round two had no edit_file (see _WRITE_OVERWRITES_PLAIN).
    """
    scaffolded = syntax_gate or change_report
    return (
        _WRITE_INTRO
        + (_WRITE_OVERWRITES_USE_EDIT if scaffolded else _WRITE_OVERWRITES_PLAIN)
        + (_WRITE_GATE if syntax_gate else "")
        + (_WRITE_REPORT if change_report else "")
        + _PATHS_RELATIVE
    )


def _edit_file_description(syntax_gate: bool) -> str:
    """edit_file's description for this flag. True is today's text exactly."""
    return _EDIT_INTRO + (_EDIT_GATE if syntax_gate else "") + _PATHS_RELATIVE


# --- the tools ------------------------------------------------------------------

class ReadFile(Tool):
    name = "read_file"
    description = (
        "Read a UTF-8 text file from the workspace and return its contents. "
        "Paths are relative to the workspace root."
    )
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "file path, relative to workspace root"}},
        "required": ["path"],
    }

    def __init__(self, workspace: Workspace):
        self.ws = workspace

    def run(self, arguments: dict) -> str:
        path = self.ws.resolve(arguments["path"])  # raises if outside sandbox
        if not path.is_file():
            return f"Error: not a file: {self.ws.relative(path)}"
        data = path.read_text(encoding="utf-8", errors="replace")
        if len(data) > MAX_BYTES:
            return data[:MAX_BYTES] + f"\n... [truncated at {MAX_BYTES} chars]"
        return data


class WriteFile(Tool):
    name = "write_file"
    # The class attribute is the default tool's text; __init__ sets the instance's own
    # from its flags, and the registry reads the instance (see _write_file_description).
    description = _write_file_description(syntax_gate=True, change_report=True)
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "file path, relative to workspace root"},
            "content": {"type": "string", "description": "the complete new text of the file"},
        },
        "required": ["path", "content"],
    }

    def __init__(self, workspace: Workspace, syntax_gate: bool = True, change_report: bool = True):
        """`syntax_gate` and `change_report` switch the two Week 4b behaviours off.

        Both default to on, and default_tools() passes neither, so every existing
        caller gets the tool the module docstring describes. They exist for the
        capability ladder (benchmark/ladder), whose level 3 is the round-two harness:
        WriteFile(ws, syntax_gate=False, change_report=False) is that tool — a broken
        .py goes to disk, and the result is "wrote <path>: N chars" — so the gate and
        the report can be measured as their own rung rather than assumed to be worth
        what round three suggested. The description the model reads follows the
        flags, because a description that promises a gate the tool does not have is
        a lie the model would act on.
        """
        self.ws = workspace
        self.syntax_gate = syntax_gate
        self.change_report = change_report
        self.description = _write_file_description(syntax_gate, change_report)

    def run(self, arguments: dict) -> str:
        path = self.ws.resolve(arguments["path"])  # raises if outside sandbox
        rel = self.ws.relative(path)
        content = arguments["content"]

        # Gate first: nothing on disk changes (not even a parent directory) unless
        # the content is acceptable. With the gate off (level 3 of the ladder) the
        # content goes to disk as it is, broken or not, exactly as round two's did.
        if self.syntax_gate:
            refusal = _syntax_refusal(rel, content, verb="write")
            if refusal:
                return refusal
        if path.is_dir():
            return f"Error: is a directory: {rel}"

        # The change report needs the old file, and is worked out BEFORE the write.
        # It parses that file, which the gate never saw (it may have reached the disk
        # through run_bash or a checkout), and a parse can fail. Anything that can
        # fail after the file has changed tells the model the write failed when it
        # succeeded — the model then retries, and the repeated-call note does not
        # even catch it because the second result differs. So: compute first, write
        # last. With the report off nothing is read: the round-two tool did not look.
        old = None
        removed: list[str] = []
        if self.change_report:
            # Decoded for the report only — the old text is never written back, so an
            # undecodable byte becoming U+FFFD costs nothing here. The raw bytes are
            # decoded (rather than read_text) so CRLF is not translated and the size
            # reported is the size on disk.
            old = path.read_bytes().decode("utf-8", errors="replace") if path.is_file() else None
            if old is not None and _is_python(rel):
                removed = _removed_top_level(old, content)
        path.parent.mkdir(parents=True, exist_ok=True)
        # newline="": the content goes to disk byte for byte, CRLF included.
        path.write_text(content, encoding="utf-8", newline="")

        if not self.change_report:
            # The round-two report: how many characters landed, and not a word about
            # what they replaced. This is the message that said "Wrote 179 chars" over
            # a file that had just lost two functions (module docstring).
            return f"wrote {rel}: {len(content)} chars"
        if old is None:
            return f"created {rel}: {len(content)} chars"

        report = f"overwrote {rel}: {len(old)} -> {len(content)} chars"
        if removed:
            report += (
                f"; REMOVED top-level: {', '.join(removed)}. "
                "If that was not intended, restore them with edit_file."
            )
        return report


class EditFile(Tool):
    name = "edit_file"
    # The default tool's text; the instance's own is set from its flag in __init__.
    description = _edit_file_description(syntax_gate=True)
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "file path, relative to workspace root"},
            "old": {"type": "string",
                    "description": "the exact text to replace (must occur exactly once; "
                                   "must differ from `new`)"},
            "new": {"type": "string", "description": "the text to put in its place"},
        },
        "required": ["path", "old", "new"],
    }

    def __init__(self, workspace: Workspace, syntax_gate: bool = True):
        """`syntax_gate` False skips the Python compile check; see WriteFile.__init__.

        No level of the ladder runs edit_file without its gate — level 4 turns the
        tool and its gate on together — but the flag is there so the gate can be
        measured on its own should the question come up, and so the two tools that
        share the gate share the knob.
        """
        self.ws = workspace
        self.syntax_gate = syntax_gate
        self.description = _edit_file_description(syntax_gate)

    def run(self, arguments: dict) -> str:
        path = self.ws.resolve(arguments["path"])  # raises if outside sandbox
        rel = self.ws.relative(path)
        old, new = arguments["old"], arguments["new"]

        if not path.is_file():
            return f"Error: not a file: {rel}"
        if old == "":
            return "Error: `old` is empty; give the exact text to replace."

        text = _read_utf8(path)
        if text is None:
            return (
                f"Error: {rel} is not UTF-8 text; edit_file only edits UTF-8 files. "
                "The file is unchanged."
            )

        # The file's own line ending wins: both snippets are converted to it, so the
        # match and the write use it and the rest of the file keeps its bytes.
        nl = _line_ending(text)
        old, new = _with_line_ending(old, nl), _with_line_ending(new, nl)

        if old == new:
            # Round three: a model "fixed" code that was already right and was told
            # "replaced 1 occurrence". An edit that changes nothing is not an edit,
            # and saying so is what tells the model the file already reads that way.
            note = f" {rel} already contains that text, so no edit is needed." if old in text else ""
            return f"Error: old and new are identical; nothing changed.{note}"

        count = text.count(old)
        adjusted = False
        if count > 1:
            return (
                f"Error: `old` occurs {count} times in {rel}; include more surrounding "
                "lines so it matches exactly once."
            )
        if count == 1:
            start = text.index(old)
            end = start + len(old)
        else:
            # Not there verbatim. Before giving up, try it with the indentation of
            # both sides ignored — the round-three loop (see _match_reindented).
            match = _match_reindented(text, old, nl)
            if match is None:
                return f"Error: `old` not found in {rel}.{_not_found_hint(text, old)}"
            start, end, margin = match
            new = _reindent(new, margin, nl)
            adjusted = True

        updated = text[:start] + new + text[end:]

        if updated == text:
            # Gap B by another road. The `old == new` guard above only catches
            # literally identical snippets; here `old` was mis-indented, `new`
            # differed from it only in indentation, and re-indenting `new` to the
            # file's margin gave back the very block on disk — a model "fixing"
            # indentation that was already right. Reporting "replaced 1 occurrence"
            # would tell it the file changed when nothing did.
            return (
                f"Error: this edit changes nothing in {rel}: `old` matched only after "
                "adjusting indentation and `new` re-indents to the same text. The file "
                "is unchanged."
            )

        if self.syntax_gate:
            refusal = _syntax_refusal(rel, updated, verb="edit")
            if refusal:
                return refusal

        path.write_text(updated, encoding="utf-8", newline="")  # newline="": no translation
        # Report where the replacement now sits, so the model can read_file and check.
        first, last = (
            _line_span(updated, start, new) if new else _line_span(text, start, text[start:end])
        )
        report = f"edited {rel}: replaced 1 occurrence (lines {first}-{last})"
        if adjusted:
            report += (
                "; matched after adjusting indentation (`old` was indented differently "
                "from the file, so `new` was re-indented to match)"
            )
        return report


class ListDir(Tool):
    name = "list_dir"
    description = (
        "List the entries of a directory in the workspace (directories shown with a "
        "trailing slash). Defaults to the workspace root. Paths are relative to root."
    )
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "directory path; defaults to '.'"}},
    }

    def __init__(self, workspace: Workspace):
        self.ws = workspace

    def run(self, arguments: dict) -> str:
        path = self.ws.resolve(arguments.get("path", "."))  # raises if outside sandbox
        if not path.is_dir():
            return f"Error: not a directory: {self.ws.relative(path)}"
        entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
        if not entries:
            return f"(empty directory: {self.ws.relative(path)})"
        return "\n".join(e.name + ("/" if e.is_dir() else "") for e in entries)


class Grep(Tool):
    name = "grep"
    description = (
        "Search for a regular-expression pattern in the workspace. `path` may be a "
        "directory (searched recursively) or a single file. Returns matching lines as "
        "path:line_number:line. Use this to find where something is defined or used."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "a Python regular expression"},
            "path": {"type": "string",
                     "description": "a directory to search recursively, or one file; defaults to '.'"},
        },
        "required": ["pattern"],
    }

    def __init__(self, workspace: Workspace, timeout: float = GREP_TIMEOUT):
        self.ws = workspace
        self.timeout = timeout

    def run(self, arguments: dict) -> str:
        root = self.ws.resolve(arguments.get("path", "."))  # raises if outside sandbox
        regex = re.compile(arguments["pattern"])

        # What to search depends on what `path` is. The first version always did
        # root.rglob("*") — "everything inside this directory" — which is right for a
        # directory but yields NOTHING for a file, so grep(pattern, path="calc.py")
        # answered "(no matches)" even when calc.py contained the pattern. A wrong
        # "no matches" is worse than an error: the model believes it and reasons from
        # a false fact. (Found in the model spike: qwen3:8b hit it six times.)
        if root.is_file():
            files = [root]
        elif root.is_dir():
            files = sorted(root.rglob("*"))
        else:
            return f"Error: no such file or directory: {self.ws.relative(root)}"

        hits: list[str] = []
        try:
            with _time_limit(self.timeout):
                for file in files:
                    # rglob hands back every entry under root, and an entry may be a
                    # symlink pointing ANYWHERE: is_file() and read_text() both follow
                    # it without a word. Resolving only the starting path made grep the
                    # one file tool that could read ~/.ssh/id_rsa through a link planted
                    # in the workspace — and grep never asks permission. So each
                    # candidate goes through ws.resolve() like the path of every other
                    # tool, and one that lands outside the sandbox is skipped.
                    try:
                        real = self.ws.resolve(file)
                    except WorkspaceError:
                        continue
                    if not real.is_file():
                        continue
                    try:
                        text = real.read_text(encoding="utf-8", errors="strict")
                    except (UnicodeDecodeError, OSError):
                        continue  # skip binary / unreadable files
                    for lineno, line in enumerate(text.splitlines(), start=1):
                        if regex.search(line):
                            hits.append(f"{self.ws.relative(file)}:{lineno}:{line.strip()}")
                            if len("\n".join(hits)) > MAX_BYTES:
                                hits.append("... [truncated: too many matches]")
                                return "\n".join(hits)
        except _OutOfTime:
            # The pattern is the model's, and one with nested repetition backtracks
            # exponentially (see _time_limit). Hand back what was found, then say why
            # the search stopped, in terms the model can fix.
            hits.append(
                f"Error: grep stopped after {self.timeout:g}s: pattern "
                f"{arguments['pattern']!r} is too expensive to match (nested repetition "
                "such as (a+)+ backtracks exponentially). Simplify the pattern, or "
                "search one file."
            )
            return "\n".join(hits)
        return "\n".join(hits) if hits else f"(no matches for {arguments['pattern']!r})"
