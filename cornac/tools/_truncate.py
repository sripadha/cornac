"""Head+tail truncation for long tool output.

Why this exists: tool output goes straight into the model's context window, so it has
to be capped — a test suite or a build can print megabytes. The obvious cap,
`text[:limit]`, is the wrong one. It keeps the *start* of the output and throws away
the *end*, which is exactly where the useful part lives: a Python traceback ends with
the actual exception, pytest prints its pass/fail summary last, compilers put the
error count at the bottom. An agent that only sees the first N chars of a failing
test run has no idea what failed.

So we keep both ends. The head shows what the command was doing (which tests ran,
what it printed first); the tail shows how it ended (the error, the summary, the
exit). A notice in the middle tells the model how much was dropped, so it can decide
to re-run with narrower output (e.g. `pytest -q tests/test_one.py`) if it needs to
see the missing part.
"""

from __future__ import annotations


def truncate_head_tail(text: str, head: int, tail: int, label: str = "output") -> str:
    """Return `text` unchanged if it fits in head+tail chars; otherwise keep the first
    `head` chars and the last `tail` chars, with a notice between them saying how
    much was omitted. `label` names what was cut (e.g. "output", "stderr")."""
    # A limit of zero (or less) means "keep nothing from that side". Guarding it
    # matters most for the tail: text[-0:] is text[0:] — the WHOLE string — so an
    # unguarded tail=0 would return the notice followed by all the output it had just
    # claimed to cut.
    head = max(head, 0)
    tail = max(tail, 0)
    if len(text) <= head + tail:
        return text
    dropped = len(text) - head - tail
    notice = (
        f"\n... [{label} truncated: {dropped} chars omitted; "
        f"showing first {head} and last {tail}] ...\n"
    )
    tail_part = text[-tail:] if tail else ""
    return text[:head] + notice + tail_part
