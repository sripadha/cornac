"""Verifier for the `research` spike task: right file, right number?

Ground truth is fixed by the fixture: notes/memory-experiments.txt, line 4, says that
gradient checkpointing "cut memory by 37%". Both halves must appear in the final
answer — the file name and the number. Naming the file without the figure means the
agent found the note but did not read it; 37 without the file name could be a lucky
guess. No other note in the fixture contains the number 37.
"""

from __future__ import annotations

import re
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent / "fixture"
EXPECTED_FILE = "memory-experiments.txt"
EXPECTED_NUMBER = "37"

# The number as a standalone figure. \b37\b was the first attempt and it is too
# loose: it matches inside "37.5%", "2.37 GB" and "0.37". The lookbehind rejects a
# digit or a decimal point before it (2.37, 137); the lookahead rejects a digit after
# it, with or without a decimal point in between (370, 37.5). "37", "37%",
# "37 percent" and a sentence-final "37." all still match.
NUMBER_PATTERN = re.compile(rf"(?<![\d.]){EXPECTED_NUMBER}(?!\.?\d)")

# Every note in the fixture except the right one, read off the fixture directory so
# this list cannot drift from it. Used to catch a hedged answer.
OTHER_NOTES = sorted(
    p.name for p in (FIXTURE / "notes").iterdir() if p.is_file() and p.name != EXPECTED_FILE
)


def _hedged_file(text: str) -> str | None:
    """The first other note named in a "File:" slot, or None.

    The prompt asks for the answer as "File: <name>. Memory saving: <value>". A reply
    like "File: data-loading.txt or memory-experiments.txt" contains the right name
    and would pass a plain substring check, but it is a hedge, not an answer. So the
    text after every "File:" — up to the "Memory saving" part, or the end of the
    line — must not name any other note. Another note mentioned elsewhere ("data-
    loading.txt also says 'checkpoint' but means saved weights") is fine: that is an
    explanation, not a hedge, and the spike should not fail a model for showing its
    reasoning. `text` is already lowercased by the caller.
    """
    for match in re.finditer(r"file:\s*([^\n]*)", text):
        slot = match.group(1).split("memory saving")[0]
        for other in OTHER_NOTES:
            if other in slot:
                return other
    return None


def verify(workspace_dir: str, final_text: str) -> tuple[bool, str]:
    if not final_text:
        return False, "no final answer (the run produced no text)"
    text = final_text.lower()
    has_file = EXPECTED_FILE in text
    has_number = NUMBER_PATTERN.search(text) is not None
    if has_file and has_number:
        other = _hedged_file(text)
        if other:
            return False, f"final answer hedges between {EXPECTED_FILE} and {other}"
        return True, f"final answer names {EXPECTED_FILE} and {EXPECTED_NUMBER}%"
    missing = []
    if not has_file:
        missing.append(f"the file name ({EXPECTED_FILE})")
    if not has_number:
        missing.append(f"the memory saving ({EXPECTED_NUMBER}%)")
    return False, "final answer is missing " + " and ".join(missing)
