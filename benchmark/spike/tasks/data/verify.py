"""Verifier for the `data` spike task: did the agent report the right revenue total?

The ground truth is fixed by the fixture: sales.csv has forty rows, and
sum(units * price) over them is 121764.46. It is hardcoded here rather than recomputed
from the workspace on purpose — the workspace is the AGENT's copy, and an agent that
edits sales.csv must not be able to move the goalposts.

Forty rows, not six. The first version of the fixture had six, and a verbose model
read the file once and then multiplied the six rows by hand in a page of prose — and
got it right. That pass said nothing about whether the model can drive run_python or
run_bash, which is what the spike is measuring. Forty non-round prices times
two-digit unit counts is past what any model does reliably in its head, so a pass now
means a tool did the arithmetic (runs.jsonl's `tools_used` shows which one).
"""

from __future__ import annotations

import re

# sum(units * price) over the 40 rows of fixture/sales.csv, computed with Decimal
# arithmetic; the float sum a model's run_python would produce agrees to the cent.
# The rubric in task.md quotes this number too — change both or neither.
EXPECTED = "121764.46"


def verify(workspace_dir: str, final_text: str) -> tuple[bool, str]:
    """Pass iff the exact total appears in the agent's final answer.

    The number may be written as 121764.46, 121,764.46, $121764.46 or $121,764.46 —
    those are all the same answer, so separators and a currency sign are stripped
    first. Anything else (121764.5, 121764, 121764.460) is not the answer: the prompt
    asks for two decimals, and a verifier that forgave rounding would also forgive a
    model that never computed the number.
    """
    if not final_text:
        return False, "no final answer (the run produced no text)"
    text = final_text.replace(",", "").replace("$", "")
    # Guard both sides so the digits inside 1121764.46 or 121764.461 do not count.
    if re.search(rf"(?<![\d.]){re.escape(EXPECTED)}(?!\d)", text):
        return True, f"final answer contains {EXPECTED}"
    return False, f"final answer does not contain {EXPECTED}"
