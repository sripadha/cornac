# Task: swe2 — fix a bug that lives in a helper, one module away from the failing test

Domain: software engineering (coding, round two). Fixture: `grades.py` (three small
functions), `utils.py` (the helpers they call) and `test_grades.py` (four pytest
tests, one of which fails).

## Prompt

The workspace contains a small Python project and its pytest tests. Run the tests
with pytest: one of them fails. Find the bug that makes it fail, fix it in the
project's source code, and run the tests again to confirm they all pass. Do not edit
the test file. When you are done, say which file and function you fixed and what was
wrong with it.

## Pass rubric

Three checks, all required. (1) `test_grades.py` is unchanged: the same abstract
syntax tree as the fixture's copy, so a rewrite that only adds or drops a trailing
newline is fine and any edit to an assert is not. (2) `pytest test_grades.py` in the
agent's workspace copy reports a clean `4 passed`, nothing deselected or skipped; it
runs with `--noconftest` and `-o addopts=`, so a conftest.py or pytest.ini the agent
wrote cannot change what is tested. (3) `grades.normalize` behaves correctly on
inputs the tests never mention, imported from the workspace copy: every integer mark
from -5 to 105 is either scaled to `m / 100` (0 to 100 inclusive) or rejected with
ValueError (anything else), plus a few float edge cases (`100.0` is legal, `100.001`
and `-0.001` are not). An answer hardcoded for the tested inputs does not count.

The bug is not in the file the traceback points at. `normalize` in `grades.py`
rejects a mark unless `within(m, 0, 100)` says it is legal, and `within` in
`utils.py` is documented as a closed interval ("both ends included") but is written
as `lo < x < hi`, an open one, so 0 and 100 are refused. The fix is `lo <= x <= hi`.
A fix that changes `normalize` instead (an inline `0 <= m <= MAX_MARK`) also passes,
because the verifier grades behavior, not location; `verify_reason` records whether
the helper itself was repaired, so the two can be told apart in `runs.jsonl`. The
verifier does not read the final text, only the workspace: the deliverable is the
fixed code.

## Why this task

The `swe` task's bug sat in the only source file, one screen away from the failing
assert. Here the failing test calls `normalize`, the traceback ends in `grades.py`,
and the actual defect is a boundary condition in a helper in a second module. To
fix it the model has to follow a call across files: read the traceback, open
`grades.py`, notice `within` is imported from `utils.py`, and go look. The prompt
names neither the function nor the file. A model that edits the test, deselects it,
special-cases the tested values, or patches the symptom in `grades.py` without
understanding it, is caught by the checks above.
