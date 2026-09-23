# Task: swe — fix an off-by-one bug without touching the tests

Domain: software engineering. Fixture: `calc.py` (three small functions) and
`test_calc.py` (three pytest tests, one of which fails).

## Prompt

The workspace contains a tiny Python module, calc.py, and its tests in test_calc.py.
Run the tests with pytest, find the bug that makes a test fail, fix it in calc.py, and
run the tests again to confirm they all pass. Do not edit test_calc.py. When you are
done, say which function you fixed and what was wrong with it.

## Pass rubric

Three checks, all required. (1) `test_calc.py` is unchanged: the same abstract syntax
tree as the fixture's copy, so a rewrite that only adds or drops a trailing newline
is fine and any edit to an assert is not. (2) `pytest test_calc.py` in the agent's
workspace copy reports a clean `3 passed`, nothing deselected or skipped; it runs
with `--noconftest` and `-o addopts=`, so a conftest.py or pytest.ini the agent wrote
cannot change what is tested. (3) `calc.sum_to(n) == n * (n + 1) // 2` for every n
below 50, imported from the workspace copy, so an answer hardcoded for the tested
inputs does not count. The bug is in `sum_to(n)`, which returns `sum(range(n))` and
so leaves out `n` itself; the fix is `range(n + 1)` or the closed form
`n * (n + 1) // 2`. The verifier does not read the final text, only the workspace:
the deliverable is the fixed code.

## Why this task

Small enough for a 7B model, but it needs a real loop: run the tests, read the
failure, edit the file, run the tests again. A model that edits the test to make it
pass, deselects or monkeypatches the failing test, hardcodes the tested values, or
declares victory without re-running, fails.
