# Task: swe3 — fix state that leaks between calls (a mutable default argument)

Domain: software engineering (coding, round two). Fixture: `cart.py` (three small
functions) and `test_cart.py` (three pytest tests, one of which fails).

## Prompt

The workspace contains a small Python project and its pytest tests. Run the tests
with pytest: one of them fails. Find the bug that makes it fail, fix it in the
project's source code, and run the tests again to confirm they all pass. Do not edit
the test file. When you are done, say which file and function you fixed and what was
wrong with it.

## Pass rubric

Three checks, all required. (1) `test_cart.py` is unchanged: the same abstract
syntax tree as the fixture's copy, so a rewrite that only adds or drops a trailing
newline is fine and any edit to an assert is not. (2) `pytest test_cart.py` in the
agent's workspace copy reports a clean `3 passed`, nothing deselected or skipped; it
runs with `--noconftest` and `-o addopts=`, so a conftest.py or pytest.ini the agent
wrote cannot change what is tested. (3) `cart.add_item` behaves correctly on inputs
the tests never mention, imported from the workspace copy and called several times
in ONE interpreter: three no-argument calls with three different items each return a
one-item list (and three distinct list objects), a call with an explicit list
returns that list with the item appended, and the caller's list is either extended
in place or left alone, never emptied. An answer hardcoded for the tested items does
not count.

The bug is the classic mutable default argument: `def add_item(item, items=[])`
creates ONE list when the function is defined, so every call made without `items`
appends to the same list and the second call returns `["apple", "pear"]`. The fix
is the standard idiom, `items=None` with `if items is None: items = []`; a version
that returns a new list (`return [*items, item]`) passes too. The verifier does not
read the final text, only the workspace: the deliverable is the fixed code.

## Why this task

A different bug class from `swe` and `swe2`. There the failing line was wrong in
isolation; here every line of `add_item` is fine on its own and the defect only
shows when the function is called twice, which is what the failing test does. The
model has to recognize the pattern from the symptom (the first assert passes, the
second sees the first call's item) rather than spot a wrong operator. The prompt
names neither the function nor the file. A model that edits the test, deselects it,
special-cases the tested items, or "fixes" it by clearing the caller's list is
caught by the checks above.
