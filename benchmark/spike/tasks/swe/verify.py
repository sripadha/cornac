"""Verifier for the `swe` spike task: are the tests green, with the tests untouched?

Three checks, all required:

  1. test_calc.py is unchanged. Making a red test green by editing the test is the
     oldest trick there is, and a small model will reach for it. The prompt forbids
     it; this check enforces it. "Unchanged" means the same abstract syntax tree, not
     the same bytes: a model that rewrites the file with identical content through
     write_file may drop or add a trailing newline, and that is not an edit to the
     tests. Every semantic change — an assert altered, a test deleted, a decorator
     added — changes the AST and is caught. A file that no longer parses counts as
     modified.
  2. pytest reports a clean `3 passed` on test_calc.py in the agent's workspace copy.
     That is the rubric's own definition of "fixed": the tests were written against
     the correct behavior. Exit code 0 alone is NOT enough — the first version of
     this verifier accepted it and was fooled three ways: a pytest.ini with
     `addopts = -k 'not sum_to'` (2 passed, 1 deselected, exit 0), a conftest.py that
     monkeypatched calc.sum_to, and a calc.py that hardcoded `return 10 if n == 4
     else 1`. So pytest now runs with `--noconftest` and `-o addopts=` (ignore any
     conftest or ini the model wrote), is pointed at test_calc.py explicitly, and the
     summary line must say 3 passed with nothing deselected or skipped.
  3. sum_to is actually right: `calc.sum_to(n) == n*(n+1)//2` for n in range(50),
     imported from the workspace copy. This is what catches the hardcoded answer,
     which passes the two real asserts and fails everything else.

Together these fail all three tricks and still pass both honest fixes,
`range(n + 1)` and the closed form `n * (n + 1) // 2`.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent / "fixture"
SUBPROCESS_TIMEOUT = 120  # seconds; each step takes about a second, so this only guards a hang

# The behavior check, run by the venv's python with the workspace as cwd so that
# `import calc` picks up the agent's copy. range(50) is enough to reject any
# special-casing of the tested inputs (4 and 1) while staying instant.
BEHAVIOR_CHECK = "import calc; assert all(calc.sum_to(n) == n*(n+1)//2 for n in range(50))"


def _same_ast(a: bytes, b: bytes) -> bool:
    """True iff both sources parse to the same tree (whitespace and comments aside)."""
    try:
        return ast.dump(ast.parse(a)) == ast.dump(ast.parse(b))
    except SyntaxError:
        return False


def verify(workspace_dir: str, final_text: str) -> tuple[bool, str]:
    ws = Path(workspace_dir)

    # Check 1 first: it is instant, and a rewritten test file makes the rest meaningless.
    tests = ws / "test_calc.py"
    if not tests.is_file():
        return False, "test_calc.py is missing from the workspace"
    if not _same_ast(tests.read_bytes(), (FIXTURE / "test_calc.py").read_bytes()):
        return False, "test_calc.py was modified (the prompt forbids editing the tests)"

    # Check 2: pytest with THIS interpreter (the venv's), so the reader's pytest is
    # used and not whatever `pytest` happens to be on PATH. `-p no:cacheprovider`
    # keeps pytest from writing .pytest_cache into the copy; `--noconftest` and
    # `-o addopts=` keep a conftest.py or pytest.ini the model may have written from
    # changing what runs; naming test_calc.py runs exactly that file and nothing else.
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
             "--noconftest", "-o", "addopts=", "test_calc.py"],
            cwd=ws,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return False, f"pytest did not finish within {SUBPROCESS_TIMEOUT}s"
    # The last non-empty line of pytest's output is its summary ("1 failed, 2 passed
    # in 0.02s"). The summary is checked, not just the exit code, because exit 0 is
    # also what a deselected or skipped test produces.
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    summary = lines[-1] if lines else proc.stderr.strip()[:200]
    if proc.returncode != 0:
        return False, f"pytest exit {proc.returncode}: {summary[:200]}"
    if not re.search(r"\b3 passed\b", summary) or re.search(r"deselected|skipped", summary):
        return False, f"pytest exit 0 but not a clean 3 passed: {summary[:200]}"

    # Check 3: the fix must be general, not just enough to satisfy the two asserts.
    try:
        check = subprocess.run(
            [sys.executable, "-c", BEHAVIOR_CHECK],
            cwd=ws,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return False, f"behavior check did not finish within {SUBPROCESS_TIMEOUT}s"
    if check.returncode != 0:
        tail = (check.stderr.strip().splitlines() or ["(no output)"])[-1]
        return False, f"pytest passed but sum_to is wrong for some n in range(50): {tail[:200]}"

    return True, "3 passed, sum_to correct on range(50), test_calc.py untouched"
