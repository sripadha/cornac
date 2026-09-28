"""The capability ladder: one frozen model, six levels of help, one table.

This package holds the experiment the whole project is built around. The same small
model (qwen3.5:4b) fixes the same three broken programs six times over, and each time
it is given exactly one more thing than the time before: nothing, then a system
prompt, then one shell tool, then the full tool set, then the harness's loop guards,
then sub-agents. Counting how often it succeeds at each level shows what each piece
of scaffolding is worth, in one number per level.

Two modules:

  levels.py      the six levels, each a `Level` whose run() builds the model call or
                 the agent for that level (the single source of truth for what each
                 level adds);
  run_ladder.py  the runner: fresh fixture copy per run, grading by the task's own
                 verify.py, one JSON line per run, the results table.

It is a package (this file exists) only so that tools which look for __init__.py treat
the directory as a unit; the runner and the levels are loaded by file path, the way
the model spike is (see tests/conftest.py), because they are scripts that live outside
the cornac package and are not meant to be pip-installed.
"""
