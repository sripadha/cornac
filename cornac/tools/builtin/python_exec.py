"""Python tool: run_python.

Runs a snippet of Python in a separate process, with the workspace root as the
working directory, and returns whatever the snippet printed (plus any error). Useful
for data-analysis tasks: load a CSV, compute a mean, filter rows.

The snippet runs like a notebook cell: if its last statement is a bare expression,
the value is echoed as repr(), exactly as Jupyter/IPython do. Why: the weak local
model (Qwen 2.5 7B) keeps ending its snippet with `total_revenue` instead of
`print(total_revenue)` — the habit of someone who lives in notebooks. Under a plain
`python -c` that value is computed and thrown away, the tool reports "(no output)",
and the model gives up. Echoing the final expression meets the model where it is.
This is scaffolding, which is the project's thesis. The runner that implements the
rule — and keeps tracebacks pointing at the model's own line numbers — lives in
cornac/tools/_pyrunner.py.

Like run_bash, this executes arbitrary code and is gated by the Week 3 permission
system (defaults to "ask"). We run it as a subprocess (not exec() in-process) for two
reasons: a crash or infinite loop can't take down the agent, and a timeout can kill it
— together with anything the snippet itself started (see cornac/tools/_subprocess.py).

This is NOT a security sandbox — a determined snippet can still read files the user can
read. Real isolation (containers, seccomp) is out of scope for v1; the honest controls
are the permission prompt, the subprocess timeout, and the workspace cwd.

The description spells out that a snippet runs in the workspace and does not edit
files. In the model spike (benchmark/spike/) one model redefined the buggy function
inside a run_python snippet, saw its own test pass, and reported the file fixed — the
file on disk was untouched. A snippet is a scratch cell; edit_file (a few lines) or
write_file (the whole file) is how a file changes, and the description says so.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from cornac.tools._pyrunner import RUNNER_SOURCE
from cornac.tools._subprocess import partial_output, run_with_timeout
from cornac.tools._truncate import truncate_head_tail
from cornac.tools.base import Tool
from cornac.tools.workspace import Workspace

DEFAULT_TIMEOUT = 30

# Same head+tail policy as run_bash: a Python traceback ends with the exception line,
# so cutting the tail would hide the one thing the model most needs to see.
HEAD_CHARS = 40_000
TAIL_CHARS = 10_000


class RunPython(Tool):
    name = "run_python"
    description = (
        "Run a Python 3 code snippet in the workspace root and return whatever it "
        "prints to stdout (and any error). Import what you need; print your results. "
        "The value of a final bare expression is printed too, like a notebook cell. "
        "A non-zero exit status is reported as '(exit code N)'. The snippet already "
        "runs in the workspace (no cd needed) and does NOT edit source files: a "
        "function redefined here lives only in the snippet. To change a file use "
        "edit_file (a few lines) or write_file (the whole file)."
    )
    input_schema = {
        "type": "object",
        "properties": {"code": {"type": "string", "description": "Python source to execute"}},
        "required": ["code"],
    }

    def __init__(self, workspace: Workspace, timeout: float = DEFAULT_TIMEOUT):
        self.ws = workspace
        self.timeout = timeout

    def run(self, arguments: dict) -> str:
        code = arguments["code"]
        source_file = self._stash_source(code)
        try:
            proc = run_with_timeout(
                # The venv's own python, so libs are available. `-c` runs the runner
                # program; its one argument is where the snippet waits (see below).
                [sys.executable, "-c", RUNNER_SOURCE, str(source_file)],
                cwd=self.ws.root,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            # Keep what the snippet printed before it hung (a stack of progress
            # lines, a partial traceback) — it is usually the clue the model needs.
            message = f"Error: code timed out after {self.timeout}s"
            partial = self._combine(*partial_output(exc)).strip()
            if partial:
                message += f"\n--- partial output ---\n{partial}"
            return message
        finally:
            # The runner deletes the file the moment it has read it. This covers the
            # runs that never got that far: the interpreter failed to start, or the
            # timeout killed the child first.
            source_file.unlink(missing_ok=True)

        result = self._combine(proc.stdout or "", proc.stderr or "").strip()
        if proc.returncode != 0:
            # A snippet that ends in `sys.exit(3)` or `os._exit(5)` is signalling
            # failure on purpose, and without this line the model would read whatever
            # it printed as success. run_bash prefixes the exit code on every run,
            # as a shell user expects; here only a failure is marked, so the happy
            # path stays a clean echo — `4`, not `(exit code 0)\n4`. A negative
            # number is Popen's way of saying "killed by that signal" (os.abort()
            # shows up as -6, SIGABRT). An uncaught exception exits 1, so a traceback
            # carries the prefix too — consistent, and harmless.
            result = f"(exit code {proc.returncode})\n{result}".rstrip()
        return result or "(no output)"

    def _stash_source(self, code: str) -> Path:
        """Write the snippet to a throwaway file in the system temp dir; return its path.

        Why a file: the snippet must not travel on stdin — a snippet may call
        input(), and its stdin has to stay whatever the agent's is — nor in argv,
        where the kernel caps a single argument at ~128 KB and `ps` shows it to
        everyone. An environment variable would have meant threading an `env`
        parameter through run_with_timeout, or editing os.environ process-wide, for
        one caller's benefit. A file is plain.

        Why the system temp dir, not the workspace: the workspace is the model's
        directory, not the tool's. It may well be read-only — a dataset folder, a
        checkout owned by someone else — and the plain `python -c` this replaces ran
        fine there, so run_python must too. It also means nothing of ours ever
        appears among the model's files, not even for a moment. The runner gets an
        absolute path, reads it, and deletes it before the snippet runs; cwd is set
        separately by Popen, so where the file lives has no bearing on what the
        snippet sees. tempfile creates it mode 0600, so only this user can read the
        model's code while it waits.
        """
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            prefix=".cornac-run-",
            suffix=".py",
            delete=False,
        ) as f:
            try:
                f.write(code)
            except BaseException:
                # The file exists from the moment NamedTemporaryFile returns, but the
                # cleanup in run() only starts once it has the path. A write that
                # fails here — a lone surrogate such as '\ud83d', which a model's JSON
                # can carry and UTF-8 cannot encode — must not leave an empty stash
                # behind. Remove it and let the error reach the model as before.
                Path(f.name).unlink(missing_ok=True)
                raise
        return Path(f.name)

    @staticmethod
    def _combine(out: str, err: str) -> str:
        """stdout first, then stderr under its own heading, trimmed head+tail."""
        result = out
        if err:
            result += "\n--- stderr ---\n" + err
        return truncate_head_tail(result, HEAD_CHARS, TAIL_CHARS)
