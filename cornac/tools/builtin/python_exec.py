"""Python tool: run_python.

Runs a snippet of Python in a separate process, with the workspace root as the
working directory, and returns whatever the snippet printed (plus any error). Useful
for data-analysis tasks: load a CSV, compute a mean, filter rows.

Like run_bash, this executes arbitrary code and is gated by the Week 3 permission
system (defaults to "ask"). We run it as a subprocess (not exec() in-process) for two
reasons: a crash or infinite loop can't take down the agent, and a timeout can kill it.

This is NOT a security sandbox — a determined snippet can still read files the user can
read. Real isolation (containers, seccomp) is out of scope for v1; the honest controls
are the permission prompt, the subprocess timeout, and the workspace cwd.
"""

from __future__ import annotations

import subprocess
import sys

from cornac.tools.base import Tool
from cornac.tools.workspace import Workspace

DEFAULT_TIMEOUT = 30
MAX_OUTPUT = 50_000


class RunPython(Tool):
    name = "run_python"
    description = (
        "Run a Python 3 code snippet from the workspace root and return whatever it "
        "prints to stdout (and any error). Import what you need; print your results."
    )
    input_schema = {
        "type": "object",
        "properties": {"code": {"type": "string", "description": "Python source to execute"}},
        "required": ["code"],
    }

    def __init__(self, workspace: Workspace, timeout: int = DEFAULT_TIMEOUT):
        self.ws = workspace
        self.timeout = timeout

    def run(self, arguments: dict) -> str:
        code = arguments["code"]
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code],   # the venv's own python, so libs are available
                cwd=self.ws.root,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return f"Error: code timed out after {self.timeout}s"

        out = (proc.stdout or "")
        err = (proc.stderr or "")
        result = out
        if err:
            result += ("\n--- stderr ---\n" + err)
        if len(result) > MAX_OUTPUT:
            result = result[:MAX_OUTPUT] + "\n... [output truncated]"
        return result.strip() or "(no output)"
