"""Shell tool: run_bash.

Runs a shell command with the workspace root as its working directory. This is the
most powerful — and most dangerous — built-in: a command can do anything the user can.
Two structural guards live here:

  - cwd is pinned to the workspace root (commands start inside the sandbox)
  - a timeout prevents a hung command from freezing the agent

The *real* safety gate for run_bash is the Week 3 permission system (this tool will
default to "ask"). Until then, treat it as trusted-local-use only. We deliberately do
NOT try to sanitize the command string here — partial blocklists give false confidence;
the honest control is permission prompts plus the workspace cwd.
"""

from __future__ import annotations

import subprocess

from cornac.tools.base import Tool
from cornac.tools.workspace import Workspace

DEFAULT_TIMEOUT = 30  # seconds
MAX_OUTPUT = 50_000   # chars of combined stdout+stderr to return


class RunBash(Tool):
    name = "run_bash"
    description = (
        "Run a bash command from the workspace root and return its combined "
        "stdout/stderr and exit code. Use for running tests, git, build tools, etc."
    )
    input_schema = {
        "type": "object",
        "properties": {"command": {"type": "string", "description": "the bash command to run"}},
        "required": ["command"],
    }

    def __init__(self, workspace: Workspace, timeout: int = DEFAULT_TIMEOUT):
        self.ws = workspace
        self.timeout = timeout

    def run(self, arguments: dict) -> str:
        command = arguments["command"]
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=self.ws.root,           # pin to the sandbox directory
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {self.timeout}s"

        out = (proc.stdout or "") + (proc.stderr or "")
        if len(out) > MAX_OUTPUT:
            out = out[:MAX_OUTPUT] + "\n... [output truncated]"
        return f"(exit code {proc.returncode})\n{out}".rstrip()
