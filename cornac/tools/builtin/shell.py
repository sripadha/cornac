"""Shell tool: run_bash.

Runs a shell command with the workspace root as its working directory. This is the
most powerful — and most dangerous — built-in: a command can do anything the user can.
Two structural guards live here:

  - cwd is pinned to the workspace root (commands start inside the sandbox)
  - a timeout prevents a hung command from freezing the agent — and kills the whole
    process group, not just the shell, so "timed out" means "stopped" (see
    cornac/tools/_subprocess.py for why that distinction matters)

The *real* safety gate for run_bash is the Week 3 permission system (this tool will
default to "ask"). Until then, treat it as trusted-local-use only. We deliberately do
NOT try to sanitize the command string here — partial blocklists give false confidence;
the honest control is permission prompts plus the workspace cwd.
"""

from __future__ import annotations

import subprocess

from cornac.tools._subprocess import partial_output, run_with_timeout
from cornac.tools._truncate import truncate_head_tail
from cornac.tools.base import Tool
from cornac.tools.workspace import Workspace

DEFAULT_TIMEOUT = 30  # seconds

# Long output is cut to head+tail rather than just head: tracebacks and test summaries
# live at the END of a command's output, and the model needs to see them. See
# cornac/tools/_truncate.py for the reasoning.
HEAD_CHARS = 40_000   # chars kept from the start of combined stdout+stderr
TAIL_CHARS = 10_000   # chars kept from the end


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

    def __init__(self, workspace: Workspace, timeout: float = DEFAULT_TIMEOUT):
        self.ws = workspace
        self.timeout = timeout

    def run(self, arguments: dict) -> str:
        command = arguments["command"]
        try:
            proc = run_with_timeout(
                command,
                shell=True,
                cwd=self.ws.root,           # pin to the sandbox directory
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            # What the command printed before it hung is often the whole diagnosis
            # (the test that stalled, the prompt it was waiting on), so it is shown
            # too — head+tail-trimmed like any other output.
            message = f"Error: command timed out after {self.timeout}s"
            out, err = partial_output(exc)
            partial = truncate_head_tail(out + err, HEAD_CHARS, TAIL_CHARS).rstrip()
            if partial:
                message += f"\n--- partial output ---\n{partial}"
            return message

        out = (proc.stdout or "") + (proc.stderr or "")
        out = truncate_head_tail(out, HEAD_CHARS, TAIL_CHARS)
        return f"(exit code {proc.returncode})\n{out}".rstrip()
