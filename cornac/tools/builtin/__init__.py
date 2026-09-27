"""Built-in tools + a helper to assemble the standard set.

`default_tools(workspace)` returns all eleven built-ins wired to a workspace, ready to
drop into a ToolRegistry. File/shell/python tools are confined to the workspace root;
web tools are unconfined (they hit the network). The clock tool needs no workspace.
spawn_agent needs the *agent* instead, and gets it when an Agent is built with the
registry that holds it (see tools/builtin/spawn.py).
"""

from __future__ import annotations

from cornac.tools.base import Tool
from cornac.tools.builtin.clock import get_current_time
from cornac.tools.builtin.files import EditFile, Grep, ListDir, ReadFile, WriteFile
from cornac.tools.builtin.python_exec import RunPython
from cornac.tools.builtin.shell import RunBash
from cornac.tools.builtin.spawn import SpawnAgent  # sub-agents (Week 4b-B)
from cornac.tools.builtin.web import web_fetch, web_search
from cornac.tools.workspace import Workspace


def default_tools(workspace: Workspace | str = ".") -> list[Tool]:
    """All eleven built-in tools, confined to the given workspace where relevant."""
    ws = workspace if isinstance(workspace, Workspace) else Workspace(workspace)
    return [
        ReadFile(ws),
        WriteFile(ws),
        EditFile(ws),  # right after write_file: the model sees the two side by side
        ListDir(ws),
        Grep(ws),
        RunBash(ws),
        RunPython(ws),
        web_search,
        web_fetch,
        get_current_time,
        # --- sub-agents (Week 4b-B) ---
        # Last: it is bound to its agent by Agent.__init__, not to the workspace.
        SpawnAgent(),
        # --- end sub-agents ---
    ]


__all__ = [
    "default_tools",
    "get_current_time",
    "ReadFile",
    "WriteFile",
    "EditFile",
    "ListDir",
    "Grep",
    "RunBash",
    "RunPython",
    "web_search",
    "web_fetch",
    "SpawnAgent",  # sub-agents (Week 4b-B)
]
