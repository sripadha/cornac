"""Built-in tools + a helper to assemble the standard set.

`default_tools(workspace)` returns all eight built-ins wired to a workspace, ready to
drop into a ToolRegistry. File/shell/python tools are confined to the workspace root;
web tools are unconfined (they hit the network). The clock tool needs no workspace.
"""

from __future__ import annotations

from cornac.tools.base import Tool
from cornac.tools.builtin.clock import get_current_time
from cornac.tools.builtin.files import Grep, ListDir, ReadFile, WriteFile
from cornac.tools.builtin.python_exec import RunPython
from cornac.tools.builtin.shell import RunBash
from cornac.tools.builtin.web import web_fetch, web_search
from cornac.tools.workspace import Workspace


def default_tools(workspace: Workspace | str = ".") -> list[Tool]:
    """All eight built-in tools, confined to the given workspace where relevant."""
    ws = workspace if isinstance(workspace, Workspace) else Workspace(workspace)
    return [
        ReadFile(ws),
        WriteFile(ws),
        ListDir(ws),
        Grep(ws),
        RunBash(ws),
        RunPython(ws),
        web_search,
        web_fetch,
        get_current_time,
    ]


__all__ = [
    "default_tools",
    "get_current_time",
    "ReadFile",
    "WriteFile",
    "ListDir",
    "Grep",
    "RunBash",
    "RunPython",
    "web_search",
    "web_fetch",
]
