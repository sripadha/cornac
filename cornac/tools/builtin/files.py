"""File tools: read_file, write_file, list_dir, grep.

These are class-based tools (subclassing Tool directly) rather than @tool functions,
because each one needs a Workspace to confine it to the sandbox root. The constructor
takes the Workspace; run() validates every path through ws.resolve() before touching
disk, so a path that escapes the root becomes a clean error result (Layer 2) the model
can recover from — never an actual read/write outside the sandbox.
"""

from __future__ import annotations

import re

from cornac.tools.base import Tool
from cornac.tools.workspace import Workspace

MAX_BYTES = 100_000  # cap on how much a single read/grep returns, to protect context


class ReadFile(Tool):
    name = "read_file"
    description = (
        "Read a UTF-8 text file from the workspace and return its contents. "
        "Paths are relative to the workspace root."
    )
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "file path, relative to workspace root"}},
        "required": ["path"],
    }

    def __init__(self, workspace: Workspace):
        self.ws = workspace

    def run(self, arguments: dict) -> str:
        path = self.ws.resolve(arguments["path"])  # raises if outside sandbox
        if not path.is_file():
            return f"Error: not a file: {self.ws.relative(path)}"
        data = path.read_text(encoding="utf-8", errors="replace")
        if len(data) > MAX_BYTES:
            return data[:MAX_BYTES] + f"\n... [truncated at {MAX_BYTES} chars]"
        return data


class WriteFile(Tool):
    name = "write_file"
    description = (
        "Write text to a file in the workspace, creating parent directories as needed. "
        "Overwrites if the file exists. Paths are relative to the workspace root."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "file path, relative to workspace root"},
            "content": {"type": "string", "description": "the text to write"},
        },
        "required": ["path", "content"],
    }

    def __init__(self, workspace: Workspace):
        self.ws = workspace

    def run(self, arguments: dict) -> str:
        path = self.ws.resolve(arguments["path"])  # raises if outside sandbox
        path.parent.mkdir(parents=True, exist_ok=True)
        content = arguments["content"]
        path.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} chars to {self.ws.relative(path)}"


class ListDir(Tool):
    name = "list_dir"
    description = (
        "List the entries of a directory in the workspace (directories shown with a "
        "trailing slash). Defaults to the workspace root. Paths are relative to root."
    )
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "directory path; defaults to '.'"}},
    }

    def __init__(self, workspace: Workspace):
        self.ws = workspace

    def run(self, arguments: dict) -> str:
        path = self.ws.resolve(arguments.get("path", "."))  # raises if outside sandbox
        if not path.is_dir():
            return f"Error: not a directory: {self.ws.relative(path)}"
        entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
        if not entries:
            return f"(empty directory: {self.ws.relative(path)})"
        return "\n".join(e.name + ("/" if e.is_dir() else "") for e in entries)


class Grep(Tool):
    name = "grep"
    description = (
        "Search for a regular-expression pattern across text files in the workspace "
        "(recursively from the given directory). Returns matching lines as "
        "path:line_number:line. Use this to find where something is defined or used."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "a Python regular expression"},
            "path": {"type": "string", "description": "directory to search from; defaults to '.'"},
        },
        "required": ["pattern"],
    }

    def __init__(self, workspace: Workspace):
        self.ws = workspace

    def run(self, arguments: dict) -> str:
        root = self.ws.resolve(arguments.get("path", "."))  # raises if outside sandbox
        regex = re.compile(arguments["pattern"])
        hits: list[str] = []
        for file in sorted(root.rglob("*")):
            if not file.is_file():
                continue
            try:
                text = file.read_text(encoding="utf-8", errors="strict")
            except (UnicodeDecodeError, OSError):
                continue  # skip binary / unreadable files
            for lineno, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    hits.append(f"{self.ws.relative(file)}:{lineno}:{line.strip()}")
                    if len("\n".join(hits)) > MAX_BYTES:
                        hits.append("... [truncated: too many matches]")
                        return "\n".join(hits)
        return "\n".join(hits) if hits else f"(no matches for {arguments['pattern']!r})"
