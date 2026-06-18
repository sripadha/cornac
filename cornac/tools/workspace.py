"""Workspace — the sandbox that confines file/shell tools to one directory.

Why this exists: an agent decides on its own which files to read, write, and which
shell commands to run. Letting it roam the entire filesystem is asking for trouble
(`read_file("/etc/shadow")`, `run_bash("rm -rf ~")`). A Workspace pins every tool to
a single root directory and rejects any path that tries to escape it.

This is *defense in depth*, not the whole defense. Week 3 adds a permission system
(ask/allow/deny per tool). The Workspace is the structural guardrail underneath that:
even an "allowed" tool can only act inside the sandbox.

Tools receive a Workspace and call `ws.resolve(path)` to turn a user/model-supplied
path into a real, validated absolute path — or raise if it points outside the root.
"""

from __future__ import annotations

from pathlib import Path


class WorkspaceError(Exception):
    """Raised when a path would escape the workspace root. The registry catches this
    and feeds it back to the model as an error result (so it can self-correct)."""


class Workspace:
    def __init__(self, root: str | Path = "."):
        # resolve() makes the root absolute and collapses any "..", giving us a
        # canonical path to compare against later.
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise WorkspaceError(f"workspace root is not a directory: {self.root}")

    def resolve(self, path: str | Path) -> Path:
        """Turn a (possibly relative) path into an absolute path *inside* the root.

        Raises WorkspaceError if the result would land outside the root — this is the
        line of defense against `../../` traversal and absolute paths like /etc/passwd.
        """
        # A path from the model may be relative (to the root) or absolute. Join it to
        # the root, then fully resolve it (collapsing "..", following the tree).
        candidate = (self.root / path).resolve()

        # The candidate is safe iff root is one of its parents (or it *is* root).
        if candidate != self.root and self.root not in candidate.parents:
            raise WorkspaceError(
                f"path {str(path)!r} escapes the workspace root ({self.root})"
            )
        return candidate

    def relative(self, path: Path) -> str:
        """Render an absolute path back as workspace-relative, for tidy tool output."""
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)
