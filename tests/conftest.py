"""Fixtures shared by more than one test module.

`spike` loads benchmark/spike/run_spike.py as a module. The runner is a script that
lives outside the package (its main() sits behind an __name__ guard), so it cannot be
imported by name; importlib loads it from its path. The workspace-hint tests and the
runner's own tests both need it, hence one fixture here rather than a copy in each.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_SPIKE = REPO_ROOT / "benchmark" / "spike" / "run_spike.py"


@pytest.fixture(scope="module")
def spike():
    """The runner as a module.

    Registered in sys.modules before it runs: the runner's @dataclass resolves its
    `from __future__ import annotations` strings through sys.modules[__module__].
    """
    name = "run_spike_under_test"
    spec = importlib.util.spec_from_file_location(name, RUN_SPIKE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        yield module
    finally:
        sys.modules.pop(name, None)
