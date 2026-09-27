"""`python -m cornac "task"` — the module form of the command line.

Python runs this file when the package itself is named on the command line. It
does nothing but hand over to cornac.cli.main(), so that `python -m cornac` and the
installed `cornac` script (pyproject.toml, [project.scripts]) are the same program
with the same exit codes. Keeping it to one call also means there is exactly one
place — cli.py — where the command line is defined and tested.
"""

import sys

from cornac.cli import main

if __name__ == "__main__":
    sys.exit(main())
