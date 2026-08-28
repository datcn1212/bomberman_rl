"""Diagnostics: run one when a result is not what you expected.

Subcommands:

    policy     decode the table into a readable policy
    bombs      what the agent's bombs actually achieved
    symmetry   how much of the table is one situation seen twice
    curves     learning curves from the training logs

    python3 tools/diagnose.py <subcommand> [args...]

Named `diagnose` rather than `inspect` on purpose. A module called `inspect.py`
in a directory that lands on sys.path shadows the standard library's `inspect`,
which `dataclasses` imports, and every tool in this folder then fails with an
error that points nowhere near the cause.

Each subcommand lives in tools/diagnose/ and can also be run directly; this file
exists so that `tools/` shows one entry per job rather than one per script.
"""

import runpy
import sys
from pathlib import Path

SUBCOMMANDS = ["policy", "bombs", "symmetry", "curves"]


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in SUBCOMMANDS:
        print(__doc__)
        return 1
    sub = sys.argv[1]
    sys.argv = ["tools/diagnose/%s.py" % sub] + sys.argv[2:]
    runpy.run_path(str(Path(__file__).resolve().parent / "diagnose" / (sub + ".py")),
                   run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
