"""Turning a trained seed into a submission.

    python3 tools/ship.py choose    pick which seed to ship, on held-out arenas
    python3 tools/ship.py final     decisive measurement of the shipped model
    python3 tools/ship.py check     the assignment's hard constraints

Each subcommand is a script in tools/ship/ and runs standalone too; this just
gives tools/ one entry per job.
"""

import runpy
import sys
from pathlib import Path

SUBCOMMANDS = ["choose", "final", "check"]


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in SUBCOMMANDS:
        print(__doc__)
        return 1
    sub = sys.argv[1]
    sys.argv = ["tools/ship/%s.py" % sub] + sys.argv[2:]
    runpy.run_path(str(Path(__file__).resolve().parent / "ship" / (sub + ".py")),
                   run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
