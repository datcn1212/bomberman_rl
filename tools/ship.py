"""The pipeline that turns a trained seed into a submission.

Subcommands:

    choose     pick which seed to ship, on held-out arenas
    final      the decisive measurement of the shipped model
    gate       EXACT mode, the only way to count win rate
    check      the assignment's hard constraints
    latency    worst-case decision time

    python3 tools/ship.py <subcommand> [args...]

Each subcommand lives in tools/ship/ and can also be run directly; this file
exists so that `tools/` shows one entry per job rather than one per script.
"""

import runpy
import sys
from pathlib import Path

SUBCOMMANDS = ["choose", "final", "gate", "check", "latency"]


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
