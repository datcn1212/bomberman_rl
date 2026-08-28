"""Baselines. Without these a score has no meaning.

Subcommands:

    baselines  the four agents the framework ships with
    winrate    those agents at our own sample size

    python3 tools/reference.py <subcommand> [args...]

Each subcommand lives in tools/reference/ and can also be run directly; this file
exists so that `tools/` shows one entry per job rather than one per script.
"""

import runpy
import sys
from pathlib import Path

SUBCOMMANDS = ['baselines', 'winrate']


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in SUBCOMMANDS:
        print(__doc__)
        return 1
    sub = sys.argv[1]
    sys.argv = ["tools/reference/%s.py" % sub] + sys.argv[2:]
    runpy.run_path(str(Path(__file__).resolve().parent / "reference" / (sub + ".py")),
                   run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
