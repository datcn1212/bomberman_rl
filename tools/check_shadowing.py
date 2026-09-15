"""Fail if any module in `tools/` shadows something Python can already import.

Two collisions were introduced by grouping the tools into folders, and both were
expensive to diagnose because the traceback names neither file:

  * `tools/inspect.py` shadowed the standard library's `inspect`, which
    `dataclasses` imports, so *every* tool in the folder failed;
  * `tools/ship/select.py` shadowed the built-in `select`, which
    `concurrent.futures` needs, so the latency tool failed while claiming the
    agent had infinite worst-case think time.

A glob over the standard library's `*.py` files misses the second: `select` is a
C extension in lib-dynload. `importlib.util.find_spec` is the authoritative test
and is what this uses, run from outside the repository so that our own modules
are not on the path.

    python3 tools/check_shadowing.py
"""

import importlib.util
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

PROBE = """
import importlib.util, pathlib, sys, json
tools, root = sys.argv[1], sys.argv[2]
names = sorted({p.stem for p in pathlib.Path(tools).rglob("*.py")} - {"__init__"})
own = {p.stem for p in pathlib.Path(root).glob("*.py")}
bad = {}
for n in names:
    why = []
    try:
        if importlib.util.find_spec(n) is not None:
            why.append("shadows an importable module")
    except Exception:
        pass
    if n in own:
        why.append("shadows a framework module at the repository root")
    if why:
        bad[n] = why
print(json.dumps(bad))
"""


def main():
    # Run from /tmp so that `tools/` and the repository root are not on sys.path
    # and cannot mask the very collision being looked for.
    proc = subprocess.run([sys.executable, "-c", PROBE, str(ROOT / "tools"), str(ROOT)],
                          cwd="/tmp", capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stderr[-2000:])
        return 1
    import json
    bad = json.loads(proc.stdout)
    for name, why in sorted(bad.items()):
        print("  [SHADOW] %-16s %s" % (name, "; ".join(why)))
    if bad:
        print("\n%d module name(s) shadow something importable - rename them" % len(bad))
        return 1
    print("no module in tools/ shadows an importable name")
    return 0


if __name__ == "__main__":
    sys.exit(main())
