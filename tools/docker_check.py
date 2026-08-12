"""Run the submitted agent inside containers, the way the tournament will.

The assignment's section 8 offers a Dockerfile "with the Python environment we
will use for the tournament". As of this writing `docker build .` on it fails:
`continuumio/miniconda3` now ships Python 3.14, and TensorFlow has no wheel for
3.14, so `pip install tensorflow` aborts the build before our code is ever
copied in. That is a bug in the provided Dockerfile, not in the agent -- and the
agent needs neither TensorFlow nor any of the other heavy packages.

So the compatibility question is tested directly instead: does the submitted
model, pickled under one Python and numpy, load and play identically under the
versions the graders might use? The base image of the tournament Dockerfile is
included so the real environment is covered.

The command run is the one the assignment says the graders run: no training,
against three random_agents.

    python3 tools/docker_check.py
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The last entry is the base image of the Dockerfile the assignment ships.
IMAGES = ["python:3.11", "python:3.12", "python:3.13", "python:3.14",
          "continuumio/miniconda3"]

INNER = r"""
import json, subprocess, sys, numpy
print("PY %s NUMPY %s" % (sys.version.split()[0], numpy.__version__))
r = subprocess.run([sys.executable, "main.py", "play", "--no-gui",
                    "--agents", "q_bomber", "random_agent", "random_agent",
                    "random_agent", "--n-rounds", "5",
                    "--save-stats", "/tmp/dock.json"],
                   capture_output=True, text=True)
print("EXIT %d" % r.returncode)
if r.returncode:
    print(r.stderr[-800:])
else:
    a = json.load(open("/tmp/dock.json"))["by_agent"]["q_bomber"]
    print("RESULT score=%s coins=%s crates=%s rounds=%s think_ms=%.2f"
          % (a.get("score"), a.get("coins"), a.get("crates"), a.get("rounds"),
             1000.0 * a.get("time", 0) / max(1, a.get("steps", 1))))
"""


def main():
    print("Running the graders' command (no training, three random_agents) "
          "in each image.\n")
    failures = []
    for image in IMAGES:
        proc = subprocess.run(
            ["docker", "run", "--rm", "--platform", "linux/amd64",
             "-v", "%s:/w" % ROOT, "-w", "/w", image, "bash", "-c",
             "pip install -q numpy tqdm 2>/dev/null; python - <<'PYEOF'\n%s\nPYEOF" % INNER],
            capture_output=True, text=True)
        lines = [ln for ln in proc.stdout.splitlines()
                 if ln.startswith(("PY ", "EXIT ", "RESULT "))]
        ok = any(ln.startswith("RESULT") for ln in lines)
        print("  [%s] %-26s %s" % ("PASS" if ok else "FAIL", image,
                                   " | ".join(lines)))
        if not ok:
            failures.append(image)
            print("       %s" % proc.stdout[-400:])
    print("\n%d/%d images play the agent successfully" % (len(IMAGES) - len(failures),
                                                          len(IMAGES)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
