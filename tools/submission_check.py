"""Submission-readiness checks for agent_code/tabular_q.

Everything here is a rule the tournament enforces or a failure mode that has
already cost this project time. Run from the repository root:

    python3 tools/submission_check.py
"""

import ast
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

AGENT_DIR = ROOT / "agent_code" / "tabular_q"
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print("  [%s] %-52s %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)
    return ok


# --- 14.1 plays a tournament-shaped game without errors ---------------------

def check_plays_like_the_tournament():
    """No --train, three opponents, and crucially no --silence-errors: with it,
    an exception inside the agent is swallowed and the agent simply forfeits
    every step, which looks like a bad policy rather than a crash."""
    proc = subprocess.run(
        [sys.executable, "main.py", "play", "--no-gui",
         "--agents", "tabular_q", "random_agent", "random_agent", "random_agent",
         "--n-rounds", "5", "--save-stats", "results/submission_check.json"],
        cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        return check("plays 4-agent game with errors unsilenced", False,
                     proc.stderr.strip().splitlines()[-1][:90] if proc.stderr else "")
    stats = json.loads((ROOT / "results" / "submission_check.json").read_text())
    me = stats["by_agent"].get("tabular_q", {})
    return check("plays 4-agent game with errors unsilenced", True,
                 "score %s over %s rounds" % (me.get("score"), me.get("rounds")))


def check_plays_against_rule_based():
    proc = subprocess.run(
        [sys.executable, "main.py", "play", "--no-gui", "--my-agent", "tabular_q",
         "--n-rounds", "5", "--save-stats", "results/submission_check_rb.json"],
        cwd=ROOT, capture_output=True, text=True)
    return check("plays --my-agent (vs three rule_based_agents)",
                 proc.returncode == 0,
                 "" if proc.returncode == 0 else proc.stderr.strip()[-90:])


# --- 14.2 think time --------------------------------------------------------

def check_think_time():
    """The penalty for overrunning is cumulative, so the maximum matters, not
    the mean. Measured from the framework's own per-step timing."""
    stats = json.loads((ROOT / "results" / "submission_check_rb.json").read_text())
    me = stats["by_agent"]["tabular_q"]
    mean_ms = 1000.0 * me["time"] / max(1, me["steps"])
    proc = subprocess.run([sys.executable, "tools/benchmark_latency.py",
                           str(AGENT_DIR / "model.pkl")],
                          cwd=ROOT, capture_output=True, text=True)
    worst = re.search(r"full decision step\s+mean\s+([\d.]+) ms\s+p99\s+([\d.]+) ms\s+max\s+([\d.]+) ms",
                      proc.stdout)
    worst_ms = float(worst.group(3)) if worst else float("inf")
    return check("worst-case act() under the 0.5 s budget", worst_ms < 500.0,
                 "in-game mean %.2f ms, synthetic worst case %.2f ms" % (mean_ms, worst_ms))


# --- 14.3 the tournament's own settings -------------------------------------

def check_settings_untouched():
    proc = subprocess.run(["git", "diff", "--stat", "HEAD", "--",
                           "settings.py", "environment.py", "items.py",
                           "agents.py", "main.py"],
                          cwd=ROOT, capture_output=True, text=True)
    dirty = proc.stdout.strip()
    proc2 = subprocess.run(["git", "diff", "--stat", "d941dbd", "--",
                            "settings.py", "environment.py", "items.py",
                            "agents.py", "main.py"],
                           cwd=ROOT, capture_output=True, text=True)
    since_start = proc2.stdout.strip()
    return check("framework files unmodified since the first commit",
                 not dirty and not since_start,
                 since_start.replace("\n", "; ")[:80] if since_start else "")


# --- 14.4 rules about what the agent may contain -----------------------------

FORBIDDEN_MODULES = {"multiprocessing", "concurrent", "subprocess", "threading"}
FORBIDDEN_CALLS = ("Pool", "ProcessPoolExecutor", "ThreadPoolExecutor", "Thread")


def _imported_modules(tree):
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    return modules


def check_no_parallelism():
    """Parsed, not grepped: the module docstring of callbacks.py says the word
    'multiprocessing' precisely because it does not use it, and a text search
    flags that."""
    hits = []
    for path in sorted(AGENT_DIR.glob("*.py")):
        tree = ast.parse(path.read_text())
        for module in _imported_modules(tree) & FORBIDDEN_MODULES:
            hits.append("%s imports %s" % (path.name, module))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "attr", getattr(node.func, "id", ""))
                if name in FORBIDDEN_CALLS:
                    hits.append("%s calls %s" % (path.name, name))
    return check("no multiprocessing anywhere in the agent", not hits, ", ".join(hits))


def check_no_absolute_paths():
    pattern = re.compile(r"""["'](/(Users|home|mnt|tmp|var)/|[A-Z]:\\)""")
    hits = [path.name for path in sorted(AGENT_DIR.glob("*.py"))
            if pattern.search(path.read_text())]
    return check("no absolute paths in the agent", not hits, ", ".join(hits))


def check_no_network_or_writes_outside_agent():
    bad = ("urllib", "requests", "socket", "http.client")
    hits = [path.name for path in sorted(AGENT_DIR.glob("*.py"))
            if any(b in path.read_text() for b in bad)]
    return check("no network access in the agent", not hits, ", ".join(hits))


def check_imports_are_available():
    """Everything the agent imports must exist in the tournament image. numpy
    and settings are guaranteed; anything else has to be declared."""
    allowed_stdlib = {"json", "os", "pickle", "collections", "pathlib", "math",
                      "csv", "time", "sys", "itertools", "random", "copy"}
    external = set()
    for path in sorted(AGENT_DIR.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                external.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                external.add(node.module.split(".")[0])
    project = {"settings", "events", "agent_code"}
    third_party = external - allowed_stdlib - project
    known_present = {"numpy", "scipy", "sklearn"}
    undeclared = third_party - known_present
    requirements = AGENT_DIR / "requirements.txt"
    if undeclared and not requirements.exists():
        return check("third-party imports are available or declared", False,
                     "undeclared: %s" % sorted(undeclared))
    return check("third-party imports are available or declared", True,
                 "third-party: %s" % (sorted(third_party) or "none"))


# --- 14.5 the model loads without depending on the working directory ---------

def check_model_loads_the_way_the_framework_loads_it():
    """The framework chdir's into the agent folder before every event, so the
    agent uses a relative path. This reproduces exactly that: cwd = agent dir,
    no TQ_CONFIG set, load `model.pkl`."""
    model_path = AGENT_DIR / "model.pkl"
    if not model_path.exists():
        return check("model.pkl loads with cwd = agent dir", False, "missing")
    script = (
        "import sys; sys.path.insert(0, %r)\n"
        "from agent_code.tabular_q import model\n"
        "m = model.QModel.load('model.pkl')\n"
        "print('%%d rows, flags %%s' %% (len(m), m.feature_flags))\n" % str(ROOT))
    proc = subprocess.run([sys.executable, "-c", script], cwd=str(AGENT_DIR),
                          capture_output=True, text=True,
                          env={k: v for k, v in os.environ.items() if k != "TQ_CONFIG"})
    return check("model.pkl loads with cwd = agent dir", proc.returncode == 0,
                 proc.stdout.strip() or proc.stderr.strip().splitlines()[-1][:90])


def check_defaults_match_the_submitted_model():
    """The tournament sets no TQ_CONFIG, so the dataclass defaults are what the
    agent plays with. A model trained with different feature flags would be fed
    a state it has never seen -- and nothing would report it."""
    from agent_code.tabular_q import config, model
    cfg = config.load()
    trained = getattr(model.QModel.load(str(AGENT_DIR / "model.pkl")),
                      "feature_flags", None)
    if not trained:
        return check("defaults match the flags the model was trained with", False,
                     "model carries no feature_flags")
    mismatch = {k: (v, getattr(cfg, k, "<absent>"))
                for k, v in trained.items() if getattr(cfg, k, None) != v}
    return check("defaults match the flags the model was trained with",
                 not mismatch, str(mismatch)[:90])


def main():
    print("Submission readiness for agent_code/tabular_q\n")
    print("play:")
    check_plays_like_the_tournament()
    check_plays_against_rule_based()
    print("timing:")
    check_think_time()
    print("rules:")
    check_settings_untouched()
    check_no_parallelism()
    check_no_absolute_paths()
    check_no_network_or_writes_outside_agent()
    check_imports_are_available()
    print("model:")
    check_model_loads_from_anywhere()
    check_defaults_match_the_submitted_model()

    failed = [name for name, ok, _ in RESULTS if not ok]
    print("\n%d/%d checks pass" % (len(RESULTS) - len(failed), len(RESULTS)))
    if failed:
        print("FAILED: %s" % ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
