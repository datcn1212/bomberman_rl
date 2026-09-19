"""Submission-readiness checks for one agent.

Each check is either a tournament rule or a failure mode that already cost
time once. Timing calls the real act(), not a rebuild of what it does, so
masking and everything else counts.

    python3 tools/ship.py check
    python3 tools/ship.py check --agent linear_q
"""

import argparse
import ast
import importlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import settings as s  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print("  [%s] %-52s %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)
    return ok


# --- plays a tournament-shaped game -----------------------------------------

def check_plays_like_the_tournament(agent):
    """No --train, three opponents, and no --silence-errors: with that flag an
    exception inside the agent is swallowed and it just forfeits every step,
    which reads as a bad policy instead of a crash."""
    proc = subprocess.run(
        [sys.executable, "main.py", "play", "--no-gui",
         "--agents", agent, "random_agent", "random_agent", "random_agent",
         "--n-rounds", "5", "--save-stats", "results/submission_check.json"],
        cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        return check("plays 4-agent game with errors unsilenced", False,
                     proc.stderr.strip().splitlines()[-1][:90] if proc.stderr else "")
    stats = json.loads((ROOT / "results" / "submission_check.json").read_text())
    me = stats["by_agent"].get(agent, {})
    return check("plays 4-agent game with errors unsilenced", True,
                 "score %s over %s rounds" % (me.get("score"), me.get("rounds")))


def check_plays_against_rule_based(agent):
    proc = subprocess.run(
        [sys.executable, "main.py", "play", "--no-gui", "--my-agent", agent,
         "--n-rounds", "5", "--save-stats", "results/submission_check_rb.json"],
        cwd=ROOT, capture_output=True, text=True)
    return check("plays --my-agent (vs three rule_based_agents)",
                 proc.returncode == 0,
                 "" if proc.returncode == 0 else proc.stderr.strip()[-90:])


# --- timing ------------------------------------------------------------------

def worst_case_states(n=400, seed=7):
    """Boards that hit every expensive path: dense crates so the target BFS runs
    long, bombs ticking so the danger schedule and time-expanded escape search
    are both active, opponents so their tiles are collected too."""
    rng = np.random.default_rng(seed)
    states = []
    for _ in range(n):
        field = np.zeros((s.COLS, s.ROWS), dtype=int)
        field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = -1
        for x in range(s.COLS):
            for y in range(s.ROWS):
                if (x + 1) * (y + 1) % 2 == 1:
                    field[x, y] = -1

        free = [(x, y) for x in range(1, s.COLS - 1) for y in range(1, s.ROWS - 1)
                if field[x, y] == 0]
        for (x, y) in free:
            if rng.random() < s.SCENARIOS["classic"]["CRATE_DENSITY"]:
                field[x, y] = 1

        # clear a pocket so the agent has somewhere to stand and search from
        pos = free[int(rng.integers(len(free)))]
        for (dx, dy) in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (2, 0), (0, 2)):
            x, y = pos[0] + dx, pos[1] + dy
            if 0 < x < s.COLS - 1 and 0 < y < s.ROWS - 1 and field[x, y] == 1:
                field[x, y] = 0

        open_tiles = [(x, y) for (x, y) in free if field[x, y] == 0 and (x, y) != pos]
        rng.shuffle(open_tiles)
        bombs = [(open_tiles[i], int(rng.integers(0, s.BOMB_TIMER)))
                 for i in range(min(4, len(open_tiles)))]
        # deliberately stand inside a blast: that is what triggers the escape
        # search, the most expensive branch in features.py
        bombs.append((pos, int(rng.integers(0, s.BOMB_TIMER))))

        explosion_map = np.zeros((s.COLS, s.ROWS))
        for (x, y) in open_tiles[7:12]:
            explosion_map[x, y] = 1

        states.append({
            "round": 1, "step": 100, "field": field,
            "self": ("me", 0, True, pos),
            "others": [("o%d" % i, 0, True, xy)
                       for i, xy in enumerate(open_tiles[4:7])],
            "bombs": bombs,
            "coins": list(open_tiles[12:15]),
            "explosion_map": explosion_map, "user_input": None,
        })
    return states


class _QuietLogger:
    def info(self, *a, **k):
        pass

    def debug(self, *a, **k):
        pass


def measure_act(agent):
    """Milliseconds per act() call on worst-case boards. Returns (mean, max)."""
    agent_dir = ROOT / "agent_code" / agent
    callbacks = importlib.import_module("agent_code.%s.callbacks" % agent)

    cfg_path = agent_dir / "latency_config.json"
    cfg_path.write_text(json.dumps({"model_path": str(agent_dir / "model.pkl"),
                                    "continue_from": None, "log_path": None}))
    env_var = "TQ_CONFIG" if agent == "tabular_q" else "LQ_CONFIG"
    os.environ[env_var] = str(cfg_path)

    holder = type("Agent", (), {})()
    holder.train = False
    holder.logger = _QuietLogger()
    callbacks.setup(holder)

    states = worst_case_states()
    for st in states[:20]:            # warm up; first touch is not representative
        callbacks.act(holder, st)

    times = []
    for st in states:
        t0 = time.perf_counter()
        callbacks.act(holder, st)
        times.append((time.perf_counter() - t0) * 1000.0)

    cfg_path.unlink(missing_ok=True)
    del os.environ[env_var]
    return statistics.mean(times), max(times)


def check_think_time(agent):
    """The overrun penalty is cumulative, borrowing from the next step, so
    the maximum matters, not the mean."""
    mean_ms, worst_ms = measure_act(agent)
    budget_ms = s.TIMEOUT * 1000
    return check("worst-case act() under the %.1f s budget" % s.TIMEOUT,
                 worst_ms < budget_ms,
                 "mean %.3f ms, worst %.3f ms (%.2f%% of budget)"
                 % (mean_ms, worst_ms, 100 * worst_ms / budget_ms))


# --- the tournament's own files ----------------------------------------------

def check_settings_untouched():
    framework = ["settings.py", "environment.py", "items.py", "agents.py", "main.py"]
    dirty = subprocess.run(["git", "diff", "--stat", "HEAD", "--", *framework],
                           cwd=ROOT, capture_output=True, text=True).stdout.strip()
    since_start = subprocess.run(["git", "diff", "--stat", "d941dbd", "--", *framework],
                                 cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return check("framework files unmodified since the first commit",
                 not dirty and not since_start,
                 since_start.replace("\n", "; ")[:80] if since_start else "")


# --- what the agent is allowed to contain ------------------------------------

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


def check_no_parallelism(agent_dir):
    """Parsed, not grepped: a docstring can mention multiprocessing precisely
    because the file does not use it, and a text search flags that."""
    hits = []
    for path in sorted(agent_dir.glob("*.py")):
        tree = ast.parse(path.read_text())
        for module in _imported_modules(tree) & FORBIDDEN_MODULES:
            hits.append("%s imports %s" % (path.name, module))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "attr", getattr(node.func, "id", ""))
                if name in FORBIDDEN_CALLS:
                    hits.append("%s calls %s" % (path.name, name))
    return check("no multiprocessing anywhere in the agent", not hits, ", ".join(hits))


def check_no_absolute_paths(agent_dir):
    pattern = re.compile(r"""["'](/(Users|home|mnt|tmp|var)/|[A-Z]:\\)""")
    hits = [path.name for path in sorted(agent_dir.glob("*.py"))
            if pattern.search(path.read_text())]
    return check("no absolute paths in the agent", not hits, ", ".join(hits))


def check_no_network(agent_dir):
    bad = ("urllib", "requests", "socket", "http.client")
    hits = [path.name for path in sorted(agent_dir.glob("*.py"))
            if any(b in path.read_text() for b in bad)]
    return check("no network access in the agent", not hits, ", ".join(hits))


def check_imports_are_available(agent_dir):
    """Everything the agent imports has to exist in the tournament image. numpy
    and the framework modules are guaranteed; anything else must be declared."""
    allowed_stdlib = {"json", "os", "pickle", "collections", "pathlib", "math",
                      "csv", "time", "sys", "itertools", "random", "copy",
                      "dataclasses", "functools", "heapq"}
    external = set()
    for path in sorted(agent_dir.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                external.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                external.add(node.module.split(".")[0])

    third_party = external - allowed_stdlib - {"settings", "events", "agent_code"}
    undeclared = third_party - {"numpy", "scipy", "sklearn"}
    if undeclared and not (agent_dir / "requirements.txt").exists():
        return check("third-party imports are available or declared", False,
                     "undeclared: %s" % sorted(undeclared))
    return check("third-party imports are available or declared", True,
                 "third-party: %s" % (sorted(third_party) or "none"))


# --- the model ---------------------------------------------------------------

def check_model_loads_like_the_framework_does(agent, agent_dir):
    """The framework chdir's into the agent folder before every event, so the
    agent uses a relative path. Reproduce exactly that: cwd = agent dir, no
    config env var set, load model.pkl."""
    if not (agent_dir / "model.pkl").exists():
        return check("model.pkl loads with cwd = agent dir", False, "missing")

    cls = "QModel" if agent == "tabular_q" else "LinearQModel"
    script = (
        "import sys; sys.path.insert(0, %r)\n"
        "from agent_code.%s import model\n"
        "m = model.%s.load('model.pkl')\n"
        "print('loaded, flags %%s' %% (m.feature_flags,))\n" % (str(ROOT), agent, cls))
    env = {k: v for k, v in os.environ.items()
           if k not in ("TQ_CONFIG", "LQ_CONFIG")}
    proc = subprocess.run([sys.executable, "-c", script], cwd=str(agent_dir),
                          capture_output=True, text=True, env=env)
    return check("model.pkl loads with cwd = agent dir", proc.returncode == 0,
                 proc.stdout.strip() or proc.stderr.strip().splitlines()[-1][:90])


def check_defaults_match_the_model(agent, agent_dir):
    """The tournament sets no config file, so the dataclass defaults are what
    the agent plays with. A model trained under different feature flags would be
    fed a state it has never seen, and nothing would report it."""
    config = importlib.import_module("agent_code.%s.config" % agent)
    model = importlib.import_module("agent_code.%s.model" % agent)
    cls = model.QModel if agent == "tabular_q" else model.LinearQModel

    cfg = config.Config()          # defaults, not config.load()
    trained = getattr(cls.load(str(agent_dir / "model.pkl")), "feature_flags", None)
    if not trained:
        return check("defaults match the flags the model was trained with", False,
                     "model carries no feature_flags")
    mismatch = {k: (v, getattr(cfg, k, "<absent>"))
                for k, v in trained.items() if getattr(cfg, k, None) != v}
    return check("defaults match the flags the model was trained with",
                 not mismatch, str(mismatch)[:90])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default="tabular_q")
    args = p.parse_args()
    agent_dir = ROOT / "agent_code" / args.agent

    print("Submission readiness for agent_code/%s\n" % args.agent)
    print("play:")
    check_plays_like_the_tournament(args.agent)
    check_plays_against_rule_based(args.agent)
    print("timing:")
    check_think_time(args.agent)
    print("rules:")
    check_settings_untouched()
    check_no_parallelism(agent_dir)
    check_no_absolute_paths(agent_dir)
    check_no_network(agent_dir)
    check_imports_are_available(agent_dir)
    print("model:")
    check_model_loads_like_the_framework_does(args.agent, agent_dir)
    check_defaults_match_the_model(args.agent, agent_dir)

    failed = [name for name, ok, _ in RESULTS if not ok]
    print("\n%d/%d checks pass" % (len(RESULTS) - len(failed), len(RESULTS)))
    if failed:
        print("FAILED: %s" % ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
