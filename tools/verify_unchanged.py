"""Prove that a change leaves behaviour untouched, before spending hours on it.

Any change described as an optimisation, a refactor, or a new option that
defaults to off is a claim: that the agent trained afterwards is the same agent.
The claim is cheap to test and expensive to get wrong - a change that quietly
alters what the agent observes costs every experiment run after it, plus their
controls.

This trains a few hundred episodes twice on identical seeds, under two configs,
and compares the learned tables entry by entry. Identical tables mean identical
trajectories, identical updates and identical exploration draws.

    # a change that should alter nothing at all
    python3 tools/verify_unchanged.py

    # a new option that must be inert while it is off
    python3 tools/verify_unchanged.py --b '{"use_bomb_safety": false}'

    # a sanity check that the comparison can actually fail
    python3 tools/verify_unchanged.py --b '{"gamma": 0.9}'
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BASE = {
    "n_episodes": 300,
    "eps_decay_episodes": 200,
    "gamma": 0.995,
    "use_symmetry": True,
    "use_opponent_blocking": True,
}


def train(overrides, tag, seed, opponents, episodes):
    work = Path(tempfile.mkdtemp(prefix="verify_%s_" % tag))
    cfg = dict(BASE)
    cfg.update(overrides)
    cfg["n_episodes"] = episodes
    cfg["model_path"] = str(work / "model.pkl")
    cfg["log_path"] = str(work / "log.csv")
    cfg["continue_from"] = None
    cfg_path = work / "config.json"
    cfg_path.write_text(json.dumps(cfg))

    cmd = [sys.executable, "main.py", "play", "--no-gui",
           "--agents", "tabular_q", *opponents,
           "--train", "1", "--scenario", "classic",
           "--n-rounds", str(episodes), "--seed", str(seed)]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          env=dict(os.environ, TQ_CONFIG=str(cfg_path)))
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-2000:])
    return work / "model.pkl"


def compare(path_a, path_b):
    from agent_code.tabular_q.model import QModel
    a, b = QModel.load(str(path_a)), QModel.load(str(path_b))
    if set(a.q) != set(b.q):
        only_a, only_b = set(a.q) - set(b.q), set(b.q) - set(a.q)
        return False, ("different rows visited: %d only in A, %d only in B"
                       % (len(only_a), len(only_b)))
    worst, where = 0.0, None
    for index in a.q:
        diff = float(abs(a.q[index] - b.q[index]).max())
        if diff > worst:
            worst, where = diff, index
    if worst > 0:
        return False, "largest Q difference %.6g at row %s" % (worst, where)
    return True, "%d rows identical" % len(a.q)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", default="{}", help="JSON overrides for the first run")
    p.add_argument("--b", default="{}", help="JSON overrides for the second run")
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--seeds", type=int, nargs="+", default=[1001, 1002])
    p.add_argument("--solo", action="store_true",
                   help="train alone instead of against three rule_based agents")
    args = p.parse_args()

    opponents = [] if args.solo else ["rule_based_agent"] * 3
    ok_all = True
    for seed in args.seeds:
        a = train(json.loads(args.a), "a", seed, opponents, args.episodes)
        b = train(json.loads(args.b), "b", seed, opponents, args.episodes)
        ok, detail = compare(a, b)
        ok_all &= ok
        print("  seed %d: %s  (%s)" % (seed, "IDENTICAL" if ok else "DIFFERS", detail))

    print("\nVERDICT:", "unchanged" if ok_all else "BEHAVIOUR CHANGED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
