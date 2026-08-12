"""Random hyperparameter search with a hard budget.

Random rather than grid: with eight parameters a grid is unaffordable, and
random search covers each individual axis better than a grid of the same size
when only some axes matter. Every config that is tried is written to
experiments/registry.csv, including the bad ones -- those are the evidence that
the chosen value is a peak and not just the first thing that worked.

Each candidate is trained on a reduced budget and evaluated in FAST mode; the
winner is meant to be retrained at full budget afterwards.

    python3 tools/sweep.py --exp-id s7 --configs 20 --seeds 2
"""

import argparse
import json
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

# The base configuration every candidate starts from: the design S2-S5 arrived
# at. Only the swept keys below are allowed to differ.
BASE = {
    "alpha_schedule": "visit",
    "reward_moved_into_danger": 0.0,
    "reward_stayed_in_danger": 0.0,
    "reward_escaped_danger": 0.0,
    "state_backoff": True,
    # Ablation A6 measured that the opponent dimensions *cost* 33% of the Task 4
    # score while leaving kills unchanged, so the search starts from the
    # opponent-blind variant. See report.md 6.2.
    "use_opponent_features": False,
}

# Axes and their candidate values. Ranges are centred on the current value and
# widened in the direction the S5 diagnosis points: the Task 4 gap is entirely
# a suicide rate, so the death penalty, the trapped penalty and the step size
# all get room to move.
SPACE = {
    "alpha": [0.1, 0.2, 0.3],
    "alpha_half_life": [1000.0, 2000.0, 5000.0],
    "gamma": [0.9, 0.95, 0.99],
    "eps_end": [0.02, 0.05, 0.10],
    "reward_killed_self": [-5.0, -8.0, -12.0],
    "reward_trapped": [-0.5, -1.0, -2.0],
    "reward_bomb_wasted": [-0.1, -0.2, -0.4],
    "reward_crate": [0.2, 0.3, 0.5],
}


def sample(rng):
    return {key: values[int(rng.integers(len(values)))] for key, values in SPACE.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", default="s7")
    parser.add_argument("--configs", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=2000,
                        help="per curriculum phase; reduced for the search")
    parser.add_argument("--eval-seeds", type=int, default=15)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    phases = ["loot-crate:%d" % args.episodes, "classic:%d" % args.episodes,
              "classic:%d@rule_based_agent" % args.episodes]

    seen, results = set(), []
    for index in range(args.configs):
        for _ in range(50):
            candidate = sample(rng)
            key = tuple(sorted(candidate.items()))
            if key not in seen:
                seen.add(key)
                break
        overrides = dict(BASE, **candidate)
        exp_id = "%s_cfg%02d" % (args.exp_id, index)
        print("\n=== %s %s ===" % (exp_id, json.dumps(candidate, sort_keys=True)), flush=True)
        (ROOT / "experiments" / exp_id).mkdir(parents=True, exist_ok=True)
        (ROOT / "experiments" / exp_id / "candidate.json").write_text(
            json.dumps(candidate, indent=2, sort_keys=True))

        started = time.time()
        proc = subprocess.run(
            [sys.executable, "tools/train.py", "--exp-id", exp_id,
             "--phases", *phases,
             "--seeds", *[str(s + 1) for s in range(args.seeds)],
             "--workers", str(args.workers),
             "--overrides", json.dumps(overrides),
             "--phase-overrides", json.dumps([{}, {"eps_start": 0.25},
                                              {"eps_start": 0.25}]),
             "--eval-scenario", "classic",
             "--eval-opponents", "rule_based_agent",
             "--eval-mode", "fast", "--eval-rounds", "15",
             "--eval-seeds", str(args.eval_seeds),
             "--note", "S7 random search candidate"],
            cwd=ROOT, capture_output=True, text=True)
        if proc.returncode != 0:
            print("  FAILED:\n%s" % proc.stderr[-1500:], flush=True)
            continue

        scores = _read_scores(exp_id, args.seeds)
        if not scores:
            continue
        mean_score = st.mean(s for s, _ in scores)
        mean_suicide = st.mean(x for _, x in scores)
        results.append((mean_score, mean_suicide, exp_id, candidate))
        print("  score %.3f  suicide %.3f  (%.0fs)"
              % (mean_score, mean_suicide, time.time() - started), flush=True)

    print("\n\n=== %s ranked by mean score ===" % args.exp_id)
    for score, suicide, exp_id, candidate in sorted(results, reverse=True):
        print("  %6.3f  suicide %.3f  %-12s %s"
              % (score, suicide, exp_id, json.dumps(candidate, sort_keys=True)))


def _read_scores(exp_id, n_seeds):
    import csv
    registry = ROOT / "experiments" / "registry.csv"
    with open(registry) as fh:
        rows = [r for r in csv.DictReader(fh) if r["exp_id"].startswith(exp_id + "_seed")]
    return [(float(r["mean_score"]), float(r["suicide_rate"])) for r in rows[-n_seeds:]]


if __name__ == "__main__":
    main()
