"""Random hyperparameter search with successive halving.

Nine phases of feature and reward work moved the `classic` score less than one
hyperparameter that had been left at its first guess, so the parameters deserve
a search of their own rather than another round of hand-tuning.

Two stages, because a full-budget run of every candidate is not affordable:

  screen   every candidate on a few seeds and a shorter budget, to rank them
  confirm  the best few at the full protocol, where the ranking is trusted

The screen is deliberately noisy and is used only to order candidates. Nothing
is concluded from it - this log has already shown twice that a small sample
reverses (a wasted-bomb penalty at t = 2.6 on five seeds fell to t = 1.7 on ten).

    python3 tools/hpsearch.py --candidates 12 --screen-seeds 3 --confirm-top 3
"""

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# The configuration every arm starts from: everything this log has established.
BASE = {
    "gamma": 0.995,
    "use_symmetry": True,
    "use_opponent_blocking": True,
    "eps_decay_episodes": 2000,
}

# Values are sampled around the current setting, not around the previous
# branch's tuned values: Phase 13 showed the optimum moves when the surrounding
# configuration changes, so borrowing them would import the wrong context.
SPACE = {
    "alpha": [0.05, 0.1, 0.2, 0.3],
    "alpha_half_life": [500.0, 1000.0, 2000.0, 4000.0],
    "eps_end": [0.02, 0.05, 0.10],
    "eps_decay_episodes": [1000, 2000, 3000],
    "gamma": [0.99, 0.995, 0.999],
    "reward_crate": [0.15, 0.3, 0.6],
    "reward_coin_found": [0.0, 0.1, 0.3],
    "reward_step": [-0.003, -0.01, -0.03],
    "reward_wait": [-0.02, -0.05, -0.15],
}

OPPONENTS = ["rule_based_agent"] * 3
PHASE = "classic:%d@" + "+".join(OPPONENTS)


def sample(rng, n):
    """The default first, then n random draws, all distinct."""
    seen = {json.dumps({}, sort_keys=True)}
    out = [{}]
    while len(out) < n + 1:
        # Pick by index: rng.choice returns numpy scalars, which do not
        # survive json.dumps and would reach the agent as a different type.
        cand = {k: v[int(rng.integers(len(v)))] for k, v in SPACE.items()}
        cand = {k: v for k, v in cand.items()
                if BASE.get(k, object()) != v}
        key = json.dumps(cand, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
    return out


def run(exp_id, overrides, seeds, episodes, workers, eval_workers, note):
    cfg = dict(BASE)
    cfg.update(overrides)
    cmd = [sys.executable, "tools/train.py", "--exp-id", exp_id,
           "--phases", PHASE % episodes,
           "--seeds", *[str(s) for s in seeds],
           "--overrides", json.dumps(cfg),
           "--eval-scenario", "classic", "--eval-mode", "fast",
           "--eval-rounds", "20",
           "--eval-opponents", *OPPONENTS,
           "--workers", str(workers), "--eval-workers", str(eval_workers),
           "--note", note]
    log = Path("/tmp/%s.log" % exp_id)
    with open(log, "w") as fh:
        subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    return read_scores(log)


def read_scores(log):
    text = log.read_text()
    if "===" not in text:
        return None
    tail = text[text.rindex("==="):]
    m = re.search(r"  mean_score\s+[\-0-9.]+\s+\(per seed: (.*?), spread", tail)
    return [float(x) for x in m.group(1).split(", ")] if m else None


def main():
    import numpy as np

    p = argparse.ArgumentParser()
    p.add_argument("--candidates", type=int, default=12)
    p.add_argument("--screen-seeds", type=int, default=3)
    p.add_argument("--screen-episodes", type=int, default=4000)
    p.add_argument("--confirm-top", type=int, default=3)
    p.add_argument("--confirm-seeds", type=int, default=10)
    p.add_argument("--confirm-episodes", type=int, default=6000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260815)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    candidates = sample(rng, args.candidates)

    print("=== screen: %d candidates, %d seeds, %d episodes ===\n"
          % (len(candidates), args.screen_seeds, args.screen_episodes), flush=True)
    screened = []
    for i, cand in enumerate(candidates):
        exp_id = "hp_s%02d" % i
        scores = run(exp_id, cand, list(range(1, args.screen_seeds + 1)),
                     args.screen_episodes, args.workers, args.workers,
                     "hp screen %d: %s" % (i, json.dumps(cand, sort_keys=True)))
        mean = statistics.mean(scores) if scores else float("nan")
        screened.append((mean, i, cand))
        print("  %-9s %6.3f   %s" % (exp_id, mean,
                                     json.dumps(cand, sort_keys=True) or "(default)"),
              flush=True)

    screened.sort(reverse=True, key=lambda r: (r[0] if r[0] == r[0] else -1))
    print("\n=== confirm: top %d at %d seeds, %d episodes ===\n"
          % (args.confirm_top, args.confirm_seeds, args.confirm_episodes), flush=True)
    for mean, i, cand in screened[:args.confirm_top]:
        exp_id = "hp_c%02d" % i
        scores = run(exp_id, cand, list(range(1, args.confirm_seeds + 1)),
                     args.confirm_episodes, 10, 8,
                     "hp confirm %d: %s" % (i, json.dumps(cand, sort_keys=True)))
        if scores:
            se = statistics.stdev(scores) / len(scores) ** 0.5
            print("  %-9s %6.3f +/- %.3f   %s"
                  % (exp_id, statistics.mean(scores), se,
                     json.dumps(cand, sort_keys=True) or "(default)"), flush=True)


if __name__ == "__main__":
    main()
