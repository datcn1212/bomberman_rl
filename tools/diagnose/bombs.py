"""Measure what the agent's bombs actually achieve.

The aggregate crate count hides the distribution: an agent that drops thirty
bombs hitting one crate each and an agent that drops ten hitting three each look
similar in "crates per round" but are very different policies. This replays a
trained model and records, for every bomb it chooses to drop, how many crates
the blast would cover and whether an escape route existed at that moment.

Run from the repository root:
    python3 tools/analyse_bombs.py experiments/p3_danger --seeds 1 2 3 4 5
"""

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tools.evaluate import EVAL_SEEDS  # noqa: E402

OUT = ROOT / "results" / "bombs"


def collect(exp_dir, seed, scenario, rounds):
    OUT.mkdir(parents=True, exist_ok=True)
    model = ROOT / exp_dir / ("seed%d" % seed) / "model.pkl"
    if not model.is_file():
        raise FileNotFoundError(model)
    cfg_path = OUT / ("cfg_seed%d.json" % seed)
    cfg_path.write_text(json.dumps(
        {"model_path": str(model), "continue_from": None, "log_path": None}))
    log_path = OUT / ("bombs_seed%d.csv" % seed)
    if log_path.exists():
        log_path.unlink()

    env = dict(os.environ, TQ_CONFIG=str(cfg_path), TQ_BOMBLOG=str(log_path))
    proc = subprocess.run(
        [sys.executable, "main.py", "play", "--no-gui", "--agents", "tabular_q",
         "--scenario", scenario, "--n-rounds", str(rounds),
         "--seed", str(EVAL_SEEDS[0]), "--train", "0"],
        cwd=ROOT, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-3000:])
    if not log_path.exists():
        return []
    rows = []
    with open(log_path) as fh:
        for line in fh:
            step, payload, escape = line.strip().split(",")
            rows.append((int(step), int(payload), int(escape)))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("exp_dir")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--scenario", default="loot-crate")
    p.add_argument("--rounds", type=int, default=10)
    args = p.parse_args()

    print("| seed | bombs | hit 0 crates | hit 1 | hit 2+ | mean payload | no escape |")
    print("|---|---|---|---|---|---|---|")
    totals = Counter()
    total_bombs = 0
    payload_sum = 0
    for seed in args.seeds:
        rows = collect(args.exp_dir, seed, args.scenario, args.rounds)
        n = len(rows)
        zero = sum(1 for _, pay, _ in rows if pay == 0)
        one = sum(1 for _, pay, _ in rows if pay == 1)
        many = sum(1 for _, pay, _ in rows if pay >= 2)
        trapped = sum(1 for _, _, esc in rows if esc == 0)
        mean_pay = (sum(pay for _, pay, _ in rows) / n) if n else float("nan")
        print("| %d | %d | %d (%.0f%%) | %d (%.0f%%) | %d (%.0f%%) | %.2f | %d (%.1f%%) |"
              % (seed, n, zero, 100 * zero / n if n else 0,
                 one, 100 * one / n if n else 0,
                 many, 100 * many / n if n else 0, mean_pay,
                 trapped, 100 * trapped / n if n else 0))
        totals["zero"] += zero
        totals["one"] += one
        totals["many"] += many
        totals["trapped"] += trapped
        total_bombs += n
        payload_sum += sum(pay for _, pay, _ in rows)

    if total_bombs:
        print("| **all** | %d | %d (%.0f%%) | %d (%.0f%%) | %d (%.0f%%) | %.2f | %d (%.1f%%) |"
              % (total_bombs, totals["zero"], 100 * totals["zero"] / total_bombs,
                 totals["one"], 100 * totals["one"] / total_bombs,
                 totals["many"], 100 * totals["many"] / total_bombs,
                 payload_sum / total_bombs,
                 totals["trapped"], 100 * totals["trapped"] / total_bombs))


if __name__ == "__main__":
    main()
