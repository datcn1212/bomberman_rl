"""Evaluate models that are already trained, without retraining them.

Useful when an evaluation pass is interrupted, and for measuring an existing
model under a different scenario or against different opponents than the run it
came from. Seeds already present in experiments/registry.csv are skipped unless
--force is given.

    python3 tools/eval_models.py --exp s3c_n5 --scenario classic
    python3 tools/eval_models.py --exp s5_final --scenario classic \\
        --opponents rule_based_agent --mode exact
"""

import argparse
import csv
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.evaluate import (EVAL_SEEDS, REGISTRY, evaluate,  # noqa: E402
                            format_metrics, register)

EXPERIMENTS = ROOT / "experiments"


def already_done(exp_id):
    if not REGISTRY.exists():
        return set()
    with open(REGISTRY) as fh:
        return {row["exp_id"] for row in csv.DictReader(fh)
                if row["exp_id"].startswith(exp_id)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True)
    parser.add_argument("--scenario", default="classic")
    parser.add_argument("--opponents", nargs="*", default=[])
    parser.add_argument("--mode", default="fast", choices=["fast", "exact"])
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--suffix", default="",
                        help="appended to the exp_id written to the registry, so "
                             "the same models can be logged under several setups")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    done = already_done(args.exp)
    results = []
    for seed_dir in sorted((EXPERIMENTS / args.exp).glob("seed*")):
        config_path = seed_dir / "eval_config.json"
        model_path = seed_dir / "model.pkl"
        if not model_path.exists():
            print("skip %s: no model" % seed_dir.name)
            continue
        if not config_path.exists():
            config_path.write_text('{"model_path": "%s"}' % model_path.resolve())
        exp_id = "%s_%s%s" % (args.exp, seed_dir.name, args.suffix)
        if exp_id in done and not args.force:
            print("skip %s: already in registry" % exp_id)
            continue
        metrics = evaluate("q_bomber", args.opponents, args.scenario, mode=args.mode,
                           seeds=EVAL_SEEDS, n_rounds=(1 if args.mode == "exact"
                                                       else args.rounds),
                           tag=exp_id, extra_env={"QB_CONFIG": str(config_path)},
                           workers=args.workers)
        print(format_metrics(metrics), flush=True)
        register(metrics, exp_id=exp_id, note=args.note)
        results.append(metrics)

    if len(results) > 1:
        print("\n=== %s%s over %d seeds ===" % (args.exp, args.suffix, len(results)))
        for key in ("mean_score", "mean_coins", "mean_kills", "suicide_rate",
                    "mean_bombs", "mean_crates", "win_rate", "survival_rate"):
            values = [m[key] for m in results]
            print("  %-14s %.3f +/- %.3f" % (key, st.mean(values), st.stdev(values)))


if __name__ == "__main__":
    main()
