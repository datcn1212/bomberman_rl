"""Evaluate models that are already on disk.

Training and evaluation are separate costs, and only evaluation is cheap. When a
run dies in its evaluation phase the trained models are still there, so this
re-runs just the measurement instead of the whole experiment.

    python3 tools/eval_existing.py --exp-id p13_no_sym \\
        --overrides '{"gamma": 0.999, "exploration": "max_boltzmann"}'
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.evaluate import (EVAL_SEEDS, evaluate, format_metrics,  # noqa: E402
                            register)

EXPERIMENTS = ROOT / "experiments"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-id", required=True)
    p.add_argument("--agent", default="tabular_q")
    p.add_argument("--overrides", default="{}")
    p.add_argument("--scenario", default="classic")
    p.add_argument("--opponents", nargs="*", default=[])
    p.add_argument("--mode", default="fast", choices=["fast", "exact"])
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--note", default="re-evaluated from stored models")
    args = p.parse_args()

    overrides = json.loads(args.overrides)
    results = []
    for seed in args.seeds:
        seed_dir = EXPERIMENTS / args.exp_id / ("seed%d" % seed)
        model_path = seed_dir / "model.pkl"
        if not model_path.is_file():
            print("  seed %d: no model, skipped" % seed, flush=True)
            continue
        cfg = dict(overrides)
        cfg["model_path"] = str(model_path)
        cfg["continue_from"] = None
        cfg["log_path"] = None
        cfg_path = seed_dir / "eval_config.json"
        cfg_path.write_text(json.dumps(cfg, indent=2, sort_keys=True))

        tag = "%s_seed%d" % (args.exp_id, seed)
        metrics = evaluate(args.agent, args.opponents, args.scenario,
                           mode=args.mode, seeds=EVAL_SEEDS, n_rounds=args.rounds,
                           tag=tag, extra_env={"TQ_CONFIG": str(cfg_path)},
                           workers=args.workers)
        print(format_metrics(metrics), flush=True)
        register(metrics, exp_id=tag, note=args.note)
        results.append(metrics)

    if not results:
        raise SystemExit("no models found for %s" % args.exp_id)

    print("\n=== %s across %d seeds ===" % (args.exp_id, len(results)))
    for key in ("mean_score", "mean_coins", "mean_kills", "suicide_rate",
                "mean_bombs", "mean_crates", "win_rate", "survival_rate",
                "mean_steps"):
        values = [m[key] for m in results]
        print("  %-14s %.3f   (per seed: %s, spread %.3f)"
              % (key, statistics.mean(values),
                 ", ".join("%.3f" % v for v in values),
                 max(values) - min(values)))


if __name__ == "__main__":
    main()
