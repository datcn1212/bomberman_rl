"""Run the EXACT-mode evaluation used for task gates.

FAST mode plays many rounds per process and can only report means, because the
stats file aggregates them. EXACT mode plays one round per process, so each file
describes a single round: that is what makes an honest win rate possible, and
survival too, by comparing the agent's own step count against the round length.

    python3 tools/ship.py gate --exp s2g_slowdecay --scenario classic
    python3 tools/ship.py gate --exp s5_final --scenario classic \\
        --opponents rule_based_agent
"""

import argparse
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tools.evaluate import EVAL_SEEDS, evaluate, format_metrics, register  # noqa: E402

EXPERIMENTS = ROOT / "experiments"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True)
    parser.add_argument("--scenario", default="classic")
    parser.add_argument("--opponents", nargs="*", default=[])
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--note", default="gate check")
    args = parser.parse_args()

    exp_dir = EXPERIMENTS / args.exp
    seed_dirs = sorted(exp_dir.glob("seed*"))
    if args.seeds:
        seed_dirs = [exp_dir / ("seed%d" % s) for s in args.seeds]

    results = []
    for seed_dir in seed_dirs:
        config_path = seed_dir / "eval_config.json"
        if not config_path.exists():
            print("skipping %s: no eval_config.json" % seed_dir.name)
            continue
        metrics = evaluate("tabular_q", args.opponents, args.scenario, mode="exact",
                           seeds=EVAL_SEEDS, n_rounds=1,
                           tag="%s_%s_gate" % (args.exp, seed_dir.name),
                           extra_env={"TQ_CONFIG": str(config_path)}, workers=5)
        print(format_metrics(metrics), flush=True)
        register(metrics, exp_id="%s_%s_exact" % (args.exp, seed_dir.name),
                 note=args.note)
        results.append(metrics)

    if not results:
        return
    print("\n=== %s, EXACT over %d seeds x %d rounds ==="
          % (args.exp, len(results), len(EVAL_SEEDS)))
    for key in ("mean_score", "mean_coins", "suicide_rate", "survival_rate",
                "win_rate", "mean_bombs", "mean_crates", "mean_steps"):
        values = [m[key] for m in results]
        spread = "%.3f" % st.stdev(values) if len(values) > 1 else "-"
        print("  %-14s %.3f +/- %s   (%s)"
              % (key, st.mean(values), spread,
                 ", ".join("%.3f" % v for v in values)))


if __name__ == "__main__":
    main()
