"""Pick which trained seed to ship, on evaluation arenas not used to rank them.

Choosing the highest-scoring seed on the same arenas that produced the ranking
inflates the reported result: with a run-to-run spread of 0.09 on this board, the
gap between the best and median seed is mostly noise, and selecting on it is
selecting noise. So selection uses a fresh block of evaluation seeds, and the
number finally reported comes from the original block, which played no part in
the choice.

    python3 tools/ship.py choose --exp-id final_tabular_q
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tools.evaluate import evaluate, register  # noqa: E402

EXPERIMENTS = ROOT / "experiments"
SELECTION_SEEDS = list(range(9101, 9131))   # disjoint from EVAL_SEEDS 9001..9030


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-id", default="final_tabular_q")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    opponents = ["rule_based_agent"] * 3
    results = []
    for seed in args.seeds:
        model = EXPERIMENTS / args.exp_id / ("seed%d" % seed) / "model.pkl"
        if not model.is_file():
            continue
        cfg_path = model.parent / "selection_config.json"
        cfg_path.write_text(json.dumps(
            {"model_path": str(model), "continue_from": None, "log_path": None}))
        m = evaluate("tabular_q", opponents, "classic", mode="fast",
                     seeds=SELECTION_SEEDS, n_rounds=args.rounds,
                     tag="select_%s_seed%d" % (args.exp_id, seed),
                     extra_env={"TQ_CONFIG": str(cfg_path)}, workers=args.workers)
        register(m, exp_id="select_%s_seed%d" % (args.exp_id, seed),
                 note="model selection on held-out arenas 9101-9130")
        results.append((m["mean_score"], seed, m))
        print("  seed %2d   score %.3f   coins %.3f   kills %.3f   suicide %.3f"
              % (seed, m["mean_score"], m["mean_coins"], m["mean_kills"],
                 m["suicide_rate"]), flush=True)

    results.sort(reverse=True)
    best_score, best_seed, _ = results[0]
    scores = [r[0] for r in results]
    print("\nselection block 9101-9130: mean %.3f, sd %.3f, range %.3f - %.3f"
          % (statistics.mean(scores), statistics.stdev(scores), min(scores), max(scores)))
    print("chosen: seed %d (score %.3f on the selection block)" % (best_seed, best_score))
    print("\nThe figure to report for this model is its score on 9001-9030, which "
          "was not used to choose it.")
    return best_seed


if __name__ == "__main__":
    main()
