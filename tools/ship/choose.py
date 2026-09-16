"""Pick which trained seed to ship, ranked on arenas that don't report the result.

Choosing the best seed on the same arenas we then quote inflates the number:
with a run-to-run spread of ~0.09 on this board the gap between best and median
seed is mostly noise, and picking on it is picking noise. So selection runs on
a separate block of seeds, and the figure we report comes from the usual block,
which played no part in the choice.

    python3 tools/ship.py choose --exp-id final_tabular_q
    python3 tools/ship.py choose --exp-id lq_p6_budget12k --agent linear_q
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tools.evaluate import config_env_var, evaluate, register  # noqa: E402

EXPERIMENTS = ROOT / "experiments"
SELECTION_SEEDS = list(range(9101, 9131))   # disjoint from EVAL_SEEDS 9001..9030


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-id", default="final_tabular_q")
    p.add_argument("--agent", default="tabular_q")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--opponents", nargs="+", default=["rule_based_agent"] * 3)
    p.add_argument("--scenario", default="classic")
    args = p.parse_args()

    results = []
    for seed in args.seeds:
        model = EXPERIMENTS / args.exp_id / ("seed%d" % seed) / "model.pkl"
        if not model.is_file():
            continue
        cfg_path = model.parent / "selection_config.json"
        cfg_path.write_text(json.dumps(
            {"model_path": str(model), "continue_from": None, "log_path": None}))

        tag = "select_%s_seed%d" % (args.exp_id, seed)
        m = evaluate(args.agent, args.opponents, args.scenario, mode="fast",
                     seeds=SELECTION_SEEDS, n_rounds=args.rounds, tag=tag,
                     extra_env={config_env_var(args.agent): str(cfg_path)},
                     workers=args.workers)
        register(m, exp_id=tag, note="model selection on held-out arenas 9101-9130")
        results.append((m["mean_score"], seed, m))
        print("  seed %2d   score %.3f   coins %.3f   kills %.3f   suicide %.3f"
              % (seed, m["mean_score"], m["mean_coins"], m["mean_kills"],
                 m["suicide_rate"]), flush=True)

    if not results:
        raise SystemExit("no models found for %s" % args.exp_id)

    results.sort(reverse=True)
    best_score, best_seed, _ = results[0]
    scores = [r[0] for r in results]
    print("\nselection block 9101-9130, %s vs %s: mean %.3f, sd %.3f, range %.3f - %.3f"
          % (args.agent, "+".join(sorted(set(args.opponents))),
             statistics.mean(scores),
             statistics.stdev(scores) if len(scores) > 1 else 0.0,
             min(scores), max(scores)))
    print("chosen: seed %d (score %.3f on the selection block)" % (best_seed, best_score))
    print("\nReport this model's score on 9001-9030, which was not used to choose it.")
    return best_seed


if __name__ == "__main__":
    main()
