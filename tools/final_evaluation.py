"""Final measurement of the submitted agent.

Two things this does that the per-stage evaluations do not.

1. **Held-out validation of the selection.** The submitted model was picked as
   the best of eleven training runs *on* EVAL_SEEDS, so its score on those seeds
   is optimistic by construction. Everything here is re-measured on a disjoint
   seed block so the reported numbers are not the ones used to choose.

2. **All four tasks with the shipped artefact**, exactly as the tournament will
   load it: no QB_CONFIG, no overrides, just agent_code/q_bomber/model.pkl.

    python3 tools/final_evaluation.py
"""

import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.evaluate import EVAL_SEEDS, evaluate, format_metrics, register  # noqa: E402

# Disjoint from EVAL_SEEDS (9001..9030) and from every training seed (1000+).
HOLDOUT_SEEDS = [seed + 500 for seed in EVAL_SEEDS]

SETUPS = [
    ("Task 1  coin-heaven solo", "coin-heaven", [], "fast", 20),
    ("Task 2  classic solo", "classic", [], "fast", 20),
    ("Task 2  classic solo (exact)", "classic", [], "exact", 1),
    ("Task 3  vs peaceful", "classic", ["peaceful_agent"], "exact", 1),
    ("Task 3  vs coin_collector", "classic", ["coin_collector_agent"], "exact", 1),
    ("Task 4  vs one rule_based", "classic", ["rule_based_agent"], "exact", 1),
    ("Task 4  vs three rule_based", "classic",
     ["rule_based_agent", "rule_based_agent", "rule_based_agent"], "exact", 1),
]


def main():
    print("Submitted agent, measured on held-out seeds %d..%d\n"
          % (HOLDOUT_SEEDS[0], HOLDOUT_SEEDS[-1]))
    rows = []
    for label, scenario, opponents, mode, rounds in SETUPS:
        metrics = evaluate("q_bomber", opponents, scenario, mode=mode,
                           seeds=HOLDOUT_SEEDS, n_rounds=rounds,
                           tag="final_%s" % label.split()[1].lower().replace("-", ""),
                           workers=5)
        print("%s\n%s\n" % (label, format_metrics(metrics)), flush=True)
        register(metrics, exp_id="FINAL", note=label)
        rows.append((label, metrics))

    print("\n| setup | score | coins | kills | suicide | survival | win rate |")
    print("|---|---|---|---|---|---|---|")
    for label, m in rows:
        win = "-" if m["win_rate"] != m["win_rate"] else "%.3f" % m["win_rate"]
        surv = "-" if m["survival_rate"] != m["survival_rate"] else "%.3f" % m["survival_rate"]
        print("| %s | %.2f +/- %.2f | %.2f | %.3f | %.3f | %s | %s |"
              % (label, m["mean_score"], m["score_std"], m["mean_coins"],
                 m["mean_kills"], m["suicide_rate"], surv, win))


if __name__ == "__main__":
    main()
