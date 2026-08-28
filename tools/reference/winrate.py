"""Re-measure the reference agents' win rate at the same sample size we report
for our own agent.

A win rate over 30 rounds carries a standard error near 0.09, which is far too
wide to say whether 0.460 beats 0.433. Our own figures are five seeds of 30
rounds, so this repeats the 30-seed EXACT protocol five times as well, on
disjoint seed blocks, for a like-for-like comparison.

    python3 tools/reference.py winrate
"""

import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tools.evaluate import EVAL_SEEDS, evaluate, register  # noqa: E402


def main():
    for agent in ("rule_based_agent", "coin_collector_agent"):
        rates, scores = [], []
        for repeat in range(5):
            metrics = evaluate(agent, ["rule_based_agent"], "classic", mode="exact",
                               seeds=[seed + repeat * 100 for seed in EVAL_SEEDS],
                               n_rounds=1, tag="ref_%s_r%d" % (agent, repeat),
                               workers=5)
            register(metrics, exp_id="S0ref_%s_rep%d" % (agent, repeat),
                     note="reference win rate at matched sample size")
            rates.append(metrics["win_rate"])
            scores.append(metrics["mean_score"])
        print("%-22s vs rule_based_agent over 5x30 rounds: "
              "win %.3f +/- %.3f   score %.3f +/- %.3f   %s"
              % (agent, st.mean(rates), st.stdev(rates),
                 st.mean(scores), st.stdev(scores),
                 ", ".join("%.3f" % r for r in rates)), flush=True)


if __name__ == "__main__":
    main()
