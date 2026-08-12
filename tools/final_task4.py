"""Task 4 win rate of the submitted agent at a sample size that can support the
claim.

Thirty rounds give a standard error near 0.09 on a win rate, which cannot
separate 0.53 from 0.50. This plays five disjoint held-out blocks of 30, for 150
rounds, and reports the spread across blocks. The same protocol is applied to
`rule_based_agent` against a copy of itself so the two are compared like for
like from the same board slot.

    python3 tools/final_task4.py
"""

import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.evaluate import EVAL_SEEDS, evaluate, register  # noqa: E402


def measure(agent, label):
    rates, scores, suicides = [], [], []
    for block in range(5):
        offset = 500 + block * 40
        metrics = evaluate(agent, ["rule_based_agent"], "classic", mode="exact",
                           seeds=[seed + offset for seed in EVAL_SEEDS], n_rounds=1,
                           tag="final_task4_%s_b%d" % (agent, block), workers=5)
        register(metrics, exp_id="FINAL-task4-%s" % agent,
                 note="held-out block %d" % block)
        rates.append(metrics["win_rate"])
        scores.append(metrics["mean_score"])
        suicides.append(metrics["suicide_rate"])
    print("%-24s win %.3f +/- %.3f (se %.3f)   score %.2f +/- %.2f   suicide %.3f"
          % (label, st.mean(rates), st.stdev(rates), st.stdev(rates) / 5 ** 0.5,
             st.mean(scores), st.stdev(scores), st.mean(suicides)), flush=True)
    print("     per block: %s" % ", ".join("%.3f" % r for r in rates), flush=True)
    return st.mean(rates)


def main():
    print("Task 4, 5 disjoint held-out blocks of 30 rounds each\n")
    measure("q_bomber", "q_bomber (submitted)")
    measure("rule_based_agent", "rule_based_agent (self)")


if __name__ == "__main__":
    main()
