"""Settle whether the submitted agent actually beats rule_based_agent.

At 150 rounds the difference between the two win rates was +0.033 with a
standard error of 0.042 -- consistent with anything from "clearly worse" to
"clearly better". This runs 20 disjoint blocks of 30 rounds per agent (600
rounds each), which brings the standard error of the difference to roughly
0.02 and can resolve a 0.04 gap.

Both agents are measured in the same board slot, against one
`rule_based_agent`, on the same seed blocks, so the comparison controls for any
positional advantage.

    python3 tools/decisive_head_to_head.py
"""

import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.evaluate import EVAL_SEEDS, evaluate, register  # noqa: E402

BLOCKS = 20


def measure(agent, tag):
    wins, scores, suicides = [], [], []
    for block in range(BLOCKS):
        offset = 2000 + block * 40
        metrics = evaluate(agent, ["rule_based_agent"], "classic", mode="exact",
                           seeds=[seed + offset for seed in EVAL_SEEDS], n_rounds=1,
                           tag="decisive_%s_b%02d" % (tag, block), workers=5)
        register(metrics, exp_id="DECISIVE-%s" % tag,
                 note="block %d of %d" % (block, BLOCKS))
        wins.append(metrics["win_rate"])
        scores.append(metrics["mean_score"])
        suicides.append(metrics["suicide_rate"])
        print("  block %2d  win %.3f  score %.2f" % (block, wins[-1], scores[-1]),
              flush=True)
    return wins, scores, suicides


def report(label, wins, scores, suicides):
    n = len(wins)
    print("%-34s win %.4f +/- %.4f (se)   score %.3f   suicide %.3f   %d rounds"
          % (label, st.mean(wins), st.stdev(wins) / n ** 0.5,
             st.mean(scores), st.mean(suicides), n * len(EVAL_SEEDS)), flush=True)


def main():
    print("q_bomber (submitted):", flush=True)
    qw, qs, qsu = measure("q_bomber", "q_bomber")
    print("\nrule_based_agent against a copy of itself:", flush=True)
    rw, rs, rsu = measure("rule_based_agent", "rule_based")

    print()
    report("q_bomber (submitted)", qw, qs, qsu)
    report("rule_based_agent (self)", rw, rs, rsu)

    diff = st.mean(qw) - st.mean(rw)
    se = (st.stdev(qw) ** 2 / len(qw) + st.stdev(rw) ** 2 / len(rw)) ** 0.5
    print("\nwin-rate difference %+.4f, standard error %.4f -> %.2f se"
          % (diff, se, abs(diff) / se))
    verdict = ("q_bomber is better" if diff > 2 * se else
               "rule_based_agent is better" if diff < -2 * se else
               "indistinguishable: the two are at parity")
    print("VERDICT: %s" % verdict)

    sdiff = st.mean(qs) - st.mean(rs)
    sse = (st.stdev(qs) ** 2 / len(qs) + st.stdev(rs) ** 2 / len(rs)) ** 0.5
    print("score difference %+.3f, standard error %.3f -> %.2f se"
          % (sdiff, sse, abs(sdiff) / sse))


if __name__ == "__main__":
    main()
