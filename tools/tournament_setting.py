"""The four-agent measurement, which is the one the tournament actually runs.

Every head-to-head number reported so far is 1v1: two agents on the board. The
tournament plays four. Those are different games -- with four agents the board
is more crowded, three independent bombers are creating danger, and the chance
level for "highest score in the round" is 1/4 rather than 1/2.

So the reference has to change too: a `rule_based_agent` in a game of four
`rule_based_agent`s should win about a quarter of rounds, and that is what makes
our number meaningful.

600 rounds per configuration, in 20 disjoint blocks of 30, both agents in the
same board slot.

    python3 tools/tournament_setting.py
"""

import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.evaluate import EVAL_SEEDS, evaluate, register  # noqa: E402

BLOCKS = 20
THREE_OPPONENTS = ["rule_based_agent"] * 3


def measure(agent, tag):
    wins, draws, scores, suicides = [], [], [], []
    for block in range(BLOCKS):
        offset = 4000 + block * 40
        metrics = evaluate(agent, THREE_OPPONENTS, "classic", mode="exact",
                           seeds=[seed + offset for seed in EVAL_SEEDS], n_rounds=1,
                           tag="tourney_%s_b%02d" % (tag, block), workers=5)
        register(metrics, exp_id="TOURNEY-%s" % tag,
                 note="four-agent block %d of %d" % (block, BLOCKS))
        wins.append(metrics["win_rate"])
        draws.append(metrics["draw_rate"])
        scores.append(metrics["mean_score"])
        suicides.append(metrics["suicide_rate"])
        print("  block %2d  win %.3f  score %.2f" % (block, wins[-1], scores[-1]),
              flush=True)
    return wins, draws, scores, suicides


def main():
    print("q_bomber + three rule_based_agent:", flush=True)
    qw, qd, qs, qsu = measure("q_bomber", "q_bomber")
    print("\nfour rule_based_agent (first slot measured):", flush=True)
    rw, rd, rs, rsu = measure("rule_based_agent", "rule_based")

    print("\n%-34s %18s %8s %8s %9s" % ("agent (first slot of four)", "win rate",
                                        "draw", "score", "suicide"))
    for label, w, d, s, su in [("q_bomber (submitted)", qw, qd, qs, qsu),
                               ("rule_based_agent", rw, rd, rs, rsu)]:
        print("%-34s %.4f +/- %-6.4f %8.3f %8.3f %9.3f"
              % (label, st.mean(w), st.stdev(w) / len(w) ** 0.5,
                 st.mean(d), st.mean(s), st.mean(su)), flush=True)
    print("%-34s %.4f" % ("chance level with four equal agents", 0.25))

    diff = st.mean(qw) - st.mean(rw)
    se = (st.stdev(qw) ** 2 / len(qw) + st.stdev(rw) ** 2 / len(rw)) ** 0.5
    print("\nwin-rate difference %+.4f, standard error %.4f -> %.2f se"
          % (diff, se, abs(diff) / se))
    print("VERDICT: %s" % ("q_bomber is better" if diff > 2 * se else
                           "rule_based_agent is better" if diff < -2 * se else
                           "indistinguishable"))


if __name__ == "__main__":
    main()
