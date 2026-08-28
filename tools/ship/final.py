"""The decisive evaluation of the shipped model.

EXACT mode, one process per round, so survival and win rate are counted rather
than inferred, and 600 rounds per setting because this log does not make a
head-to-head claim on less. Arenas 9001 onwards; the model was chosen on a
disjoint block, so none of these rounds took part in selecting it.

    python3 tools/final_eval.py
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tools.evaluate import evaluate, format_metrics, register  # noqa: E402

AGENT_DIR = ROOT / "agent_code" / "tabular_q"

SETTINGS = [
    ("task2_solo", "classic", []),
    ("task4_1v1", "classic", ["rule_based_agent"]),
    ("task4_four", "classic", ["rule_based_agent"] * 3),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=600)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--reference", action="store_true",
                   help="measure rule_based_agent in the same settings")
    args = p.parse_args()

    cfg_path = AGENT_DIR / "final_eval_config.json"
    cfg_path.write_text(json.dumps(
        {"model_path": str(AGENT_DIR / "model.pkl"),
         "continue_from": None, "log_path": None}))

    seeds = list(range(9001, 9001 + args.rounds))
    for name, scenario, opponents in SETTINGS:
        for agent, extra in ([("tabular_q", {"TQ_CONFIG": str(cfg_path)})] +
                             ([("rule_based_agent", None)] if args.reference else [])):
            m = evaluate(agent, opponents, scenario, mode="exact", seeds=seeds,
                         n_rounds=1, tag="final_%s_%s" % (name, agent),
                         extra_env=extra, workers=args.workers)
            print("\n[%s] %s" % (name, agent))
            print(format_metrics(m), flush=True)
            register(m, exp_id="FINAL-%s-%s" % (name, agent),
                     note="decisive evaluation, %d rounds, exact mode" % args.rounds)


if __name__ == "__main__":
    main()
