"""The decisive evaluation of a shipped model.

EXACT mode, one process per round, so win rate and survival are counted rather
than inferred, and 600 rounds per setting because a head-to-head claim on less
isn't worth making. Arenas are 9001-9600, so they contain the 30 arenas
`choose` ranks seeds on (9101-9130); report_tabular_q.md checks that leaving
those out changes nothing that matters.

    python3 tools/ship.py final --agent tabular_q --reference
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tools.evaluate import config_env_var, evaluate, format_metrics, register  # noqa: E402

SETTINGS = [
    ("task2_solo", "classic", []),
    ("task4_1v1", "classic", ["rule_based_agent"]),
    ("task4_four", "classic", ["rule_based_agent"] * 3),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default="tabular_q")
    p.add_argument("--rounds", type=int, default=600)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--reference", action="store_true",
                   help="also measure rule_based_agent in the same settings")
    args = p.parse_args()

    agent_dir = ROOT / "agent_code" / args.agent
    cfg_path = agent_dir / "final_eval_config.json"
    cfg_path.write_text(json.dumps(
        {"model_path": str(agent_dir / "model.pkl"),
         "continue_from": None, "log_path": None}))

    seeds = list(range(9001, 9001 + args.rounds))
    runs = [(args.agent, {config_env_var(args.agent): str(cfg_path)})]
    if args.reference:
        runs.append(("rule_based_agent", None))

    for name, scenario, opponents in SETTINGS:
        for agent, extra in runs:
            m = evaluate(agent, opponents, scenario, mode="exact", seeds=seeds,
                         n_rounds=1, tag="final_%s_%s" % (name, agent),
                         extra_env=extra, workers=args.workers)
            print("\n[%s] %s" % (name, agent))
            print(format_metrics(m), flush=True)
            register(m, exp_id="FINAL-%s-%s" % (name, agent),
                     note="decisive evaluation, %d rounds, exact mode" % args.rounds)


if __name__ == "__main__":
    main()
