"""Measure the four agents shipped with the framework under our own evaluation
protocol, so every later result has a permanent reference point measured the
same way. Run from the repository root: python3 tools/run_baselines.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.evaluate import EVAL_SEEDS, evaluate, format_metrics, register  # noqa: E402

AGENTS = ["random_agent", "peaceful_agent", "coin_collector_agent", "rule_based_agent"]

CONFIGS = [
    # name,          scenario,      opponents,              mode,    rounds
    ("task1_solo", "coin-heaven", [], "fast", 20),
    ("task2_solo", "classic", [], "fast", 20),
    ("task4_1v1", "classic", ["rule_based_agent"], "fast", 20),
    ("task4_1v1", "classic", ["rule_based_agent"], "exact", 1),
]


def main():
    rows = []
    for agent in AGENTS:
        for cfg_name, scenario, opponents, mode, n_rounds in CONFIGS:
            # An agent cannot be its own opponent under a distinct name here;
            # rule_based_agent vs rule_based_agent is still a meaningful
            # self-play reference, so keep it.
            tag = f"baseline_{agent}_{cfg_name}"
            m = evaluate(agent, opponents, scenario, mode=mode,
                         seeds=EVAL_SEEDS, n_rounds=n_rounds,
                         tag=tag, workers=4)
            print(format_metrics(m), flush=True)
            print("-" * 70, flush=True)
            register(m, exp_id="S0-baseline", note=cfg_name)
            rows.append((agent, cfg_name, mode, m))

    print("\n\n=== markdown summary ===\n")
    print("| agent | config | mode | mean_score | mean_coins | suicide_rate | win_rate | survival | mean_steps |")
    print("|---|---|---|---|---|---|---|---|---|")
    for agent, cfg, mode, m in rows:
        print(f"| `{agent}` | {cfg} | {mode} | {m['mean_score']:.2f} +/- {m['score_std']:.2f} "
              f"| {m['mean_coins']:.2f} | {m['suicide_rate']:.3f} "
              f"| {m['win_rate']:.3f} | {m['survival_rate']:.3f} | {m['mean_steps']:.0f} |")


if __name__ == "__main__":
    main()
