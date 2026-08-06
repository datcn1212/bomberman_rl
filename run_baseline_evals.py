"""Eval 3 baseline (khong can train) cho so sanh voi q_linear da hoc:
- q_linear voi trong so ngau nhien (chua hoc gi)
- random_agent
- rule_based_agent

Cong cu phat trien, khong nam trong agent_code/q_linear/.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
AGENT_DIR = ROOT / "agent_code" / "q_linear"
RESULTS_DIR = ROOT / "results"
LOG_DIR = ROOT / "logs" / "refeval"

EVAL_ROUNDS = 1000
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
N_FEATURES = 9


def make_untrained_weights():
    rng = np.random.default_rng(0)
    beta = rng.normal(0, 0.01, size=(len(ACTIONS), N_FEATURES)).astype(np.float32)
    path = AGENT_DIR / "weights_untrained.npy"
    np.save(path, beta)
    return path


def evaluate(agent_name, tag, weights_path=None):
    import os
    stats_path = RESULTS_DIR / f"eval_{tag}.json"
    env = os.environ.copy()
    if weights_path is not None:
        env["Q_LINEAR_MODEL_PATH"] = str(weights_path)
    subprocess.run(
        [sys.executable, "main.py", "play", "--agents", agent_name, "--no-gui",
         "--scenario", "coin-heaven", "--n-rounds", str(EVAL_ROUNDS),
         "--save-stats", str(stats_path), "--log-dir", str(LOG_DIR)],
        cwd=ROOT, env=env, check=True,
    )
    data = json.loads(stats_path.read_text())
    rounds = list(data["by_round"].values())
    coins = [r["coins"] for r in rounds]
    steps = [r["steps"] for r in rounds]
    n = len(rounds)
    print(f"[{tag}] coins_mean={sum(coins) / n:.2f} steps_mean={sum(steps) / n:.1f} n_rounds={n}")


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    weights = make_untrained_weights()
    evaluate("q_linear", "untrained", weights)
    evaluate("random_agent", "random_agent")
    evaluate("rule_based_agent", "rule_based_agent")
