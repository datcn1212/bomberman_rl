"""Train + eval MOT config cho agent q_linear o Task 2 (loot-crate/classic).

Cong cu phat trien, khong nam trong agent_code/q_linear/.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
AGENT_DIR = ROOT / "agent_code" / "q_linear"
RESULTS_DIR = ROOT / "results"

NAME = os.environ["CFG_NAME"]
SCENARIO = os.environ.get("SCENARIO", "loot-crate")
ALPHA = os.environ.get("Q_LINEAR_ALPHA", "0.01")
GAMMA = os.environ.get("Q_LINEAR_GAMMA", "0.95")
TRAIN_EPISODES = int(os.environ.get("TRAIN_EPISODES", "20000"))
EVAL_ROUNDS = int(os.environ.get("EVAL_ROUNDS", "1000"))
BUFFER_SIZE = os.environ.get("Q_LINEAR_BUFFER_SIZE", "2000")
BATCH_SIZE = os.environ.get("Q_LINEAR_BATCH_SIZE", "32")

WEIGHTS_PATH = AGENT_DIR / f"weights_{NAME}.npy"
LOG_PATH = AGENT_DIR / f"log_{NAME}.csv"
STATS_PATH = RESULTS_DIR / f"eval_{NAME}.json"
LOG_DIR = ROOT / "logs" / NAME


def run(cmd, env_overrides):
    env = os.environ.copy()
    env.update(env_overrides)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    run(
        [sys.executable, "main.py", "play", "--agents", "q_linear", "--train", "1",
         "--no-gui", "--scenario", SCENARIO, "--n-rounds", str(TRAIN_EPISODES),
         "--log-dir", str(LOG_DIR)],
        {
            "Q_LINEAR_MODEL_PATH": str(WEIGHTS_PATH),
            "Q_LINEAR_LOG_PATH": str(LOG_PATH),
            "Q_LINEAR_TOTAL_EPISODES": str(TRAIN_EPISODES),
            "Q_LINEAR_ALPHA": ALPHA,
            "Q_LINEAR_GAMMA": GAMMA,
            "Q_LINEAR_USE_SHAPING": "1",
            "Q_LINEAR_BUFFER_SIZE": BUFFER_SIZE,
            "Q_LINEAR_BATCH_SIZE": BATCH_SIZE,
        },
    )
    t1 = time.time()

    run(
        [sys.executable, "main.py", "play", "--agents", "q_linear", "--no-gui",
         "--scenario", SCENARIO, "--n-rounds", str(EVAL_ROUNDS),
         "--save-stats", str(STATS_PATH), "--log-dir", str(LOG_DIR)],
        {"Q_LINEAR_MODEL_PATH": str(WEIGHTS_PATH)},
    )
    t2 = time.time()

    data = json.loads(STATS_PATH.read_text())
    rounds = list(data["by_round"].values())
    coins = [r["coins"] for r in rounds]
    steps = [r["steps"] for r in rounds]
    suicides = [r.get("suicides", 0) for r in rounds]
    n = len(rounds)

    print(f"[{NAME}] scenario={SCENARIO} train={t1 - t0:.1f}s eval={t2 - t1:.1f}s "
          f"coins_mean={sum(coins)/n:.2f} steps_mean={sum(steps)/n:.1f} "
          f"suicide_rate={sum(suicides)/n:.3f} n_rounds={n}")


if __name__ == "__main__":
    sys.exit(main())
