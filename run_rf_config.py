"""Train + eval MOT config cho agent q_rf (Random Forest fitted-Q).

Ho tro ca 3 kieu: 1 pha (1 scenario, khong doi thu), 2 pha (curriculum
loot-crate -> classic), 3 pha (them 1 pha co doi thu cho Task 3/4). Chon bang
bien moi truong PHASES.

Luu y ve doc ket qua: --save-stats cua framework gom coins/kills/suicides cua
TAT CA agent theo tung round (xem environment.py end_round), nen khi co doi thu
phai doc 'by_agent' thay vi 'by_round' de lay dung so lieu cua rieng q_rf.

Cong cu phat trien, khong nam trong agent_code/q_rf/.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
AGENT_DIR = ROOT / "agent_code" / "q_rf"
RESULTS_DIR = ROOT / "results"

NAME = os.environ["CFG_NAME"]
PHASES = os.environ.get("PHASES", "1")           # "1", "2" hoac "3"
SCENARIO = os.environ.get("SCENARIO", "loot-crate")  # chi dung khi PHASES=1
OPPONENT = os.environ.get("OPPONENT", "")        # rong = khong co doi thu
EPISODES = int(os.environ.get("EPISODES", "1500"))
EVAL_ROUNDS = int(os.environ.get("EVAL_ROUNDS", "300"))
REFIT_EVERY = os.environ.get("Q_RF_REFIT_EVERY", "100")

MODEL_PATH = AGENT_DIR / f"model_{NAME}.pkl"
STATS_PATH = RESULTS_DIR / f"eval_{NAME}.json"
LOG_DIR = ROOT / "logs" / NAME


def run(cmd, env_overrides):
    env = os.environ.copy()
    env.update(env_overrides)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def train_phase(scenario, episodes, log_name, opponent="", continue_training=False):
    agents = ["q_rf"] + ([opponent] if opponent else [])
    run(
        [sys.executable, "main.py", "play", "--agents", *agents, "--train", "1",
         "--no-gui", "--scenario", scenario, "--n-rounds", str(episodes),
         "--log-dir", str(LOG_DIR)],
        {
            "Q_RF_MODEL_PATH": str(MODEL_PATH),
            "Q_RF_LOG_PATH": str(AGENT_DIR / f"log_{NAME}_{log_name}.csv"),
            "Q_RF_TOTAL_EPISODES": str(episodes),
            "Q_RF_REFIT_EVERY": REFIT_EVERY,
            **({"Q_RF_CONTINUE": "1"} if continue_training else {}),
        },
    )


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if PHASES == "1":
        train_phase(SCENARIO, EPISODES, "phase1")
        eval_scenario = SCENARIO
    elif PHASES == "2":
        train_phase("loot-crate", EPISODES, "phase1")
        train_phase("classic", EPISODES, "phase2", continue_training=True)
        eval_scenario = "classic"
    else:  # "3"
        train_phase("loot-crate", EPISODES, "phase1")
        train_phase("classic", EPISODES, "phase2", continue_training=True)
        train_phase("classic", EPISODES, "phase3", opponent=OPPONENT, continue_training=True)
        eval_scenario = "classic"

    t1 = time.time()

    eval_agents = ["q_rf"] + ([OPPONENT] if (PHASES == "3" and OPPONENT) else [])
    run(
        [sys.executable, "main.py", "play", "--agents", *eval_agents, "--no-gui",
         "--scenario", eval_scenario, "--n-rounds", str(EVAL_ROUNDS),
         "--save-stats", str(STATS_PATH), "--log-dir", str(LOG_DIR)],
        {"Q_RF_MODEL_PATH": str(MODEL_PATH)},
    )
    t2 = time.time()

    data = json.loads(STATS_PATH.read_text())
    n = EVAL_ROUNDS
    if PHASES == "3" and OPPONENT:
        stats = data["by_agent"]["q_rf"]
        coins, kills = stats.get("coins", 0), stats.get("kills", 0)
        suicides, steps = stats.get("suicides", 0), stats.get("steps", 0)
    else:
        rounds = list(data["by_round"].values())
        coins = sum(r["coins"] for r in rounds)
        kills = sum(r["kills"] for r in rounds)
        suicides = sum(r.get("suicides", 0) for r in rounds)
        steps = sum(r["steps"] for r in rounds)

    print(f"[{NAME}] phases={PHASES} scenario={eval_scenario} vs={OPPONENT or 'none'} "
          f"train={t1 - t0:.1f}s eval={t2 - t1:.1f}s "
          f"coins_mean={coins/n:.2f} kills_mean={kills/n:.3f} steps_mean={steps/n:.1f} "
          f"suicide_rate={suicides/n:.3f} n_rounds={n}")


if __name__ == "__main__":
    sys.exit(main())
