"""3-phase curriculum train + eval cho agent q_tabular o Task 3: san
peaceful_agent hoac coin_collector_agent tren scenario classic.

Phase 1: loot-crate mot minh (nang luc nhat-coin/pha-crate). Phase 2: classic
mot minh (nang luc tranh-nguy-hiem/thoat rieng cho mat do crate cua classic,
KHONG co doi thu -- day chinh la cau hinh da kiem chung tot nhat cho Task 2:
~2.44 coin/9, xem bao_cao_tabularQ.md muc 3.4-e). Phase 3: classic cung doi
thu, tiep tuc cung 1 bang -- chi hoc them hanh vi san duoi TREN NEN mot chinh
sach song sot da vung, thay vi hoc ca 2 thu cung luc trong 1 phase.

Ly do doi tu 2-phase sang 3-phase: 2-phase (loot-crate -> classic+doi thu
truc tiep) cho ket qua te hon rieng (suicide 89.6-98.0% qua 3 seed, xem
bao_cao_tabularQ.md) so voi phase-2-rieng da kiem chung (suicide ~74%) --
nghi ngo viec hoc truy duoi (doi hoi dinh vi mao hiem hon) CUNG LUC voi hoc
tranh-bom lam co che thoat hiem chua kip hoi tu truoc khi phai chiu them do
kho cua viec duoi bat.

Luu y: --save-stats cua framework gom coins/kills/suicides theo TONG CA HAI
agent trong moi round (xem environment.py end_round), khong tach rieng duoc.
Dung 'by_agent' (tong ca qua trinh chay, khong chia theo round) de doc dung
so lieu cua rieng q_tabular.

Cong cu phat trien, khong nam trong agent_code/q_tabular/.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
AGENT_DIR = ROOT / "agent_code" / "q_tabular"
RESULTS_DIR = ROOT / "results"

NAME = os.environ["CFG_NAME"]
OPPONENT = os.environ.get("OPPONENT", "peaceful_agent")
PHASE1_EPISODES = int(os.environ.get("PHASE1_EPISODES", "8000"))
PHASE2_EPISODES = int(os.environ.get("PHASE2_EPISODES", "8000"))
PHASE3_EPISODES = int(os.environ.get("PHASE3_EPISODES", "8000"))
EVAL_ROUNDS = int(os.environ.get("EVAL_ROUNDS", "500"))

TABLE_PATH = AGENT_DIR / f"table_{NAME}.npy"
LOG1_PATH = AGENT_DIR / f"log_{NAME}_phase1.csv"
LOG2_PATH = AGENT_DIR / f"log_{NAME}_phase2.csv"
LOG3_PATH = AGENT_DIR / f"log_{NAME}_phase3.csv"
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
        [sys.executable, "main.py", "play", "--agents", "q_tabular", "--train", "1",
         "--no-gui", "--scenario", "loot-crate", "--n-rounds", str(PHASE1_EPISODES),
         "--log-dir", str(LOG_DIR)],
        {
            "Q_TABULAR_MODEL_PATH": str(TABLE_PATH),
            "Q_TABULAR_LOG_PATH": str(LOG1_PATH),
            "Q_TABULAR_TOTAL_EPISODES": str(PHASE1_EPISODES),
        },
    )
    t1 = time.time()

    run(
        [sys.executable, "main.py", "play", "--agents", "q_tabular", "--train", "1",
         "--no-gui", "--scenario", "classic", "--n-rounds", str(PHASE2_EPISODES),
         "--log-dir", str(LOG_DIR)],
        {
            "Q_TABULAR_MODEL_PATH": str(TABLE_PATH),
            "Q_TABULAR_LOG_PATH": str(LOG2_PATH),
            "Q_TABULAR_TOTAL_EPISODES": str(PHASE2_EPISODES),
            "Q_TABULAR_CONTINUE": "1",
        },
    )
    t2 = time.time()

    run(
        [sys.executable, "main.py", "play", "--agents", "q_tabular", OPPONENT, "--train", "1",
         "--no-gui", "--scenario", "classic", "--n-rounds", str(PHASE3_EPISODES),
         "--log-dir", str(LOG_DIR)],
        {
            "Q_TABULAR_MODEL_PATH": str(TABLE_PATH),
            "Q_TABULAR_LOG_PATH": str(LOG3_PATH),
            "Q_TABULAR_TOTAL_EPISODES": str(PHASE3_EPISODES),
            "Q_TABULAR_CONTINUE": "1",
        },
    )
    t3 = time.time()

    run(
        [sys.executable, "main.py", "play", "--agents", "q_tabular", OPPONENT, "--no-gui",
         "--scenario", "classic", "--n-rounds", str(EVAL_ROUNDS),
         "--save-stats", str(STATS_PATH), "--log-dir", str(LOG_DIR)],
        {"Q_TABULAR_MODEL_PATH": str(TABLE_PATH)},
    )
    t4 = time.time()

    data = json.loads(STATS_PATH.read_text())
    stats = data["by_agent"]["q_tabular"]
    n = EVAL_ROUNDS
    coins = stats.get("coins", 0)
    kills = stats.get("kills", 0)
    suicides = stats.get("suicides", 0)
    steps = stats.get("steps", 0)

    print(f"[{NAME}] vs={OPPONENT} phase1={t1 - t0:.1f}s phase2={t2 - t1:.1f}s "
          f"phase3={t3 - t2:.1f}s eval={t4 - t3:.1f}s "
          f"coins_mean={coins/n:.2f} kills_mean={kills/n:.2f} steps_mean={steps/n:.1f} "
          f"suicide_rate={suicides/n:.3f} n_rounds={n}")


if __name__ == "__main__":
    sys.exit(main())
