"""
experiment.py

Experiment director for the agent training and validation.

Flow:
    1. Train during TRAIN_CHUNK games.
    2. Save q_table.pkl.
    3. Evaluation with epsilon=0 and learning_rate=0.
    4. Evaluation results and statistics.
    5.Repeat until TOTAL_TRAINING_EPISODES.

"""

import csv
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
import json
from ruamel.yaml import YAML

import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

AGENT = "tabular_q_agent"

# Total training
TOTAL_TRAINING_EPISODES = 100

# Episodes until evaluation
TRAIN_CHUNK = 10

# Evaluation episodes for each evaluation phase
VALIDATION_EPISODES = 1

# Hyperparameters
EPSILON = 0.8
LEARNING_RATE = 0.20
GAMMA = 0.90

# Directories
EXPERIMENT_NAME = "coin_heaven_eps_08_lr_02_gamma_09"
RESULTS_DIR = Path("experiments") / EXPERIMENT_NAME
CONFIG_FILE = Path("agent_code/tabular_q_agent/config.yaml")

Q_TABLE_FILE = Path("agent_code/tabular_q_agent/q_table.pkl")
TRAINING_REWARD_FILE = Path("agent_code/tabular_q_agent/reward_progress.txt")
EVALUATION_FILE = Path("results/evaluation_01000.json")


# ============================================================
# COMMANDS
# ============================================================

BASE_COMMAND = [
    "python",
    "main.py",
    "play",
    "--agents",
    AGENT,
    "--no-gui",
    "--scenario",
    "coin-heaven",
]

TRAIN_COMMAND = BASE_COMMAND + [
    "--train",
    "1",
    "--n-rounds",
    "{episodes}",
]

EVAL_ROUNDS_FLAG = "--n-rounds"

EVAL_COMMAND = BASE_COMMAND + [
    EVAL_ROUNDS_FLAG,
    "{episodes}",
    "--save-stats", "results/evaluation_01000.json"
]


# ============================================================
# UTILS
# ============================================================

def run_command(command, env):
    """Runs a command and handles the experiment execution."""

    print("\n$ " + " ".join(command))

    subprocess.run(
        command,
        env=env,
        check=True,
    )


def make_environment():
    """Make environment and modify the agent's config file"""

    env = os.environ.copy()

    modify_config()

    return env

def modify_config(epsilon=EPSILON, gamma=GAMMA, learning_rate=LEARNING_RATE):
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2) 

    with open(CONFIG_FILE, 'r', encoding='utf-8') as file:
        datos = yaml.load(file)

    datos["epsilon"] = epsilon
    datos["gamma"] = gamma
    datos["learning_rate"] = learning_rate

    with open(CONFIG_FILE, 'w', encoding='utf-8') as file:
        yaml.dump(datos, file)

    print("Updated agent config!")


def ensure_results_dir():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def count_lines(filename):
    if not filename.exists():
        return 0

    with open(filename, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def read_new_training_rewards(previous_lines):
    """
    Reads rewards generated on the last training section.
    """

    if not TRAINING_REWARD_FILE.exists():
        return []

    with open(TRAINING_REWARD_FILE, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    new_lines = lines[previous_lines:]

    return [float(x) for x in new_lines]


def save_q_table(checkpoint):
    """Saves an independet copy of the Q-table."""

    if not Q_TABLE_FILE.exists():
        raise FileNotFoundError(
            f"{Q_TABLE_FILE} doesn't exist. "
            "Training didn't generate a Q-table."
        )

    destination = RESULTS_DIR / f"q_table_{checkpoint:05d}.pkl"
    shutil.copy2(Q_TABLE_FILE, destination)


def reset_evaluation_file():

    if EVALUATION_FILE.exists():
        EVALUATION_FILE.unlink()


def read_evaluation_results(filename, agent_name):
    with open(filename, "r") as f:
        results = json.load(f)

    rounds = results["by_round"]

    coins = [
        stats["coins"]
        for stats in rounds.values()
    ]

    steps = [
        stats["steps"]
        for stats in rounds.values()
    ]

    kills = [
        stats["kills"]
        for stats in rounds.values()
    ]

    suicides = [
        stats["suicides"]
        for stats in rounds.values()
    ]

    score = results["by_agent"][agent_name]["score"]

    return {
        "coins": coins,
        "steps": steps,
        "kills": kills,
        "suicides": suicides,
        "score": score,
    }


def save_evaluation_csv(checkpoint, rows):

    filename = RESULTS_DIR / f"evaluation_{checkpoint:05d}.csv"

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "reward",
                "score",
                "coins",
                "survival_steps",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)


def append_summary(checkpoint, rows):
    """Appends row with measured metrics."""

    filename = RESULTS_DIR / "validation_summary.csv"

    file_exists = filename.exists()

    rewards = [r["reward"] for r in rows]
    scores = [r["score"] for r in rows]
    coins = [r["coins"] for r in rows]
    survival = [r["survival_steps"] for r in rows]

    row = {
        "training_episodes": checkpoint,
        "validation_episodes": len(rows),

        "reward_mean": statistics.mean(rewards),
        "reward_median": statistics.median(rewards),
        "reward_std": statistics.stdev(rewards) if len(rewards) > 1 else 0.0,

        "score_mean": statistics.mean(scores),
        "coins_mean": statistics.mean(coins),
        "survival_steps_mean": statistics.mean(survival),
    }

    with open(filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def plot_validation():
    """Plot cummulative validation plots."""

    filename = RESULTS_DIR / "validation_summary.csv"

    if not filename.exists():
        return

    rows = []

    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows.extend(reader)

    if not rows:
        return

    x = [int(r["training_episodes"]) for r in rows]

    metrics = [
        ("reward_mean", "Reward medio", "validation_reward.png"),
        ("score_mean", "Score medio", "validation_score.png"),
        ("coins_mean", "Monedas recogidas", "validation_coins.png"),
        (
            "survival_steps_mean",
            "Pasos de supervivencia",
            "validation_survival.png",
        ),
    ]

    for key, ylabel, output_name in metrics:

        y = [float(r[key]) for r in rows]

        plt.figure(figsize=(10, 5))
        plt.plot(x, y, marker="o")

        plt.xlabel("Training episodes")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs training")

        plt.grid(True)
        plt.tight_layout()

        plt.savefig(
            RESULTS_DIR / output_name,
            dpi=150,
        )

        plt.close()


def plot_training_rewards(all_rewards):
    """Observed rewards during training."""

    if not all_rewards:
        return

    x = range(1, len(all_rewards) + 1)

    plt.figure(figsize=(10, 5))

    plt.plot(
        x,
        all_rewards,
        alpha=0.25,
        label="Reward",
    )

    window = min(50, len(all_rewards))

    if len(all_rewards) >= window:

        moving_average = []

        for i in range(window, len(all_rewards) + 1):
            chunk = all_rewards[i - window:i]
            moving_average.append(
                sum(chunk) / len(chunk)
            )

        plt.plot(
            range(window, len(all_rewards) + 1),
            moving_average,
            label=f"Media móvil ({window})",
        )

    plt.xlabel("Training eposiodes")
    plt.ylabel("Reward")
    plt.title("Reward during training")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "training_reward.png",
        dpi=150,
    )

    plt.close()


# ============================================================
# EXPERIMENT
# ============================================================

def main():

    ensure_results_dir()

    env = make_environment()

    if TOTAL_TRAINING_EPISODES % TRAIN_CHUNK != 0:
        raise ValueError(
            "TOTAL_TRAINING_EPISODES should be multiple of TRAIN_CHUNK."
        )

    print("=" * 60)
    print("TABULAR-Q EXP")
    print("=" * 60)

    print(f"Agent:       {AGENT}")
    print(f"Epsilon:      {EPSILON}")
    print(f"Learning rate:{LEARNING_RATE}")
    print(f"Gamma:        {GAMMA}")
    print(f"Training:{TOTAL_TRAINING_EPISODES}")
    print(f"Checkpoint:   cada {TRAIN_CHUNK}")
    print(f"Validation:   {VALIDATION_EPISODES} partidas")
    print(f"Results:   {RESULTS_DIR}")
    print("=" * 60)

    if Q_TABLE_FILE.exists():

        answer = input(
            f"\n{Q_TABLE_FILE} already exists. "
            "¿Delete and start from scratch? [y/N]: "
        )

        if answer.lower() == "y":
            Q_TABLE_FILE.unlink()
        else:
            print(
                "Using existing Q-table"
            )

    training_lines = count_lines(TRAINING_REWARD_FILE)

    all_training_rewards = []

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    for checkpoint in range(
        TRAIN_CHUNK,
        TOTAL_TRAINING_EPISODES + 1,
        TRAIN_CHUNK,
    ):

        print("\n" + "#" * 60)
        print(
            f"CHECKPOINT {checkpoint}/{TOTAL_TRAINING_EPISODES}"
        )
        print("#" * 60)

        # ====================================================
        # 1. TRAIN
        # ====================================================

        train_command = [
            part.format(episodes=TRAIN_CHUNK)
            for part in TRAIN_COMMAND
        ]

        run_command(train_command, env)

        new_rewards = read_new_training_rewards(training_lines)

        training_lines += len(new_rewards)
        all_training_rewards.extend(new_rewards)

        plot_training_rewards(all_training_rewards)

        # ====================================================
        # 2. SAVE Q-TABLE
        # ====================================================

        save_q_table(checkpoint)

        # ====================================================
        # 3. EVAL
        # ====================================================

        reset_evaluation_file()

        eval_command = [
            part.format(episodes=VALIDATION_EPISODES)
            for part in EVAL_COMMAND
        ]

        print(
            f"\nEvaluating policy with "
            f"epsilon=0 for {VALIDATION_EPISODES} episodes..."
        )

        eval_env = env.copy()
        eval_env["RL_EPSILON"] = "0.0"
        eval_env["RL_LR"] = "0.0"

        run_command(eval_command, eval_env)

        # ====================================================
        # 4. METRICS
        # ====================================================
        validation_dic = read_evaluation_results(EVALUATION_FILE, AGENT)

        # if len(validation_dic) != VALIDATION_EPISODES:
        #     print(
        #         f"ADVERTENCIA: se esperaban "
        #         f"{VALIDATION_EPISODES} partidas de evaluación, "
        #         f"pero se encontraron {len(validation_rows)}."
        #     )

        # save_evaluation_csv(
        #     checkpoint,
        #     validation_rows,
        # )

        # append_summary(
        #     checkpoint,
        #     validation_rows,
        # )

        plot_validation()

        # rewards = [
        #     reward
        #     for reward in validation_dic["rewards"]
        # ]

        coins = [
            coin
            for coin in validation_dic["coins"]
        ]

        print("\nValidation results:")
        print("-----------------------")
        # print(f"Reward medio: {statistics.mean(rewards):.2f}")
        # print(f"Reward std:   {statistics.stdev(rewards):.2f}"
        #       if len(rewards) > 1 else "Reward std:   0.00")
        print(f"Avg coins: {statistics.mean(coins):.2f}")


    print("\n" + "=" * 60)
    print("FINISHED EXPERIMENT")
    print("=" * 60)

    print(f"Saved results:")
    print(f"    {RESULTS_DIR}")

    print("\nMain files:")
    print("    validation_summary.csv")
    print("    training_reward.png")
    print("    validation_reward.png")
    print("    validation_score.png")
    print("    validation_coins.png")
    print("    validation_survival.png")
    print("    q_table_XXXXX.pkl")


if __name__ == "__main__":
    main()
