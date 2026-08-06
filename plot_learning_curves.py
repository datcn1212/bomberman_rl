"""Ve learning curve (running-mean coins collected) cho ca 5 config.
Cong cu phat trien, khong nam trong agent_code/q_linear/.
"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONFIGS = ["baseline", "alpha_high", "alpha_low", "gamma_low", "no_shaping"]


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return [int(r["episode"]) for r in rows], [float(r["coins_collected"]) for r in rows]


def running_mean(values, window):
    out = []
    for i in range(len(values)):
        lo = max(0, i - window)
        chunk = values[lo:i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


fig, ax = plt.subplots(figsize=(9, 5.5))
for name in CONFIGS:
    ep, coins = load(f"agent_code/q_linear/log_{name}.csv")
    smoothed = running_mean(coins, window=200)
    ax.plot(ep, smoothed, label=name, linewidth=1.5)

ax.axhline(50, color="gray", linestyle="--", linewidth=1, label="tran (50 coin)")
ax.set_xlabel("episode")
ax.set_ylabel("coins collected (trung binh truot 200 episode)")
ax.set_title("Q-linear learning curve - Task 1 (coin-heaven)")
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("learning_curve.png", dpi=130)
print("saved learning_curve.png")
