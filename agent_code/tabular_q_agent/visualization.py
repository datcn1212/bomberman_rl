import matplotlib.pyplot as plt
import numpy as np

with open("reward_progress.txt", "r") as f:
    rewards = [float(line.strip()) for line in f.readlines()]

# denoise
window_size = 10
moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')

# plot
plt.figure(figsize=(10, 5))
plt.plot(rewards, alpha=0.3, label="Round reward", color="blue")
plt.plot(range(window_size-1, len(rewards)), moving_avg, label=f"Moving average ({window_size} rondas)", color="red", linewidth=2)
plt.title("Agent training progress")
plt.xlabel("Round")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)
plt.savefig("progress_plot.png")
plt.show()
