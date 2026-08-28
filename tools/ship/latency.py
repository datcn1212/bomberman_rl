"""Worst-case timing of one decision step.

The think-time penalty is cumulative (report.md 0.2): overrunning does not cost
one step, it borrows from the next one, and an agent that keeps overrunning gets
skipped entirely until the debt is paid. So the number that matters is the
maximum, not the mean, measured on the most expensive board state we can build:
full crate density, several bombs ticking, opponents present.

Run from the repository root:  python3 tools/benchmark_latency.py [model.pkl]
"""

import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import settings as s  # noqa: E402
from agent_code.tabular_q import model as qmodel  # noqa: E402
from agent_code.tabular_q.features import FLAGS, observe  # noqa: E402


def worst_case_states(n=400, seed=7):
    """Boards that stress every code path: dense crates so the target BFS runs
    long, several bombs so the danger schedule and the time-expanded escape
    search are both active, opponents so their goal set is built too."""
    rng = np.random.default_rng(seed)
    states = []
    for _ in range(n):
        field = np.zeros((s.COLS, s.ROWS), dtype=int)
        field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = -1
        for x in range(s.COLS):
            for y in range(s.ROWS):
                if (x + 1) * (y + 1) % 2 == 1:
                    field[x, y] = -1
        free = [(x, y) for x in range(1, s.COLS - 1) for y in range(1, s.ROWS - 1)
                if field[x, y] == 0]
        for (x, y) in free:
            if rng.random() < s.SCENARIOS["classic"]["CRATE_DENSITY"]:
                field[x, y] = 1
        # Clear a pocket so the agent has somewhere to stand and search from.
        pos = free[int(rng.integers(len(free)))]
        for (dx, dy) in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (2, 0), (0, 2)):
            x, y = pos[0] + dx, pos[1] + dy
            if 0 < x < s.COLS - 1 and 0 < y < s.ROWS - 1 and field[x, y] == 1:
                field[x, y] = 0

        open_tiles = [(x, y) for (x, y) in free if field[x, y] == 0 and (x, y) != pos]
        rng.shuffle(open_tiles)
        bombs = [(open_tiles[i], int(rng.integers(0, s.BOMB_TIMER)))
                 for i in range(min(4, len(open_tiles)))]
        # Put the agent inside a blast on purpose: that is what triggers the
        # time-expanded escape search, the most expensive path in features.py.
        # Without this the benchmark would only ever time the cheap branch.
        bombs.append((pos, int(rng.integers(0, s.BOMB_TIMER))))
        others = open_tiles[4:7]
        explosion_map = np.zeros((s.COLS, s.ROWS))
        for (x, y) in open_tiles[7:12]:
            explosion_map[x, y] = 1
        states.append({
            "round": 1, "step": 100, "field": field,
            "self": ("me", 0, True, pos),
            "others": [("o%d" % i, 0, True, xy) for i, xy in enumerate(others)],
            "bombs": bombs,
            "coins": [xy for xy in open_tiles[12:15]],
            "explosion_map": explosion_map, "user_input": None,
        })
    return states


def lookup(q, obs, use_symmetry):
    """Exactly what act() does after observe(): index the table, read the row."""
    index = qmodel.canonical(obs)[0] if use_symmetry else qmodel.encode(obs)
    return q.values(index)


def main():
    states = worst_case_states()
    q = qmodel.QModel()
    if len(sys.argv) > 1:
        q = qmodel.QModel.load(sys.argv[1])
    stored = dict(getattr(q, "feature_flags", None) or {})
    use_symmetry = bool(stored.pop("use_symmetry", False))
    # Honour every flag the model was trained with, or the measurement leaves out
    # work the agent actually does at play time.
    FLAGS.update(stored)

    # Warm up so import and first-touch costs do not pollute the measurement.
    for st in states[:20]:
        lookup(q, observe(st), use_symmetry)

    feature_times, total_times = [], []
    for st in states:
        t0 = time.perf_counter()
        obs = observe(st)
        t1 = time.perf_counter()
        lookup(q, obs, use_symmetry)
        t2 = time.perf_counter()
        feature_times.append((t1 - t0) * 1000.0)
        total_times.append((t2 - t0) * 1000.0)

    def report(name, values):
        values = sorted(values)
        p99 = values[int(0.99 * (len(values) - 1))]
        print(f"{name:<22} mean {statistics.mean(values):6.2f} ms   "
              f"p99 {p99:6.2f} ms   max {values[-1]:6.2f} ms   "
              f"({values[-1] / (s.TIMEOUT * 1000) * 100:.2f}% of the {s.TIMEOUT}s budget)")

    print(f"worst-case boards: {len(states)}, model rows: {len(q)}, "
          f"symmetry: {use_symmetry}")
    report("feature extraction", feature_times)
    report("full decision step", total_times)
    budget_ms = s.TIMEOUT * 1000
    print("\nVERDICT:", "PASS" if max(total_times) < budget_ms else "FAIL",
          f"(max {max(total_times):.2f} ms vs {budget_ms:.0f} ms)")


if __name__ == "__main__":
    main()
