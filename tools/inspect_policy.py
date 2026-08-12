"""Decode a tabular model into a readable policy.

A table is small enough to be read directly, which makes it possible to explain
a failure instead of guessing at it: for every state the greedy action, the
margin over the runner-up, and how often the state was updated during training.

Run from the repository root:
    python3 tools/inspect_policy.py experiments/p1_nobomb/seed3/model.pkl
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_code.tabular_q.features import ACTIONS  # noqa: E402
from agent_code.tabular_q.model import (  # noqa: E402
    RADIX_TARGET_DIR, QModel)

DIR_NAMES = ["none", "UP", "RIGHT", "DOWN", "LEFT"]


def decode(index):
    """Inverse of model.encode: table row -> (move_status, target_dir)."""
    target_dir = index % RADIX_TARGET_DIR
    move = index // RADIX_TARGET_DIR
    move_status = tuple((move >> i) & 1 for i in range(4))
    return move_status, target_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--only-reachable", action="store_true",
                   help="skip states never updated during training")
    p.add_argument("--legal", type=int, default=len(ACTIONS),
                   help="number of actions the policy may use")
    args = p.parse_args()

    model = QModel.load(args.model)
    print("visited states: %d of %d"
          % (np.count_nonzero(model.seen.sum(axis=1)), model.q.shape[0]))
    print()
    print("| state | free U,R,D,L | coin dir | greedy | margin | visits | agrees |")
    print("|---|---|---|---|---|---|---|")

    disagree = 0
    for index in range(model.q.shape[0]):
        visits = int(model.seen[index].sum())
        if args.only_reachable and visits == 0:
            continue
        move_status, target_dir = decode(index)
        values = model.q[index][:args.legal]
        best = int(np.argmax(values))
        order = np.sort(values)[::-1]
        margin = float(order[0] - order[1]) if len(order) > 1 else float("nan")

        # The minimal state makes the intended policy explicit: walk towards the
        # nearest coin. Any state whose greedy action disagrees is a candidate
        # explanation for a failure.
        agrees = "-"
        if target_dir != 0:
            want = target_dir - 1
            agrees = "yes" if best == want else "NO"
            if best != want and visits > 0:
                disagree += 1

        print("| %d | %d,%d,%d,%d | %s | %s | %.4f | %d | %s |"
              % (index, move_status[0], move_status[1], move_status[2],
                 move_status[3], DIR_NAMES[target_dir], ACTIONS[best],
                 margin, visits, agrees))

    print()
    print("visited states whose greedy action is not the coin direction: %d" % disagree)


if __name__ == "__main__":
    main()
