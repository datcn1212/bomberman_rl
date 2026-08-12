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
from agent_code.tabular_q.model import LAYOUT, QModel  # noqa: E402

DIR_NAMES = ["none", "UP", "RIGHT", "DOWN", "LEFT", "here"]
MOVE_NAMES = ["blocked", "safe", "lethal"]
KIND_NAMES = ["crate", "coin", "opponent"]
BOMB_NAMES = ["none", "pointless", "useful", "trapped"]


def decode(index):
    """Inverse of model.encode: table row -> the feature tuple."""
    parts = []
    for radix in reversed(LAYOUT):
        parts.append(index % radix)
        index //= radix
    move, t_here, target_dir, target_kind, escape_dir, bomb_opt = reversed(parts)
    move_status = []
    for _ in range(4):
        move_status.append(move % 3)
        move //= 3
    return (tuple(reversed(move_status)), t_here, target_dir, target_kind,
            escape_dir, bomb_opt)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--only-reachable", action="store_true",
                   help="skip states never updated during training")
    p.add_argument("--legal", type=int, default=len(ACTIONS),
                   help="number of actions the policy may use")
    args = p.parse_args()

    model = QModel.load(args.model)
    print("visited states: %d" % len(model))
    print()
    print("| state | U,R,D,L | t_here | target | kind | escape | bomb | greedy | margin | visits | flag |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")

    disagree = 0
    for index in sorted(model.q):
        visits = int(model.visits(index).sum())
        move_status, t_here, target_dir, target_kind, escape_dir, bomb_opt = decode(index)
        values = model.values(index)[:args.legal]
        best = int(np.argmax(values))
        order = np.sort(values)[::-1]
        margin = float(order[0] - order[1]) if len(order) > 1 else float("nan")

        # A state that offers an escape but whose greedy action walks into a
        # tile that is about to burn is the failure worth finding.
        flag = "-"
        if escape_dir != 0 and best < 4 and move_status[best] == 2:
            flag = "WALKS INTO BLAST"
            disagree += 1

        print("| %d | %s | %d | %s | %s | %s | %s | %s | %.4f | %d | %s |"
              % (index, ",".join(MOVE_NAMES[m][0] for m in move_status), t_here,
                 DIR_NAMES[target_dir], KIND_NAMES[target_kind],
                 DIR_NAMES[escape_dir], BOMB_NAMES[bomb_opt], ACTIONS[best],
                 margin, visits, flag))

    print()
    print("visited states whose greedy action steps into a lethal tile: %d" % disagree)


if __name__ == "__main__":
    main()
