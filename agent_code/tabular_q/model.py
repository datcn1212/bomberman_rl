"""The Q table and the encoding that indexes it.

The state is a tuple of small integers, so it maps onto a single table row by
mixed-radix encoding. LAYOUT records the radix of every component; it is stored
inside the pickle and checked on load, because changing the state definition
while older models are still around otherwise fails silently -- every lookup
misses, the agent still runs, and the numbers just quietly get worse.
"""

import pickle

import numpy as np

from .features import ACTIONS, observe

RADIX_MOVE = 16          # 4 neighbours, free or blocked
RADIX_TARGET_DIR = 5     # none, up, right, down, left

LAYOUT = (RADIX_MOVE, RADIX_TARGET_DIR)
N_STATES = int(np.prod(LAYOUT))
N_ACTIONS = len(ACTIONS)


def encode(obs):
    """Mixed-radix index of an Observation."""
    move = (obs.move_status[0] | obs.move_status[1] << 1
            | obs.move_status[2] << 2 | obs.move_status[3] << 3)
    return move * RADIX_TARGET_DIR + obs.target_dir


class QModel:
    def __init__(self):
        self.layout = LAYOUT
        self.q = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)
        self.seen = np.zeros((N_STATES, N_ACTIONS), dtype=np.int64)

    def values(self, state):
        return self.q[state]

    def update(self, state, action, target, alpha):
        self.seen[state, action] += 1
        self.q[state, action] += alpha * (target - self.q[state, action])

    def save(self, path):
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path):
        with open(path, "rb") as fh:
            model = pickle.load(fh)
        if getattr(model, "layout", None) != LAYOUT:
            raise ValueError(
                "model at %s was trained with layout %s but this code expects %s; "
                "loading it would make every state lookup miss silently."
                % (path, getattr(model, "layout", None), LAYOUT))
        return model


def state_of(game_state):
    return encode(observe(game_state))
