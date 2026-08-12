"""The Q table and the encoding that indexes it.

The state is a tuple of small integers, so it maps onto a single table row by
mixed-radix encoding. LAYOUT records the radix of every component and
FEATURE_VERSION records the *meaning* of those components; both are stored
inside the pickle and checked on load. The radix alone is not enough: a change
that redefines what a value means without changing how many values there are
would otherwise pass unnoticed, and every affected lookup would silently return
a row trained for something else.

The table is stored **sparsely**. Measured after Phase 3: of 43740 addressable
rows only about 630 are ever visited (1.4%), because most feature combinations
are geometrically impossible - a tile cannot be blocked on all four sides and
still offer an escape in one of them. A dense array spends 98.6% of a 4.2 MB
pickle on zeros, and the Phase 4 encoding would have made that 17 MB.
"""

import pickle

import numpy as np

from .features import ACTIONS, observe

RADIX_MOVE = 81          # 4 neighbours x {blocked, free-safe, free-lethal}
RADIX_T_HERE = 5         # decisions until this tile burns; 0 = it does not
RADIX_TARGET_DIR = 6     # none, 4 directions, or "already there"
RADIX_TARGET_KIND = 3    # crate, coin, opponent
RADIX_ESCAPE_DIR = 6     # none, 4 directions, or "staying put is safe"
RADIX_BOMB_OPT = 4       # unavailable, pointless, useful, self-trapping

LAYOUT = (RADIX_MOVE, RADIX_T_HERE, RADIX_TARGET_DIR, RADIX_TARGET_KIND,
          RADIX_ESCAPE_DIR, RADIX_BOMB_OPT)
FEATURE_VERSION = 4

N_STATES = int(np.prod(LAYOUT))
N_ACTIONS = len(ACTIONS)

# Returned for rows that have never been updated. Never mutated: callers only
# read from it, and every write goes through update(), which allocates first.
_UNSEEN = np.zeros(N_ACTIONS, dtype=np.float64)
_UNSEEN.flags.writeable = False
_UNSEEN_COUNTS = np.zeros(N_ACTIONS, dtype=np.int64)
_UNSEEN_COUNTS.flags.writeable = False


def encode(obs):
    """Mixed-radix index of an Observation."""
    move = 0
    for i in range(4):
        move = move * 3 + obs.move_status[i]
    index = move
    index = index * RADIX_T_HERE + obs.t_here
    index = index * RADIX_TARGET_DIR + obs.target_dir
    index = index * RADIX_TARGET_KIND + obs.target_kind
    index = index * RADIX_ESCAPE_DIR + obs.escape_dir
    index = index * RADIX_BOMB_OPT + obs.bomb_opt
    return index


class QModel:
    def __init__(self):
        self.layout = LAYOUT
        self.feature_version = FEATURE_VERSION
        self.q = {}
        self.seen = {}

    def __len__(self):
        return len(self.q)

    def values(self, state):
        row = self.q.get(state)
        return _UNSEEN if row is None else row

    def visits(self, state):
        row = self.seen.get(state)
        return _UNSEEN_COUNTS if row is None else row

    def update(self, state, action, target, alpha):
        row = self.q.get(state)
        if row is None:
            row = np.zeros(N_ACTIONS, dtype=np.float64)
            self.q[state] = row
            self.seen[state] = np.zeros(N_ACTIONS, dtype=np.int64)
        self.seen[state][action] += 1
        row[action] += alpha * (target - row[action])

    def effective_alpha(self, state, action, cfg):
        """Step size for one update.

        With "visit" the step size decays as alpha / (1 + n/half_life), where n
        counts updates of this (state, action) pair specifically. Rare pairs keep
        learning fast while pairs seen tens of thousands of times settle down,
        which a global schedule cannot do. Phase 2 measured what happens without
        it: the estimate never converges and the policy is decided by where the
        random walk sits when training stops.
        """
        if cfg.alpha_schedule == "constant":
            return cfg.alpha
        if cfg.alpha_schedule == "visit":
            n = self.visits(state)[action]
            return cfg.alpha / (1.0 + n / cfg.alpha_half_life)
        raise ValueError("unknown alpha_schedule %r" % cfg.alpha_schedule)

    def save(self, path):
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path):
        with open(path, "rb") as fh:
            model = pickle.load(fh)
        found = (getattr(model, "layout", None), getattr(model, "feature_version", None))
        if found != (LAYOUT, FEATURE_VERSION):
            raise ValueError(
                "model at %s was trained with layout/version %s but this code "
                "expects %s; loading it would make every state lookup miss "
                "silently." % (path, found, (LAYOUT, FEATURE_VERSION)))
        return model


def state_of(game_state):
    return encode(observe(game_state))
