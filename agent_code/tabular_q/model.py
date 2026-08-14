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


def encode_parts(move_status, t_here, target_dir, target_kind, escape_dir, bomb_opt):
    """Mixed-radix index of a raw feature tuple."""
    move = 0
    for i in range(4):
        move = move * 3 + move_status[i]
    index = move
    index = index * RADIX_T_HERE + t_here
    index = index * RADIX_TARGET_DIR + target_dir
    index = index * RADIX_TARGET_KIND + target_kind
    index = index * RADIX_ESCAPE_DIR + escape_dir
    index = index * RADIX_BOMB_OPT + bomb_opt
    return index


def encode(obs):
    return encode_parts(obs.move_status, obs.t_here, obs.target_dir,
                        obs.target_kind, obs.escape_dir, obs.bomb_opt)


# --- dihedral symmetry ----------------------------------------------------
# The arena and the rules are symmetric under D4 (four rotations x two
# reflections) and every feature is expressed relative to the agent, so a board
# rotated by 90 degrees is the same situation. The encoding uses absolute
# directions, though, so it lands on a different row: measured on a trained
# model, ~660 visited rows collapse into ~171 orbits, and the median row is
# updated 98 times where its orbit is updated 506 times.
#
# DIRS is (UP, RIGHT, DOWN, LEFT). A 90-degree clockwise rotation sends
# (dx, dy) -> (-dy, dx), mapping index i to (i + 1) % 4. A mirror in the
# vertical axis fixes UP and DOWN and swaps RIGHT and LEFT.
_ROT = (1, 2, 3, 0)
_MIRROR = (0, 3, 2, 1)


def _compose(p, q):
    return tuple(p[q[i]] for i in range(4))


def _build_d4():
    perms, rot = [], (0, 1, 2, 3)
    for _ in range(4):
        perms.append(rot)
        perms.append(_compose(_MIRROR, rot))
        rot = _compose(_ROT, rot)
    return tuple(perms)


D4 = _build_d4()


def _relabel(perm, obs):
    """Rewrite one observation's directions under a D4 element."""
    status = [0] * 4
    for i in range(4):
        status[perm[i]] = obs.move_status[i]

    def direction(value):
        # DIR_NONE (0) and DIR_HERE (5) carry no direction.
        return value if value in (0, 5) else perm[value - 1] + 1

    return encode_parts(status, obs.t_here, direction(obs.target_dir),
                        obs.target_kind, direction(obs.escape_dir), obs.bomb_opt)


def canonical(obs):
    """Smallest index in the observation's D4 orbit, and the element reaching it.

    The permutation has to come back with the index: it relabels directions, so
    the caller must translate between real move actions and the actions of the
    canonical frame.
    """
    best_index, best_perm = None, None
    for perm in D4:
        index = _relabel(perm, obs)
        if best_index is None or index < best_index:
            best_index, best_perm = index, perm
    return best_index, best_perm


IDENTITY = (0, 1, 2, 3)


def invert(perm):
    inverse = [0] * 4
    for i in range(4):
        inverse[perm[i]] = i
    return tuple(inverse)


def to_frame(perm, action):
    """Real action index -> its index in the canonical frame."""
    return perm[action] if action < 4 else action


def from_frame(perm, action):
    """Canonical-frame action index -> the real action index."""
    return invert(perm)[action] if action < 4 else action


class QModel:
    def __init__(self, feature_flags=None):
        self.layout = LAYOUT
        self.feature_version = FEATURE_VERSION
        # Recorded so a model can never be evaluated with a different
        # observation than it was trained on.
        self.feature_flags = dict(feature_flags or {})
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


# The framework hands the same game state to the agent more than once per step:
# act() sees s_t, then game_events_occurred sees s_t again as `old` and s_(t+1)
# as `new`, which act() will see next. observe() is a pure function of the state,
# so memoising on (round, step) returns identical results while cutting the two
# breadth-first searches from three evaluations per step to one.
_CACHE_KEY = None
_CACHE_VALUE = None


def observe_and_encode(game_state, use_symmetry):
    """The observation, its table index, and the frame that index is written in.

    With symmetry off the frame is the identity and the index is the plain
    encoding, so the two modes differ by exactly one lookup.
    """
    global _CACHE_KEY, _CACHE_VALUE
    # The agent's own name and position are part of the key: two instances of
    # this agent in one game share this module, and share the same (round, step),
    # but observe different states.
    me = game_state["self"]
    key = (game_state["round"], game_state["step"], me[0], me[3], use_symmetry)
    if key == _CACHE_KEY:
        return _CACHE_VALUE

    obs = observe(game_state)
    if use_symmetry:
        index, perm = canonical(obs)
        value = (index, obs, perm)
    else:
        value = (encode(obs), obs, IDENTITY)
    _CACHE_KEY, _CACHE_VALUE = key, value
    return value
