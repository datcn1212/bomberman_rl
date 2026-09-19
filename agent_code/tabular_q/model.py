"""Q table + encoding from observation to row index.
State is a few small ints, mixed-radix packed into one row each. 
LAYOUT (radices) and FEATURE_VERSION (value meanings) are pickled and checked on load.
Radices alone miss a change in meaning, and lookups would silently hit rows trained for something else.
"""

import pickle

import numpy as np

from .features import ACTIONS, observe

RADIX_MOVE = 81          # 4 neighbours x {blocked, free-safe, free-lethal}
RADIX_T_HERE = 5
RADIX_TARGET_DIR = 6     # none, 4 directions, already there
RADIX_TARGET_KIND = 3
RADIX_ESCAPE_DIR = 6
RADIX_BOMB_OPT = 5       # kept for old models, always 0 now
RADIX_OPPONENT = 3       # same
RADIX_LAST_MOVE = 5      # same

LAYOUT = (RADIX_MOVE, RADIX_T_HERE, RADIX_TARGET_DIR, RADIX_TARGET_KIND,
          RADIX_ESCAPE_DIR, RADIX_BOMB_OPT, RADIX_OPPONENT, RADIX_LAST_MOVE)
FEATURE_VERSION = 5

N_STATES = int(np.prod(LAYOUT))
N_ACTIONS = len(ACTIONS)

_UNSEEN = np.zeros(N_ACTIONS, dtype=np.float64)
_UNSEEN.flags.writeable = False
_UNSEEN_COUNTS = np.zeros(N_ACTIONS, dtype=np.int64)
_UNSEEN_COUNTS.flags.writeable = False


def encode_parts(move_status, t_here, target_dir, target_kind, escape_dir,
                 bomb_opt, opponent=0, last_move=0):
    """Pack the feature values into one integer (mixed radix)."""
    index = 0
    for i in range(4):
        index = index * 3 + move_status[i]
    index = index * RADIX_T_HERE + t_here
    index = index * RADIX_TARGET_DIR + target_dir
    index = index * RADIX_TARGET_KIND + target_kind
    index = index * RADIX_ESCAPE_DIR + escape_dir
    index = index * RADIX_BOMB_OPT + bomb_opt
    index = index * RADIX_OPPONENT + opponent
    index = index * RADIX_LAST_MOVE + last_move
    return index


def encode(obs):
    return encode_parts(obs.move_status, obs.t_here, obs.target_dir,
                        obs.target_kind, obs.escape_dir, obs.bomb_opt,
                        obs.opponent, obs.last_move)


# --- D4 symmetry ---------------------------------------------------------
# The board and the rules are symmetric under 4 rotations x 2 reflections -> a rotated board is the same situation. 
# DIRS is (UP, RIGHT, DOWN, LEFT). Rotating 90 degrees clockwise sends index i to (i+1) % 4; 
# mirroring in the vertical axis keeps UP/DOWN and swaps L/R.
_ROT = (1, 2, 3, 0)
_MIRROR = (0, 3, 2, 1)


def _compose(p, q):
    return tuple(p[q[i]] for i in range(4))


def _build_d4():
    perms = []
    rot = (0, 1, 2, 3)
    for _ in range(4):
        perms.append(rot)
        perms.append(_compose(_MIRROR, rot))
        rot = _compose(_ROT, rot)
    return tuple(perms)


D4 = _build_d4()
IDENTITY = (0, 1, 2, 3)


def _relabel(perm, obs):
    """Index of this observation after relabelling directions by perm."""
    status = [0] * 4
    for i in range(4):
        status[perm[i]] = obs.move_status[i]

    def direction(value):
        return value if value in (0, 5) else perm[value - 1] + 1   # 0/5 have no direction

    return encode_parts(status, obs.t_here, direction(obs.target_dir),
                        obs.target_kind, direction(obs.escape_dir), obs.bomb_opt,
                        obs.opponent, direction(obs.last_move))


def canonical(obs):
    """Smallest index in the orbit, plus the permutation that gets there"""
    best_index, best_perm = None, None
    for perm in D4:
        index = _relabel(perm, obs)
        if best_index is None or index < best_index:
            best_index, best_perm = index, perm
    return best_index, best_perm


def invert(perm):
    inverse = [0] * 4
    for i in range(4):
        inverse[perm[i]] = i
    return tuple(inverse)


def to_frame(perm, action):
    """Real action index -> canonical frame."""
    return perm[action] if action < 4 else action


def from_frame(perm, action):
    """Canonical frame -> real action index."""
    return invert(perm)[action] if action < 4 else action


class QModel:
    def __init__(self, feature_flags=None):
        self.layout = LAYOUT
        self.feature_version = FEATURE_VERSION
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

    def _row(self, state):
        """Row for this state, created empty if we haven't seen it before."""
        row = self.q.get(state)
        if row is None:
            row = np.zeros(N_ACTIONS, dtype=np.float64)
            self.q[state] = row
            self.seen[state] = np.zeros(N_ACTIONS, dtype=np.int64)
        return row

    def update(self, state, action, target, alpha):
        row = self._row(state)
        self.seen[state][action] += 1
        row[action] += alpha * (target - row[action])

    def add(self, state, action, amount):
        self._row(state)[action] += amount

    def note_visit(self, state, action):
        self._row(state)
        self.seen[state][action] += 1

    def effective_alpha(self, state, action, cfg):
        """alpha / (1 + n/half_life), n = updates of this (state, action) pair"""
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
                "expects %s; every lookup would miss silently."
                % (path, found, (LAYOUT, FEATURE_VERSION)))
        return model


def observe_and_encode(game_state, use_symmetry):
    """Observation, table index, and the frame the index is written in"""
    obs = observe(game_state)
    if use_symmetry:
        index, perm = canonical(obs)
        return index, obs, perm
    return encode(obs), obs, IDENTITY
