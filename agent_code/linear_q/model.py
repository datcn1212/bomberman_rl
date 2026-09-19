"""Linear Q: Q(s, a) = phi(s) . w[:, a], plus the vectoriser that builds phi.

Same observation as tabular_q (features.py is a copy), but one-hot encodes the
categorical parts into a 33-dim vector with a shared weight matrix, instead of
one table row per state. That's the only difference between the two agents.

Table rows guarantee nothing beats a blocked direction; additive one-hot
blocks don't, so callbacks.act masks blocked moves before choosing instead of
trusting the weights to learn it.

D4 handling: pick a canonical frame, vectorise the relabelled observation,
translate the chosen action back afterwards.
"""

import pickle

import numpy as np

from .features import ACTIONS, DIR_HERE, DIR_NONE, observe

N_ACTIONS = len(ACTIONS)
FEATURE_VERSION = 3

# sizes of the one-hot blocks
N_MOVE_STATUS = 4 * 3     # 4 neighbours x {blocked, free-safe, free-lethal}
N_T_HERE = 5
N_TARGET_DIR = 6
N_TARGET_KIND = 2
N_ESCAPE_DIR = 6
N_CONTINUOUS = 1          # target distance, normalised
N_BIAS = 1

PHI_DIM = (N_MOVE_STATUS + N_T_HERE + N_TARGET_DIR + N_TARGET_KIND
           + N_ESCAPE_DIR + N_CONTINUOUS + N_BIAS)

_TARGET_DIST_CAP = 15.0

# D4, same construction as in tabular_q/model.py
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


def _direction_index(perm, value):
    """Relabel a direction field. DIR_NONE and DIR_HERE aren't directions."""
    return value if value in (DIR_NONE, DIR_HERE) else perm[value - 1] + 1


def _tie_break_key(perm, obs):
    """Orders the eight views for a deterministic pick.

    Never stored, any total order works, but reusing tabular_q's mixed-radix
    order means both agents agree on the canonical view.
    """
    status = [0] * 4
    for i in range(4):
        status[perm[i]] = obs.move_status[i]
    key = 0
    for v in status:
        key = key * 3 + v
    key = key * N_T_HERE + obs.t_here
    key = key * N_TARGET_DIR + _direction_index(perm, obs.target_dir)
    key = key * N_TARGET_KIND + obs.target_kind
    key = key * N_ESCAPE_DIR + _direction_index(perm, obs.escape_dir)
    return key


def canonical_frame(obs):
    """(perm, relabelled status, target_dir, escape_dir) for the canonical view."""
    best_key, best_perm = None, None
    for perm in D4:
        key = _tie_break_key(perm, obs)
        if best_key is None or key < best_key:
            best_key, best_perm = key, perm

    status = [0] * 4
    for i in range(4):
        status[best_perm[i]] = obs.move_status[i]
    return (best_perm, tuple(status),
            _direction_index(best_perm, obs.target_dir),
            _direction_index(best_perm, obs.escape_dir))


def invert(perm):
    inverse = [0] * 4
    for i in range(4):
        inverse[perm[i]] = i
    return tuple(inverse)


def to_frame(perm, action):
    return perm[action] if action < 4 else action


def from_frame(perm, action):
    return invert(perm)[action] if action < 4 else action


def _one_hot(index, size, out, offset):
    out[offset + index] = 1.0
    return offset + size


def vectorize(move_status, t_here, target_dir, target_kind, escape_dir, target_dist):
    """Build phi(s) from fields that are already in the canonical frame."""
    phi = np.zeros(PHI_DIM, dtype=np.float64)

    # each neighbour gets its own 3-wide block, else "blocked" would merge across directions
    offset = 0
    for i in range(4):
        phi[offset + move_status[i]] = 1.0
        offset += 3

    offset = _one_hot(t_here, N_T_HERE, phi, offset)
    offset = _one_hot(target_dir, N_TARGET_DIR, phi, offset)
    offset = _one_hot(target_kind, N_TARGET_KIND, phi, offset)
    offset = _one_hot(escape_dir, N_ESCAPE_DIR, phi, offset)

    phi[offset] = min(target_dist, _TARGET_DIST_CAP) / _TARGET_DIST_CAP
    offset += N_CONTINUOUS
    phi[offset] = 1.0          # bias
    offset += N_BIAS

    assert offset == PHI_DIM
    return phi


def observe_and_encode(game_state, use_symmetry):
    """Returns (phi, obs, perm, status).

    status is the relabelled move_status, in the same canonical frame as phi
    and the Q values. obs.move_status alone is unrotated.

    Not cached: state before/after an action share a step number, so a cache
    keyed on it would serve a stale danger schedule.
    """
    obs = observe(game_state)
    if use_symmetry:
        perm, status, target_dir, escape_dir = canonical_frame(obs)
    else:
        perm = IDENTITY
        status, target_dir, escape_dir = obs.move_status, obs.target_dir, obs.escape_dir

    phi = vectorize(status, obs.t_here, target_dir, obs.target_kind, escape_dir,
                    obs.target_dist)
    return phi, obs, perm, status


class LinearQModel:
    def __init__(self, feature_flags=None):
        self.phi_dim = PHI_DIM
        self.feature_version = FEATURE_VERSION
        self.feature_flags = dict(feature_flags or {})
        self.w = np.zeros((PHI_DIM, N_ACTIONS), dtype=np.float64)
        # per (feature, action) update count, gated by phi[i] != 0, not per call
        self.visits = np.zeros((PHI_DIM, N_ACTIONS), dtype=np.int64)

    def values(self, phi):
        return phi @ self.w

    def update(self, phi, action, target, alpha):
        """w[:, a] += alpha * delta * phi. alpha can be scalar or (PHI_DIM,)."""
        delta = target - float(phi @ self.w[:, action])
        active = phi != 0
        self.w[:, action] += alpha * delta * phi
        self.visits[active, action] += 1

    def effective_alpha(self, phi, action, cfg):
        """alpha / (1 + n/half_life), one value per feature.

        n counts updates of the weight entry, not of a state, since one weight
        is shared across every state where the feature is active. Constant
        alpha never settles.
        """
        if cfg.alpha_schedule == "constant":
            return cfg.alpha
        if cfg.alpha_schedule == "visit":
            n = self.visits[:, action]
            return cfg.alpha / (1.0 + n / cfg.alpha_half_life)
        raise ValueError("unknown alpha_schedule %r" % cfg.alpha_schedule)

    def save(self, path):
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path):
        with open(path, "rb") as fh:
            model = pickle.load(fh)
        if model.phi_dim != PHI_DIM or model.feature_version != FEATURE_VERSION:
            raise ValueError(
                "model at %s has phi_dim/version (%d, %d), code expects (%d, %d); "
                "loading it would mix incompatible weight columns"
                % (path, model.phi_dim, model.feature_version, PHI_DIM, FEATURE_VERSION))
        return model
