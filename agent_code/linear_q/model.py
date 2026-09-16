"""Linear Q: Q(s, a) = phi(s) . w[:, a], plus the vectoriser that builds phi.

Same observation as tabular_q (features.py is a copy), but instead of one table
row per state we one-hot encode the categorical parts into a 33-dim vector and
keep a weight matrix shared by every state. That is the whole difference
between the two agents, which is what makes them comparable.

One thing the table gave us for free and this does not: with additive one-hot
blocks, nothing forces "this direction is blocked" to outrank everything else,
so a blocked move can come out on top. callbacks.act masks those out before
choosing rather than hoping the weights learn it (Phase 1).

D4 handling is the same idea as tabular_q's: pick a canonical frame, vectorise
the relabelled observation, translate the chosen action back afterwards.
"""

import pickle

import numpy as np

from .features import ACTIONS, BLOCKED, DIR_HERE, DIR_NONE, observe

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
    """Orders the eight views so we can pick one deterministically.

    Any total order would do - this is never stored - but using tabular_q's
    mixed-radix order means both agents call the same view canonical, so the
    D4 tests carry over between them.
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

    # each neighbour gets its own 3-wide block - "north is blocked" and "east is
    # blocked" are different facts and a shared block would merge them
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

    status is handed back separately because obs.move_status is the raw,
    unrotated field, while phi (and therefore the Q values) is indexed by
    canonical-frame directions. A caller that wants to know which of *those*
    directions is blocked needs the relabelled version.

    Not cached, same reason as tabular_q: the state before and after an action
    share a step number, so caching on it serves a stale danger schedule.
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
        # How often w[i, a] actually moved. An update only touches entries where
        # phi[i] != 0, so this counts per (feature, action), not per call.
        self.visits = np.zeros((PHI_DIM, N_ACTIONS), dtype=np.int64)

    def values(self, phi):
        return phi @ self.w

    def update(self, phi, action, target, alpha):
        """Semi-gradient TD step: w[:, a] += alpha * delta * phi.

        Every active feature moves, not one cell. alpha may be a scalar or a
        (PHI_DIM,) array - broadcasting covers both.
        """
        delta = target - float(phi @ self.w[:, action])
        active = phi != 0
        self.w[:, action] += alpha * delta * phi
        self.visits[active, action] += 1

    def effective_alpha(self, phi, action, cfg):
        """alpha / (1 + n/half_life), one value per feature.

        Same Robbins-Monro shape as tabular_q, but n counts updates of this
        weight entry, not of a state - a weight is shared by every state whose
        feature is active. Constant alpha never settles (Phase 1.4).
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
