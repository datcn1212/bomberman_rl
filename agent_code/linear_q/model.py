"""The weight matrix and the vector that indexes it.

Q(s, a) = phi(s) . w[:, a] - a linear approximator, replacing the tabular
agent's dict-of-rows with a dense weight matrix shared across every state.

phi(s) one-hot encodes every categorical component of the observation and adds
one normalised scalar for target distance, plus a bias term. Three slots the
observation still carries - bomb_opt, opponent, last_move - are held at a
constant value on this branch (see tabular_q/features.py, Phase 25), so
one-hot encoding them would spend three weight rows on columns that are always
1 at index 0 and never move: dead weight, not dead code. They are left out of
phi() rather than encoded uselessly, and adding them back is a one-line change
if a later phase turns any of them back on.

D4 canonicalisation is reused exactly as tabular_q established it: among the
eight rotations and reflections, `canonical()` picks the same one it always
would (smallest mixed-radix index under the relabelling), and phi() vectorises
the *relabelled* observation rather than the raw one. The choice of frame is
representation-independent - it only asks "which of eight equivalent views is
canonical", never "how is the result stored" - so the exact function tabular_q
tested is used unchanged; only what happens after picking the frame differs.
"""

import pickle

import numpy as np

from .features import ACTIONS, DIR_HERE, DIR_NONE, observe

N_ACTIONS = len(ACTIONS)
FEATURE_VERSION = 1

# --- categorical component sizes -------------------------------------------
# Four neighbours, each one of {blocked, free-safe, free-lethal}.
N_MOVE_STATUS = 4 * 3
N_T_HERE = 5          # 0..4
N_TARGET_DIR = 6      # none, 4 directions, "already there"
N_TARGET_KIND = 2      # crate, coin
N_ESCAPE_DIR = 6      # none, 4 directions, "staying put is safe"

# One normalised scalar (target_dist / cap) plus a bias term.
N_CONTINUOUS = 1
N_BIAS = 1

PHI_DIM = (N_MOVE_STATUS + N_T_HERE + N_TARGET_DIR + N_TARGET_KIND
          + N_ESCAPE_DIR + N_CONTINUOUS + N_BIAS)

_TARGET_DIST_CAP = 15.0


# --- D4, reused from tabular_q's model.py -----------------------------------
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
IDENTITY = (0, 1, 2, 3)


def _direction_index(perm, value):
    """Relabel one direction-valued field; DIR_NONE and DIR_HERE carry no direction."""
    return value if value in (DIR_NONE, DIR_HERE) else perm[value - 1] + 1


def _tie_break_key(perm, obs):
    """A total order over the eight relabelled views, for picking a canonical one.

    Only used to compare views against each other - never stored, never used as
    a table index - so any encoding that is total and deterministic works. This
    one mirrors tabular_q's mixed-radix order exactly, which keeps the two
    agents' notion of "canonical" identical and makes the D4 unit tests
    transferable between them almost unchanged.
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
    """The D4 element reaching the canonical view of `obs`, and that view itself.

    Returns (perm, relabelled_status, relabelled_target_dir, relabelled_escape_dir).
    The caller vectorises the relabelled fields directly; `to_frame`/`from_frame`
    translate a chosen action into and out of that frame, exactly as in
    tabular_q.
    """
    best_key, best_perm = None, None
    for perm in D4:
        key = _tie_break_key(perm, obs)
        if best_key is None or key < best_key:
            best_key, best_perm = key, perm
    status = [0] * 4
    for i in range(4):
        status[best_perm[i]] = obs.move_status[i]
    target_dir = _direction_index(best_perm, obs.target_dir)
    escape_dir = _direction_index(best_perm, obs.escape_dir)
    return best_perm, tuple(status), target_dir, escape_dir


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


# --- vectorisation -----------------------------------------------------

def _one_hot(index, size, out, offset):
    out[offset + index] = 1.0
    return offset + size


def vectorize(move_status, t_here, target_dir, target_kind, escape_dir, target_dist):
    """Pack the (already framed) categorical fields into phi(s)."""
    phi = np.zeros(PHI_DIM, dtype=np.float64)
    # Each neighbour gets its own 3-wide block, not a shared one: neighbour i
    # being blocked is a different fact from neighbour j being blocked, and a
    # single shared one-hot would conflate "one side is open" across sides.
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
    phi[offset] = 1.0  # bias
    offset += N_BIAS
    assert offset == PHI_DIM
    return phi


def observe_and_encode(game_state, use_symmetry):
    """The observation, its feature vector, and the frame it was vectorised in.

    Mirrors tabular_q's function of the same name and same contract: not
    memoised, for the same reason (Phase 25.4 in the tabular_q report - the
    framework labels the state before and after an action with the same step
    number, so caching on it returns a stale danger schedule).
    """
    obs = observe(game_state)
    if use_symmetry:
        perm, status, target_dir, escape_dir = canonical_frame(obs)
    else:
        perm = IDENTITY
        status, target_dir, escape_dir = obs.move_status, obs.target_dir, obs.escape_dir
    phi = vectorize(status, obs.t_here, target_dir, obs.target_kind, escape_dir,
                    obs.target_dist)
    return phi, obs, perm


class LinearQModel:
    def __init__(self, feature_flags=None):
        self.phi_dim = PHI_DIM
        self.feature_version = FEATURE_VERSION
        self.feature_flags = dict(feature_flags or {})
        self.w = np.zeros((PHI_DIM, N_ACTIONS), dtype=np.float64)

    def values(self, phi):
        return phi @ self.w

    def update(self, phi, action, target, alpha):
        """One semi-gradient TD step: w[:, a] += alpha * delta * phi.

        Every feature active in `phi` moves, not one table cell - which is the
        entire point of function approximation and also the entire mechanism by
        which a bad step size or a bad feature scale can make Q diverge, since
        a single update now touches every state that shares a feature.
        """
        delta = target - float(phi @ self.w[:, action])
        self.w[:, action] += alpha * delta * phi

    def save(self, path):
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path):
        with open(path, "rb") as fh:
            model = pickle.load(fh)
        if model.phi_dim != PHI_DIM or model.feature_version != FEATURE_VERSION:
            raise ValueError(
                "model at %s was trained with phi_dim/version (%d, %d) but this "
                "code expects (%d, %d); loading it would silently mix incompatible "
                "weight columns" % (path, model.phi_dim, model.feature_version,
                                    PHI_DIM, FEATURE_VERSION))
        return model
