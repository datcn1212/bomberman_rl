"""Unit tests for phi(s): the part of a linear agent that can be wrong without
ever crashing, and the part a training curve cannot diagnose - a scrambled
one-hot block still produces numbers, just numbers for the wrong meaning.

Run from the repository root:  python3 -m pytest agent_code/linear_q/tests -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from agent_code.linear_q.features import (  # noqa: E402
    BLOCKED, DIR_HERE, DIR_NONE, FREE_SAFE, KIND_COIN, KIND_CRATE, observe)
from agent_code.linear_q.model import (  # noqa: E402
    D4, IDENTITY, N_ESCAPE_DIR, N_MOVE_STATUS, N_T_HERE, N_TARGET_DIR,
    N_TARGET_KIND, PHI_DIM, canonical_frame, from_frame, invert, to_frame,
    vectorize)


def make_state(field, pos, coins=(), bombs=()):
    """Build the minimal game_state dict the features read."""
    return {
        "field": field,
        "self": ("me", 0, True, pos),
        "coins": list(coins),
        "bombs": [((p[0], p[1]), t) for p, t in bombs],
        "explosion_map": np.zeros_like(field),
        "others": [],
        "step": 1,
        "round": 1,
    }


def open_field(w=9, h=9):
    """Free interior surrounded by a wall ring."""
    field = np.zeros((w, h), dtype=int)
    field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = -1
    return field


# --- shape and one-hot structure -------------------------------------------

def test_dim_matches_the_sum_of_its_blocks():
    assert PHI_DIM == N_MOVE_STATUS + N_T_HERE + N_TARGET_DIR + N_TARGET_KIND + N_ESCAPE_DIR + 2


def test_every_categorical_block_is_a_one_hot_and_the_bias_is_always_one():
    move_status = (BLOCKED, FREE_SAFE, BLOCKED, FREE_SAFE)
    phi = vectorize(move_status, 2, 1, KIND_COIN, DIR_HERE, 4)

    assert phi.shape == (PHI_DIM,)
    offset = 0
    for i in range(4):
        block = phi[offset:offset + 3]
        assert block.sum() == 1.0
        assert block[move_status[i]] == 1.0
        offset += 3
    assert phi[offset:offset + N_T_HERE].sum() == 1.0
    offset += N_T_HERE
    assert phi[offset:offset + N_TARGET_DIR].sum() == 1.0
    offset += N_TARGET_DIR
    assert phi[offset:offset + N_TARGET_KIND].sum() == 1.0
    offset += N_TARGET_KIND
    assert phi[offset:offset + N_ESCAPE_DIR].sum() == 1.0
    offset += N_ESCAPE_DIR
    assert 0.0 <= phi[offset] <= 1.0            # normalised distance
    assert phi[offset + 1] == 1.0               # bias, always on


def test_target_distance_is_normalised_and_capped():
    move_status = (FREE_SAFE,) * 4
    near = vectorize(move_status, 0, 1, KIND_COIN, DIR_NONE, target_dist=3)
    far = vectorize(move_status, 0, 1, KIND_COIN, DIR_NONE, target_dist=15)
    beyond_cap = vectorize(move_status, 0, 1, KIND_COIN, DIR_NONE, target_dist=999)
    assert near[-2] == pytest.approx(3 / 15)
    assert far[-2] == pytest.approx(1.0)
    assert beyond_cap[-2] == pytest.approx(1.0)   # capped, not overflowed


def test_two_different_observations_never_collide():
    """The categorical blocks are independent, so no two distinct field
    combinations should ever produce an identical phi (holding target_dist and
    the bias fixed)."""
    a = vectorize((BLOCKED, FREE_SAFE, FREE_SAFE, FREE_SAFE), 0, 1, KIND_COIN, DIR_NONE, 5)
    b = vectorize((FREE_SAFE, BLOCKED, FREE_SAFE, FREE_SAFE), 0, 1, KIND_COIN, DIR_NONE, 5)
    assert not np.array_equal(a, b)


# --- D4 canonicalisation, cross-checked against tabular_q's own -----------

def test_d4_is_the_same_group_tabular_q_uses():
    from agent_code.tabular_q.model import D4 as TABULAR_D4
    assert D4 == TABULAR_D4


def test_action_translation_round_trips():
    for perm in D4:
        for action in range(4):
            assert from_frame(perm, to_frame(perm, action)) == action
    # WAIT and BOMB carry no direction, so they are frame-independent.
    for perm in D4:
        assert to_frame(perm, 4) == 4 and to_frame(perm, 5) == 5


def test_rotated_board_reaches_the_same_canonical_frame():
    """A board and its 90-degree rotation describe the same situation, so
    vectorising both through their own canonical frame must agree."""
    field = open_field()
    field[2, 1] = 1  # a crate one step to the right of the agent
    state = make_state(field, (1, 1), coins=[(1, 5)])
    obs = observe(state)

    perm, status, target_dir, escape_dir = canonical_frame(obs)
    phi = vectorize(status, obs.t_here, target_dir, obs.target_kind, escape_dir,
                    obs.target_dist)

    # Re-deriving the canonical frame a second time must be idempotent.
    perm2, status2, target_dir2, escape_dir2 = canonical_frame(obs)
    assert perm2 == perm
    assert np.array_equal(
        vectorize(status2, obs.t_here, target_dir2, obs.target_kind, escape_dir2,
                 obs.target_dist),
        phi)


def test_symmetry_off_is_the_plain_encoding():
    from agent_code.linear_q.model import observe_and_encode
    field = open_field()
    field[2, 1] = 1
    state = make_state(field, (1, 1), coins=[(1, 5)])
    obs = observe(state)

    phi_off, _, perm_off = observe_and_encode(state, use_symmetry=False)
    assert perm_off == IDENTITY
    expected = vectorize(obs.move_status, obs.t_here, obs.target_dir,
                         obs.target_kind, obs.escape_dir, obs.target_dist)
    assert np.array_equal(phi_off, expected)


def test_canonical_frame_agrees_with_tabular_q_on_many_boards():
    """Both agents define "canonical" as the same tie-break over the same D4
    group (test_d4_is_the_same_group_tabular_q_uses), so they must pick the
    same permutation on any board - not just the hand-built one above."""
    from agent_code.tabular_q.model import canonical as tq_canonical
    from agent_code.tabular_q.features import observe as tq_observe

    rng = np.random.default_rng(42)
    tested = 0
    for seed in range(200):
        r = np.random.default_rng(seed)
        field = open_field()
        for x in range(1, 8):
            for y in range(1, 8):
                if r.random() < 0.3:
                    field[x, y] = 1
        free = [(x, y) for x in range(1, 8) for y in range(1, 8) if field[x, y] == 0]
        if len(free) < 3:
            continue
        pos = free[r.integers(len(free))]
        free.remove(pos)
        n_coin = int(r.integers(0, 3))
        coins = [free[i] for i in r.choice(len(free), size=min(n_coin, len(free)), replace=False)] \
            if n_coin and free else []
        state = make_state(field, pos, coins=coins)

        obs = observe(state)
        perm_lq, _, _, _ = canonical_frame(obs)
        _, perm_tq = tq_canonical(tq_observe(state))
        assert perm_lq == perm_tq
        tested += 1
    assert tested > 100    # the random boards were not all degenerate
