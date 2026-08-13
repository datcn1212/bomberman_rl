"""Unit tests for the D4 canonicalisation added in Phase 12.

Folding a state onto its orbit representative also relabels the directions, so
the action the agent finally plays goes through two translations. Getting either
one wrong produces an agent that runs the right policy in the wrong frame -
which does not crash and does not look like a bug, it just plays badly.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from agent_code.tabular_q.features import ACTIONS, observe  # noqa: E402
from agent_code.tabular_q.model import (  # noqa: E402
    D4, IDENTITY, canonical, encode, from_frame, invert, observe_and_encode,
    to_frame)


def make_state(field, pos, coins=(), bombs=()):
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
    field = np.zeros((w, h), dtype=int)
    field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = -1
    return field


def rotate_field(field):
    """Rotate the board 90 degrees clockwise in (x, y) index space."""
    w, h = field.shape
    out = np.zeros((h, w), dtype=field.dtype)
    for x in range(w):
        for y in range(h):
            out[h - 1 - y, x] = field[x, y]
    return out


def rotate_point(pos, height):
    x, y = pos
    return (height - 1 - y, x)


def test_d4_is_a_group_of_eight():
    assert len({tuple(p) for p in D4}) == 8
    members = {tuple(p) for p in D4}
    for p in D4:
        for q in D4:
            assert tuple(p[q[i]] for i in range(4)) in members


def test_inverse_undoes_the_permutation():
    for p in D4:
        inv = invert(p)
        assert tuple(inv[p[i]] for i in range(4)) == IDENTITY


def test_action_translation_round_trips():
    for p in D4:
        for action in range(len(ACTIONS)):
            assert from_frame(p, to_frame(p, action)) == action


def test_wait_and_bomb_are_frame_independent():
    for p in D4:
        for action in (ACTIONS.index("WAIT"), ACTIONS.index("BOMB")):
            assert to_frame(p, action) == action
            assert from_frame(p, action) == action


def test_orbit_sizes_divide_eight():
    """Orbit-stabiliser: an orbit under a group of order 8 has size 1, 2, 4 or 8."""
    field = open_field()
    field[5, 4] = 1
    seen = set()
    for pos in [(2, 2), (4, 4), (1, 1), (4, 2)]:
        for coins in [(), ((6, 4),), ((4, 6),), ((2, 4),)]:
            for bombs in [(), (((4, 2), 2),), (((2, 4), 1),)]:
                obs = observe(make_state(field, pos, coins, bombs))
                orbit = {encode(obs)}
                index, _ = canonical(obs)
                seen.add(index)
                # Rebuild the orbit through the raw relabelling.
                from agent_code.tabular_q.model import _relabel
                orbit = {_relabel(p, obs) for p in D4}
                assert 8 % len(orbit) == 0, len(orbit)
    assert seen


def test_rotated_board_maps_to_the_same_canonical_row():
    """The point of the whole exercise: one situation, one row."""
    field = open_field()
    field[6, 4] = 1                      # crate to the east
    state = make_state(field, (4, 4), coins=[(4, 2)])
    index, _ = canonical(observe(state))

    rotated_field = field
    pos = (4, 4)
    coin = (4, 2)
    for _ in range(3):
        height = rotated_field.shape[1]
        pos = rotate_point(pos, height)
        coin = rotate_point(coin, height)
        rotated_field = rotate_field(rotated_field)
        rotated = make_state(rotated_field, pos, coins=[coin])
        assert canonical(observe(rotated))[0] == index


def test_rotating_the_board_rotates_the_recovered_action():
    """A canonical action must translate back to the physically equal move."""
    field = open_field()
    state = make_state(field, (4, 4), coins=[(6, 4)])
    obs = observe(state)
    _, perm = canonical(obs)

    rotated_field = rotate_field(field)
    height = field.shape[1]
    rotated = make_state(rotated_field, rotate_point((4, 4), height),
                         coins=[rotate_point((6, 4), height)])
    _, perm_rot = canonical(observe(rotated))

    # RIGHT on the original board is DOWN after one clockwise rotation.
    right, down = ACTIONS.index("RIGHT"), ACTIONS.index("DOWN")
    assert from_frame(perm, to_frame(perm, right)) == right
    canonical_action = to_frame(perm, right)
    assert from_frame(perm_rot, canonical_action) == down


def test_symmetry_off_is_the_plain_encoding():
    state = make_state(open_field(), (4, 4), coins=[(6, 4)])
    index, obs, perm = observe_and_encode(state, use_symmetry=False)
    assert perm == IDENTITY
    assert index == encode(obs)


def test_canonical_index_is_never_larger_than_the_plain_one():
    field = open_field()
    field[5, 4] = 1
    for pos in [(2, 2), (4, 4), (4, 2), (6, 6)]:
        obs = observe(make_state(field, pos, coins=[(6, 4)]))
        assert canonical(obs)[0] <= encode(obs)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
