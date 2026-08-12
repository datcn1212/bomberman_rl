"""Unit tests for the state encoding.

The encoding is the part of a tabular agent that can be wrong without ever
crashing, so it is tested directly on hand-built boards rather than only through
training scores.

Run from the repository root:  python3 -m pytest agent_code/tabular_q/tests -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from agent_code.tabular_q.features import DIR_NONE, Board, observe  # noqa: E402
from agent_code.tabular_q.model import N_STATES, encode  # noqa: E402


def make_state(field, pos, coins=()):
    """Build the minimal game_state dict the features read."""
    return {
        "field": field,
        "self": ("me", 0, True, pos),
        "coins": list(coins),
        "bombs": [],
        "explosion_map": np.zeros_like(field),
        "others": [],
        "step": 1,
        "round": 1,
    }


def open_field(w=7, h=7):
    """Free interior surrounded by a wall ring."""
    field = np.zeros((w, h), dtype=int)
    field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = -1
    return field


def test_move_status_marks_walls_blocked():
    field = open_field()
    obs = observe(make_state(field, (1, 1)))
    # At the top-left interior corner, UP and LEFT are the wall ring.
    assert obs.move_status == (0, 1, 1, 0)


def test_move_status_marks_crates_blocked():
    field = open_field()
    field[2, 1] = 1  # crate to the right
    obs = observe(make_state(field, (1, 1)))
    assert obs.move_status[1] == 0


def test_target_dir_points_along_shortest_path():
    field = open_field()
    obs = observe(make_state(field, (3, 3), coins=[(5, 3)]))
    assert obs.target_dir == 2  # RIGHT, shifted by one
    assert obs.target_dist == 2


def test_target_dir_is_none_without_coins():
    obs = observe(make_state(open_field(), (3, 3)))
    assert obs.target_dir == DIR_NONE
    assert obs.target_dist == -1


def test_target_dir_routes_around_a_wall():
    """Straight-line direction would be wrong here, BFS must not be."""
    field = open_field()
    field[4, 1:6] = -1        # vertical wall to the right of the agent
    field[4, 3] = -1
    obs = observe(make_state(field, (3, 3), coins=[(5, 3)]))
    # The coin is walled off on this board, so nothing is reachable.
    assert obs.target_dir == DIR_NONE

    field[4, 5] = 0           # open one gap near the bottom
    obs = observe(make_state(field, (3, 3), coins=[(5, 3)]))
    assert obs.target_dir == 3  # DOWN, towards the gap, not RIGHT


def test_target_dir_prefers_the_nearer_coin():
    field = open_field(9, 9)
    obs = observe(make_state(field, (4, 4), coins=[(4, 1), (4, 7)]))
    assert obs.target_dir == 1  # UP: distance 3 versus 3 downward is a tie...
    # ... so make it unambiguous instead of asserting a tie-break.
    obs = observe(make_state(field, (4, 4), coins=[(4, 2), (4, 7)]))
    assert obs.target_dir == 1
    obs = observe(make_state(field, (4, 4), coins=[(4, 1), (4, 6)]))
    assert obs.target_dir == 3


def test_encode_is_within_the_table():
    field = open_field()
    for pos in [(1, 1), (3, 3), (5, 5)]:
        for coins in [(), ((2, 2),), ((5, 1),)]:
            index = encode(observe(make_state(field, pos, coins)))
            assert 0 <= index < N_STATES


def test_encode_is_injective_on_its_components():
    """Different feature tuples must not collide onto the same row."""
    seen = {}
    field = open_field()
    for pos in [(1, 1), (1, 3), (3, 3), (5, 5), (3, 1)]:
        for coins in [(), ((2, 2),), ((5, 1),), ((1, 5),), ((5, 5),)]:
            obs = observe(make_state(field, pos, coins))
            key = (obs.move_status, obs.target_dir)
            index = encode(obs)
            if index in seen:
                assert seen[index] == key
            seen[index] = key


def test_board_reports_distance_zero_when_standing_on_a_coin():
    board = Board(make_state(open_field(), (3, 3), coins=[(3, 3)]))
    assert board.target_search(board.coins) == (DIR_NONE, 0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
