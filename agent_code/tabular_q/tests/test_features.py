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

from agent_code.tabular_q.features import (  # noqa: E402
    BLOCKED, DIR_HERE, DIR_NONE, FREE_SAFE, KIND_COIN, KIND_CRATE, Board,
    observe)
from agent_code.tabular_q.model import N_STATES, encode  # noqa: E402


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


def test_move_status_marks_walls_blocked():
    obs = observe(make_state(open_field(), (1, 1)))
    # At the top-left interior corner, UP and LEFT are the wall ring.
    assert obs.move_status == (BLOCKED, FREE_SAFE, FREE_SAFE, BLOCKED)


def test_move_status_marks_crates_blocked():
    field = open_field()
    field[2, 1] = 1
    obs = observe(make_state(field, (1, 1)))
    assert obs.move_status[1] == BLOCKED


def test_move_status_marks_armed_bombs_blocked():
    """A tile holding a live bomb cannot be walked onto."""
    field = open_field()
    obs = observe(make_state(field, (4, 4), bombs=[((5, 4), 3)]))
    assert obs.move_status[1] == BLOCKED


def test_target_dir_points_along_shortest_path():
    obs = observe(make_state(open_field(), (3, 3), coins=[(5, 3)]))
    assert obs.target_dir == 2          # RIGHT, shifted by one
    assert obs.target_kind == KIND_COIN
    assert obs.target_dist == 2


def test_target_dir_is_none_on_an_empty_board():
    obs = observe(make_state(open_field(), (3, 3)))
    assert obs.target_dir == DIR_NONE


def test_target_dir_routes_around_a_wall():
    """The straight-line direction is wrong here; the BFS must not be."""
    field = open_field()
    field[4, 1:8] = -1                  # wall column right of the agent
    obs = observe(make_state(field, (3, 3), coins=[(5, 3)]))
    assert obs.target_dir == DIR_NONE   # coin is walled off entirely

    field[4, 6] = 0                     # open one gap low down
    obs = observe(make_state(field, (3, 3), coins=[(5, 3)]))
    assert obs.target_dir == 3          # DOWN towards the gap, not RIGHT


def test_target_dir_prefers_the_nearer_coin():
    field = open_field(11, 11)
    assert observe(make_state(field, (5, 5), coins=[(5, 2), (5, 9)])).target_dir == 1
    assert observe(make_state(field, (5, 5), coins=[(5, 1), (5, 6)])).target_dir == 3


def test_crate_is_a_target_when_no_coin_is_reachable():
    field = open_field()
    field[6, 4] = 1
    obs = observe(make_state(field, (2, 4)))
    assert obs.target_kind == KIND_CRATE
    assert obs.target_dir == 2          # RIGHT, towards the crate


def test_standing_next_to_a_crate_reports_here():
    field = open_field()
    field[5, 4] = 1
    obs = observe(make_state(field, (4, 4)))
    assert obs.target_dir == DIR_HERE
    assert obs.target_kind == KIND_CRATE


def test_a_reachable_coin_beats_a_nearer_crate_only_when_closer():
    """Coins and crates share one search, so the nearer goal wins."""
    field = open_field(11, 11)
    field[9, 5] = 1                     # crate far to the right
    obs = observe(make_state(field, (5, 5), coins=[(6, 5)]))
    assert obs.target_kind == KIND_COIN


def test_target_search_ignores_paths_through_a_blast():
    """A coin reachable only through a tile that burns now is not routed to."""
    field = open_field(11, 11)
    field[:, 4] = -1
    field[:, 6] = -1
    field[5, 4] = 0                     # single gap north at x=5
    plain = observe(make_state(field, (3, 5), coins=[(5, 2)]))
    assert plain.target_dir == 2        # RIGHT, through the gap

    blocked = observe(make_state(field, (3, 5), coins=[(5, 2)],
                                 bombs=[((5, 5), 0)]))
    assert blocked.target_dir != 2


def test_encode_is_within_the_table():
    field = open_field()
    for pos in [(1, 1), (3, 3), (5, 5)]:
        for coins in [(), ((2, 2),), ((5, 1),)]:
            for bombs in [(), (((4, 4), 2),)]:
                index = encode(observe(make_state(field, pos, coins, bombs)))
                assert 0 <= index < N_STATES


def test_encode_is_injective_on_its_components():
    """Different feature tuples must not collide onto the same row."""
    seen = {}
    field = open_field()
    field[6, 6] = 1
    for pos in [(1, 1), (1, 3), (3, 3), (5, 5), (3, 1), (5, 6)]:
        for coins in [(), ((2, 2),), ((5, 1),), ((1, 5),)]:
            for bombs in [(), (((4, 4), 2),), (((2, 2), 0),)]:
                obs = observe(make_state(field, pos, coins, bombs))
                key = (obs.move_status, obs.t_here, obs.target_dir,
                       obs.target_kind, obs.escape_dir)
                index = encode(obs)
                if index in seen:
                    assert seen[index] == key
                seen[index] = key


def test_board_reports_here_when_standing_on_a_coin():
    board = Board(make_state(open_field(), (3, 3), coins=[(3, 3)]))
    assert board.target_search([(3, 3)]) == (DIR_HERE, KIND_COIN, 0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
