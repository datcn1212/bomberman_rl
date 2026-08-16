"""Unit tests for the danger model added in Phase 3.

The timing constants here are the ones measured in Phase 0, not the ones read
off settings.py, so these tests fail if the game's behaviour and our model of it
ever drift apart.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import settings as s  # noqa: E402
from agent_code.tabular_q.features import (  # noqa: E402
    BLOCKED, BOMB_NONE, BOMB_POINTLESS, BOMB_TRAPPED, BOMB_USEFUL, DIR_HERE,
    DIR_NONE, FLAGS, FREE_LETHAL, FREE_SAFE, HIT_CRATE, HIT_NONE,
    HIT_OPPONENT, HIT_POINTLESS, HORIZON, OPP_FAR, OPP_NEAR, OPP_NONE,
    SAFE_NONE, SAFE_ROBUST, SAFE_TIGHT, SAFE_TRAPPED, Board, blast_tiles,
    danger_schedule, observe)


@pytest.fixture
def bomb_opt_on():
    """bomb_opt is off by default (Phase 5); these tests exercise the computation."""
    FLAGS["use_bomb_opt"] = True
    yield
    FLAGS["use_bomb_opt"] = False


def open_field(w=9, h=9):
    field = np.zeros((w, h), dtype=int)
    field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = -1
    return field


def make_state(field, pos, coins=(), bombs=(), explosion=None, others=()):
    return {
        "field": field,
        "self": ("me", 0, True, pos),
        "coins": list(coins),
        "bombs": [((p[0], p[1]), t) for p, t in bombs],
        "explosion_map": np.zeros_like(field) if explosion is None else explosion,
        "others": list(others),
        "step": 1,
        "round": 1,
    }


def test_blast_reaches_bomb_power_and_stops_at_walls():
    field = open_field()
    tiles = set(blast_tiles(field, 4, 4))
    for i in range(1, s.BOMB_POWER + 1):
        assert (4 + i, 4) in tiles
        assert (4, 4 + i) in tiles
    assert (4 + s.BOMB_POWER + 1, 4) not in tiles

    field[6, 4] = -1
    tiles = set(blast_tiles(field, 4, 4))
    assert (5, 4) in tiles
    assert (6, 4) not in tiles      # the wall itself does not burn
    assert (7, 4) not in tiles      # and the ray does not continue past it


def test_blast_does_not_turn_corners():
    field = open_field()
    tiles = set(blast_tiles(field, 4, 4))
    off_axis = [(x, y) for x, y in tiles if x != 4 and y != 4]
    assert off_axis == []


def test_crates_burn_but_do_not_shield():
    field = open_field()
    field[5, 4] = 1
    tiles = set(blast_tiles(field, 4, 4))
    assert (5, 4) in tiles
    assert (6, 4) in tiles


def test_schedule_marks_the_two_lethal_steps_measured_in_phase_0():
    """A bomb with observed timer t kills on decisions t and t+1, not before."""
    field = open_field()
    for timer in range(s.BOMB_TIMER):
        lethal = danger_schedule(make_state(field, (1, 1), bombs=[((4, 4), timer)]))
        for k in range(HORIZON + 1):
            expected = k in (timer, timer + 1)
            assert bool(lethal[k][4, 4]) == expected, (timer, k)


def test_existing_explosion_is_lethal_only_now():
    field = open_field()
    explosion = np.zeros_like(field)
    explosion[4, 4] = 1
    lethal = danger_schedule(make_state(field, (1, 1), explosion=explosion))
    assert lethal[0][4, 4]
    assert not lethal[1][4, 4]


def test_hypothetical_bomb_is_lethal_four_and_five_steps_out():
    """Measured: a bomb dropped now appears next step with timer 3."""
    field = open_field()
    lethal = danger_schedule(make_state(field, (4, 4)), extra_bomb=(4, 4))
    assert not lethal[3][4, 4]
    assert lethal[s.BOMB_TIMER][4, 4]
    assert lethal[s.BOMB_TIMER + 1][4, 4]


def test_escape_reports_here_when_nothing_threatens():
    board = Board(make_state(open_field(), (4, 4)))
    assert board.escape_search() == DIR_HERE


def test_escape_finds_the_side_corridor():
    """Agent in a horizontal corridor with its own bomb; only a side gap saves it."""
    field = open_field(11, 11)
    field[:, 1] = -1
    field[:, 3] = -1
    field[1:10, 2] = 0          # horizontal corridor at y = 2
    field[5, 3] = 0             # one gap leading south
    field[5, 4] = 0
    state = make_state(field, (3, 2), bombs=[((1, 2), 3)])
    board = Board(state)
    assert board.escape_search() == 2   # RIGHT, towards the gap


def test_escape_returns_none_when_trapped():
    """Dead-end corridor with a bomb at its mouth: nothing survives."""
    field = open_field(11, 11)
    field[:, 1] = -1
    field[:, 3] = -1
    field[6:, 2] = -1           # seal the corridor: a real dead end, x = 1..5
    state = make_state(field, (4, 2), bombs=[((2, 2), 3)])
    board = Board(state)
    assert board.escape_search() == DIR_NONE


def test_escape_may_cross_a_tile_that_burns_later():
    """The point of searching (tile, step) pairs rather than tiles."""
    field = open_field(11, 11)
    state = make_state(field, (4, 4), bombs=[((4, 7), 3)])
    board = Board(state)
    # (4,6) burns at k=3 but is free at k=1, so a path through it is legal.
    assert board.lethal_at((4, 6), 3)
    assert not board.lethal_at((4, 6), 1)
    assert board.escape_search() != DIR_NONE


def test_move_status_separates_blocked_from_lethal():
    field = open_field()
    field[3, 4] = -1                                  # wall to the left
    obs = observe(make_state(field, (4, 4), bombs=[((4, 2), 1)]))
    assert obs.move_status[3] == BLOCKED              # LEFT
    assert obs.move_status[0] == FREE_LETHAL          # UP, into the blast
    assert obs.move_status[1] == FREE_SAFE            # RIGHT


def test_t_here_counts_down_to_the_blast():
    field = open_field()
    for timer, expected in [(0, 1), (1, 2), (2, 3), (3, 4)]:
        obs = observe(make_state(field, (4, 4), bombs=[((4, 4), timer)]))
        assert obs.t_here == expected, timer
    assert observe(make_state(field, (4, 4))).t_here == 0


def test_bomb_option_none_without_a_bomb(bomb_opt_on):
    field = open_field()
    field[5, 4] = 1
    state = make_state(field, (4, 4))
    state["self"] = ("me", 0, False, (4, 4))     # bombs_left = False
    assert observe(state).bomb_opt == BOMB_NONE


def test_bomb_option_pointless_when_nothing_is_in_range(bomb_opt_on):
    """The gap Phase 3 exposed: adjacent to a crate diagonally is not in range."""
    field = open_field()
    field[5, 5] = 1                              # crate on the diagonal
    obs = observe(make_state(field, (4, 4)))
    assert obs.bomb_opt == BOMB_POINTLESS


def test_bomb_option_useful_when_a_crate_is_in_range_and_escape_exists(bomb_opt_on):
    field = open_field()
    field[5, 4] = 1
    obs = observe(make_state(field, (4, 4)))
    assert obs.bomb_opt == BOMB_USEFUL


def test_bomb_option_trapped_in_a_dead_end(bomb_opt_on):
    """Bombing at the end of a short dead end leaves nowhere to run."""
    field = open_field(11, 11)
    field[:, 1] = -1
    field[:, 3] = -1
    field[4:, 2] = -1                            # corridor is x = 1..3 only
    field[3, 2] = 1                              # crate sealing the end
    obs = observe(make_state(field, (2, 2)))
    assert obs.bomb_opt == BOMB_TRAPPED


def test_bomb_option_counts_crates_through_other_crates():
    """A blast destroys every crate on its ray, not just the first."""
    field = open_field(11, 11)
    field[5, 4] = 1
    field[6, 4] = 1
    board = Board(make_state(field, (4, 4)))
    assert board.bomb_payload() == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


@pytest.fixture
def opponent_blocking_on():
    FLAGS["use_opponent_blocking"] = True
    yield
    FLAGS["use_opponent_blocking"] = False


def other(pos):
    return ("them", 0, True, pos)


def test_opponent_blocks_the_immediate_move(opponent_blocking_on):
    field = open_field()
    obs = observe(make_state(field, (4, 4), others=[other((5, 4))]))
    assert obs.move_status[1] == BLOCKED          # RIGHT is occupied
    assert obs.move_status[3] == FREE_SAFE        # LEFT is not


def test_opponent_is_ignored_without_the_flag():
    """The default has to stay exactly as it was, or every earlier result moves."""
    field = open_field()
    obs = observe(make_state(field, (4, 4), others=[other((5, 4))]))
    assert obs.move_status[1] == FREE_SAFE


def test_escape_does_not_route_through_a_body(opponent_blocking_on):
    """The failure this fixes: a corridor whose only exit is occupied."""
    field = open_field(11, 11)
    field[:, 1] = -1
    field[:, 3] = -1
    field[6:, 2] = -1                             # dead end, x = 1..5
    state = make_state(field, (4, 2), bombs=[((2, 2), 3)])
    assert Board(state).escape_search() == DIR_NONE

    # Same board, bomb further away so an escape east exists...
    field2 = open_field(11, 11)
    field2[:, 1] = -1
    field2[:, 3] = -1
    state2 = make_state(field2, (4, 2), bombs=[((1, 2), 3)])
    assert Board(state2).escape_search() == 2     # RIGHT

    # ... and now an opponent stands in it.
    state3 = make_state(field2, (4, 2), bombs=[((1, 2), 3)],
                        others=[other((5, 2))])
    assert Board(state3).escape_search() != 2


def test_opponents_do_not_block_beyond_the_first_step(opponent_blocking_on):
    """They move too, so treating a body as a permanent wall over-reports traps."""
    field = open_field(11, 11)
    state = make_state(field, (4, 4), bombs=[((4, 4), 3)],
                       others=[other((7, 4))])
    board = Board(state)
    # The tile two steps past the opponent is still reachable in the search.
    assert board.walkable((7, 4), now=True) is False
    assert board.walkable((7, 4)) is True


@pytest.fixture
def bomb_safety_on():
    FLAGS["use_bomb_safety"] = True
    yield
    FLAGS["use_bomb_safety"] = False


def test_bomb_safety_reports_no_bomb(bomb_safety_on):
    field = open_field()
    state = make_state(field, (4, 4))
    state["self"] = ("me", 0, False, (4, 4))
    assert observe(state).bomb_opt == SAFE_NONE


def test_bomb_safety_open_board_is_robust(bomb_safety_on):
    """In the open there are several ways out, so bombing is low risk."""
    obs = observe(make_state(open_field(11, 11), (5, 5)))
    assert obs.bomb_opt == SAFE_ROBUST


def test_bomb_safety_dead_end_is_trapped(bomb_safety_on):
    field = open_field(11, 11)
    field[:, 1] = -1
    field[:, 3] = -1
    field[4:, 2] = -1                       # corridor x = 1..3 only
    assert observe(make_state(field, (2, 2))).bomb_opt == SAFE_TRAPPED


def test_bomb_safety_single_corridor_is_tight(bomb_safety_on):
    """One way out is survivable but an opponent can seal it."""
    field = open_field(13, 13)
    field[:, 1] = -1
    field[:, 3] = -1                        # long corridor at y = 2
    obs = observe(make_state(field, (2, 2)))
    assert obs.bomb_opt == SAFE_TIGHT


def test_bomb_safety_counts_routes_not_just_existence(bomb_safety_on):
    field = open_field(13, 13)
    field[:, 1] = -1
    field[:, 3] = -1
    tight = Board(make_state(field, (2, 2)), extra_bomb=(2, 2)).escape_routes()
    field[2, 3] = 0                         # open a second way out
    field[2, 4] = 0
    roomier = Board(make_state(field, (2, 2)), extra_bomb=(2, 2)).escape_routes()
    assert roomier > tight


@pytest.fixture
def opponent_distance_on():
    FLAGS["use_opponent_distance"] = True
    FLAGS["use_opponent_blocking"] = True
    yield
    FLAGS["use_opponent_distance"] = False
    FLAGS["use_opponent_blocking"] = False


@pytest.fixture
def bomb_hits_on():
    FLAGS["use_bomb_hits"] = True
    yield
    FLAGS["use_bomb_hits"] = False


def test_opponent_state_none_when_alone(opponent_distance_on):
    assert observe(make_state(open_field(13, 13), (6, 6))).opponent == OPP_NONE


def test_opponent_state_near_and_far(opponent_distance_on):
    field = open_field(15, 15)
    near = observe(make_state(field, (6, 6), others=[other((6, 9))]))
    far = observe(make_state(field, (2, 2), others=[other((12, 12))]))
    assert near.opponent == OPP_NEAR      # three steps
    assert far.opponent == OPP_FAR        # twenty steps


def test_opponent_distance_is_walking_distance_not_straight_line(opponent_distance_on):
    """An opponent just across a wall is not two steps away."""
    field = open_field(15, 15)
    field[:, 7] = -1                       # full wall between the two halves
    obs = observe(make_state(field, (6, 6), others=[other((6, 8))]))
    assert obs.opponent == OPP_FAR


def test_bomb_hits_separates_opponent_from_crate(bomb_hits_on):
    field = open_field(13, 13)
    field[7, 6] = 1
    assert observe(make_state(field, (6, 6))).bomb_opt == HIT_CRATE

    plain = open_field(13, 13)
    assert observe(make_state(plain, (6, 6),
                              others=[other((8, 6))])).bomb_opt == HIT_OPPONENT


def test_bomb_hits_pointless_when_it_reaches_neither(bomb_hits_on):
    field = open_field(13, 13)
    assert observe(make_state(field, (6, 6))).bomb_opt == HIT_POINTLESS


def test_bomb_hits_reports_no_bomb(bomb_hits_on):
    field = open_field(13, 13)
    field[7, 6] = 1
    state = make_state(field, (6, 6))
    state["self"] = ("me", 0, False, (6, 6))
    assert observe(state).bomb_opt == HIT_NONE
