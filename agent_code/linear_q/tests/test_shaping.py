"""Unit tests for potential-based reward shaping.

Phase 4 (report_linear_q.md) added a second potential, toward escaping danger,
alongside the existing one toward the target. Both are Phi(s) functions of the
state alone, summed, so Ng et al.'s policy-invariance guarantee covers the
combination the same way it covers either term on its own - but the sign
convention for the escape term is easy to get backwards (t_here counts UP as
danger gets closer, unlike target_dist), so the mapping is pinned directly
here rather than trusted by inspection.

Run from the repository root:  python3 -m pytest agent_code/linear_q/tests -q
"""

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from agent_code.linear_q.features import DIR_HERE, DIR_NONE  # noqa: E402
from agent_code.linear_q.train import _potential  # noqa: E402


def make_self(shaping_weight=0.0, escape_shaping_weight=0.0, shaping_distance_cap=15):
    self = types.SimpleNamespace()
    self.cfg = types.SimpleNamespace(
        shaping_weight=shaping_weight,
        escape_shaping_weight=escape_shaping_weight,
        shaping_distance_cap=shaping_distance_cap,
    )
    return self


def make_obs(t_here=0, target_dir=DIR_NONE, target_dist=0):
    return types.SimpleNamespace(t_here=t_here, target_dir=target_dir, target_dist=target_dist)


# --- both terms off by default ----------------------------------------------

def test_both_weights_zero_gives_zero_potential_regardless_of_state():
    self = make_self()
    for t_here in range(5):
        assert _potential(self, make_obs(t_here=t_here)) == 0.0


# --- escape term: the sign convention ---------------------------------------

def test_a_safe_tile_contributes_nothing_to_the_escape_term():
    self = make_self(escape_shaping_weight=1.0)
    assert _potential(self, make_obs(t_here=0)) == 0.0


def test_urgency_increases_as_the_blast_gets_closer():
    """t_here=1 (about to detonate) must score worse than t_here=4 (still
    three decisions of room) - the opposite ordering of the raw t_here value."""
    self = make_self(escape_shaping_weight=1.0)
    phi_urgent = _potential(self, make_obs(t_here=1))
    phi_mild = _potential(self, make_obs(t_here=4))
    phi_safe = _potential(self, make_obs(t_here=0))
    assert phi_urgent < phi_mild < phi_safe


def test_escaping_in_one_step_gives_a_large_positive_shaping_reward():
    """The scenario Phase 3 diagnosed directly: standing where the tile is
    about to burn, then moving to where it never does. The shaping term for
    that transition (gamma*Phi(s') - Phi(s)) must be a clear positive reward,
    not a rounding-sized nudge."""
    self = make_self(escape_shaping_weight=0.5)
    self.cfg.gamma = 0.995
    before = make_obs(t_here=1)
    after = make_obs(t_here=0)
    shaping = self.cfg.gamma * _potential(self, after) - _potential(self, before)
    assert shaping > 1.0


def test_escape_weight_zero_is_inert_even_when_target_weight_is_not():
    """Regression: the two terms must not leak into each other through a
    shared code path."""
    self = make_self(shaping_weight=1.0, escape_shaping_weight=0.0)
    obs = make_obs(t_here=1, target_dir=1, target_dist=5)
    only_target = -1.0 * 5
    assert _potential(self, obs) == pytest.approx(only_target)


# --- both terms together ----------------------------------------------------

def test_target_and_escape_terms_sum_independently():
    self = make_self(shaping_weight=1.0, escape_shaping_weight=0.5)
    obs = make_obs(t_here=1, target_dir=1, target_dist=5)
    target_term = -1.0 * 5
    escape_term = -0.5 * 4      # t_here=1 -> urgency 4
    assert _potential(self, obs) == pytest.approx(target_term + escape_term)


def test_target_reached_and_safe_gives_the_best_possible_potential():
    self = make_self(shaping_weight=1.0, escape_shaping_weight=0.5)
    obs = make_obs(t_here=0, target_dir=DIR_HERE, target_dist=0)
    assert _potential(self, obs) == pytest.approx(0.0)
