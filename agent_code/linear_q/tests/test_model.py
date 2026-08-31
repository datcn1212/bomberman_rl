"""Unit tests for the weight matrix and the semi-gradient update.

Run from the repository root:  python3 -m pytest agent_code/linear_q/tests -q
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from agent_code.linear_q.model import PHI_DIM, N_ACTIONS, LinearQModel  # noqa: E402


def test_a_fresh_model_predicts_zero_everywhere():
    m = LinearQModel()
    phi = np.random.default_rng(0).random(PHI_DIM)
    assert np.allclose(m.values(phi), 0.0)


def test_one_update_moves_toward_the_target_by_exactly_alpha_times_delta():
    """With Q(s,a) starting at 0, delta = target, and the new prediction on the
    same phi is alpha * target * ||phi||^2 - the entire mechanism spelled out,
    not just "it moved in the right direction"."""
    m = LinearQModel()
    phi = np.zeros(PHI_DIM); phi[0] = 1.0; phi[5] = 1.0   # two active features
    m.update(phi, action=2, target=10.0, alpha=0.1)
    expected = 0.1 * 10.0 * float(phi @ phi)
    assert m.values(phi)[2] == pytest.approx(expected)


def test_update_only_touches_the_column_for_the_action_taken():
    m = LinearQModel()
    phi = np.ones(PHI_DIM)
    m.update(phi, action=1, target=5.0, alpha=0.1)
    untouched = [a for a in range(N_ACTIONS) if a != 1]
    assert np.allclose(m.w[:, untouched], 0.0)
    assert not np.allclose(m.w[:, 1], 0.0)


def test_repeated_updates_on_one_state_converge_toward_the_target():
    """Not a training run - just confirms the recursion has the right fixed
    point, so a training failure later cannot be blamed on this arithmetic."""
    m = LinearQModel()
    phi = np.zeros(PHI_DIM); phi[3] = 1.0; phi[-1] = 1.0  # a one-hot plus bias
    for _ in range(500):
        m.update(phi, action=0, target=7.0, alpha=0.05)
    assert m.values(phi)[0] == pytest.approx(7.0, abs=1e-2)


def test_a_shared_feature_moves_a_different_state_too():
    """The entire point of function approximation: two different phi vectors
    that share an active feature must influence each other's value, unlike a
    tabular row."""
    m = LinearQModel()
    phi_a = np.zeros(PHI_DIM); phi_a[3] = 1.0; phi_a[-1] = 1.0
    phi_b = np.zeros(PHI_DIM); phi_b[3] = 1.0; phi_b[10] = 1.0   # shares index 3
    before = m.values(phi_b)[0]
    m.update(phi_a, action=0, target=10.0, alpha=0.1)
    after = m.values(phi_b)[0]
    assert after != before


def test_save_load_roundtrips_the_weights():
    m = LinearQModel()
    m.update(np.ones(PHI_DIM), action=0, target=3.0, alpha=0.1)
    with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
        m.save(f.name)
        loaded = LinearQModel.load(f.name)
    assert np.array_equal(loaded.w, m.w)


def test_load_rejects_a_model_with_a_different_phi_dim():
    """Mirrors tabular_q's LAYOUT check: a silent dimension mismatch would mix
    weight columns trained for a different meaning into every prediction."""
    m = LinearQModel()
    m.phi_dim = PHI_DIM - 1
    with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
        m.save(f.name)
        with pytest.raises(ValueError):
            LinearQModel.load(f.name)
