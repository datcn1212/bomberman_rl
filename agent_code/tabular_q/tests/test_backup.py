"""Unit tests for the multi-step backups.

The n-step return and the eligibility trace are both arithmetic that no game
can check for you: a wrong discount or an off-by-one in the window produces a
model that still plays, just worse. These pin the arithmetic directly.

Run from the repository root:  python3 -m pytest agent_code/tabular_q/tests -q
"""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from agent_code.tabular_q import train as T  # noqa: E402
from agent_code.tabular_q.config import Config  # noqa: E402
from agent_code.tabular_q.model import QModel  # noqa: E402


def make_agent(**overrides):
    """A stand-in carrying only what the update functions actually touch."""
    self = types.SimpleNamespace()
    self.cfg = Config(**overrides)
    self.model = QModel()
    self.legal = np.arange(6)
    self.buffer = T.deque()
    self.traces = {}
    self.action_was_greedy = True
    self.episode_reward = 0.0
    self.episode_coins = 0
    self.episode_crates = 0
    self.bombs_dropped = 0
    self.pending = None
    return self


def step(self, state, action, reward, next_state, terminal=False):
    self.pending = T.Transition(step=0, state=state, action=action, reward=reward,
                                next_state=next_state, terminal=terminal,
                                coins=0, crates=0, bombs=0)
    T._flush(self)


# --- n-step returns --------------------------------------------------------

def test_one_step_is_the_plain_bellman_target():
    self = make_agent(n_step=1, alpha=1.0, alpha_schedule="constant", gamma=0.9)
    self.model.add(20, 0, 5.0)                    # Q(20, UP) = 5
    step(self, 10, 0, 1.0, 20)
    # alpha 1 so Q(10,UP) lands exactly on r + gamma * max Q(20, .)
    assert self.model.values(10)[0] == pytest.approx(1.0 + 0.9 * 5.0)


def test_n_step_window_accumulates_discounted_rewards():
    self = make_agent(n_step=3, alpha=1.0, alpha_schedule="constant", gamma=0.5)
    self.model.add(40, 0, 8.0)                    # bootstrap value at the end
    step(self, 10, 0, 1.0, 20)                    # buffered, window not full
    assert self.model.values(10)[0] == 0.0
    step(self, 20, 0, 2.0, 30)                    # still buffered
    assert self.model.values(10)[0] == 0.0
    step(self, 30, 0, 4.0, 40)                    # window full -> first fires
    expected = 1.0 + 0.5 * 2.0 + 0.25 * 4.0 + 0.125 * 8.0
    assert self.model.values(10)[0] == pytest.approx(expected)


def test_terminal_truncates_the_window_and_drains_the_buffer():
    self = make_agent(n_step=5, alpha=1.0, alpha_schedule="constant", gamma=0.5)
    step(self, 10, 0, 1.0, 20)
    step(self, 20, 0, 2.0, None, terminal=True)
    # No bootstrap past a terminal transition, and nothing is left waiting.
    assert self.model.values(10)[0] == pytest.approx(1.0 + 0.5 * 2.0)
    assert self.model.values(20)[0] == pytest.approx(2.0)
    assert len(self.buffer) == 0


def test_a_buffered_transition_is_never_lost():
    self = make_agent(n_step=4, alpha=1.0, alpha_schedule="constant", gamma=1.0)
    for i in range(3):
        step(self, 10 + i, 0, 1.0, 20 + i)
    assert len(self.buffer) == 3
    step(self, 13, 0, 1.0, None, terminal=True)
    assert len(self.buffer) == 0
    for i in range(3):
        assert self.model.values(10 + i)[0] > 0.0


# --- Watkins's Q(lambda) ---------------------------------------------------

def test_lambda_credits_the_earlier_state_through_the_trace():
    self = make_agent(td_lambda=0.9, alpha=1.0, alpha_schedule="constant", gamma=1.0)
    step(self, 10, 0, 0.0, 20)                    # no reward yet
    first = self.model.values(10)[0]
    step(self, 20, 0, 1.0, 30)                    # reward arrives one step later
    # The trace is still alive, so the reward reaches back to (10, UP).
    assert self.model.values(10)[0] > first


def test_a_non_greedy_action_cuts_the_trace():
    self = make_agent(td_lambda=0.9, alpha=1.0, alpha_schedule="constant", gamma=1.0)
    self.action_was_greedy = False
    step(self, 10, 0, 0.0, 20)
    assert self.traces == {}
    before = self.model.values(10)[0]
    step(self, 20, 0, 1.0, 30)
    assert self.model.values(10)[0] == before     # nothing reached back


def test_traces_do_not_survive_the_episode():
    self = make_agent(td_lambda=0.9, alpha=1.0, alpha_schedule="constant", gamma=1.0)
    step(self, 10, 0, 0.0, 20)
    step(self, 20, 0, 1.0, None, terminal=True)
    assert self.traces == {}


def test_trace_updates_do_not_inflate_visit_counts():
    """Only the pair actually taken counts as visited.

    Counting every traced pair would collapse the visit-decayed step size for
    states the agent has barely seen.
    """
    self = make_agent(td_lambda=0.9, alpha=1.0, alpha_schedule="constant", gamma=1.0)
    step(self, 10, 0, 0.0, 20)
    step(self, 20, 0, 1.0, 30)
    assert self.model.visits(10)[0] == 1
    assert self.model.visits(20)[0] == 1


def test_both_backups_at_once_is_refused():
    self = make_agent()
    self.cfg = Config(n_step=3, td_lambda=0.5)
    self.episode = 0
    with pytest.raises(ValueError):
        T.setup_training(self)


def test_each_traced_pair_uses_its_own_step_size():
    """Regression: sharing one alpha across the trace diverges.

    A pair the agent has visited thousands of times must keep its own decayed
    step size even when the TD error arrives from a freshly discovered state.
    Borrowing the visitor's alpha lets settled entries take full-sized steps and
    sends Q to NaN a few thousand episodes into training.
    """
    self = make_agent(td_lambda=0.9, alpha=1.0, alpha_schedule="visit",
                      alpha_half_life=1.0, gamma=1.0)
    # (10, UP) is well travelled; (20, UP) is new.
    for _ in range(100):
        self.model.note_visit(10, 0)

    step(self, 10, 0, 0.0, 20)          # puts (10, UP) into the trace
    settled_before = self.model.values(10)[0]
    fresh_before = self.model.values(20)[0]
    step(self, 20, 0, 1.0, 30)          # TD error arrives from the new state

    settled_move = abs(self.model.values(10)[0] - settled_before)
    fresh_move = abs(self.model.values(20)[0] - fresh_before)
    # Same delta, same-order traces: the gap can only come from the step size.
    assert fresh_move > 10 * settled_move
