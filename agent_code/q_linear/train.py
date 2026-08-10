import os
import csv
import random
from collections import deque

import numpy as np

import events as e
from .features import (state_to_features, nearest_coin_distance, nearest_crate_spot_distance,
                        stuck_ratio, in_danger, ACTIONS, POSITION_HISTORY_LENGTH)

ALPHA = float(os.environ.get("Q_LINEAR_ALPHA", "0.01"))
GAMMA = float(os.environ.get("Q_LINEAR_GAMMA", "0.95"))
# n-step TD: bootstrap off the state N_STEP steps ahead instead of 1 step
# ahead, using the N_STEP actually-observed rewards in between. N_STEP=1
# reduces exactly to the original 1-step update (see _store). Intuition:
# with 1-step bootstrapping, a reward earned downstream (e.g. a coin found
# only after several crates are cleared) has to propagate back one gradient
# step at a time through the chain of intermediate states before it affects
# the Q-value of the state that started that chain -- slow, and under linear
# function approximation each hop adds approximation error (part of why the
# "deadly triad" -- function approximation + bootstrapping + off-policy max
# -- destabilized earlier Task 2 attempts). Summing N_STEP real rewards
# before bootstrapping shortens that chain and leans less on the
# (error-prone) bootstrap term.
N_STEP = int(os.environ.get("Q_LINEAR_N_STEP", "3"))
GAMMA_N = GAMMA ** N_STEP
EPS_START = float(os.environ.get("Q_LINEAR_EPS_START", "1.0"))
EPS_END = float(os.environ.get("Q_LINEAR_EPS_END", "0.05"))
EPS_DECAY_FRAC = float(os.environ.get("Q_LINEAR_EPS_DECAY_FRAC", "0.8"))
TOTAL_EPISODES = int(os.environ.get("Q_LINEAR_TOTAL_EPISODES", "20000"))
USE_SHAPING = os.environ.get("Q_LINEAR_USE_SHAPING", "1") == "1"
MODEL_PATH = os.environ.get("Q_LINEAR_MODEL_PATH", "q_linear_weights.npy")
LOG_PATH = os.environ.get("Q_LINEAR_LOG_PATH", "training_log.csv")
BUFFER_SIZE = int(os.environ.get("Q_LINEAR_BUFFER_SIZE", "2000"))
BATCH_SIZE = int(os.environ.get("Q_LINEAR_BATCH_SIZE", "32"))
CRATE_POTENTIAL_WEIGHT = float(os.environ.get("Q_LINEAR_CRATE_WEIGHT", "0.3"))
COIN_POTENTIAL_WEIGHT = float(os.environ.get("Q_LINEAR_COIN_WEIGHT", "1.0"))
STUCK_PENALTY_WEIGHT = float(os.environ.get("Q_LINEAR_STUCK_PENALTY", "0.3"))
DANGER_PENALTY_WEIGHT = float(os.environ.get("Q_LINEAR_DANGER_PENALTY", "1.0"))

REWARDS = {
    e.COIN_COLLECTED: 1.0,
    e.INVALID_ACTION: -1.0,
    e.CRATE_DESTROYED: 2.0,
    e.KILLED_SELF: -5.0,
    e.GOT_KILLED: -5.0,
    # No SURVIVED_ROUND: an unconditional reward for reaching the step cap
    # made "stand still and do nothing" strictly profitable regardless of
    # whether the agent ever engaged with a crate — the death penalty alone
    # already teaches survival without rewarding passivity.
    # WAITED gets an explicit small penalty: since potential Phi(s) is
    # always <= 0, a self-transition (WAIT, same state) yields shaped
    # reward = gamma*Phi(s) - Phi(s) = Phi(s)*(gamma-1), which is POSITIVE
    # whenever Phi(s) < 0 -- i.e. standing still earns "free" reward from
    # the shaping formula alone. This penalty cancels that out.
    e.WAITED: -0.1,
}


def setup_training(self):
    self.episode_counter = 0
    self.epsilon = EPS_START
    self.round_reward = 0.0
    self.round_crates = 0
    self.best_crate_distance = float('inf')
    self.buffer = deque(maxlen=BUFFER_SIZE)
    self.n_step_buffer = deque()  # sliding window of (phi, action_idx, reward); drained at round end
    with open(LOG_PATH, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "coins_collected", "crates_destroyed", "total_reward", "epsilon"])


def _potential(self, game_state):
    if game_state is None:
        return 0.0  # terminal potential = 0 by convention (Ng et al.)
    # Crate term uses the BEST (smallest) distance-to-a-crate-spot achieved
    # so far this episode, not the current distance. A crate-adjacent tile
    # does not disappear just by standing next to it (unlike a coin, which
    # is consumed on contact) — so a live-distance potential can be "farmed"
    # for free by repeatedly approaching and retreating from any crate,
    # since each approach yields +reward and each retreat yields a smaller
    # -reward, and with gamma close to 1 the closed 2-step cycle's Bellman
    # fixed point amplifies to Q ~= (r_approach + gamma*r_retreat)/(1-gamma^2)
    # -- verified numerically against a real stuck run: implied rewards of
    # +0.317/-0.283 matched the live-distance formula's +0.285/-0.285 almost
    # exactly. Using the running-best distance makes retreating a no-op
    # (best-so-far cannot get worse), which removes the farmable cycle.
    crate_dist = nearest_crate_spot_distance(game_state)
    if crate_dist is not None:
        self.best_crate_distance = min(self.best_crate_distance, crate_dist)
    crate_term = -self.best_crate_distance if self.best_crate_distance < float('inf') else 0.0

    # Coin term stays live-distance: a coin is consumed on contact, so it
    # cannot be farmed the same way, and Task 1 already validated this form.
    coin_dist = nearest_coin_distance(game_state)
    coin_term = -coin_dist if coin_dist is not None else 0.0

    return CRATE_POTENTIAL_WEIGHT * crate_term + COIN_POTENTIAL_WEIGHT * coin_term


def _stuck_penalty(self, new_game_state):
    """Direct (non-potential-based) penalty proportional to how repetitive
    the trajectory would look if the agent ends up at new_game_state's
    position. Added because the stuck_ratio FEATURE alone (see features.py)
    was learned with the correct sign but too small a magnitude: at
    saturation (stuck_ratio=0.75) the oscillating actions still out-scored
    genuinely useful ones by a wide margin (verified: Q=0.53 for the
    repeating move vs Q=-3.5 for BOMB at the same state). A feature only
    provides a signal the model MIGHT learn to weight strongly enough; a
    direct reward term forces the issue.
    """
    if new_game_state is None:
        return 0.0
    new_pos = new_game_state['self'][3]
    hypothetical_history = (list(self.position_history) + [new_pos])[-POSITION_HISTORY_LENGTH:]
    return stuck_ratio(hypothetical_history)


def _shaped_reward(self, old_game_state, new_game_state, events):
    reward = sum(REWARDS.get(ev, 0.0) for ev in events)
    if USE_SHAPING:
        # Evaluate old_state's potential FIRST: _potential() has a side
        # effect (updating self.best_crate_distance), so evaluating
        # new_state first would let new_state's (later-in-time) distance
        # improve best_crate_distance before old_state's potential is
        # computed, corrupting the history ordering.
        old_potential = _potential(self, old_game_state)
        new_potential = _potential(self, new_game_state)
        reward += GAMMA * new_potential - old_potential
    reward -= STUCK_PENALTY_WEIGHT * _stuck_penalty(self, new_game_state)
    # Direct penalty for BEING in a bomb blast right now, not just for the
    # eventual KILLED_SELF/GOT_KILLED event. The terminal death penalty is
    # correct but 3-4 steps delayed (BOMB_TIMER), so it only reaches the
    # "should I escape right now" decision via multi-step TD bootstrapping,
    # which needs many sampled danger episodes to propagate reliably.
    # Observed directly: with only the delayed penalty, a state correctly
    # flagged in_danger_now=1 (with a safe escape direction available)
    # still had Q(WAIT)=-0.85 > Q(safe direction)=-2.66 -- the agent wasn't
    # getting a strong enough immediate signal to move. A same-step penalty
    # gives dense, immediate feedback instead of relying on delayed credit
    # assignment for a state the agent has had little training exposure to
    # (it rarely engaged with bombs at all before the stuck-loop was fixed).
    if new_game_state is not None and in_danger(new_game_state):
        reward -= DANGER_PENALTY_WEIGHT
    return reward


def _discounted_return(window):
    """GAMMA^i-weighted sum of the rewards in `window` (a sequence of
    (phi, action_idx, reward) tuples, oldest first) -- the n-step (or
    shorter, near a round's end) return starting from the window's oldest
    entry.
    """
    return sum(GAMMA ** i * r for i, (_, _, r) in enumerate(window))


def _store(self, old_state, self_action, new_state, reward):
    # Reuse the exact feature vector act() computed for old_state (same
    # object, per the framework's `store_game_state` contract) instead of
    # recomputing it — guarantees the stuck_ratio bit matches what the
    # agent actually saw when it chose self_action, and avoids a redundant
    # BFS/feature pass.
    phi = self.last_features
    action_idx = ACTIONS.index(self_action)
    self.n_step_buffer.append((phi, action_idx, reward))
    if len(self.n_step_buffer) < N_STEP:
        return  # window not full yet; wait for more rewards before emitting
    # Window just reached N_STEP using only REAL observed rewards up to and
    # including this step, so if this step is terminal (new_state is None)
    # no bootstrap is needed regardless -- the return is already complete.
    if new_state is not None:
        phi_next = state_to_features(new_state, stuck_ratio(self.position_history))
    else:
        phi_next = None
    n_return = _discounted_return(self.n_step_buffer)
    phi0, action0, _ = self.n_step_buffer.popleft()  # slide the window by 1
    self.buffer.append((phi0, action0, n_return, phi_next))


def _drain_n_step_buffer(self):
    """Round just ended: `_store` only emits once the window reaches
    N_STEP entries, so up to N_STEP-1 of the most recent steps are still
    sitting in the window unemitted. Flush them one at a time, oldest
    first, each as a terminal transition (no bootstrap -- there is no more
    reward to wait for once the round is over) with a correspondingly
    shorter return.
    """
    while self.n_step_buffer:
        n_return = _discounted_return(self.n_step_buffer)
        phi0, action0, _ = self.n_step_buffer.popleft()
        self.buffer.append((phi0, action0, n_return, None))


def _train_step(self):
    if len(self.buffer) < BATCH_SIZE:
        return
    indices = random.sample(range(len(self.buffer)), BATCH_SIZE)
    for i in indices:
        phi, action_idx, n_return, phi_next = self.buffer[i]
        if phi_next is not None:
            target = n_return + GAMMA_N * float(np.max(self.beta @ phi_next))
        else:
            target = n_return
        q_sa = float(self.beta[action_idx] @ phi)
        self.beta[action_idx] += ALPHA * (target - q_sa) * phi


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    reward = _shaped_reward(self, old_game_state, new_game_state, events)
    self.round_reward += reward
    self.round_crates += events.count(e.CRATE_DESTROYED)
    _store(self, old_game_state, self_action, new_game_state, reward)
    _train_step(self)


def end_of_round(self, last_game_state, last_action, events):
    reward = _shaped_reward(self, last_game_state, None, events)
    self.round_reward += reward
    self.round_crates += events.count(e.CRATE_DESTROYED)
    _store(self, last_game_state, last_action, None, reward)
    _drain_n_step_buffer(self)
    _train_step(self)

    coins_collected = last_game_state['self'][1]
    with open(LOG_PATH, "a", newline="") as f:
        csv.writer(f).writerow([self.episode_counter, coins_collected, self.round_crates,
                                 self.round_reward, self.epsilon])

    self.episode_counter += 1
    self.round_reward = 0.0
    self.round_crates = 0
    self.best_crate_distance = float('inf')  # reset for the next round

    decay_episodes = max(1, int(TOTAL_EPISODES * EPS_DECAY_FRAC))
    frac = min(1.0, self.episode_counter / decay_episodes)
    self.epsilon = EPS_START + frac * (EPS_END - EPS_START)

    np.save(MODEL_PATH, self.beta)
