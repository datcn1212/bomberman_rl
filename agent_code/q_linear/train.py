import os
import csv
import random
from collections import deque

import numpy as np

import events as e
from .features import state_to_features, nearest_coin_distance, nearest_crate_spot_distance, ACTIONS

ALPHA = float(os.environ.get("Q_LINEAR_ALPHA", "0.01"))
GAMMA = float(os.environ.get("Q_LINEAR_GAMMA", "0.95"))
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
    self.buffer = deque(maxlen=BUFFER_SIZE)
    with open(LOG_PATH, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "coins_collected", "crates_destroyed", "total_reward", "epsilon"])


def _potential(game_state):
    if game_state is None:
        return 0.0  # terminal potential = 0 by convention (Ng et al.)
    crate_dist = nearest_crate_spot_distance(game_state)
    crate_term = -crate_dist if crate_dist is not None else 0.0
    coin_dist = nearest_coin_distance(game_state)
    coin_term = -coin_dist if coin_dist is not None else 0.0
    return CRATE_POTENTIAL_WEIGHT * crate_term + COIN_POTENTIAL_WEIGHT * coin_term


def _shaped_reward(old_game_state, new_game_state, events):
    reward = sum(REWARDS.get(ev, 0.0) for ev in events)
    if USE_SHAPING:
        reward += GAMMA * _potential(new_game_state) - _potential(old_game_state)
    return reward


def _store(self, old_state, self_action, new_state, reward):
    # Reuse the exact feature vector act() computed for old_state (same
    # object, per the framework's `store_game_state` contract) instead of
    # recomputing it — guarantees the "recently visited" bit matches what
    # the agent actually saw when it chose self_action, and avoids a
    # redundant BFS/feature pass.
    phi = self.last_features
    action_idx = ACTIONS.index(self_action)
    if new_state is not None:
        new_pos = new_state['self'][3]
        new_recently_visited = new_pos in self.position_history
        phi_next = state_to_features(new_state, new_recently_visited)
    else:
        phi_next = None
    self.buffer.append((phi, action_idx, reward, phi_next))


def _train_step(self):
    if len(self.buffer) < BATCH_SIZE:
        return
    indices = random.sample(range(len(self.buffer)), BATCH_SIZE)
    for i in indices:
        phi, action_idx, reward, phi_next = self.buffer[i]
        if phi_next is not None:
            target = reward + GAMMA * float(np.max(self.beta @ phi_next))
        else:
            target = reward
        q_sa = float(self.beta[action_idx] @ phi)
        self.beta[action_idx] += ALPHA * (target - q_sa) * phi


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    reward = _shaped_reward(old_game_state, new_game_state, events)
    self.round_reward += reward
    self.round_crates += events.count(e.CRATE_DESTROYED)
    _store(self, old_game_state, self_action, new_game_state, reward)
    _train_step(self)


def end_of_round(self, last_game_state, last_action, events):
    reward = _shaped_reward(last_game_state, None, events)
    self.round_reward += reward
    self.round_crates += events.count(e.CRATE_DESTROYED)
    _store(self, last_game_state, last_action, None, reward)
    _train_step(self)

    coins_collected = last_game_state['self'][1]
    with open(LOG_PATH, "a", newline="") as f:
        csv.writer(f).writerow([self.episode_counter, coins_collected, self.round_crates,
                                 self.round_reward, self.epsilon])

    self.episode_counter += 1
    self.round_reward = 0.0
    self.round_crates = 0

    decay_episodes = max(1, int(TOTAL_EPISODES * EPS_DECAY_FRAC))
    frac = min(1.0, self.episode_counter / decay_episodes)
    self.epsilon = EPS_START + frac * (EPS_END - EPS_START)

    np.save(MODEL_PATH, self.beta)
