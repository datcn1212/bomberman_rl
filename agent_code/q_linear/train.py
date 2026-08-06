import os
import csv
import numpy as np

import events as e
from .features import state_to_features, nearest_coin_distance, ACTIONS

ALPHA = float(os.environ.get("Q_LINEAR_ALPHA", "0.05"))
GAMMA = float(os.environ.get("Q_LINEAR_GAMMA", "0.95"))
EPS_START = float(os.environ.get("Q_LINEAR_EPS_START", "1.0"))
EPS_END = float(os.environ.get("Q_LINEAR_EPS_END", "0.05"))
EPS_DECAY_FRAC = float(os.environ.get("Q_LINEAR_EPS_DECAY_FRAC", "0.8"))
TOTAL_EPISODES = int(os.environ.get("Q_LINEAR_TOTAL_EPISODES", "20000"))
USE_SHAPING = os.environ.get("Q_LINEAR_USE_SHAPING", "1") == "1"
MODEL_PATH = os.environ.get("Q_LINEAR_MODEL_PATH", "q_linear_weights.npy")
LOG_PATH = os.environ.get("Q_LINEAR_LOG_PATH", "training_log.csv")

REWARDS = {
    e.COIN_COLLECTED: 1.0,
    e.INVALID_ACTION: -1.0,
}


def setup_training(self):
    self.episode_counter = 0
    self.epsilon = EPS_START
    self.round_reward = 0.0
    with open(LOG_PATH, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "coins_collected", "total_reward", "epsilon"])


def _potential(game_state):
    if game_state is None:
        return 0.0  # terminal potential = 0 by convention (Ng et al.)
    dist = nearest_coin_distance(game_state)
    return -dist if dist is not None else 0.0


def _shaped_reward(old_game_state, new_game_state, events):
    reward = sum(REWARDS.get(ev, 0.0) for ev in events)
    if USE_SHAPING:
        reward += GAMMA * _potential(new_game_state) - _potential(old_game_state)
    return reward


def _update(self, old_state, self_action, new_state, reward):
    phi = state_to_features(old_state)
    action_idx = ACTIONS.index(self_action)
    q_sa = float(self.beta[action_idx] @ phi)

    if new_state is not None:
        target = reward + GAMMA * float(np.max(self.beta @ state_to_features(new_state)))
    else:
        target = reward

    self.beta[action_idx] += ALPHA * (target - q_sa) * phi


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    reward = _shaped_reward(old_game_state, new_game_state, events)
    self.round_reward += reward
    _update(self, old_game_state, self_action, new_game_state, reward)


def end_of_round(self, last_game_state, last_action, events):
    reward = _shaped_reward(last_game_state, None, events)
    self.round_reward += reward
    _update(self, last_game_state, last_action, None, reward)

    coins_collected = last_game_state['self'][1]
    with open(LOG_PATH, "a", newline="") as f:
        csv.writer(f).writerow([self.episode_counter, coins_collected, self.round_reward, self.epsilon])

    self.episode_counter += 1
    self.round_reward = 0.0

    decay_episodes = max(1, int(TOTAL_EPISODES * EPS_DECAY_FRAC))
    frac = min(1.0, self.episode_counter / decay_episodes)
    self.epsilon = EPS_START + frac * (EPS_END - EPS_START)

    np.save(MODEL_PATH, self.beta)
