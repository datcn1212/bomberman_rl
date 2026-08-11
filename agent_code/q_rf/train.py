import os
import csv
import pickle
from collections import deque

import numpy as np
from sklearn.ensemble import RandomForestRegressor

import events as e
from .features import (state_to_features, nearest_coin_distance, nearest_crate_spot_distance,
                       can_bomb_opponent_now, stuck_ratio, in_danger, ACTIONS,
                       POSITION_HISTORY_LENGTH)
from .callbacks import q_values

GAMMA = float(os.environ.get("Q_RF_GAMMA", "0.95"))
EPS_START = float(os.environ.get("Q_RF_EPS_START", "1.0"))
EPS_END = float(os.environ.get("Q_RF_EPS_END", "0.05"))
EPS_DECAY_FRAC = float(os.environ.get("Q_RF_EPS_DECAY_FRAC", "0.8"))
TOTAL_EPISODES = int(os.environ.get("Q_RF_TOTAL_EPISODES", "2000"))
USE_SHAPING = os.environ.get("Q_RF_USE_SHAPING", "1") == "1"
MODEL_PATH = os.environ.get("Q_RF_MODEL_PATH", "q_rf_model.pkl")
LOG_PATH = os.environ.get("Q_RF_LOG_PATH", "training_log.csv")

# Fitted-Q iteration refits the forests from scratch every REFIT_EVERY episodes
# instead of nudging weights each step. That is the whole point of this model:
# a batch method sidesteps the "moving target" instability of online
# bootstrapping that made the linear agent unstable (see bao_cao_task2.md), at
# the cost of learning in coarse jumps rather than continuously.
REFIT_EVERY = int(os.environ.get("Q_RF_REFIT_EVERY", "100"))
BUFFER_SIZE = int(os.environ.get("Q_RF_BUFFER_SIZE", "60000"))
N_ESTIMATORS = int(os.environ.get("Q_RF_N_ESTIMATORS", "20"))
MAX_DEPTH = int(os.environ.get("Q_RF_MAX_DEPTH", "12"))
MIN_SAMPLES_LEAF = int(os.environ.get("Q_RF_MIN_SAMPLES_LEAF", "5"))
# Inner FQI passes per refit. Each pass propagates reward information exactly
# ONE bootstrap step further back through the state space, because the targets
# for pass k are built from the forests fitted in pass k-1. With one pass per
# refit and a refit every 100 episodes, a 1500-episode run only ever propagates
# values 15 steps -- far too short to connect "drop a bomb here" to the coin it
# eventually reveals. Measured directly: coins per episode crawled from 0.00 to
# only 0.58 over 1500 episodes and then regressed. Running several passes over
# the same buffer is the standard Fitted-Q Iteration formulation and costs no
# extra environment interaction.
FQI_PASSES = int(os.environ.get("Q_RF_FQI_PASSES", "6"))

CRATE_POTENTIAL_WEIGHT = float(os.environ.get("Q_RF_CRATE_WEIGHT", "0.3"))
COIN_POTENTIAL_WEIGHT = float(os.environ.get("Q_RF_COIN_WEIGHT", "1.0"))
STUCK_PENALTY_WEIGHT = float(os.environ.get("Q_RF_STUCK_PENALTY", "0.5"))
DANGER_PENALTY_WEIGHT = float(os.environ.get("Q_RF_DANGER_PENALTY", "1.0"))
BOMB_NEAR_OPPONENT_BONUS = float(os.environ.get("Q_RF_BOMB_NEAR_OPPONENT", "1.0"))

# Reward design carried over unchanged from q_linear/q_tabular. These choices
# describe the MDP, not the value representation, and were settled by a long
# debugging effort documented in bao_cao_task2.md and bao_cao_tabularQ.md;
# changing them here would make the three models incomparable.
REWARDS = {
    e.COIN_COLLECTED: 1.0,
    e.INVALID_ACTION: -1.0,
    e.CRATE_DESTROYED: 2.0,
    e.KILLED_SELF: -5.0,
    e.GOT_KILLED: -5.0,
    e.WAITED: -0.1,
    e.KILLED_OPPONENT: 5.0,
}


def setup_training(self):
    self.episode_counter = 0
    self.epsilon = EPS_START
    self.round_reward = 0.0
    self.round_crates = 0
    self.round_kills = 0
    self.best_crate_distance = float('inf')
    self.buffer = deque(maxlen=BUFFER_SIZE)
    with open(LOG_PATH, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "coins_collected", "crates_destroyed", "opponents_killed",
                                "total_reward", "epsilon", "buffer_size"])


def _potential(self, game_state):
    if game_state is None:
        return 0.0
    crate_dist = nearest_crate_spot_distance(game_state)
    if crate_dist is not None:
        self.best_crate_distance = min(self.best_crate_distance, crate_dist)
    crate_term = -self.best_crate_distance if self.best_crate_distance < float('inf') else 0.0

    coin_dist = nearest_coin_distance(game_state)
    coin_term = -coin_dist if coin_dist is not None else 0.0

    return CRATE_POTENTIAL_WEIGHT * crate_term + COIN_POTENTIAL_WEIGHT * coin_term


def _stuck_penalty(self, new_game_state):
    if new_game_state is None:
        return 0.0
    new_pos = new_game_state['self'][3]
    hypothetical = (list(self.position_history) + [new_pos])[-POSITION_HISTORY_LENGTH:]
    return stuck_ratio(hypothetical)


def _shaped_reward(self, old_game_state, new_game_state, events):
    reward = sum(REWARDS.get(ev, 0.0) for ev in events)
    if USE_SHAPING:
        old_potential = _potential(self, old_game_state)
        new_potential = _potential(self, new_game_state)
        reward += GAMMA * new_potential - old_potential
    reward -= STUCK_PENALTY_WEIGHT * _stuck_penalty(self, new_game_state)
    if new_game_state is not None and in_danger(new_game_state):
        reward -= DANGER_PENALTY_WEIGHT
    if (old_game_state is not None and e.BOMB_DROPPED in events
            and can_bomb_opponent_now(old_game_state)):
        reward += BOMB_NEAR_OPPONENT_BONUS
    return reward


def _store(self, self_action, new_game_state, reward):
    phi = self.last_features
    action_idx = ACTIONS.index(self_action)
    if new_game_state is not None:
        phi_next = state_to_features(new_game_state, stuck_ratio(self.position_history))
    else:
        phi_next = None
    self.buffer.append((phi, action_idx, reward, phi_next))


def _fit_one_pass(self, states, actions, rewards, next_states, non_terminal):
    """A single Fitted-Q Iteration pass over the buffer: build regression
    targets from the CURRENT forests, then train fresh forests on them.

    The bootstrap term is evaluated once, up front, and stays frozen while the
    new forests are fitted -- that is what makes this a batch method and avoids
    the moving-target instability that plagued the online linear agent.
    """
    targets = rewards.copy()
    if self.forests is not None and len(non_terminal) > 0:
        # One batched predict per action; the buffer holds tens of thousands
        # of rows, so per-row calls would dominate the runtime.
        next_q = np.stack([forest.predict(next_states) for forest in self.forests], axis=1)
        targets[non_terminal] += GAMMA * next_q.max(axis=1)

    new_forests = []
    for action_idx in range(len(ACTIONS)):
        mask = actions == action_idx
        forest = RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            n_jobs=1,  # tournament rules forbid multiprocessing; keep training honest too
        )
        if mask.sum() >= MIN_SAMPLES_LEAF:
            forest.fit(states[mask], targets[mask])
        elif self.forests is not None:
            # Too few samples for this action this round -- keep the previous
            # forest rather than dropping the action's Q-function entirely.
            forest = self.forests[action_idx]
        else:
            # Nothing fitted yet and no data: fit a constant so predict() works.
            forest.fit(states[:1], targets[:1])
        new_forests.append(forest)

    self.forests = new_forests


def _refit(self):
    """Run FQI_PASSES fitted-Q iterations over the current buffer."""
    if len(self.buffer) == 0:
        return

    states = np.array([t[0] for t in self.buffer], dtype=np.float32)
    actions = np.array([t[1] for t in self.buffer], dtype=np.int64)
    rewards = np.array([t[2] for t in self.buffer], dtype=np.float32)
    non_terminal = np.array([i for i, t in enumerate(self.buffer) if t[3] is not None],
                            dtype=np.int64)
    next_states = (np.array([self.buffer[i][3] for i in non_terminal], dtype=np.float32)
                   if len(non_terminal) > 0 else np.empty((0, states.shape[1]), dtype=np.float32))

    for _ in range(FQI_PASSES):
        _fit_one_pass(self, states, actions, rewards, next_states, non_terminal)


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    reward = _shaped_reward(self, old_game_state, new_game_state, events)
    self.round_reward += reward
    self.round_crates += events.count(e.CRATE_DESTROYED)
    self.round_kills += events.count(e.KILLED_OPPONENT)
    _store(self, self_action, new_game_state, reward)


def end_of_round(self, last_game_state, last_action, events):
    reward = _shaped_reward(self, last_game_state, None, events)
    self.round_reward += reward
    self.round_crates += events.count(e.CRATE_DESTROYED)
    self.round_kills += events.count(e.KILLED_OPPONENT)
    _store(self, last_action, None, reward)

    coins_collected = last_game_state['self'][1]
    with open(LOG_PATH, "a", newline="") as f:
        csv.writer(f).writerow([self.episode_counter, coins_collected, self.round_crates,
                                self.round_kills, self.round_reward, self.epsilon,
                                len(self.buffer)])

    self.episode_counter += 1
    self.round_reward = 0.0
    self.round_crates = 0
    self.round_kills = 0
    self.best_crate_distance = float('inf')

    decay_episodes = max(1, int(TOTAL_EPISODES * EPS_DECAY_FRAC))
    frac = min(1.0, self.episode_counter / decay_episodes)
    self.epsilon = EPS_START + frac * (EPS_END - EPS_START)

    if self.episode_counter % REFIT_EVERY == 0:
        _refit(self)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.forests, f)
