import os
import csv

import numpy as np

import events as e
from .features import (encode_state, nearest_coin_distance, nearest_crate_spot_distance,
                        can_bomb_opponent_now, stuck_ratio, in_danger, ACTIONS,
                        POSITION_HISTORY_LENGTH)

ALPHA = float(os.environ.get("Q_TABULAR_ALPHA", "0.2"))
GAMMA = float(os.environ.get("Q_TABULAR_GAMMA", "0.95"))
EPS_START = float(os.environ.get("Q_TABULAR_EPS_START", "1.0"))
EPS_END = float(os.environ.get("Q_TABULAR_EPS_END", "0.05"))
EPS_DECAY_FRAC = float(os.environ.get("Q_TABULAR_EPS_DECAY_FRAC", "0.8"))
TOTAL_EPISODES = int(os.environ.get("Q_TABULAR_TOTAL_EPISODES", "20000"))
USE_SHAPING = os.environ.get("Q_TABULAR_USE_SHAPING", "1") == "1"
MODEL_PATH = os.environ.get("Q_TABULAR_MODEL_PATH", "q_tabular_table.npy")
LOG_PATH = os.environ.get("Q_TABULAR_LOG_PATH", "training_log.csv")
CRATE_POTENTIAL_WEIGHT = float(os.environ.get("Q_TABULAR_CRATE_WEIGHT", "0.3"))
COIN_POTENTIAL_WEIGHT = float(os.environ.get("Q_TABULAR_COIN_WEIGHT", "1.0"))
STUCK_PENALTY_WEIGHT = float(os.environ.get("Q_TABULAR_STUCK_PENALTY", "0.5"))
DANGER_PENALTY_WEIGHT = float(os.environ.get("Q_TABULAR_DANGER_PENALTY", "1.0"))
BOMB_NEAR_OPPONENT_BONUS = float(os.environ.get("Q_TABULAR_BOMB_NEAR_OPPONENT", "1.0"))

# Same reward/shaping design as q_linear/train.py -- these choices are about
# the MDP itself (which events matter, how potential-based shaping and the
# stuck/danger penalties are computed), not about the value representation,
# and were only reached after extensively debugging a farmable potential-shaping
# cycle and a state-aliasing oscillation trap on the linear model. There is no
# reason to re-derive that from scratch just because the Q-value representation
# changed from a linear weight vector to a table -- reusing it, not
# re-deriving it. Not using n-step returns here either: a n-step experiment on
# the linear model showed it amplifies the danger penalty's backward reach
# (an action 2 steps before triggering `in_danger` gets blamed as strongly as
# the action that actually caused it), collapsing bombing to near-zero;
# tabular Q-learning here uses the same plain 1-step update as a precaution.
REWARDS = {
    e.COIN_COLLECTED: 1.0,
    e.INVALID_ACTION: -1.0,
    e.CRATE_DESTROYED: 2.0,
    e.KILLED_SELF: -5.0,
    e.GOT_KILLED: -5.0,
    e.WAITED: -0.1,
    # Task 3/4 addition: weighted 5x COIN_COLLECTED to match the actual
    # tournament scoring ratio (coin = 1 point, kill = 5 points per
    # settings/final_project.pdf), not a value picked by feel.
    e.KILLED_OPPONENT: 5.0,
}


def setup_training(self):
    self.episode_counter = 0
    self.epsilon = EPS_START
    self.round_reward = 0.0
    self.round_crates = 0
    self.round_kills = 0
    self.best_crate_distance = float('inf')
    with open(LOG_PATH, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "coins_collected", "crates_destroyed", "opponents_killed",
                                 "total_reward", "epsilon"])


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
    hypothetical_history = (list(self.position_history) + [new_pos])[-POSITION_HISTORY_LENGTH:]
    return stuck_ratio(hypothetical_history)


def _shaped_reward(self, old_game_state, new_game_state, events):
    reward = sum(REWARDS.get(ev, 0.0) for ev in events)
    if USE_SHAPING:
        old_potential = _potential(self, old_game_state)
        new_potential = _potential(self, new_game_state)
        reward += GAMMA * new_potential - old_potential
    reward -= STUCK_PENALTY_WEIGHT * _stuck_penalty(self, new_game_state)
    if new_game_state is not None and in_danger(new_game_state):
        reward -= DANGER_PENALTY_WEIGHT
    # Dense bonus for actually dropping a bomb while an opponent is in blast
    # range, separate from the sparse KILLED_OPPONENT event. Motivation: a
    # crate is stationary, so CRATE_DESTROYED alone already gives frequent
    # feedback for "bombing here was good"; an opponent moves every step
    # (peaceful_agent moves randomly on EVERY turn), so it can easily wander
    # out of the blast before the BOMB_TIMER=4 fuse goes off even when the
    # agent's positioning was correct -- KILLED_OPPONENT alone was too rare
    # to learn from (0.01-0.03 kills/round average across 3 different state
    # designs and 8000-24000 training episodes). This rewards the SETUP
    # (being positioned to threaten a kill), which happens far more often
    # than the kill itself, independent of whether the opponent escapes.
    if (old_game_state is not None and e.BOMB_DROPPED in events
            and can_bomb_opponent_now(old_game_state)):
        reward += BOMB_NEAR_OPPONENT_BONUS
    return reward


def _update(self, state, action_idx, reward, next_state):
    if next_state is not None:
        target = reward + GAMMA * float(np.max(self.q_table[next_state]))
    else:
        target = reward
    q_sa = self.q_table[state, action_idx]
    self.q_table[state, action_idx] += ALPHA * (target - q_sa)


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    reward = _shaped_reward(self, old_game_state, new_game_state, events)
    self.round_reward += reward
    self.round_crates += events.count(e.CRATE_DESTROYED)
    self.round_kills += events.count(e.KILLED_OPPONENT)
    next_state = encode_state(new_game_state, stuck_ratio(self.position_history))
    _update(self, self.last_state, ACTIONS.index(self_action), reward, next_state)


def end_of_round(self, last_game_state, last_action, events):
    reward = _shaped_reward(self, last_game_state, None, events)
    self.round_reward += reward
    self.round_crates += events.count(e.CRATE_DESTROYED)
    self.round_kills += events.count(e.KILLED_OPPONENT)
    _update(self, self.last_state, ACTIONS.index(last_action), reward, None)

    coins_collected = last_game_state['self'][1]
    with open(LOG_PATH, "a", newline="") as f:
        csv.writer(f).writerow([self.episode_counter, coins_collected, self.round_crates,
                                 self.round_kills, self.round_reward, self.epsilon])

    self.episode_counter += 1
    self.round_reward = 0.0
    self.round_crates = 0
    self.round_kills = 0
    self.best_crate_distance = float('inf')

    decay_episodes = max(1, int(TOTAL_EPISODES * EPS_DECAY_FRAC))
    frac = min(1.0, self.episode_counter / decay_episodes)
    self.epsilon = EPS_START + frac * (EPS_END - EPS_START)

    np.save(MODEL_PATH, self.q_table)
