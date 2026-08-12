"""Q-learning update, driven by the framework's training callbacks.

The delivery pattern was measured in Phase 0 (tools/verify_events.py) and
dictates the structure here:

  * when the agent survives, the last step arrives twice -- once through
    `game_events_occurred` and once through `end_of_round`, the second time with
    SURVIVED_ROUND attached;
  * when the agent dies, the fatal step arrives *only* through `end_of_round`.

So transitions are staged rather than learned on arrival. A staged transition is
flushed when the next one shows up, and `end_of_round` either replaces it (same
step number -- it is the re-delivery) or flushes it first (different step number
-- the agent died and this is a genuinely new transition).
"""

import csv

import numpy as np

import events as e

from .features import ACTIONS
from .model import state_of


class Transition:
    __slots__ = ("step", "state", "action", "reward", "next_state", "terminal")

    def __init__(self, step, state, action, reward, next_state, terminal):
        self.step = step
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.terminal = terminal


def setup_training(self):
    self.episode = 0
    self.pending = None
    self.episode_reward = 0.0
    self.episode_coins = 0
    self.log_rows = []


def reward_from(self, events):
    """Map a step's events onto a scalar reward."""
    cfg = self.cfg
    reward = cfg.reward_step
    if e.COIN_COLLECTED in events:
        reward += cfg.reward_coin
    if e.INVALID_ACTION in events:
        reward += cfg.reward_invalid
    if e.WAITED in events:
        reward += cfg.reward_wait
    if e.KILLED_SELF in events:
        reward += cfg.reward_killed_self
    return reward


def _flush(self):
    """Apply the Q-learning update for the staged transition, if any."""
    t = self.pending
    self.pending = None
    if t is None:
        return
    if t.terminal:
        target = t.reward
    else:
        # Max over legal actions only. With BOMB masked out its row stays at
        # zero, and a plain max would bootstrap from that zero whenever every
        # legal action is worth less -- an optimistic target for an action the
        # agent is not even allowed to take.
        nxt = self.model.values(t.next_state)[self.legal]
        target = t.reward + self.cfg.gamma * float(np.max(nxt))
    self.model.update(t.state, t.action, target, self.cfg.alpha)


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    if old_game_state is None or self_action is None:
        return
    _flush(self)
    reward = reward_from(self, events)
    self.episode_reward += reward
    self.episode_coins += events.count(e.COIN_COLLECTED)
    self.pending = Transition(
        step=old_game_state["step"],
        state=state_of(old_game_state),
        action=ACTIONS.index(self_action),
        reward=reward,
        next_state=state_of(new_game_state),
        terminal=False,
    )


def end_of_round(self, last_game_state, last_action, events):
    if last_action is not None:
        if self.pending is not None and self.pending.step == last_game_state["step"]:
            # Re-delivery of a step already staged: keep the end_of_round
            # version, which carries the complete event list.
            self.pending = None
        else:
            _flush(self)
        reward = reward_from(self, events)
        self.episode_reward += reward
        self.episode_coins += events.count(e.COIN_COLLECTED)
        self.pending = Transition(
            step=last_game_state["step"],
            state=state_of(last_game_state),
            action=ACTIONS.index(last_action),
            reward=reward,
            next_state=None,
            terminal=True,
        )
        _flush(self)

    self.episode += 1
    _decay_epsilon(self)
    _log_episode(self, last_game_state, events)

    self.episode_reward = 0.0
    self.episode_coins = 0

    if self.episode >= self.cfg.n_episodes or self.episode % 500 == 0:
        self.model.save(self.cfg.model_path)
        _write_log(self)


def _decay_epsilon(self):
    cfg = self.cfg
    frac = min(1.0, self.episode / max(1, cfg.eps_decay_episodes))
    self.epsilon = cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)


def _log_episode(self, last_game_state, events):
    self.log_rows.append({
        "episode": self.episode,
        "steps": last_game_state["step"] if last_game_state else 0,
        "score": last_game_state["self"][1] if last_game_state else 0,
        "coins": self.episode_coins,
        "reward": round(self.episode_reward, 4),
        "epsilon": round(self.epsilon, 4),
        "killed_self": int(e.KILLED_SELF in events),
        "survived": int(e.SURVIVED_ROUND in events),
        "states_visited": int(np.count_nonzero(self.model.seen.sum(axis=1))),
    })


def _write_log(self):
    if not self.cfg.log_path or not self.log_rows:
        return
    with open(self.cfg.log_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(self.log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(self.log_rows)
