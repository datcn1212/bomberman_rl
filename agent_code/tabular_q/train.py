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

from . import features
from .features import ACTIONS, DIR_NONE, Board
from .model import observe_and_encode, to_frame


class Transition:
    __slots__ = ("step", "state", "action", "reward", "next_state", "terminal", "coins", "crates", "bombs")

    def __init__(self, step, state, action, reward, next_state, terminal, coins, crates, bombs):
        self.step = step
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.terminal = terminal
        self.coins = coins
        self.crates = crates
        self.bombs = bombs


def setup_training(self):
    # Recorded on the model so it can never be evaluated with a different
    # observation, or a different frame, than it was trained with.
    self.model.feature_flags = dict(features.FLAGS)
    self.model.feature_flags["use_symmetry"] = self.cfg.use_symmetry
    self.episode = 0
    self.pending = None
    self.episode_reward = 0.0
    self.episode_coins = 0
    self.episode_crates = 0
    self.bombs_dropped = 0
    self.log_rows = []


def _potential(self, obs):
    """Phi(s) = -w * distance to the nearest target.

    Capped so that "no target reachable" and "target very far" are the same
    value; an unbounded potential would make the shaping term jump whenever the
    last coin on a board is collected.
    """
    if not self.cfg.shaping_weight:
        return 0.0
    dist = self.cfg.shaping_distance_cap
    if obs.target_dir != DIR_NONE:
        dist = min(obs.target_dist, self.cfg.shaping_distance_cap)
    return -self.cfg.shaping_weight * dist


def _wasted_bomb(old_game_state, events):
    """True when the agent dropped a bomb whose blast covers no crate.

    Evaluated on the state the bomb was dropped from, so the penalty lands on
    the transition that made the decision rather than four steps later, where
    Phase 4 showed it cannot be attributed.
    """
    if e.BOMB_DROPPED not in events:
        return False
    return Board(old_game_state).bomb_payload() == 0


def reward_from(self, events, old_game_state=None):
    """Map a step's events onto a scalar reward.

    Crates and revealed coins are counted per occurrence, because one
    well-placed bomb can destroy several at once and a flat bonus would make a
    bomb that clears one crate worth as much as a bomb that clears four.
    """
    cfg = self.cfg
    reward = cfg.reward_step
    reward += cfg.reward_coin * events.count(e.COIN_COLLECTED)
    reward += cfg.reward_crate * events.count(e.CRATE_DESTROYED)
    reward += cfg.reward_coin_found * events.count(e.COIN_FOUND)
    if e.INVALID_ACTION in events:
        reward += cfg.reward_invalid
    if e.WAITED in events:
        reward += cfg.reward_wait
    if e.KILLED_SELF in events:
        reward += cfg.reward_killed_self
    elif e.GOT_KILLED in events:
        # KILLED_SELF always comes with GOT_KILLED; charging both would double
        # the penalty for suicide relative to being killed by someone else.
        reward += cfg.reward_got_killed
    if e.SURVIVED_ROUND in events:
        reward += cfg.reward_survived
    if old_game_state is not None and _wasted_bomb(old_game_state, events):
        reward += cfg.reward_bomb_wasted
    return reward


def _flush(self):
    """Apply the Q-learning update for the staged transition, if any."""
    t = self.pending
    self.pending = None
    if t is None:
        return
    # Episode statistics are accumulated here rather than where the callback
    # arrives, so that a transition dropped as a re-delivery is not counted.
    # Counting on arrival is what makes a 50-coin board report 51 coins.
    self.episode_reward += t.reward
    self.episode_coins += t.coins
    self.episode_crates += t.crates
    self.bombs_dropped += t.bombs
    if t.terminal:
        target = t.reward
    else:
        # Max over legal actions only. With BOMB masked out its row stays at
        # zero, and a plain max would bootstrap from that zero whenever every
        # legal action is worth less -- an optimistic target for an action the
        # agent is not even allowed to take.
        nxt = self.model.values(t.next_state)[self.legal]
        target = t.reward + self.cfg.gamma * float(np.max(nxt))
    alpha = self.model.effective_alpha(t.state, t.action, self.cfg)
    self.model.update(t.state, t.action, target, alpha)


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    if old_game_state is None or self_action is None:
        return
    _flush(self)
    old_index, old_obs, old_perm = observe_and_encode(old_game_state, self.cfg.use_symmetry)
    new_index, new_obs, _ = observe_and_encode(new_game_state, self.cfg.use_symmetry)
    shaping = (self.cfg.gamma * _potential(self, new_obs)) - _potential(self, old_obs)
    self.pending = Transition(
        step=old_game_state["step"],
        state=old_index,
        action=to_frame(old_perm, ACTIONS.index(self_action)),
        reward=reward_from(self, events, old_game_state) + shaping,
        next_state=new_index,
        terminal=False,
        coins=events.count(e.COIN_COLLECTED),
        crates=events.count(e.CRATE_DESTROYED),
        bombs=events.count(e.BOMB_DROPPED),
    )


def end_of_round(self, last_game_state, last_action, events):
    if last_action is not None:
        if self.pending is not None and self.pending.step == last_game_state["step"]:
            # Re-delivery of a step already staged: keep the end_of_round
            # version, which carries the complete event list.
            self.pending = None
        else:
            _flush(self)
        # Terminal transition: Ng et al. require Phi(terminal) = 0 for the
        # policy-invariance guarantee, so the shaping term is just -Phi(s).
        last_index, last_obs, last_perm = observe_and_encode(
            last_game_state, self.cfg.use_symmetry)
        self.pending = Transition(
            step=last_game_state["step"],
            state=last_index,
            action=to_frame(last_perm, ACTIONS.index(last_action)),
            reward=reward_from(self, events, last_game_state) - _potential(self, last_obs),
            next_state=None,
            terminal=True,
            coins=events.count(e.COIN_COLLECTED),
            crates=events.count(e.CRATE_DESTROYED),
            bombs=events.count(e.BOMB_DROPPED),
        )
        _flush(self)

    self.episode += 1
    _decay_epsilon(self)
    _log_episode(self, last_game_state, events)

    self.episode_reward = 0.0
    self.episode_coins = 0
    self.episode_crates = 0
    self.bombs_dropped = 0

    if self.episode >= self.cfg.n_episodes or self.episode % 500 == 0:
        self.model.save(self.cfg.model_path)
        _write_log(self)


def _decay_epsilon(self):
    cfg = self.cfg
    frac = min(1.0, self.episode / max(1, cfg.eps_decay_episodes))
    self.epsilon = cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)


def _log_episode(self, last_game_state, events):
    trace = {}
    if self.cfg.trace_state >= 0:
        row = self.model.values(self.cfg.trace_state)
        trace = {"q_%s" % ACTIONS[i]: round(float(row[i]), 5) for i in range(len(ACTIONS))}
    self.log_rows.append({
        "episode": self.episode,
        "steps": last_game_state["step"] if last_game_state else 0,
        "score": last_game_state["self"][1] if last_game_state else 0,
        "coins": self.episode_coins,
        "crates": self.episode_crates,
        "bombs": self.bombs_dropped,
        "reward": round(self.episode_reward, 4),
        "epsilon": round(self.epsilon, 4),
        "killed_self": int(e.KILLED_SELF in events),
        "survived": int(e.SURVIVED_ROUND in events),
        "states_visited": len(self.model),
        **trace,
    })


def _write_log(self):
    if not self.cfg.log_path or not self.log_rows:
        return
    with open(self.cfg.log_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(self.log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(self.log_rows)
