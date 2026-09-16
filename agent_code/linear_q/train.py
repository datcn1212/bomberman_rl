"""Training callbacks: one-step semi-gradient TD.

The staging pattern is the same as tabular_q/train.py and exists for the same
reason (a framework quirk, measured in Phase 0):

  - surviving, the last step arrives twice, the second time with SURVIVED_ROUND;
  - dying, the fatal step arrives only through end_of_round.

So transitions are staged and flushed when the next one arrives. No n-step or
Q(lambda) here - this branch never got past the one-step baseline.
"""

import csv

import numpy as np

import events as e

from .features import ACTIONS, DIR_NONE, Board
from .model import observe_and_encode, to_frame


class Transition:
    __slots__ = ("step", "phi", "action", "reward", "next_phi", "terminal",
                 "coins", "crates", "bombs")

    def __init__(self, step, phi, action, reward, next_phi, terminal,
                 coins, crates, bombs):
        self.step = step
        self.phi = phi
        self.action = action
        self.reward = reward
        self.next_phi = next_phi
        self.terminal = terminal
        self.coins = coins
        self.crates = crates
        self.bombs = bombs


def setup_training(self):
    self.model.feature_flags["use_symmetry"] = self.cfg.use_symmetry
    self.episode = 0
    self.pending = None
    self.episode_reward = 0.0
    self.episode_coins = 0
    self.episode_crates = 0
    self.bombs_dropped = 0
    self.log_rows = []


def _potential(self, obs):
    """Phi(s) = -w_target * target distance - w_escape * danger urgency.

    Two independent potentials added together. Ng et al. holds for any Phi(s),
    and a sum of valid potentials is valid too, so adding the escape term meant
    no change at the call sites.

    Careful with the sign: t_here is 0 when the tile never burns and 1..4
    counting down to the blast, so a *smaller* nonzero t_here is worse - the
    opposite of target_dist. The remap below flips it so that for both terms,
    higher potential means better.
    """
    phi = 0.0
    if self.cfg.shaping_weight:
        dist = self.cfg.shaping_distance_cap
        if obs.target_dir != DIR_NONE:
            dist = min(obs.target_dist, self.cfg.shaping_distance_cap)
        phi -= self.cfg.shaping_weight * dist

    if self.cfg.escape_shaping_weight:
        urgency = 0 if obs.t_here == 0 else (5 - obs.t_here)
        phi -= self.cfg.escape_shaping_weight * urgency
    return phi


def _wasted_bomb(old_game_state, events):
    if e.BOMB_DROPPED not in events:
        return False
    return Board(old_game_state).bomb_payload() == 0


def _is_trapped(game_state):
    return Board(game_state).escape_search() == DIR_NONE


def _bomb_without_escape(old_game_state, events):
    if e.BOMB_DROPPED not in events:
        return False
    pos = old_game_state["self"][3]
    return Board(old_game_state, extra_bomb=pos).escape_search() == DIR_NONE


def reward_from(self, events, old_game_state=None):
    """Scalar reward for one step's events. Crates and coins count per occurrence."""
    cfg = self.cfg
    reward = cfg.reward_step
    reward += cfg.reward_coin * events.count(e.COIN_COLLECTED)
    reward += cfg.reward_kill * events.count(e.KILLED_OPPONENT)
    reward += cfg.reward_crate * events.count(e.CRATE_DESTROYED)
    reward += cfg.reward_coin_found * events.count(e.COIN_FOUND)

    if e.INVALID_ACTION in events:
        reward += cfg.reward_invalid
    if e.WAITED in events:
        reward += cfg.reward_wait
    if e.KILLED_SELF in events:
        reward += cfg.reward_killed_self
    elif e.GOT_KILLED in events:
        # elif: KILLED_SELF comes with GOT_KILLED, don't charge both
        reward += cfg.reward_got_killed
    if e.SURVIVED_ROUND in events:
        reward += cfg.reward_survived

    if old_game_state is not None:
        if _wasted_bomb(old_game_state, events):
            reward += cfg.reward_bomb_wasted
        if cfg.reward_trapped and _is_trapped(old_game_state):
            reward += cfg.reward_trapped
        if cfg.reward_bomb_no_escape and _bomb_without_escape(old_game_state, events):
            reward += cfg.reward_bomb_no_escape
    return reward


def _best_next(self, phi):
    """max Q over legal actions only, so a masked BOMB column can't be the target."""
    return float(np.max(self.model.values(phi)[self.legal]))


def _flush(self):
    t = self.pending
    self.pending = None
    if t is None:
        return

    # counted here, not on arrival, so a re-delivered transition isn't double-counted
    self.episode_reward += t.reward
    self.episode_coins += t.coins
    self.episode_crates += t.crates
    self.bombs_dropped += t.bombs

    target = t.reward if t.terminal else t.reward + self.cfg.gamma * _best_next(self, t.next_phi)
    alpha = self.model.effective_alpha(t.phi, t.action, self.cfg)
    self.model.update(t.phi, t.action, target, alpha)


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    if old_game_state is None or self_action is None:
        return
    _flush(self)

    old_phi, old_obs, old_perm, _ = observe_and_encode(old_game_state, self.cfg.use_symmetry)
    new_phi, new_obs, _, _ = observe_and_encode(new_game_state, self.cfg.use_symmetry)
    shaping = (self.cfg.gamma * _potential(self, new_obs)) - _potential(self, old_obs)

    self.pending = Transition(
        step=old_game_state["step"],
        phi=old_phi,
        action=to_frame(old_perm, ACTIONS.index(self_action)),
        reward=reward_from(self, events, old_game_state) + shaping,
        next_phi=new_phi,
        terminal=False,
        coins=events.count(e.COIN_COLLECTED),
        crates=events.count(e.CRATE_DESTROYED),
        bombs=events.count(e.BOMB_DROPPED),
    )


def end_of_round(self, last_game_state, last_action, events):
    if last_action is not None:
        if self.pending is not None and self.pending.step == last_game_state["step"]:
            # re-delivery of a step we already staged; keep this version, it has
            # the complete event list
            self.pending = None
        else:
            _flush(self)

        last_phi, last_obs, last_perm, _ = observe_and_encode(
            last_game_state, self.cfg.use_symmetry)
        # Phi(terminal) must be 0 for the policy-invariance guarantee, so the
        # shaping term on the terminal step is just -Phi(s).
        self.pending = Transition(
            step=last_game_state["step"],
            phi=last_phi,
            action=to_frame(last_perm, ACTIONS.index(last_action)),
            reward=reward_from(self, events, last_game_state) - _potential(self, last_obs),
            next_phi=None,
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
        "weight_norm": round(float(np.linalg.norm(self.model.w)), 4),
    })


def _write_log(self):
    if not self.cfg.log_path or not self.log_rows:
        return
    with open(self.cfg.log_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(self.log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(self.log_rows)
