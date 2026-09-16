"""Training callbacks: the Q-learning update itself.

One quirk of the framework shapes this whole file (measured in Phase 0):

  - if the agent survives, the last step arrives twice, once through
    game_events_occurred and again through end_of_round with SURVIVED_ROUND;
  - if it dies, the fatal step arrives only through end_of_round.

So we stage a transition instead of learning on arrival, and flush it when the
next one turns up. end_of_round either replaces the staged one (same step
number = it's the re-delivery) or flushes it first (different step = the agent
died and this is a new transition).
"""

import csv
from collections import deque

import numpy as np

import events as e

from . import features
from .features import ACTIONS, DIR_NONE, Board
from .model import observe_and_encode, to_frame


class Transition:
    __slots__ = ("step", "state", "action", "reward", "next_state", "terminal",
                 "coins", "crates", "bombs")

    def __init__(self, step, state, action, reward, next_state, terminal,
                 coins, crates, bombs):
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
    # Store the flags on the model so it can't later be evaluated with a
    # different observation than it was trained with.
    self.model.feature_flags = dict(features.FLAGS)
    self.model.feature_flags["use_symmetry"] = self.cfg.use_symmetry

    if self.cfg.n_step > 1 and self.cfg.td_lambda > 0.0:
        raise ValueError("n_step and td_lambda solve the same problem; pick one")

    self.episode = 0
    self.pending = None
    self.buffer = deque()      # n-step window
    self.traces = {}           # Q(lambda) eligibility traces
    self.action_was_greedy = True
    self.episode_reward = 0.0
    self.episode_coins = 0
    self.episode_crates = 0
    self.bombs_dropped = 0
    self.log_rows = []


def _potential(self, obs):
    """Phi(s) = -w * distance to nearest target, capped.

    The cap makes "no target" and "target very far" the same value, otherwise
    the shaping term jumps when the last coin gets collected.
    """
    if not self.cfg.shaping_weight:
        return 0.0
    dist = self.cfg.shaping_distance_cap
    if obs.target_dir != DIR_NONE:
        dist = min(obs.target_dist, self.cfg.shaping_distance_cap)
    return -self.cfg.shaping_weight * dist


def _wasted_bomb(old_game_state, events):
    """Bomb dropped whose blast hits no crate.

    Checked on the state we bombed from, so the penalty lands on the decision
    and not four steps later where it can't be attributed any more (Phase 4).
    """
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
    """Scalar reward for one step's events.

    Crates and revealed coins count per occurrence - one good bomb can clear
    four crates, and a flat bonus would price that the same as clearing one.
    """
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
        # KILLED_SELF always comes with GOT_KILLED, so this is elif: otherwise
        # suicide costs double what being killed by someone else costs.
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


def _best_next(self, state):
    """max Q over legal actions only.

    With BOMB masked its column stays zero, and a plain max would bootstrap off
    that zero whenever the legal actions are all worth less.
    """
    return float(np.max(self.model.values(state)[self.legal]))


def _flush(self):
    """Send the staged transition to whichever update rule is configured."""
    t = self.pending
    self.pending = None
    if t is None:
        return

    # Stats are counted here, not on arrival, so a transition we drop as a
    # re-delivery isn't counted twice (that's what made 50-coin boards log 51).
    self.episode_reward += t.reward
    self.episode_coins += t.coins
    self.episode_crates += t.crates
    self.bombs_dropped += t.bombs

    if self.cfg.td_lambda > 0.0:
        _lambda_update(self, t)
        return

    self.buffer.append(t)
    if t.terminal:
        # Episode over: give everything still waiting the shortest return it
        # can get rather than no update at all.
        while self.buffer:
            _apply_n_step(self)
    elif len(self.buffer) >= self.cfg.n_step:
        _apply_n_step(self)


def _apply_n_step(self):
    """Update the oldest buffered transition using the window after it.

    G = r0 + gamma*r1 + ... + gamma^(k-1)*r(k-1) + gamma^k * max_a Q(sk, a),
    cut short if a terminal transition falls inside the window. n_step = 1 is
    ordinary one-step Q-learning.

    The intermediate actions aren't importance-corrected. That's the usual
    shortcut; Q(lambda) is the arm that handles it properly.
    """
    g = 0.0
    discount = 1.0
    bootstrap = None
    for i, t in enumerate(self.buffer):
        if i >= self.cfg.n_step:
            break
        g += discount * t.reward
        discount *= self.cfg.gamma
        if t.terminal:
            bootstrap = None
            break
        bootstrap = t.next_state
    if bootstrap is not None:
        g += discount * _best_next(self, bootstrap)

    head = self.buffer.popleft()
    alpha = self.model.effective_alpha(head.state, head.action, self.cfg)
    self.model.update(head.state, head.action, g, alpha)


def _lambda_update(self, t):
    """Watkins Q(lambda), replacing traces.

    Decay happens after the update and depends on whether the *next* action was
    greedy. The framework has already asked for that action by the time we get
    here, so self.action_was_greedy is the flag we want.
    """
    cfg = self.cfg
    self.traces[(t.state, t.action)] = 1.0

    q_sa = float(self.model.values(t.state)[t.action])
    target = t.reward if t.terminal else t.reward + cfg.gamma * _best_next(self, t.next_state)
    delta = target - q_sa

    # Each traced pair steps with its own alpha. Sharing the visited pair's
    # alpha lets a pair seen 10k times take full-size steps borrowed from some
    # fresh state - that breaks the Robbins-Monro condition and sent Q to NaN.
    for (state, action), trace in self.traces.items():
        alpha = self.model.effective_alpha(state, action, cfg)
        self.model.add(state, action, alpha * delta * trace)
    # After the updates, matching the one-step path where alpha is read before
    # update() bumps the counter.
    self.model.note_visit(t.state, t.action)

    if t.terminal:
        self.traces.clear()
        return
    if not self.action_was_greedy:
        # Watkins cuts here: past a non-greedy action the return no longer
        # estimates the greedy policy.
        self.traces.clear()
        return

    decay = cfg.gamma * cfg.td_lambda
    floor = cfg.trace_floor
    self.traces = {k: v * decay for k, v in self.traces.items() if v * decay > floor}


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
            # Same step already staged: this is the re-delivery, and the
            # end_of_round copy has the full event list, so keep that one.
            self.pending = None
        else:
            _flush(self)

        last_index, last_obs, last_perm = observe_and_encode(
            last_game_state, self.cfg.use_symmetry)
        # Ng et al. need Phi(terminal) = 0, so the shaping term here is -Phi(s).
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

    # Nothing may carry into the next episode: a leftover trace or half-full
    # window would credit the new round's first states with the old round's
    # reward. When last_action is None nothing terminal was staged, so this
    # drain isn't always a no-op.
    while self.buffer:
        _apply_n_step(self)
    self.traces.clear()

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
