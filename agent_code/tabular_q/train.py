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
from collections import deque

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
    if self.cfg.n_step > 1 and self.cfg.td_lambda > 0.0:
        raise ValueError(
            "n_step and td_lambda are two answers to the same question; "
            "set one of them, not both")
    self.episode = 0
    self.pending = None
    # n-step returns wait here until the window is full. Q(lambda) does not use
    # it; the trace dictionary carries the same information incrementally.
    self.buffer = deque()
    self.traces = {}
    self.action_was_greedy = True
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


def _is_trapped(game_state):
    """Standing inside a blast schedule with no surviving move."""
    return Board(game_state).escape_search() == DIR_NONE


def _bomb_without_escape(old_game_state, events):
    """A bomb was dropped and nothing survives the blast it creates."""
    if e.BOMB_DROPPED not in events:
        return False
    pos = old_game_state["self"][3]
    return Board(old_game_state, extra_bomb=pos).escape_search() == DIR_NONE


def reward_from(self, events, old_game_state=None):
    """Map a step's events onto a scalar reward.

    Crates and revealed coins are counted per occurrence, because one
    well-placed bomb can destroy several at once and a flat bonus would make a
    bomb that clears one crate worth as much as a bomb that clears four.
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
        # KILLED_SELF always comes with GOT_KILLED; charging both would double
        # the penalty for suicide relative to being killed by someone else.
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
    """Bootstrap value of a state: max over *legal* actions only.

    With BOMB masked out its row stays at zero, and a plain max would bootstrap
    from that zero whenever every legal action is worth less -- an optimistic
    target for an action the agent is not even allowed to take.
    """
    return float(np.max(self.model.values(state)[self.legal]))


def _flush(self):
    """Hand the staged transition to whichever update rule is configured."""
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

    if self.cfg.td_lambda > 0.0:
        _lambda_update(self, t)
        return

    self.buffer.append(t)
    if t.terminal:
        # The episode is over, so every transition still waiting gets the
        # shortest return that is available to it rather than none at all.
        while self.buffer:
            _apply_n_step(self)
    elif len(self.buffer) >= self.cfg.n_step:
        _apply_n_step(self)


def _apply_n_step(self):
    """Update the oldest buffered transition from the window that follows it.

    G = r_0 + gamma r_1 + ... + gamma^(k-1) r_(k-1) + gamma^k max_a Q(s_k, a),
    truncated at the terminal transition when one falls inside the window. With
    n_step = 1 this is exactly one-step Q-learning, which is what keeps the
    default byte-identical to what came before.

    The intermediate actions are not corrected for being off-policy. That is the
    usual practical choice, and the bias it introduces is bounded by how often
    exploration fires inside a window of n steps; Q(lambda) is the arm that
    handles the same problem the principled way.
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
    """One step of Watkins's Q(lambda) with replacing traces.

    The trace decay belongs after the update and depends on whether the *next*
    action is greedy. By the time a transition reaches here the framework has
    already asked for that next action, so `self.action_was_greedy` is exactly
    the flag the algorithm wants.
    """
    cfg = self.cfg
    key = (t.state, t.action)
    self.traces[key] = 1.0                      # replacing trace
    self.model.note_visit(t.state, t.action)

    q_sa = float(self.model.values(t.state)[t.action])
    target = t.reward if t.terminal else t.reward + cfg.gamma * _best_next(self, t.next_state)
    delta = target - q_sa

    alpha = self.model.effective_alpha(t.state, t.action, cfg)
    for (state, action), trace in self.traces.items():
        self.model.add(state, action, alpha * delta * trace)

    if t.terminal:
        self.traces.clear()
        return
    if not self.action_was_greedy:
        # Watkins cuts here: beyond a non-greedy action the return no longer
        # estimates the greedy policy, so carrying the trace further would
        # credit states for a path the target policy would not have taken.
        self.traces.clear()
        return
    decay = cfg.gamma * cfg.td_lambda
    floor = cfg.trace_floor
    self.traces = {k: v * decay for k, v in self.traces.items() if v * decay > floor}


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    if old_game_state is None or self_action is None:
        return
    _flush(self)
    old_index, old_obs, old_perm = observe_and_encode(
        old_game_state, self.cfg.use_symmetry)
    new_index, new_obs, _ = observe_and_encode(
        new_game_state, self.cfg.use_symmetry)
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

    # Nothing may survive into the next episode: a trace or a half-filled window
    # would credit the first states of the new round with the last round's
    # reward. `last_action is None` is the one path that stages no terminal
    # transition, so the drain here is not always a no-op.
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
