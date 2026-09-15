"""Hyperparameters for the tabular agent.

Every knob lives here so that an experiment is fully described by one JSON file.
`tools/train.py` writes that file and points the agent at it through the
TQ_CONFIG environment variable, which keeps the agent itself free of any
command line parsing and makes each run reproducible from its config alone.
"""

import json
import os
from dataclasses import dataclass, fields


@dataclass
class Config:
    # --- learning ---------------------------------------------------------
    alpha: float = 0.1
    gamma: float = 0.995

    # How far the reward is carried back in one update. A bomb pays off 4-5
    # decisions after it is placed (features.HORIZON), so one-step TD has to
    # push that signal back through five separate table updates, each damped by
    # alpha. n_step = 1 is plain one-step Q-learning and is the default.
    n_step: int = 1

    # Watkins's Q(lambda): eligibility traces that decay by gamma*lambda and are
    # cut whenever a non-greedy action is taken, which is what keeps the
    # off-policy target honest. 0.0 disables traces. Mutually exclusive with
    # n_step > 1 - they are two answers to the same question.
    td_lambda: float = 0.0

    # Traces below this are dropped, which bounds the dictionary without
    # changing the arithmetic in any way that a float can see.
    trace_floor: float = 1e-4
    # "constant" keeps alpha fixed, which violates the Robbins-Monro condition
    # sum(alpha^2) < inf: the estimate never settles, it random-walks around the
    # true value forever. "visit" divides by the visit count of the individual
    # (state, action) pair, which satisfies it.
    alpha_schedule: str = "visit"
    alpha_half_life: float = 1000.0

    # --- exploration ------------------------------------------------------
    # "epsilon": the exploratory action is drawn uniformly, so it is as likely
    # to be the worst action as a plausible one.
    # "max_boltzmann": the exploratory action is drawn from softmax(Q/tau), so
    # exploration concentrates on actions the agent has reason to think are
    # good. Phase 7 measured that under uniform exploration the agent learns
    # almost nothing until epsilon has decayed, wasting most of the budget.
    exploration: str = "epsilon"
    temperature: float = 0.5
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_episodes: int = 2000

    # --- action set -------------------------------------------------------
    # On a board with no crates and no opponents a bomb can only kill its owner,
    # so Task 1 can be run with BOMB removed. Phase 1 measures what that costs
    # instead of assuming it.
    allow_bomb: bool = True

    # Fold each state onto the smallest member of its D4 orbit (four rotations
    # x two reflections). Every feature is relative to the agent, so rotating
    # the board is the same situation described in a different frame; without
    # this, a horizontal and a vertical corridor are learned separately.
    use_symmetry: bool = True

    # Treat a tile an opponent stands on as blocked for the immediate move.
    # Not a new state component - it corrects what the existing ones report, so
    # it costs no resolution. Without it the escape search routes paths through
    # other agents' bodies, the move turns out invalid, and the agent stands
    # still inside a blast.
    use_opponent_blocking: bool = True


    # --- rewards ----------------------------------------------------------
    # The game's own score (1 per coin, 5 per kill) is far too sparse to learn
    # from, so the reward is assembled from the per-step event list. Every weight
    # is a config value, which makes a reward change an experiment rather than a
    # code edit.
    reward_coin: float = 1.0
    # The game scores a kill at 5 and a coin at 1, so the ratio here matches the
    # objective rather than being tuned to it.
    reward_kill: float = 5.0
    reward_crate: float = 0.3
    reward_coin_found: float = 0.1
    # Charged the moment a bomb is dropped whose blast covers no crate. Phase 4
    # measured why this is needed: the crate reward arrives four steps later, by
    # which point the trajectories of a good bomb and a useless one have merged,
    # so the delayed signal cannot be attributed back to the decision.
    reward_bomb_wasted: float = 0.0
    reward_invalid: float = -0.5
    reward_step: float = -0.01
    reward_wait: float = -0.05
    reward_killed_self: float = -5.0
    reward_got_killed: float = -5.0
    reward_survived: float = 0.0
    # Charged while standing in a blast with no escape route at all. Unlike a
    # flat death penalty this fires before the agent dies, on the step where the
    # situation is still legible, so it can be attributed to what caused it.
    reward_trapped: float = 0.0
    # Charged for dropping a bomb that leaves no escape - the decision, rather
    # than the situation it creates.
    reward_bomb_no_escape: float = 0.0

    # --- potential-based shaping -----------------------------------------
    # F(s, a, s') = gamma * Phi(s') - Phi(s) with Phi(s) = -w * distance to the
    # nearest target. Ng et al. (1999) show this form leaves the optimal policy
    # unchanged, which an ad-hoc "reward for stepping closer" bonus does not:
    # that one can be farmed by oscillating towards and away from a target.
    # Set to 0 to switch shaping off.
    shaping_weight: float = 0.0
    shaping_distance_cap: int = 15

    # --- diagnostics ------------------------------------------------------
    # Index of one table row whose Q values are written to the training log each
    # episode. Used to watch a single decision converge (or fail to).
    trace_state: int = -1

    # --- bookkeeping ------------------------------------------------------
    n_episodes: int = 6000
    seed: int = 0
    model_path: str = "model.pkl"
    continue_from: str = None
    log_path: str = None
    # Stamped by the training driver so a stored model can be traced back to the
    # exact agent code that produced it.
    git_commit: str = ""


def load():
    """Read the config named by TQ_CONFIG, falling back to the defaults."""
    cfg = Config()
    path = os.environ.get("TQ_CONFIG")
    if not path:
        return cfg
    with open(path) as fh:
        raw = json.load(fh)
    known = {f.name for f in fields(Config)}
    unknown = set(raw) - known
    if unknown:
        # A silently ignored key means an experiment that did not test what its
        # name claims, which is worse than a crash.
        raise KeyError("unknown config keys: %s" % sorted(unknown))
    for key, value in raw.items():
        setattr(cfg, key, value)
    return cfg
