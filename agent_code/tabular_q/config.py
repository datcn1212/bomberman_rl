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
    gamma: float = 0.9
    # "constant" keeps alpha fixed, which violates the Robbins-Monro condition
    # sum(alpha^2) < inf: the estimate never settles, it random-walks around the
    # true value forever. "visit" divides by the visit count of the individual
    # (state, action) pair, which satisfies it.
    alpha_schedule: str = "visit"
    alpha_half_life: float = 1000.0

    # --- exploration ------------------------------------------------------
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_episodes: int = 2000

    # --- action set -------------------------------------------------------
    # On a board with no crates and no opponents a bomb can only kill its owner,
    # so Task 1 can be run with BOMB removed. Phase 1 measures what that costs
    # instead of assuming it.
    allow_bomb: bool = True

    # Off by default. Phase 5 measured that this component carries no
    # information in the states where the bombing decision is made, while
    # fragmenting the travelling states by 25% -- which costs the movement
    # policy far more than the bombing policy gains.
    use_bomb_opt: bool = False

    # --- rewards ----------------------------------------------------------
    # The game's own score (1 per coin, 5 per kill) is far too sparse to learn
    # from, so the reward is assembled from the per-step event list. Every weight
    # is a config value, which makes a reward change an experiment rather than a
    # code edit.
    reward_coin: float = 1.0
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

    # --- diagnostics ------------------------------------------------------
    # Index of one table row whose Q values are written to the training log each
    # episode. Used to watch a single decision converge (or fail to).
    trace_state: int = -1

    # --- bookkeeping ------------------------------------------------------
    n_episodes: int = 1000
    seed: int = 0
    model_path: str = "model.pkl"
    continue_from: str = None
    log_path: str = None


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
