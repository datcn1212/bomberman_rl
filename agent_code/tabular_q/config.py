"""All hyperparameters for the tabular agent.
"""

import json
import os
from dataclasses import dataclass, fields


@dataclass
class Config:
    # learning
    alpha: float = 0.1
    gamma: float = 0.995
    n_step: int = 1          # 1 = plain one-step Q-learning
    td_lambda: float = 0.0   # Watkins Q(lambda); 0 = off. Don't use with n_step > 1
    trace_floor: float = 1e-4
    alpha_schedule: str = "visit"   # "visit" or "constant"; see Phase 2
    alpha_half_life: float = 1000.0

    # exploration
    exploration: str = "epsilon"    # or "max_boltzmann"
    temperature: float = 0.5
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_episodes: int = 2000

    allow_bomb: bool = True         # off for Task 1, where a bomb can only hurt us
    use_symmetry: bool = True       # fold states onto their D4 orbit
    use_opponent_blocking: bool = True

    # rewards
    reward_coin: float = 1.0
    reward_kill: float = 5.0
    reward_crate: float = 0.3
    reward_coin_found: float = 0.1
    reward_bomb_wasted: float = 0.0
    reward_invalid: float = -0.5
    reward_step: float = -0.01
    reward_wait: float = -0.05
    reward_killed_self: float = -5.0
    reward_got_killed: float = -5.0
    reward_survived: float = 0.0
    reward_trapped: float = 0.0
    reward_bomb_no_escape: float = 0.0

    # potential-based shaping, Phi(s) = -w * distance to nearest target.
    # 0 turns it off.
    shaping_weight: float = 0.0
    shaping_distance_cap: int = 15

    trace_state: int = -1    # log Q values of this row each episode, -1 = off

    # run bookkeeping
    n_episodes: int = 6000
    seed: int = 0
    model_path: str = "model.pkl"
    continue_from: str = None
    log_path: str = None
    git_commit: str = ""


def load():
    """Config named by TQ_CONFIG, or the defaults if it isn't set."""
    cfg = Config()
    path = os.environ.get("TQ_CONFIG")
    if not path:
        return cfg

    with open(path) as fh:
        raw = json.load(fh)

    # Crash on a typo instead of quietly running the default
    unknown = set(raw) - {f.name for f in fields(Config)}
    if unknown:
        raise KeyError("unknown config keys: %s" % sorted(unknown))

    for key, value in raw.items():
        setattr(cfg, key, value)
    return cfg
