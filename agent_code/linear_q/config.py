"""Hyperparameters for the linear agent.

Same idea as tabular_q/config.py: one JSON file per run, path passed in
LQ_CONFIG. Values were chosen by the sweeps in report_linear_q.md.
"""

import json
import os
from dataclasses import dataclass, fields


@dataclass
class Config:
    # learning
    alpha: float = 0.001
    # "visit" decays alpha per weight entry w[i, a] by how often that entry was
    # actually updated. Note this is per (feature, action), not per state like
    # in tabular_q - one weight is shared by every state using that feature.
    alpha_schedule: str = "visit"
    alpha_half_life: float = 1000.0
    gamma: float = 0.995

    # exploration
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_episodes: int = 2000
    exploration: str = "epsilon"
    temperature: float = 0.5

    allow_bomb: bool = True
    use_symmetry: bool = True
    use_opponent_blocking: bool = True

    # rewards, copied from tabular_q: reward design depends on the game, not on
    # how we approximate Q.
    reward_coin: float = 1.0
    reward_kill: float = 5.0
    reward_crate: float = 0.3
    reward_coin_found: float = 0.1
    reward_invalid: float = -0.5
    reward_step: float = -0.01
    reward_wait: float = -0.05
    reward_killed_self: float = -5.0
    reward_got_killed: float = -5.0
    reward_survived: float = 0.0    # tried in Phase 9, no measurable gain
    reward_trapped: float = 0.0
    reward_bomb_no_escape: float = 0.0
    reward_bomb_wasted: float = 0.0

    # shaping: one potential towards the target, one towards safety
    shaping_weight: float = 0.0
    shaping_distance_cap: int = 15
    escape_shaping_weight: float = 0.1

    # run bookkeeping
    n_episodes: int = 12000
    seed: int = 0
    model_path: str = "model.pkl"
    log_path: str = None
    continue_from: str = None
    git_commit: str = None


def load():
    """Config named by LQ_CONFIG, or the defaults if it isn't set."""
    cfg = Config()
    path = os.environ.get("LQ_CONFIG")
    if not path:
        return cfg

    with open(path) as fh:
        raw = json.load(fh)

    unknown = set(raw) - {f.name for f in fields(Config)}
    if unknown:
        raise KeyError("unknown config keys: %s" % sorted(unknown))

    for key, value in raw.items():
        setattr(cfg, key, value)
    return cfg
