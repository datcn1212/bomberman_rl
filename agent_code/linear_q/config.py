"""Hyperparameters for the linear agent.

Mirrors `tabular_q/config.py`'s convention: every knob lives here so an
experiment is fully described by one JSON file, written by `tools/train.py` and
pointed at through the LQ_CONFIG environment variable.

`alpha_schedule` offers only "constant" for now. A tabular agent's visit-count
decay is meaningless here - a weight is touched by every state that shares an
active feature, not by one state alone - so what a correct schedule looks like
is an open question, not a known one. Phase 1 answers it by first proving a
constant rate converges at all, the same order tabular_q's own Phase 2 followed.
"""

import json
import os
from dataclasses import dataclass, fields


@dataclass
class Config:
    # --- learning -----------------------------------------------------
    alpha: float = 0.01
    alpha_schedule: str = "constant"
    gamma: float = 0.995

    # --- exploration ----------------------------------------------------
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_episodes: int = 2000
    exploration: str = "epsilon"
    temperature: float = 0.5

    allow_bomb: bool = True
    use_symmetry: bool = True
    use_opponent_blocking: bool = True

    # --- rewards ----------------------------------------------------
    # Identical to tabular_q's defaults at the point this branch forked: reward
    # design is a property of the events and the game, not of the function
    # approximator, so there is no reason to re-derive it from scratch. Revisit
    # only if a linear-specific failure mode traces back to one of these.
    reward_coin: float = 1.0
    reward_kill: float = 5.0
    reward_crate: float = 0.3
    reward_coin_found: float = 0.1
    reward_invalid: float = -0.5
    reward_step: float = -0.01
    reward_wait: float = -0.05
    reward_killed_self: float = -5.0
    reward_got_killed: float = -5.0
    reward_survived: float = 0.0
    reward_trapped: float = 0.0
    reward_bomb_no_escape: float = 0.0
    reward_bomb_wasted: float = 0.0

    shaping_weight: float = 0.0
    shaping_distance_cap: int = 15

    # --- run plumbing -----------------------------------------------------
    n_episodes: int = 6000
    seed: int = 0
    model_path: str = "model.pkl"
    log_path: str = None
    continue_from: str = None
    git_commit: str = None


def load():
    """Read the config named by LQ_CONFIG, falling back to the defaults."""
    cfg = Config()
    path = os.environ.get("LQ_CONFIG")
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
