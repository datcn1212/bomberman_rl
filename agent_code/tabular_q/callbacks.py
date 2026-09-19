"""What the framework calls: setup() once, act() every step.
The framework chdir's into this folder before each event, so relative paths
here still work when the agent is dropped onto the tournament machine.
"""

import os
import numpy as np

from . import config
from . import features
from .features import ACTIONS
from .model import QModel, from_frame, observe_and_encode

MODEL_FILE = "model.pkl"


def setup(self):
    self.cfg = config.load()
    self.rng = np.random.default_rng(self.cfg.seed)
    self.epsilon = self.cfg.eps_start
    n = len(ACTIONS) if self.cfg.allow_bomb else ACTIONS.index("BOMB")
    self.legal = np.arange(n)
    self.bomb_log = os.environ.get("TQ_BOMBLOG")

    features.FLAGS["use_opponent_blocking"] = self.cfg.use_opponent_blocking

    source = self.cfg.continue_from if self.train else self.cfg.model_path
    if source and os.path.isfile(source):
        self.model = QModel.load(source)
        stored = dict(getattr(self.model, "feature_flags", None) or {})
        if "use_symmetry" in stored:
            self.cfg.use_symmetry = stored.pop("use_symmetry")
        if stored:
            features.FLAGS.update(stored)
        self.logger.info("loaded model from %s (flags %s)", source, features.FLAGS)
    elif self.train:
        self.model = QModel()
        self.logger.info("starting from an empty table")
    else:
        raise FileNotFoundError(
            "no model at %s; the agent cannot play without one" % source)


def act(self, game_state):
    state, _, perm = observe_and_encode(game_state, self.cfg.use_symmetry)
    values = self.model.values(state)[self.legal]

    if self.train and self.rng.random() < self.epsilon:
        choice = explore(self, values)
    else:
        choice = greedy(self.rng, values)

    # Q(lambda) cuts the trace on a non-greedy action
    self.action_was_greedy = bool(values[choice] == values.max())

    action = from_frame(perm, int(self.legal[choice]))
    if self.bomb_log is not None:
        _record_bomb(self, game_state, action)
    return ACTIONS[action]


def explore(self, values):
    if self.cfg.exploration == "epsilon":
        return int(self.rng.integers(len(values)))

    if self.cfg.exploration == "max_boltzmann":
        # subtract the max first so exp() can't overflow
        scaled = (values - values.max()) / max(self.cfg.temperature, 1e-6)
        weights = np.exp(scaled)
        total = weights.sum()
        if not np.isfinite(total) or total <= 0:
            return int(self.rng.integers(len(values)))
        return int(self.rng.choice(len(values), p=weights / total))

    raise ValueError("unknown exploration %r" % self.cfg.exploration)


def greedy(rng, values):
    best = np.flatnonzero(values == values.max())
    return int(best[0]) if len(best) == 1 else int(rng.choice(best))


def _record_bomb(self, game_state, action):
    """Log what a chosen BOMB would have hit. Off unless TQ_BOMBLOG is set."""
    if ACTIONS[action] != "BOMB" or not game_state["self"][2]:
        return
    from .features import Board
    board = Board(game_state)
    hypothetical = Board(game_state, extra_bomb=board.pos)
    with open(self.bomb_log, "a") as fh:
        fh.write("%d,%d,%d\n" % (game_state["step"], board.bomb_payload(),
                                 hypothetical.escape_search()))
