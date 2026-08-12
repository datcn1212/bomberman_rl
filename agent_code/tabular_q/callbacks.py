"""Entry points the framework calls.

The framework changes the working directory to this folder before every event,
so all paths here are relative and stay valid when the agent is dropped into
the tournament machine.
"""

import os

import numpy as np

from . import config
from .features import ACTIONS
from .model import QModel, state_of

MODEL_FILE = "model.pkl"


def setup(self):
    """Called once per round, before the first act()."""
    self.cfg = config.load()
    self.rng = np.random.default_rng(self.cfg.seed)
    self.epsilon = self.cfg.eps_start
    n = len(ACTIONS) if self.cfg.allow_bomb else ACTIONS.index("BOMB")
    self.legal = np.arange(n)
    self.bomb_log = os.environ.get("TQ_BOMBLOG")

    # Training either continues an explicit checkpoint or starts empty. Playing
    # reads `model_path`, which defaults to the file shipped next to this one;
    # the experiment harness overrides it to evaluate a specific checkpoint.
    source = self.cfg.continue_from if self.train else self.cfg.model_path
    if source and os.path.isfile(source):
        self.model = QModel.load(source)
        self.logger.info("loaded model from %s", source)
    elif self.train:
        self.model = QModel()
        self.logger.info("starting from an empty table")
    else:
        raise FileNotFoundError(
            "no model at %s; the agent cannot play without one" % source)


def act(self, game_state):
    state = state_of(game_state)
    values = self.model.values(state)[self.legal]

    if self.train and self.rng.random() < self.epsilon:
        choice = int(self.rng.integers(len(self.legal)))
    else:
        choice = greedy(self.rng, values)
    action = int(self.legal[choice])

    if self.bomb_log is not None:
        _record_bomb(self, game_state, action)

    self.last_state = state
    self.last_action = action
    return ACTIONS[action]


def _record_bomb(self, game_state, action):
    """Diagnostic: what a chosen BOMB would actually have achieved.

    Off unless TQ_BOMBLOG is set. Used to size how much of the score is lost to
    bombs that hit nothing, which is not visible in the aggregate crate count.
    """
    if ACTIONS[action] != "BOMB" or not game_state["self"][2]:
        return
    from .features import Board
    board = Board(game_state)
    hypothetical = Board(game_state, extra_bomb=board.pos)
    with open(self.bomb_log, "a") as fh:
        fh.write("%d,%d,%d\n" % (game_state["step"], board.bomb_payload(),
                                 hypothetical.escape_search()))


def greedy(rng, values):
    """Argmax with random tie-breaking.

    Plain argmax always returns the lowest index, which on an all-zero row is
    UP. That turns an unvisited state into a systematic bias rather than an
    arbitrary choice, and the bias is invisible in the reward curve.
    """
    best = np.flatnonzero(values == values.max())
    return int(best[0]) if len(best) == 1 else int(rng.choice(best))
