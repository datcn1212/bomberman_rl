"""Entry points the framework calls.

The framework changes the working directory to this folder before every event,
so all paths here are relative and stay valid when the agent is dropped into
the tournament machine.
"""

import os

import numpy as np

from . import config
from . import features
from .features import ACTIONS, BLOCKED
from .model import LinearQModel, from_frame, observe_and_encode

MODEL_FILE = "model.pkl"


def setup(self):
    """Called once per round, before the first act()."""
    self.cfg = config.load()
    self.rng = np.random.default_rng(self.cfg.seed)
    self.epsilon = self.cfg.eps_start
    n = len(ACTIONS) if self.cfg.allow_bomb else ACTIONS.index("BOMB")
    self.legal = np.arange(n)
    self.bomb_log = os.environ.get("LQ_BOMBLOG")

    features.FLAGS["use_opponent_blocking"] = self.cfg.use_opponent_blocking

    # Training either continues an explicit checkpoint or starts empty. Playing
    # reads `model_path`, which defaults to the file shipped next to this one;
    # the experiment harness overrides it to evaluate a specific checkpoint.
    source = self.cfg.continue_from if self.train else self.cfg.model_path
    if source and os.path.isfile(source):
        self.model = LinearQModel.load(source)
        stored = dict(getattr(self.model, "feature_flags", None) or {})
        if "use_symmetry" in stored:
            self.cfg.use_symmetry = stored.pop("use_symmetry")
        if stored:
            features.FLAGS.update(stored)
        self.logger.info("loaded model from %s (flags %s)", source, features.FLAGS)
    elif self.train:
        self.model = LinearQModel()
        self.logger.info("starting from a zero weight matrix")
    else:
        raise FileNotFoundError(
            "no model at %s; the agent cannot play without one" % source)


def act(self, game_state):
    # `perm` is the frame phi() was vectorised in: with symmetry on, the chosen
    # action has to be translated back out of the canonical frame before it is
    # returned to the game. `status` is move_status in that same frame, so it
    # indexes the same four directions `values` does.
    phi, _, perm, status = observe_and_encode(game_state, self.cfg.use_symmetry)
    values = self.model.values(phi)

    # A move into a wall is invalid regardless of what the model predicts for
    # it - Q(blocked direction) can still rank highest, because move_status and
    # target_dir are separate additive features and nothing forces "blocked"
    # to dominate their sum (see model.py's module docstring for how this was
    # found: a greedy policy that walked into a wall and, since the position
    # and therefore the state never changed, repeated the identical invalid
    # choice for the rest of the episode). Excluding it here is the same
    # mechanism already used for BOMB when `allow_bomb` is off, just decided
    # every step instead of once at setup: the model still computes and still
    # updates Q for a blocked direction whenever a transition visits it, it is
    # only kept out of the decision.
    candidates = [a for a in self.legal if a >= 4 or status[a] != BLOCKED]
    values = values[candidates]

    if self.train and self.rng.random() < self.epsilon:
        choice = explore(self, values)
    else:
        choice = greedy(self.rng, values)
    action = from_frame(perm, int(candidates[choice]))

    if self.bomb_log is not None:
        _record_bomb(self, game_state, action)

    return ACTIONS[action]


def _record_bomb(self, game_state, action):
    """Diagnostic: what a chosen BOMB would actually have achieved.

    Off unless LQ_BOMBLOG is set. Used to size how much of the score is lost to
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


def explore(self, values):
    """Pick an exploratory action."""
    if self.cfg.exploration == "epsilon":
        return int(self.rng.integers(len(values)))
    if self.cfg.exploration == "max_boltzmann":
        # Subtracting the max before exponentiating keeps this finite for large
        # values; it cancels out of the normalised probabilities.
        scaled = (values - values.max()) / max(self.cfg.temperature, 1e-6)
        weights = np.exp(scaled)
        total = weights.sum()
        if not np.isfinite(total) or total <= 0:
            return int(self.rng.integers(len(values)))
        return int(self.rng.choice(len(values), p=weights / total))
    raise ValueError("unknown exploration %r" % self.cfg.exploration)


def greedy(rng, values):
    """Argmax with random tie-breaking.

    Plain argmax always returns the lowest index, which on an all-zero row is
    UP. That turns an untrained model into a systematic bias rather than an
    arbitrary choice, and the bias is invisible in the reward curve.
    """
    best = np.flatnonzero(values == values.max())
    return int(best[0]) if len(best) == 1 else int(rng.choice(best))
