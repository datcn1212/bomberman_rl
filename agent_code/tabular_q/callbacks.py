"""Entry points the framework calls.

The framework changes the working directory to this folder before every event,
so all paths here are relative and stay valid when the agent is dropped into
the tournament machine.
"""

import os

import numpy as np

from . import config
from . import features
from .features import ACTIONS
from .model import QModel, from_frame, observe_and_encode

MODEL_FILE = "model.pkl"


def setup(self):
    """Called once per round, before the first act()."""
    self.cfg = config.load()
    self.rng = np.random.default_rng(self.cfg.seed)
    self.epsilon = self.cfg.eps_start
    n = len(ACTIONS) if self.cfg.allow_bomb else ACTIONS.index("BOMB")
    self.legal = np.arange(n)
    self.bomb_log = os.environ.get("TQ_BOMBLOG")
    self.obs_log = os.environ.get("TQ_OBSLOG")

    # The observation must match the one the model was trained with, so at play
    # time the model's own record wins over whatever the config happens to say.
    features.FLAGS["use_bomb_opt"] = self.cfg.use_bomb_opt
    self.use_symmetry = self.cfg.use_symmetry

    # Training either continues an explicit checkpoint or starts empty. Playing
    # reads `model_path`, which defaults to the file shipped next to this one;
    # the experiment harness overrides it to evaluate a specific checkpoint.
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
    # `perm` is the frame the row is written in: with symmetry on, the chosen
    # action has to be translated back out of the canonical frame before it is
    # returned to the game.
    state, _, perm = observe_and_encode(game_state, self.cfg.use_symmetry)
    values = self.model.values(state)[self.legal]

    if self.train and self.rng.random() < self.epsilon:
        choice = explore(self, values)
    else:
        choice = greedy(self.rng, values)
    action = from_frame(perm, int(self.legal[choice]))

    if self.bomb_log is not None:
        _record_bomb(self, game_state, action)
    if self.obs_log is not None:
        _record_observation(self, game_state)

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
    UP. That turns an unvisited state into a systematic bias rather than an
    arbitrary choice, and the bias is invisible in the reward curve.
    """
    best = np.flatnonzero(values == values.max())
    return int(best[0]) if len(best) == 1 else int(rng.choice(best))


def _record_observation(self, game_state):
    """Diagnostic: dump the full feature tuple of every step visited.

    Off unless TQ_OBSLOG is set. Used to ask whether a state component carries
    information the others do not already imply.
    """
    from .features import Board, bomb_option, observe
    obs = observe(game_state)
    # bomb_option is computed directly rather than read off the observation, so
    # the true value is recorded even when the ablation switch holds it constant.
    true_bomb_opt = bomb_option(game_state, Board(game_state))
    with open(self.obs_log, "a") as fh:
        fh.write("%d,%d,%d,%d,%d,%d,%d,%d,%d\n" % (
            game_state["round"], game_state["step"], obs.pos[0], obs.pos[1],
            obs.target_dir, obs.target_kind, true_bomb_opt, obs.target_dist,
            len(game_state["coins"])))
