import os
from collections import deque

import numpy as np

from .features import state_to_features, ACTIONS, N_FEATURES, POSITION_HISTORY_LENGTH

MODEL_PATH = os.environ.get("Q_LINEAR_MODEL_PATH", "q_linear_weights.npy")
EVAL_EPSILON = float(os.environ.get("Q_LINEAR_EVAL_EPSILON", "0.0"))


def setup(self):
    if self.train or not os.path.isfile(MODEL_PATH):
        rng = np.random.default_rng()
        self.beta = rng.normal(0, 0.01, size=(len(ACTIONS), N_FEATURES)).astype(np.float32)
    else:
        self.beta = np.load(MODEL_PATH)

    self.current_round = 0
    self.position_history = deque(maxlen=POSITION_HISTORY_LENGTH)


def act(self, game_state: dict) -> str:
    if game_state['round'] != self.current_round:
        self.current_round = game_state['round']
        self.position_history = deque(maxlen=POSITION_HISTORY_LENGTH)

    pos = game_state['self'][3]
    recently_visited = pos in self.position_history

    features = state_to_features(game_state, recently_visited)
    self.last_features = features  # reused by train.py to avoid recomputing and to stay consistent
    q_values = self.beta @ features
    chosen = ACTIONS[int(np.argmax(q_values))]

    if os.environ.get("Q_LINEAR_DEBUG") == "1":
        self.logger.info(
            f"step={game_state['step']} pos={pos} recently_visited={recently_visited} "
            f"bombs_left={game_state['self'][2]} feat={features.tolist()} "
            f"q={q_values.tolist()} argmax={chosen}"
        )

    self.position_history.append(pos)

    eps = self.epsilon if self.train else EVAL_EPSILON
    if eps > 0 and np.random.random() < eps:
        return np.random.choice(ACTIONS)

    return chosen
