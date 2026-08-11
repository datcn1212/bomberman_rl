import os
import pickle
from collections import deque

import numpy as np

from .features import state_to_features, stuck_ratio, ACTIONS, POSITION_HISTORY_LENGTH

MODEL_PATH = os.environ.get("Q_RF_MODEL_PATH", "q_rf_model.pkl")
EVAL_EPSILON = float(os.environ.get("Q_RF_EVAL_EPSILON", "0.0"))
CONTINUE_TRAINING = os.environ.get("Q_RF_CONTINUE") == "1"


def setup(self):
    # self.forests is a list of one fitted regressor per action, or None until
    # the first fitted-Q iteration has run. Unlike a weight vector or a table,
    # an unfitted forest cannot be queried at all, so `act` needs an explicit
    # "not ready yet" path -- see below.
    if os.path.isfile(MODEL_PATH) and (CONTINUE_TRAINING or not self.train):
        with open(MODEL_PATH, "rb") as f:
            self.forests = pickle.load(f)
    else:
        self.forests = None

    self.current_round = 0
    self.position_history = deque(maxlen=POSITION_HISTORY_LENGTH)


def q_values(self, features):
    """Q(s, a) for every action, as a length-6 array. Returns zeros while no
    forest is fitted yet.
    """
    if self.forests is None:
        return np.zeros(len(ACTIONS), dtype=np.float32)
    row = features.reshape(1, -1)
    return np.array([forest.predict(row)[0] for forest in self.forests], dtype=np.float32)


def act(self, game_state: dict) -> str:
    if game_state['round'] != self.current_round:
        self.current_round = game_state['round']
        self.position_history = deque(maxlen=POSITION_HISTORY_LENGTH)

    pos = game_state['self'][3]
    stuck = stuck_ratio(self.position_history)

    features = state_to_features(game_state, stuck)
    self.last_features = features  # reused by train.py so both see the same vector

    q = q_values(self, features)
    chosen = ACTIONS[int(np.argmax(q))]

    if os.environ.get("Q_RF_DEBUG") == "1":
        self.logger.info(
            f"step={game_state['step']} pos={pos} stuck_ratio={stuck:.2f} "
            f"feat={features.tolist()} q={q.tolist()} argmax={chosen}"
        )

    self.position_history.append(pos)

    eps = self.epsilon if self.train else EVAL_EPSILON
    # Before the first fit every Q is 0, so argmax would always return UP and
    # the agent would collect a useless, near-constant first batch. Act purely
    # at random until there is a model to consult.
    if self.forests is None or (eps > 0 and np.random.random() < eps):
        return np.random.choice(ACTIONS)

    return chosen
