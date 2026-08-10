import os
from collections import deque

import numpy as np

from .features import encode_state, stuck_ratio, ACTIONS, N_STATES, POSITION_HISTORY_LENGTH

MODEL_PATH = os.environ.get("Q_TABULAR_MODEL_PATH", "q_tabular_table.npy")
EVAL_EPSILON = float(os.environ.get("Q_TABULAR_EVAL_EPSILON", "0.0"))


CONTINUE_TRAINING = os.environ.get("Q_TABULAR_CONTINUE") == "1"


def setup(self):
    # Default: a fresh training run always starts from a zero table, even if
    # MODEL_PATH already exists (e.g. from a previous experiment) -- avoids
    # accidentally continuing an unrelated run. Q_TABULAR_CONTINUE=1 opts
    # into loading an existing table and training on top of it, for
    # curriculum learning (train on one scenario, keep learning on another).
    if os.path.isfile(MODEL_PATH) and (CONTINUE_TRAINING or not self.train):
        self.q_table = np.load(MODEL_PATH)
    else:
        self.q_table = np.zeros((N_STATES, len(ACTIONS)), dtype=np.float32)

    self.current_round = 0
    self.position_history = deque(maxlen=POSITION_HISTORY_LENGTH)


def act(self, game_state: dict) -> str:
    if game_state['round'] != self.current_round:
        self.current_round = game_state['round']
        self.position_history = deque(maxlen=POSITION_HISTORY_LENGTH)

    pos = game_state['self'][3]
    stuck = stuck_ratio(self.position_history)

    state = encode_state(game_state, stuck)
    self.last_state = state  # reused by train.py to avoid recomputing and to stay consistent
    q_values = self.q_table[state]
    chosen = ACTIONS[int(np.argmax(q_values))]

    if os.environ.get("Q_TABULAR_DEBUG") == "1":
        self.logger.info(
            f"step={game_state['step']} pos={pos} stuck_ratio={stuck:.2f} state={state} "
            f"q={q_values.tolist()} argmax={chosen}"
        )

    self.position_history.append(pos)

    eps = self.epsilon if self.train else EVAL_EPSILON
    if eps > 0 and np.random.random() < eps:
        return np.random.choice(ACTIONS)

    return chosen
