import os
import numpy as np

from .features import state_to_features, ACTIONS, N_FEATURES

MODEL_PATH = os.environ.get("Q_LINEAR_MODEL_PATH", "q_linear_weights.npy")


def setup(self):
    if self.train or not os.path.isfile(MODEL_PATH):
        rng = np.random.default_rng()
        self.beta = rng.normal(0, 0.01, size=(len(ACTIONS), N_FEATURES)).astype(np.float32)
    else:
        self.beta = np.load(MODEL_PATH)


def act(self, game_state: dict) -> str:
    features = state_to_features(game_state)
    q_values = self.beta @ features

    if self.train and np.random.random() < self.epsilon:
        return np.random.choice(ACTIONS)

    return ACTIONS[int(np.argmax(q_values))]
