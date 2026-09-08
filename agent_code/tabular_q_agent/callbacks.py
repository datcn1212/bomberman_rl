import os
import pickle
import random
import yaml

import numpy as np

from .q_model import TabularQModel


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Load config values and create the agent
    """

    with open('config.yaml', 'r') as file:
        cfg = yaml.safe_load(file)

    q_file = cfg["load_q_table"]
    self.epsilon = cfg["epsilon"]
    self.lr = cfg["learning_rate"]
    self.gamma = cfg["gamma"]

    if self.train and not os.path.isfile("q_table.pkl"):
        self.logger.info("Setting up model from scratch.")
        self.model = TabularQModel(learning_rate=self.lr, gamma=self.gamma, actions=ACTIONS)
    elif self.train:
        self.logger.info("Loading model from saved state.")
        self.model = TabularQModel(learning_rate=self.lr, gamma=self.gamma, actions=ACTIONS)
        with open(q_file, "rb") as file:
            self.model.load(file)
    else:
        self.logger.info("Loading model from saved state for evaluation.")
        self.model = TabularQModel(learning_rate=0.0, gamma=self.gamma, actions=ACTIONS)
        self.epsilon = 0.0
        with open(q_file, "rb") as file:
            self.model.load(file)
        


def act(self, game_state: dict) -> str:
    """
    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    features = state_to_features(game_state)
    action = self.model.choose_action(features, self.epsilon)
    return action
    

def state_to_features(game_state: dict) -> np.array:
    """
    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    if game_state is None:
        return None

    agent_pos = game_state["self"][3]
    x, y = agent_pos

    field = game_state["field"]
    coins = game_state["coins"]  
    bombs = game_state["bombs"]
    explosion_map = game_state["explosion_map"]

    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    # Feature 1: free neighboring tiles (0/1)
    valid_moves = []
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if field[nx, ny] == 0 and explosion_map[nx, ny] == 0:
            valid_moves.append(0)
        else:
            valid_moves.append(1)

    # Feature 2: nearest coin (manhattan)
    coin_dir = [0, 0]  
    if coins:
        closest_coin = min(
            coins, key=lambda c: abs(c[0] - x) + abs(c[1] - y)
        )
        cx, cy = closest_coin
        coin_dir[0] = np.sign(cx - x)  # left (-1) or right (1)
        coin_dir[1] = np.sign(cy - y)  # up (-1) or down (1)

    # Feature 3: danger from bomb? (0/1)
    in_danger = 0
    for (bx, by), timer in bombs:
        if (bx == x and abs(by - y) <= 3) or (
            by == y and abs(bx - x) <= 3
        ):
            in_danger = 1
            break

    # feature tuple for dic
    features = valid_moves + coin_dir + [in_danger]
    return tuple(features)