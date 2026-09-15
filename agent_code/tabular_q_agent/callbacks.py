import os
import pickle
import random
import yaml

from collections import deque

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
    

def state_to_features(game_state: dict) -> tuple:
    """
    """
    if game_state is None:
        return None

    agent_pos = game_state["self"][3]
    x, y = agent_pos

    field = game_state["field"]
    coins = game_state["coins"]  
    bombs = game_state["bombs"]
    explosion_map = game_state["explosion_map"]

    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)] # UP, DOWN, LEFT, RIGHT

    # Feature 1: Valid movements
    valid_moves = []
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if field[nx, ny] == 0 and explosion_map[nx, ny] == 0:
            valid_moves.append(0)
        else:
            valid_moves.append(1)

    # Feature 2 and 4: objective type and direction
    target_dir = [0, 0]
    target_type = 2 # Default 2 (no objects)
    
    crate_indices = np.argwhere(field == 1)
    crates = [tuple(c) for c in crate_indices]

    # BFS 
    bfs_result = get_closest_target_unified(agent_pos, coins, crates, field, explosion_map)
    
    if bfs_result:
        next_step, found_type = bfs_result
        target_type = found_type # 0 coind, 1 crate
        
        if next_step:
            nx, ny = next_step
            target_dir[0] = int(np.sign(nx - x))  
            target_dir[1] = int(np.sign(ny - y))
    else:
        # Fallback
        all_targets = coins + crates
        if all_targets:
            stable_targets = sorted(all_targets, key=lambda t: (t[0], t[1]))
            closest_target = min(stable_targets, key=lambda t: abs(t[0] - x) + abs(t[1] - y))
            target_dir[0] = int(np.sign(closest_target[0] - x))
            target_dir[1] = int(np.sign(closest_target[1] - y))
            target_type = 0 if closest_target in coins else 1

    # Feature 3: Danger
    in_danger = 0
    for (bx, by), timer in bombs:
        if bx == x and abs(by - y) <= 3:
            step = 1 if by > y else -1
            blocked = False
            for check_y in range(y + step, by, step):
                if field[x, check_y] != 0:  
                    blocked = True
                    break
            if not blocked:
                in_danger = 1
                break
                
        elif by == y and abs(bx - x) <= 3:
            step = 1 if bx > x else -1
            blocked = False
            for check_x in range(x + step, bx, step):
                if field[check_x, y] != 0:  
                    blocked = True
                    break
            if not blocked:
                in_danger = 1
                break

    features = valid_moves + target_dir + [in_danger] + [target_type]
    return tuple(features)


def get_closest_target_unified(start, coins, crates, field, explosion_map):
    """
    BFS search. Shortest path to the nearest object.
    Returns: (next step, type of objective)
    """
    coins_set = set(coins)
    crates_set = set(crates)
    
    queue = deque([[start]])
    visited = {start}
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    while queue:
        path = queue.popleft()
        current = path[-1]
        cx, cy = current

        if current in coins_set:
            next_step = path[1] if len(path) > 1 else None
            return next_step, 0
            
        for dx, dy in directions:
            if (cx + dx, cy + dy) in crates_set:
                next_step = path[1] if len(path) > 1 else None
                return next_step, 1

        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if (nx, ny) not in visited and field[nx, ny] == 0 and explosion_map[nx, ny] == 0:
                visited.add((nx, ny))
                new_path = list(path)
                new_path.append((nx, ny))
                queue.append(new_path)
                
    return None