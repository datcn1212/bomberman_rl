import numpy as np
from collections import deque

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
_DIRS = {'UP': (0, -1), 'RIGHT': (1, 0), 'DOWN': (0, 1), 'LEFT': (-1, 0)}
N_FEATURES = 9


def _bfs_to_nearest(field, start, targets):
    """BFS over free tiles (field == 0) from start to the closest target.

    Returns (distance, first_action): first_action is the direction name
    of the first step on the shortest path, or (None, None) if no target
    is reachable.
    """
    if len(targets) == 0:
        return None, None
    target_set = set(map(tuple, targets))
    visited = {start}
    queue = deque([(start, 0, None)])
    while queue:
        (x, y), dist, first_action = queue.popleft()
        if (x, y) in target_set:
            return dist, first_action
        for action, (dx, dy) in _DIRS.items():
            nx, ny = x + dx, y + dy
            if (nx, ny) in visited or field[nx, ny] != 0:
                continue
            visited.add((nx, ny))
            next_first_action = first_action if first_action is not None else action
            queue.append(((nx, ny), dist + 1, next_first_action))
    return None, None


def nearest_coin_distance(game_state: dict):
    """Shortest-path distance to the nearest coin, or None if no coin remains."""
    field = game_state['field']
    start = game_state['self'][3]
    coins = game_state['coins']
    dist, _ = _bfs_to_nearest(field, start, coins)
    return dist


def state_to_features(game_state: dict) -> np.ndarray:
    """9-dim vector: [4] one-hot BFS direction to nearest coin, [4] valid-move
    flags (UP/RIGHT/DOWN/LEFT), [1] bias.
    """
    field = game_state['field']
    x, y = game_state['self'][3]
    coins = game_state['coins']

    features = np.zeros(N_FEATURES, dtype=np.float32)

    _, first_action = _bfs_to_nearest(field, (x, y), coins)
    if first_action is not None:
        features[ACTIONS.index(first_action)] = 1.0

    for i, action in enumerate(['UP', 'RIGHT', 'DOWN', 'LEFT']):
        dx, dy = _DIRS[action]
        if field[x + dx, y + dy] == 0:
            features[4 + i] = 1.0

    features[8] = 1.0  # bias
    return features
