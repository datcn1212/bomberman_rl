import numpy as np
from collections import deque

import settings as s

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
_DIRS = {'UP': (0, -1), 'RIGHT': (1, 0), 'DOWN': (0, 1), 'LEFT': (-1, 0)}
_DIR_NAMES = ['UP', 'RIGHT', 'DOWN', 'LEFT']

POSITION_HISTORY_LENGTH = 8

# Discrete state = (target_dir, target_is_coin, danger_now, escape_dir, can_bomb_here, stuck_bucket).
# target_dir/escape_dir are BFS-computed and by construction always point to a
# walkable tile, so no separate per-direction valid-move flags are needed
# (unlike the linear model's feature vector, which needed them because a
# linear model cannot itself run a BFS at inference time -- a tabular Q-table
# has no such restriction, so the discretization can fold pathfinding
# straight into the state instead of exposing raw local geometry).
_DIR5 = _DIR_NAMES + ['NONE']  # 5
N_TARGET_DIR = 5
N_TARGET_COIN = 2
N_DANGER = 2
N_ESCAPE_DIR = 5
N_CAN_BOMB = 2
N_STUCK = 3
N_STATES = N_TARGET_DIR * N_TARGET_COIN * N_DANGER * N_ESCAPE_DIR * N_CAN_BOMB * N_STUCK


def stuck_ratio(position_history):
    """Same definition as the linear model's stuck_ratio: 1 - (distinct tiles
    in position_history) / len(position_history). Shared concept, not shared
    code -- q_tabular must stay self-contained in its own directory.
    """
    if len(position_history) == 0:
        return 0.0
    return 1.0 - len(set(position_history)) / len(position_history)


def _stuck_bucket(ratio):
    if ratio < 1 / 3:
        return 0  # LOW
    if ratio < 2 / 3:
        return 1  # MED
    return 2  # HIGH


def _bfs(field, start, targets):
    """BFS over free tiles (field == 0) from start to the closest of
    `targets`. Returns (distance, first_action_name) or (None, None) if
    unreachable.
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
            next_first = first_action if first_action is not None else action
            queue.append(((nx, ny), dist + 1, next_first))
    return None, None


def _crate_adjacent_tiles(field):
    width, height = field.shape
    targets = []
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            if field[x, y] != 0:
                continue
            if any(field[x + dx, y + dy] == 1 for dx, dy in _DIRS.values()):
                targets.append((x, y))
    return targets


def _danger_tiles(field, bombs, explosion_map):
    """Tiles currently threatened: inside a pending bomb's blast line
    (stopped by walls), or an active explosion. Same rule as items.py's
    Bomb.get_blast_coords.
    """
    danger = set()
    width, height = explosion_map.shape
    for x in range(width):
        for y in range(height):
            if explosion_map[x, y] > 0:
                danger.add((x, y))
    for (bx, by), _timer in bombs:
        danger.add((bx, by))
        for dx, dy in _DIRS.values():
            for i in range(1, s.BOMB_POWER + 1):
                nx, ny = bx + dx * i, by + dy * i
                if field[nx, ny] == -1:
                    break
                danger.add((nx, ny))
    return danger


def in_danger(game_state: dict) -> bool:
    field = game_state['field']
    pos = game_state['self'][3]
    danger = _danger_tiles(field, game_state['bombs'], game_state['explosion_map'])
    return pos in danger


def nearest_coin_distance(game_state: dict):
    field = game_state['field']
    start = game_state['self'][3]
    dist, _ = _bfs(field, start, game_state['coins'])
    return dist


def nearest_crate_spot_distance(game_state: dict):
    field = game_state['field']
    start = game_state['self'][3]
    dist, _ = _bfs(field, start, _crate_adjacent_tiles(field))
    return dist


def _safe_tiles(field, danger, width, height):
    return [(x, y) for x in range(width) for y in range(height)
            if field[x, y] == 0 and (x, y) not in danger]


def encode_state(game_state: dict, stuck: float) -> int:
    """Map a game_state (+ trajectory-derived stuck ratio) to one integer in
    [0, N_STATES) -- a row index into the Q-table. Uses mixed-radix encoding
    (each sub-feature gets a fixed "digit base") instead of enumerating all
    combinations into a list and linear-scanning it: O(1) instead of O(N_STATES)
    per lookup, and no need to materialize/store the full combination list.
    """
    field = game_state['field']
    x, y = game_state['self'][3]
    coins = game_state['coins']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    has_bomb = game_state['self'][2]

    if len(coins) > 0:
        _, action = _bfs(field, (x, y), coins)
        target_dir_idx = _DIR_NAMES.index(action) if action is not None else 4
        target_is_coin = 1
    else:
        _, action = _bfs(field, (x, y), _crate_adjacent_tiles(field))
        target_dir_idx = _DIR_NAMES.index(action) if action is not None else 4
        target_is_coin = 0

    danger = _danger_tiles(field, bombs, explosion_map)
    danger_now = (x, y) in danger

    if danger_now:
        width, height = field.shape
        safe = _safe_tiles(field, danger, width, height)
        _, escape_action = _bfs(field, (x, y), safe)
        escape_dir_idx = _DIR_NAMES.index(escape_action) if escape_action is not None else 4
    else:
        escape_dir_idx = 4  # NONE: escaping is not the relevant question right now

    # A stricter version of this flag (simulate placing the bomb, require a
    # verified escape within BOMB_TIMER-1 steps) was tried and measured
    # WORSE on both loot-crate and classic, across all 6 runs (3 seeds x 2
    # scenarios): coins dropped (9.92->2.15 on loot-crate, 1.46->1.23 on
    # classic) and suicide rose sharply (58.8%->74.7%, 66.7%->97.8%). Most
    # likely explanation: making can_bomb_here=1 fire in fewer states
    # diluted the training signal for "bombing is good here" more than it
    # removed signal for genuinely bad spots -- a debug trace had already
    # shown the escape DIRECTION is followed correctly 316/319 times (99%)
    # once danger_now=1, so direction-following was never the bottleneck.
    # Reverted to the simple, empirically-better version.
    adjacent_crate = any(field[x + dx, y + dy] == 1 for dx, dy in _DIRS.values())
    can_bomb_here = bool(has_bomb and adjacent_crate and not danger_now)

    stuck_idx = _stuck_bucket(stuck)

    idx = target_dir_idx
    idx = idx * N_TARGET_COIN + target_is_coin
    idx = idx * N_DANGER + int(danger_now)
    idx = idx * N_ESCAPE_DIR + escape_dir_idx
    idx = idx * N_CAN_BOMB + int(can_bomb_here)
    idx = idx * N_STUCK + stuck_idx
    return idx
