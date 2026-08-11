import numpy as np
from collections import deque

import settings as s

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
_DIRS = {'UP': (0, -1), 'RIGHT': (1, 0), 'DOWN': (0, 1), 'LEFT': (-1, 0)}
_DIR_NAMES = ['UP', 'RIGHT', 'DOWN', 'LEFT']

N_FEATURES = 22
POSITION_HISTORY_LENGTH = 8

# Continuous feature vector, unlike q_tabular's discrete state index. A random
# forest splits on raw thresholds, so distances can stay as numbers instead of
# being bucketed -- it can learn "act differently when the coin is 2 tiles away
# vs 9 tiles away" on its own, which the tabular encoding threw away and the
# linear model could only express as one fixed slope.


def stuck_ratio(position_history):
    """1 - (distinct tiles visited recently) / (recent steps). 0 when every
    step covered new ground, near 1 when pacing over a few tiles.
    """
    if len(position_history) == 0:
        return 0.0
    return 1.0 - len(set(position_history)) / len(position_history)


def _bfs(field, start, targets, avoid=frozenset()):
    """BFS over free tiles from start to the closest of `targets`, treating
    tiles in `avoid` as blocked. Returns (distance, first_action_name), or
    (None, None) if unreachable.
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
            if (nx, ny) in visited or field[nx, ny] != 0 or (nx, ny) in avoid:
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
    """Tiles inside a pending bomb's blast lines (walls stop the blast) or
    already burning. Matches Bomb.get_blast_coords in items.py.
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


def _opponent_positions(game_state):
    return [pos for (_name, _score, _bomb, pos) in game_state['others']]


def _blast_reaches_opponent(field, pos, opponents):
    """Would a bomb dropped at `pos` right now cover an opponent?"""
    opponents = set(opponents)
    if pos in opponents:
        return True
    x, y = pos
    for dx, dy in _DIRS.values():
        for i in range(1, s.BOMB_POWER + 1):
            nx, ny = x + dx * i, y + dy * i
            if field[nx, ny] == -1:
                break
            if (nx, ny) in opponents:
                return True
    return False


def in_danger(game_state: dict) -> bool:
    field = game_state['field']
    pos = game_state['self'][3]
    return pos in _danger_tiles(field, game_state['bombs'], game_state['explosion_map'])


def can_bomb_opponent_now(game_state: dict) -> bool:
    field = game_state['field']
    pos = game_state['self'][3]
    return _blast_reaches_opponent(field, pos, _opponent_positions(game_state))


def nearest_coin_distance(game_state: dict):
    field = game_state['field']
    start = game_state['self'][3]
    danger = _danger_tiles(field, game_state['bombs'], game_state['explosion_map'])
    dist, _ = _bfs(field, start, game_state['coins'], avoid=danger)
    return dist


def nearest_crate_spot_distance(game_state: dict):
    field = game_state['field']
    start = game_state['self'][3]
    danger = _danger_tiles(field, game_state['bombs'], game_state['explosion_map'])
    dist, _ = _bfs(field, start, _crate_adjacent_tiles(field), avoid=danger)
    return dist


def _one_hot_dir(action_name):
    vec = np.zeros(4, dtype=np.float32)
    if action_name is not None:
        vec[_DIR_NAMES.index(action_name)] = 1.0
    return vec


# A distance that stands in for "no reachable target". Kept well above any real
# board distance (17x17 board) so tree splits can separate "unreachable" from
# "far but reachable" cleanly.
_UNREACHABLE = 50.0


def state_to_features(game_state: dict, stuck: float = 0.0) -> np.ndarray:
    """22-dim continuous vector:
      [0:4]   one-hot direction toward nearest coin (danger-avoiding BFS)
      [4]     distance to that coin (_UNREACHABLE if none)
      [5:9]   one-hot direction toward nearest crate-adjacent tile
      [9]     distance to that tile (_UNREACHABLE if none)
      [10:14] per-direction "moving here steps into a blast/explosion"
      [14:18] per-direction "this direction is walkable"
      [18]    standing in danger right now
      [19]    dropping a bomb here would hit a crate and is currently safe
      [20]    dropping a bomb here would cover an opponent
      [21]    stuck ratio over the recent trajectory

    All target-seeking BFS calls route AROUND currently-dangerous tiles. This
    is not an optimization -- it fixes a failure mode found by death-sequence
    tracing on q_tabular: with a danger-blind BFS, the moment an agent finishes
    escaping its own bomb the target direction points straight back into the
    blast it just left (a crate it was walking toward is usually right next to
    where the bomb went off), and it walks back in and dies. Escape routing is
    deliberately NOT danger-avoiding for the same reason as in q_tabular: a
    blast is a straight corridor, so leaving it normally requires crossing one
    or two more blast tiles first -- only the destination has to be safe.
    """
    field = game_state['field']
    x, y = game_state['self'][3]
    coins = game_state['coins']
    has_bomb = game_state['self'][2]

    danger = _danger_tiles(field, game_state['bombs'], game_state['explosion_map'])
    danger_now = (x, y) in danger

    features = np.zeros(N_FEATURES, dtype=np.float32)

    coin_dist, coin_action = _bfs(field, (x, y), coins, avoid=danger)
    features[0:4] = _one_hot_dir(coin_action)
    features[4] = coin_dist if coin_dist is not None else _UNREACHABLE

    crate_dist, crate_action = _bfs(field, (x, y), _crate_adjacent_tiles(field), avoid=danger)
    features[5:9] = _one_hot_dir(crate_action)
    features[9] = crate_dist if crate_dist is not None else _UNREACHABLE

    for i, action in enumerate(_DIR_NAMES):
        dx, dy = _DIRS[action]
        nx, ny = x + dx, y + dy
        features[10 + i] = 1.0 if (nx, ny) in danger else 0.0
        features[14 + i] = 1.0 if field[nx, ny] == 0 else 0.0

    features[18] = 1.0 if danger_now else 0.0

    adjacent_crate = any(field[x + dx, y + dy] == 1 for dx, dy in _DIRS.values())
    features[19] = 1.0 if (has_bomb and adjacent_crate and not danger_now) else 0.0

    opponents = _opponent_positions(game_state)
    features[20] = 1.0 if (has_bomb and not danger_now
                           and _blast_reaches_opponent(field, (x, y), opponents)) else 0.0

    features[21] = float(stuck)
    return features
