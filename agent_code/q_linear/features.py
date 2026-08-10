import numpy as np
from collections import deque

import settings as s

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
_DIRS = {'UP': (0, -1), 'RIGHT': (1, 0), 'DOWN': (0, 1), 'LEFT': (-1, 0)}
N_FEATURES = 20
POSITION_HISTORY_LENGTH = 8


def stuck_ratio(position_history):
    """1 - (distinct tiles in position_history) / len(position_history).
    0 if empty. Shared by callbacks.py and train.py so both compute the
    "spatial stuck-ness" feature the same way.
    """
    if len(position_history) == 0:
        return 0.0
    return 1.0 - len(set(position_history)) / len(position_history)


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


def _crate_adjacent_tiles(field):
    """Free tiles that have at least one neighboring crate — good bomb-drop
    spots.
    """
    width, height = field.shape
    targets = []
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            if field[x, y] != 0:
                continue
            if any(field[x + dx, y + dy] == 1 for dx, dy in _DIRS.values()):
                targets.append((x, y))
    return targets


def _target_tiles(field, coins):
    """Coins if any are visible; otherwise free tiles adjacent to a crate
    (good bomb-drop spots) — generalizes Task 1's "seek nearest coin" to
    also work before any coin has been revealed. Used only for the scalar
    potential function; state_to_features keeps coin-seeking and
    crate-seeking as two separate feature blocks (see below) since a
    linear model cannot otherwise tell the two situations apart when
    they'd share the same one-hot direction dimensions.
    """
    if len(coins) > 0:
        return coins
    return _crate_adjacent_tiles(field)


def _danger_tiles(field, bombs, explosion_map):
    """Tiles currently threatened: inside a pending bomb's blast line
    (stopped by walls, not by crates — matches Bomb.get_blast_coords in
    items.py), or already an active explosion.
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
    """Whether the agent's CURRENT position is inside a bomb blast line or
    active explosion. Exposed separately from state_to_features so train.py
    can use it to compute a direct reward penalty, not just a feature.
    """
    field = game_state['field']
    pos = game_state['self'][3]
    danger = _danger_tiles(field, game_state['bombs'], game_state['explosion_map'])
    return pos in danger


def nearest_target_distance(game_state: dict):
    """Shortest-path distance to the nearest target (coin, or crate-adjacent
    tile if no coin is visible yet), or None if none is reachable.
    """
    field = game_state['field']
    start = game_state['self'][3]
    targets = _target_tiles(field, game_state['coins'])
    dist, _ = _bfs_to_nearest(field, start, targets)
    return dist


def nearest_coin_distance(game_state: dict):
    """Shortest-path distance to the nearest COIN only (None if no coin is
    visible — does NOT fall back to crates).
    """
    field = game_state['field']
    start = game_state['self'][3]
    dist, _ = _bfs_to_nearest(field, start, game_state['coins'])
    return dist


def nearest_crate_spot_distance(game_state: dict):
    """Shortest-path distance to the nearest free tile adjacent to a crate
    (None if none reachable). Kept as a separate function from
    `nearest_coin_distance` (rather than one switching `nearest_target_distance`)
    so reward shaping can use both as an ADDITIVE potential — always having
    a crate-seeking gradient present, with a coin-seeking term added on top
    once a coin is visible, instead of hard-switching between the two
    (switching caused either a large one-off penalty at the moment of
    revealing a coin, or — when coin-only — zero shaping signal at all
    whenever no coin is visible yet, which let a "stand still and do
    nothing" policy look as good as actively bombing crates).
    """
    field = game_state['field']
    start = game_state['self'][3]
    dist, _ = _bfs_to_nearest(field, start, _crate_adjacent_tiles(field))
    return dist


def state_to_features(game_state: dict, stuck_ratio: float = 0.0) -> np.ndarray:
    """20-dim vector: [4] one-hot BFS direction to nearest COIN (all zero if
    none visible), [4] one-hot BFS direction to nearest crate-adjacent tile
    (all zero if a coin is visible), [4] valid-move flags, [4] "would moving
    this direction be dangerous", [1] "am I in danger now", [1] "is dropping
    a bomb here worthwhile", [1] "spatial stuck-ness" (see below), [1] bias.

    Coin-seeking and crate-seeking are kept as two separate 4-dim blocks
    (not one shared block, as in the Task 1 version) because a linear model
    cannot otherwise distinguish "walk onto this tile" (coin) from "walk
    here then drop a bomb" (crate) when both share the same one-hot
    direction dimensions.

    `stuck_ratio` must be computed by the caller from the agent's own
    trajectory history, not from `game_state` alone — see `callbacks.py`'s
    `act()` and `train.py`'s `_store()`. It is `1 - (distinct tiles in the
    last POSITION_HISTORY_LENGTH steps) / POSITION_HISTORY_LENGTH`: 0 when
    every recent step covered new ground, close to 1 when recent steps kept
    revisiting a small handful of tiles.

    An earlier version used a binary "have I been at this EXACT tile
    recently" flag instead. That caught one specific oscillating tile pair
    but not the underlying problem: many different (x, y) positions with
    similar local surroundings (no coin visible, same crate direction, same
    wall pattern) produce an IDENTICAL feature vector ("aliasing"), so a
    deterministic policy can cycle between any such aliased pair — fixing
    one specific pair just relocated the cycle to a different one with the
    same shape (verified: (15,15)<->(15,14), then (1,1)<->(2,1), then
    (15,1)<->(14,1) across successive fixes). A continuous, tile-identity
    independent "am I covering new ground" signal generalizes across all
    such aliased pairs instead of chasing them one at a time.
    """
    field = game_state['field']
    x, y = game_state['self'][3]
    coins = game_state['coins']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    has_bomb = game_state['self'][2]

    features = np.zeros(N_FEATURES, dtype=np.float32)

    if len(coins) > 0:
        _, coin_action = _bfs_to_nearest(field, (x, y), coins)
        if coin_action is not None:
            features[ACTIONS.index(coin_action)] = 1.0
    else:
        crate_targets = _crate_adjacent_tiles(field)
        _, crate_action = _bfs_to_nearest(field, (x, y), crate_targets)
        if crate_action is not None:
            features[4 + ACTIONS.index(crate_action)] = 1.0

    for i, action in enumerate(['UP', 'RIGHT', 'DOWN', 'LEFT']):
        dx, dy = _DIRS[action]
        if field[x + dx, y + dy] == 0:
            features[8 + i] = 1.0

    danger = _danger_tiles(field, bombs, explosion_map)
    for i, action in enumerate(['UP', 'RIGHT', 'DOWN', 'LEFT']):
        dx, dy = _DIRS[action]
        if (x + dx, y + dy) in danger:
            features[12 + i] = 1.0

    in_danger_now = (x, y) in danger
    features[16] = 1.0 if in_danger_now else 0.0

    adjacent_crate = any(field[x + dx, y + dy] == 1 for dx, dy in _DIRS.values())
    features[17] = 1.0 if (has_bomb and adjacent_crate and not in_danger_now) else -1.0

    features[18] = float(stuck_ratio)

    features[19] = 1.0  # bias
    return features
