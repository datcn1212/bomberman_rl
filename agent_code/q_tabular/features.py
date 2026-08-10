import numpy as np
from collections import deque

import settings as s

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
_DIRS = {'UP': (0, -1), 'RIGHT': (1, 0), 'DOWN': (0, 1), 'LEFT': (-1, 0)}
_DIR_NAMES = ['UP', 'RIGHT', 'DOWN', 'LEFT']

POSITION_HISTORY_LENGTH = 8

# Discrete state = (target_dir, target_is_coin, danger_now, escape_dir, can_bomb_here, stuck_bucket, opponent_dir).
# target_dir/escape_dir are BFS-computed and by construction always point to a
# walkable tile, so no separate per-direction valid-move flags are needed
# (unlike the linear model's feature vector, which needed them because a
# linear model cannot itself run a BFS at inference time -- a tabular Q-table
# has no such restriction, so the discretization can fold pathfinding
# straight into the state instead of exposing raw local geometry).
# opponent_dir added for Task 3/4 (hunting/fighting opponents) -- kept as a
# SEPARATE dimension from target_dir rather than folded into the same
# coin/crate priority chain, so the learned Q-values (not a hand-coded
# priority order) decide when hunting an opponent is worth it relative to
# coins/crates.
_DIR5 = _DIR_NAMES + ['NONE']  # 5
N_TARGET_DIR = 5
N_TARGET_COIN = 2
N_DANGER = 2
N_ESCAPE_DIR = 5
N_CAN_BOMB = 2
N_STUCK = 3
N_OPPONENT_DIR = 5
N_CAN_BOMB_OPPONENT = 2
N_STATES = (N_TARGET_DIR * N_TARGET_COIN * N_DANGER * N_ESCAPE_DIR * N_CAN_BOMB
            * N_STUCK * N_OPPONENT_DIR * N_CAN_BOMB_OPPONENT)


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


def _bfs(field, start, targets, avoid=frozenset()):
    """BFS over free tiles (field == 0) from start to the closest of
    `targets`, additionally treating any tile in `avoid` as blocked (used to
    route around currently-dangerous tiles, not just walls/crates). Returns
    (distance, first_action_name) or (None, None) if unreachable.
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


def can_bomb_opponent_now(game_state: dict) -> bool:
    """Whether an opponent is within blast range of a bomb dropped at the
    agent's CURRENT position right now, regardless of whether the agent
    actually has a bomb available -- exposed for train.py to reward the
    "well-positioned to hit an opponent" moment directly (see _can_bomb_opponent
    below and its use in encode_state for the full has_bomb/danger-gated
    version used in the state itself).
    """
    field = game_state['field']
    pos = game_state['self'][3]
    return _can_bomb_opponent(field, pos, _opponent_positions(game_state))


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


def _opponent_positions(game_state: dict):
    return [pos for (_name, _score, _bomb, pos) in game_state['others']]


def nearest_opponent_distance(game_state: dict):
    field = game_state['field']
    start = game_state['self'][3]
    dist, _ = _bfs(field, start, _opponent_positions(game_state))
    return dist


def _can_bomb_opponent(field, pos, opponents):
    """Whether an opponent is currently within the blast lines of a bomb
    dropped at `pos` right now (own tile plus straight lines up to
    BOMB_POWER, stopped by walls -- same rule as _danger_tiles). Needed
    because `can_bomb_here` (see below) only fires next to a CRATE: without
    a separate opponent-facing signal, the state has no way to tell "an
    opponent is in blast range" from "nothing useful is in blast range",
    even though `opponent_dir` already points the agent toward them --
    walking up to an opponent guided by opponent_dir but with no dedicated
    "bomb now" signal turned out to make hunting nearly unlearnable in
    practice (0.01 kills/round average over 3 seeds, 8000 training episodes
    with an opponent present).
    """
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

    # Computed BEFORE target_dir/escape_dir so both can route around
    # currently-dangerous tiles, not just walls/crates -- see the note on
    # `avoid` below for why this matters.
    danger = _danger_tiles(field, bombs, explosion_map)
    danger_now = (x, y) in danger

    # `avoid=danger` here is the fix for a concrete, debug-trace-confirmed
    # death pattern: the agent escapes a blast correctly (danger_now flips
    # back to 0 one tile outside the blast line), but the tile it just left
    # is still lingering-dangerous for one more round (explosion_map stays
    # >0). Without `avoid`, target_dir's BFS only dodges walls/crates, so as
    # soon as danger_now=0 it can point STRAIGHT BACK toward the coin/crate
    # it was pursuing before bombing -- which is usually right next to where
    # it just came from -- walking the agent right back into the still-live
    # blast. Every death round examined via Q_TABULAR_DEBUG showed exactly
    # this: correct escape for 2-3 steps, then danger_now=0 followed
    # immediately by a move back toward the target that re-entered danger.
    if len(coins) > 0:
        _, action = _bfs(field, (x, y), coins, avoid=danger)
        target_dir_idx = _DIR_NAMES.index(action) if action is not None else 4
        target_is_coin = 1
    else:
        _, action = _bfs(field, (x, y), _crate_adjacent_tiles(field), avoid=danger)
        target_dir_idx = _DIR_NAMES.index(action) if action is not None else 4
        target_is_coin = 0

    if danger_now:
        # NOT avoid=danger here, unlike target_dir/opponent_dir above: the
        # agent is already standing inside a blast line, and a blast is a
        # straight corridor, so the only way out routinely requires walking
        # through 1-2 MORE currently-dangerous tiles of that same corridor
        # before reaching a genuinely safe one. Blocking all danger tiles
        # during this specific search (tried first) made escape_dir turn
        # into NONE almost every time danger_now=1 -- confirmed by a
        # catastrophic regression (suicide_rate 100%, average round length
        # 11 steps) the moment this was tried. `_safe_tiles` already
        # guarantees the DESTINATION isn't dangerous; only that needs to be
        # true, not every tile of the path leading there.
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

    opponents = _opponent_positions(game_state)
    _, opponent_action = _bfs(field, (x, y), opponents, avoid=danger)
    opponent_dir_idx = _DIR_NAMES.index(opponent_action) if opponent_action is not None else 4

    can_bomb_opponent = bool(has_bomb and not danger_now and _can_bomb_opponent(field, (x, y), opponents))

    idx = target_dir_idx
    idx = idx * N_TARGET_COIN + target_is_coin
    idx = idx * N_DANGER + int(danger_now)
    idx = idx * N_ESCAPE_DIR + escape_dir_idx
    idx = idx * N_CAN_BOMB + int(can_bomb_here)
    idx = idx * N_STUCK + stuck_idx
    idx = idx * N_OPPONENT_DIR + opponent_dir_idx
    idx = idx * N_CAN_BOMB_OPPONENT + int(can_bomb_opponent)
    return idx
