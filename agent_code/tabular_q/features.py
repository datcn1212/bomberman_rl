"""Turning a game_state into the small tuple the agent learns from.
Design point: danger is not a property of a tile, it is a property of a (tile, step) pair. 
A tile that burns in three steps can be crossed now. 
So instead of an "is dangerous" flag we build a schedule lethal[k] = which tiles kill you k decisions from now, and search over it.

Timing:
  bomb with timer t  -> lethal at k = t and k = t+1
  explosion_map > 0  -> lethal at k = 0 only
  bomb dropped now   -> shows up next step with timer 3, so lethal at k = 4, 5
"""

from collections import deque

import numpy as np

import settings as s

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]

# DIRS[i] is the offset for ACTIONS[i]
DIRS = ((0, -1), (1, 0), (0, 1), (-1, 0))

HORIZON = s.BOMB_TIMER + s.EXPLOSION_TIMER

# move_status
BLOCKED, FREE_SAFE, FREE_LETHAL = 0, 1, 2
# target_dir / escape_dir: 0 = none, 1..4 = DIRS, 5 = stay put
DIR_NONE, DIR_HERE = 0, 5
# target_kind
KIND_CRATE, KIND_COIN = 0, 1

# Slots kept in the encoding but no longer filled.
BOMB_NONE = 0
OPP_NONE = 0
FLAGS = {"use_opponent_blocking": False}


def blast_tiles(field, x, y):
    """Tiles a bomb at (x, y) burns. Rays stop at walls and don't turn corners."""
    tiles = [(x, y)]
    for dx, dy in DIRS:
        for i in range(1, s.BOMB_POWER + 1):
            cx, cy = x + dx * i, y + dy * i
            if not (0 <= cx < field.shape[0] and 0 <= cy < field.shape[1]):
                break
            if field[cx, cy] == -1:
                break
            tiles.append((cx, cy))
    return tiles


def danger_schedule(game_state, extra_bomb=None):
    """lethal[k][x, y]: standing on (x, y) in k decisions is fatal"""
    field = game_state["field"]
    lethal = np.zeros((HORIZON + 1,) + field.shape, dtype=bool)

    for (bx, by), timer in game_state["bombs"]:
        tiles = blast_tiles(field, bx, by)
        for k in (timer, timer + 1):
            if 0 <= k <= HORIZON:
                for tx, ty in tiles:
                    lethal[k][tx, ty] = True

    lethal[0] |= game_state["explosion_map"] > 0

    if extra_bomb is not None:
        for tx, ty in blast_tiles(field, *extra_bomb):
            for k in (s.BOMB_TIMER, s.BOMB_TIMER + 1):
                if k <= HORIZON:
                    lethal[k][tx, ty] = True
    return lethal


class Board:
    """One step of the game: what we can walk on, what will kill us, where things are"""

    def __init__(self, game_state, extra_bomb=None):
        self.field = game_state["field"]
        self.pos = game_state["self"][3]
        self.coins = game_state["coins"]
        self.bomb_tiles = {tuple(pos) for pos, _ in game_state["bombs"]}
        # Only used for the immediate move: two agents can't share a tile
        if FLAGS["use_opponent_blocking"]:
            self.other_tiles = {tuple(other[3]) for other in game_state["others"]}
        else:
            self.other_tiles = set()
        self.lethal = danger_schedule(game_state, extra_bomb)
        self.width, self.height = self.field.shape

    def is_free(self, x, y):
        return self.field[x, y] == 0 and (x, y) not in self.bomb_tiles

    def walkable(self, tile, now=False):
        x, y = tile
        if not (0 <= x < self.width and 0 <= y < self.height and self.is_free(x, y)):
            return False
        return not (now and (x, y) in self.other_tiles)

    def neighbours(self, x, y):
        for dx, dy in DIRS:
            nxt = (x + dx, y + dy)
            if self.walkable(nxt):
                yield nxt

    def lethal_at(self, tile, k):
        return bool(self.lethal[min(k, HORIZON)][tile[0], tile[1]])

    def escape_search(self):
        """First move of a path that survives the whole horizon. BFS over (tile, step), not just tile."""
        if self.lethal_at(self.pos, 0):
            return DIR_NONE          # blast lands here this step, too late
        if all(not self.lethal_at(self.pos, k) for k in range(HORIZON + 1)):
            return DIR_HERE

        options = [((self.pos[0] + dx, self.pos[1] + dy), i + 1)
                   for i, (dx, dy) in enumerate(DIRS)]
        options.append((self.pos, DIR_HERE))

        queue = deque()
        seen = set()
        for tile, first_move in options:
            if tile != self.pos and not self.walkable(tile, now=True):
                continue
            if self.lethal_at(tile, 1) or (tile, 1) in seen:
                continue
            seen.add((tile, 1))
            queue.append((tile, 1, first_move))

        while queue:
            tile, k, first_move = queue.popleft()
            if k >= HORIZON:
                return first_move
            for nxt in list(self.neighbours(*tile)) + [tile]:
                if not self.lethal_at(nxt, k + 1) and (nxt, k + 1) not in seen:
                    seen.add((nxt, k + 1))
                    queue.append((nxt, k + 1, first_move))
        return DIR_NONE

    def steps_until_lethal(self):
        """Decisions left before this tile burns; 0 if it never does."""
        for k in range(HORIZON + 1):
            if self.lethal_at(self.pos, k):
                return min(k + 1, 4)
        return 0

    def target_search(self, coins):
        """Nearest coin, or nearest tile next to a crate. Returns (dir, kind, dist)"""
        coin_goals = set(coins)
        if self.pos in coin_goals:
            return DIR_HERE, KIND_COIN, 0
        if self._next_to_crate(self.pos):
            return DIR_HERE, KIND_CRATE, 0

        queue = deque()
        seen = {self.pos}
        for index, (dx, dy) in enumerate(DIRS):
            nxt = (self.pos[0] + dx, self.pos[1] + dy)
            if self.walkable(nxt, now=True) and not self._soon_lethal(nxt):
                seen.add(nxt)
                queue.append((nxt, index + 1, 1))

        while queue:
            tile, first_move, dist = queue.popleft()
            if tile in coin_goals:
                return first_move, KIND_COIN, min(dist, 15)
            if self._next_to_crate(tile):
                return first_move, KIND_CRATE, min(dist, 15)
            for nxt in self.neighbours(*tile):
                if nxt not in seen and not self._soon_lethal(nxt):
                    seen.add(nxt)
                    queue.append((nxt, first_move, dist + 1))
        return DIR_NONE, KIND_COIN, 0

    def bomb_payload(self):
        """Crates a bomb dropped here would destroy."""
        return sum(1 for tx, ty in blast_tiles(self.field, *self.pos)
                   if self.field[tx, ty] == 1)

    def _next_to_crate(self, tile):
        x, y = tile
        for dx, dy in DIRS:
            cx, cy = x + dx, y + dy
            if 0 <= cx < self.width and 0 <= cy < self.height and self.field[cx, cy] == 1:
                return True
        return False

    def _soon_lethal(self, tile):
        return self.lethal_at(tile, 0) or self.lethal_at(tile, 1)


class Observation:
    __slots__ = ("move_status", "t_here", "target_dir", "target_kind",
                 "escape_dir", "bomb_opt", "opponent", "last_move",
                 "target_dist", "pos")

    def __init__(self, move_status, t_here, target_dir, target_kind,
                 escape_dir, bomb_opt, target_dist, pos,
                 opponent=OPP_NONE, last_move=DIR_NONE):
        self.move_status = move_status
        self.t_here = t_here
        self.target_dir = target_dir
        self.target_kind = target_kind
        self.escape_dir = escape_dir
        self.bomb_opt = bomb_opt
        self.opponent = opponent
        self.last_move = last_move
        self.target_dist = target_dist
        self.pos = pos


def crate_positions(field):
    return [(int(x), int(y)) for x, y in zip(*np.nonzero(field == 1))]


def observe(game_state):
    board = Board(game_state)
    x, y = board.pos

    move_status = []
    for dx, dy in DIRS:
        nxt = (x + dx, y + dy)
        if not board.walkable(nxt, now=True):
            move_status.append(BLOCKED)
        elif board.lethal_at(nxt, 1):
            move_status.append(FREE_LETHAL)
        else:
            move_status.append(FREE_SAFE)

    target_dir, target_kind, target_dist = board.target_search(board.coins)

    # bomb_opt / opponent / last_move are held constant
    return Observation(
        move_status=tuple(move_status),
        t_here=board.steps_until_lethal(),
        target_dir=target_dir,
        target_kind=target_kind,
        escape_dir=board.escape_search(),
        bomb_opt=BOMB_NONE,
        opponent=OPP_NONE,
        last_move=DIR_NONE,
        target_dist=target_dist,
        pos=board.pos,
    )
