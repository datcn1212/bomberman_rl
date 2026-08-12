"""State representation.

Phase 3 scope: crates and the agent's own bombs. The Phase 0 measurements fix
the timing model used here:

  * a bomb observed with timer `t` detonates at the end of the step `t` decisions
    from now, and its tiles are lethal on that decision and on the next one;
  * a tile with `explosion_map > 0` is lethal on the current decision only;
  * a bomb dropped now becomes visible next step with timer 3, so its blast is
    lethal 4 and 5 decisions from now.

The consequence is that danger is not a property of a tile but of a
**(tile, step) pair**, so the features carry a schedule rather than a flag.
"""

from collections import deque

import numpy as np

import settings as s

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]

# Index i of DIRS is the offset of ACTIONS[i], so a direction index and a move
# action index are the same number everywhere in this agent.
DIRS = ((0, -1), (1, 0), (0, 1), (-1, 0))

# How far ahead the danger schedule is built. A bomb dropped now is lethal 5
# decisions from now, so the horizon has to cover at least that.
HORIZON = s.BOMB_TIMER + s.EXPLOSION_TIMER

# move_status values
BLOCKED, FREE_SAFE, FREE_LETHAL = 0, 1, 2

# target_dir / escape_dir values: 0 = none, 1..4 = DIRS, 5 = stay where you are
DIR_NONE, DIR_HERE = 0, 5

# target_kind values
KIND_CRATE, KIND_COIN = 0, 1


def blast_tiles(field, x, y):
    """Tiles a bomb at (x, y) would burn: straight rays stopped by walls.

    Measured in Phase 0: rays never turn corners and stop *at* the first wall.
    Crates burn but do not shield tiles behind them, because the crate is
    destroyed by the same blast.
    """
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
    """lethal[k][x, y] is True when standing on (x, y) k decisions from now kills.

    `extra_bomb` adds a hypothetical bomb dropped by the agent on this step, so
    the same routine answers both "am I safe" and "would bombing here be safe".
    """
    field = game_state["field"]
    lethal = np.zeros((HORIZON + 1,) + field.shape, dtype=bool)

    for (bx, by), timer in game_state["bombs"]:
        tiles = blast_tiles(field, bx, by)
        for k in (timer, timer + 1):
            if 0 <= k <= HORIZON:
                for tx, ty in tiles:
                    lethal[k][tx, ty] = True

    # A blast already on the board kills anything standing in it right now.
    explosion = game_state["explosion_map"]
    lethal[0] |= explosion > 0

    if extra_bomb is not None:
        tiles = blast_tiles(field, *extra_bomb)
        for k in (s.BOMB_TIMER, s.BOMB_TIMER + 1):
            if k <= HORIZON:
                for tx, ty in tiles:
                    lethal[k][tx, ty] = True
    return lethal


class Board:
    """One game step: what is walkable, what is dangerous, and where things are."""

    def __init__(self, game_state, extra_bomb=None):
        self.field = game_state["field"]
        self.pos = game_state["self"][3]
        self.coins = game_state["coins"]
        self.bomb_tiles = {tuple(pos) for pos, _ in game_state["bombs"]}
        self.lethal = danger_schedule(game_state, extra_bomb)
        self.width, self.height = self.field.shape

    def is_free(self, x, y):
        """Walkable this step: not a wall, not a crate, not an armed bomb."""
        return self.field[x, y] == 0 and (x, y) not in self.bomb_tiles

    def walkable(self, tile):
        x, y = tile
        return (0 <= x < self.width and 0 <= y < self.height and self.is_free(x, y))

    def neighbours(self, x, y):
        for dx, dy in DIRS:
            nxt = (x + dx, y + dy)
            if self.walkable(nxt):
                yield nxt

    def lethal_at(self, tile, k):
        return bool(self.lethal[min(k, HORIZON)][tile[0], tile[1]])

    def escape_search(self):
        """Is there a sequence of moves that survives the whole horizon?

        Searched over (tile, step) pairs rather than tiles: a tile that burns at
        step 3 may still be crossed at step 1, and a tile that is safe now may
        be a death trap in two steps. A tile-only search cannot express either.

        Returns the first move of a surviving path: DIR_HERE when standing still
        already survives, 1..4 for a direction, DIR_NONE when nothing survives.
        """
        if self.lethal_at(self.pos, 0):
            # Already standing in a blast that goes off this step; nothing to do.
            return DIR_NONE
        if all(not self.lethal_at(self.pos, k) for k in range(HORIZON + 1)):
            # Standing still already survives. Reported first so the agent is not
            # told to move when there is no reason to.
            return DIR_HERE

        # Staying put is seeded as an option too, because waiting one step for a
        # blast to clear is sometimes the only route out of a corridor. It is
        # seeded *last* so that a real move wins any tie: reporting "stay" when
        # the agent must actually leave within the horizon would be misleading.
        queue = deque()
        seen = set()
        options = [((self.pos[0] + dx, self.pos[1] + dy), i + 1)
                   for i, (dx, dy) in enumerate(DIRS)]
        options.append((self.pos, DIR_HERE))
        for tile, first_move in options:
            if tile != self.pos and not self.walkable(tile):
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
        """How many decisions until the current tile burns; 0 when it never does."""
        for k in range(HORIZON + 1):
            if self.lethal_at(self.pos, k):
                return min(k + 1, 4)
        return 0

    def target_search(self, coins):
        """Shortest path to the nearest coin, or to a tile next to a crate.

        One search, two goal tests: standing on a coin, or standing next to a
        crate (a crate cannot be walked onto, so being adjacent to it is what
        "reaching" it means). Breadth-first, so the first goal found is the
        nearest one, and a coin wins a tie because it is tested first.

        Tiles that burn now or on the next step are treated as walls. Routing a
        path through them is exactly what makes an agent walk into a blast on
        its way to a coin.
        """
        coin_goals = set(coins)
        if self.pos in coin_goals:
            return DIR_HERE, KIND_COIN, 0
        if self._next_to_crate(self.pos):
            return DIR_HERE, KIND_CRATE, 0

        queue = deque()
        seen = {self.pos}
        for index, (dx, dy) in enumerate(DIRS):
            nxt = (self.pos[0] + dx, self.pos[1] + dy)
            if self.walkable(nxt) and not self._soon_lethal(nxt):
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

    def _next_to_crate(self, tile):
        x, y = tile
        for dx, dy in DIRS:
            cx, cy = x + dx, y + dy
            if (0 <= cx < self.width and 0 <= cy < self.height
                    and self.field[cx, cy] == 1):
                return True
        return False

    def _soon_lethal(self, tile):
        return self.lethal_at(tile, 0) or self.lethal_at(tile, 1)


class Observation:
    __slots__ = ("move_status", "t_here", "target_dir", "target_kind",
                 "escape_dir", "target_dist", "pos")

    def __init__(self, move_status, t_here, target_dir, target_kind,
                 escape_dir, target_dist, pos):
        self.move_status = move_status
        self.t_here = t_here
        self.target_dir = target_dir
        self.target_kind = target_kind
        self.escape_dir = escape_dir
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
        if not board.walkable(nxt):
            move_status.append(BLOCKED)
        elif board.lethal_at(nxt, 1):
            move_status.append(FREE_LETHAL)
        else:
            move_status.append(FREE_SAFE)

    target_dir, target_kind, target_dist = board.target_search(board.coins)
    return Observation(
        move_status=tuple(move_status),
        t_here=board.steps_until_lethal(),
        target_dir=target_dir,
        target_kind=target_kind,
        escape_dir=board.escape_search(),
        target_dist=target_dist,
        pos=board.pos,
    )
