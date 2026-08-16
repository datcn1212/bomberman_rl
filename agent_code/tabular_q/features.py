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

# bomb_opt values: what dropping a bomb on this tile would accomplish.
# Phase 3 measured that 11% of the agent's bombs destroyed nothing, with a
# per-seed range of 0% to 28%, because the state could not tell a blast that
# clears four crates from one that clears none.
BOMB_NONE, BOMB_POINTLESS, BOMB_USEFUL, BOMB_TRAPPED = 0, 1, 2, 3

# bomb_safety reuses the same slot with a different question: not "would this
# bomb achieve anything" but "how much room would be left to escape it". With
# opponents on the board the escape route computed at bomb time can be walked
# into by someone else, so a bomb with a single way out and a bomb with three
# are very different risks - and the binary escape_dir cannot tell them apart.
SAFE_NONE, SAFE_TRAPPED, SAFE_TIGHT, SAFE_ROBUST = 0, 1, 2, 3

# bomb_opt, five-way variant: separates a bomb that reaches an opponent from one
# that only clears crates. The four-way version tested earlier had no such
# category, so "is a bomb that can hit someone worth distinguishing" was never
# actually asked.
HIT_NONE, HIT_POINTLESS, HIT_CRATE, HIT_OPPONENT, HIT_TRAPPED = 0, 1, 2, 3, 4

# How far an opponent has to be before it stops mattering, in BFS steps.
OPPONENT_NEAR = 5
OPP_NONE, OPP_FAR, OPP_NEAR = 0, 1, 2

# Which optional state components are switched on. Set once at setup and stored
# inside the model, so that a model is always evaluated with the same
# observation it was trained on. Holding a component at a constant value
# collapses it out of the encoding without changing LAYOUT, which is what makes
# a clean ablation possible.
FLAGS = {"use_bomb_opt": False, "use_opponent_blocking": False,
         "use_bomb_safety": False, "use_exact_escape": False,
         "use_bomb_hits": False, "use_opponent_distance": False,
         "use_last_move": False}


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
        # Two agents cannot share a tile, so moving into an occupied one is an
        # invalid action: the agent stays put. Ignoring that is how a planned
        # escape route ends with the agent standing still inside a blast.
        self.other_tiles = ({tuple(other[3]) for other in game_state["others"]}
                            if FLAGS["use_opponent_blocking"] else set())
        self.lethal = danger_schedule(game_state, extra_bomb)
        self.width, self.height = self.field.shape

    def is_free(self, x, y):
        """Walkable this step: not a wall, not a crate, not an armed bomb."""
        return self.field[x, y] == 0 and (x, y) not in self.bomb_tiles

    def walkable(self, tile, now=False):
        """`now` also excludes tiles an opponent is standing on.

        Only the *immediate* move is blocked by a body. Beyond that the
        opponents have moved too, and treating them as permanent walls would
        make the agent believe it is trapped when it is not.
        """
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

    def survival_table(self):
        """survive[k][x, y]: standing there at step k, is the horizon reachable.

        Backward induction from the horizon, one pass over (tile, step). A
        breadth-first search that shares one `seen` set across several starting
        moves is not equivalent: whichever branch reaches a tile first claims it,
        so a route that needed that tile is reported as blocked even when it is
        open. This has no such failure and costs less, because each cell is
        settled once instead of being re-explored per starting move.
        """
        free = (self.field == 0)
        for bx, by in self.bomb_tiles:
            free[bx, by] = False

        survive = [None] * (HORIZON + 1)
        survive[HORIZON] = free & ~self.lethal[HORIZON]
        for k in range(HORIZON - 1, 0, -1):
            nxt = survive[k + 1]
            reachable = nxt.copy()                      # stand still
            reachable[:-1, :] |= nxt[1:, :]             # step right
            reachable[1:, :] |= nxt[:-1, :]             # step left
            reachable[:, :-1] |= nxt[:, 1:]             # step down
            reachable[:, 1:] |= nxt[:, :-1]             # step up
            survive[k] = free & ~self.lethal[k] & reachable
        return survive

    def escape_routes(self):
        """How many distinct first moves survive the whole horizon.

        The binary version answers "is there a way out"; this answers "how many",
        which is what separates a bomb that a moving opponent can seal off from
        one that it cannot.
        """
        survive = self.survival_table()
        routes = 0
        for first in list(range(1, 5)) + [DIR_HERE]:
            tile = self._first_step_tile(first)
            if tile is not None and survive[1][tile[0], tile[1]]:
                routes += 1
        return routes

    def _first_step_tile(self, first_move):
        """Where a given opening move lands, or None if it is not legal now."""
        if first_move == DIR_HERE:
            return self.pos
        dx, dy = DIRS[first_move - 1]
        tile = (self.pos[0] + dx, self.pos[1] + dy)
        return tile if self.walkable(tile, now=True) else None

    def _survives_starting_with(self, first_move):
        if first_move == DIR_HERE:
            start = self.pos
        else:
            dx, dy = DIRS[first_move - 1]
            start = (self.pos[0] + dx, self.pos[1] + dy)
            if not self.walkable(start, now=True):
                return False
        if self.lethal_at(start, 1):
            return False

        queue = deque([(start, 1)])
        seen = {(start, 1)}
        while queue:
            tile, k = queue.popleft()
            if k >= HORIZON:
                return True
            for nxt in list(self.neighbours(*tile)) + [tile]:
                if not self.lethal_at(nxt, k + 1) and (nxt, k + 1) not in seen:
                    seen.add((nxt, k + 1))
                    queue.append((nxt, k + 1))
        return False

    def escape_search(self):
        if FLAGS["use_exact_escape"]:
            return self._escape_exact()
        return self._escape_frontier()

    def _escape_exact(self):
        """First opening move of a surviving route, from the survival table."""
        if self.lethal_at(self.pos, 0):
            return DIR_NONE
        if all(not self.lethal_at(self.pos, k) for k in range(HORIZON + 1)):
            return DIR_HERE
        survive = self.survival_table()
        for first in list(range(1, 5)) + [DIR_HERE]:
            tile = self._first_step_tile(first)
            if tile is not None and survive[1][tile[0], tile[1]]:
                return first
        return DIR_NONE

    def _escape_frontier(self):
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
        """How many decisions until the current tile burns; 0 when it never does."""
        for k in range(HORIZON + 1):
            if self.lethal_at(self.pos, k):
                return min(k + 1, 4)
        return 0

    def target_search(self, coins, goals=None):
        """Shortest path to the nearest coin, or to a tile next to a crate.

        One search, two goal tests: standing on a coin, or standing next to a
        crate (a crate cannot be walked onto, so being adjacent to it is what
        "reaching" it means). Breadth-first, so the first goal found is the
        nearest one, and a coin wins a tie because it is tested first.

        Tiles that burn now or on the next step are treated as walls. Routing a
        path through them is exactly what makes an agent walk into a blast on
        its way to a coin.
        """
        coin_goals = set(goals) if goals is not None else set(coins)
        crate_goals = goals is None
        if self.pos in coin_goals:
            return DIR_HERE, KIND_COIN, 0
        if crate_goals and self._next_to_crate(self.pos):
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
            if crate_goals and self._next_to_crate(tile):
                return first_move, KIND_CRATE, min(dist, 15)
            for nxt in self.neighbours(*tile):
                if nxt not in seen and not self._soon_lethal(nxt):
                    seen.add(nxt)
                    queue.append((nxt, first_move, dist + 1))
        return DIR_NONE, KIND_COIN, 0

    def bomb_payload(self):
        """How many crates a bomb dropped here would destroy."""
        return sum(1 for tx, ty in blast_tiles(self.field, *self.pos)
                   if self.field[tx, ty] == 1)

    def opponent_state(self, others):
        """Whether the nearest opponent is absent, far, or within reach.

        Distance is the walking distance, not the straight line: an opponent two
        tiles away across a wall is not two steps away.
        """
        if not others:
            return OPP_NONE
        direction, _, dist = self.target_search([], goals=set(others))
        if direction == DIR_NONE:
            return OPP_FAR
        return OPP_NEAR if dist <= OPPONENT_NEAR else OPP_FAR

    def bomb_hits(self, game_state):
        """What a bomb dropped here would reach: nothing, crates, or an opponent."""
        if not game_state["self"][2]:
            return HIT_NONE
        tiles = set(blast_tiles(self.field, *self.pos))
        others = {tuple(o[3]) for o in game_state["others"]}
        crates = sum(1 for tx, ty in tiles if self.field[tx, ty] == 1)
        if not crates and not (tiles & others):
            return HIT_POINTLESS
        hypothetical = Board(game_state, extra_bomb=self.pos)
        if hypothetical.escape_search() == DIR_NONE:
            return HIT_TRAPPED
        return HIT_OPPONENT if (tiles & others) else HIT_CRATE

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


def bomb_safety(game_state, board):
    """How robust the escape would be after dropping a bomb here."""
    if not game_state["self"][2]:
        return SAFE_NONE
    hypothetical = Board(game_state, extra_bomb=board.pos)
    routes = hypothetical.escape_routes()
    if routes == 0:
        return SAFE_TRAPPED
    return SAFE_TIGHT if routes == 1 else SAFE_ROBUST


def bomb_option(game_state, board):
    """What dropping a bomb on the current tile would accomplish.

    The escape question is answered against the *hypothetical* danger schedule
    that includes the bomb being considered, which is the only way to know
    whether the agent would still have a way out after dropping it.
    """
    if not game_state["self"][2]:
        return BOMB_NONE
    if board.bomb_payload() == 0:
        return BOMB_POINTLESS
    hypothetical = Board(game_state, extra_bomb=board.pos)
    if hypothetical.escape_search() == DIR_NONE:
        return BOMB_TRAPPED
    return BOMB_USEFUL


def crate_positions(field):
    return [(int(x), int(y)) for x, y in zip(*np.nonzero(field == 1))]


def observe(game_state, last_move=DIR_NONE):
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
    others = [tuple(o[3]) for o in game_state["others"]]
    return Observation(
        move_status=tuple(move_status),
        t_here=board.steps_until_lethal(),
        target_dir=target_dir,
        target_kind=target_kind,
        escape_dir=board.escape_search(),
        bomb_opt=(board.bomb_hits(game_state) if FLAGS["use_bomb_hits"]
                  else bomb_safety(game_state, board) if FLAGS["use_bomb_safety"]
                  else bomb_option(game_state, board) if FLAGS["use_bomb_opt"]
                  else BOMB_NONE),
        opponent=(board.opponent_state(others) if FLAGS["use_opponent_distance"]
                  else OPP_NONE),
        last_move=last_move if FLAGS["use_last_move"] else DIR_NONE,
        target_dist=target_dist,
        pos=board.pos,
    )
