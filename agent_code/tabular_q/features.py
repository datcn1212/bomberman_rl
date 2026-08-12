"""State representation.

Phase 1 scope: coin collecting on a board without crates or bombs. The state
answers only two questions:

  * which of the four neighbouring tiles can I step onto?
  * which way is the nearest coin?

That is deliberately the smallest description that makes the task solvable, so
that later phases can be judged against a baseline whose failure modes are
understood rather than against an already complicated agent.
"""

from collections import deque

import numpy as np

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]

# Index i of DIRS is the offset of ACTIONS[i], so a direction index and a move
# action index are the same number everywhere in this agent.
DIRS = ((0, -1), (1, 0), (0, 1), (-1, 0))

# `target_dir` reserves 0 for "no target reachable" and shifts the four
# directions up by one.
DIR_NONE = 0


class Board:
    """Static view of one game step: what is walkable, and where things are."""

    def __init__(self, game_state):
        self.field = game_state["field"]
        self.pos = game_state["self"][3]
        self.coins = game_state["coins"]
        self.width, self.height = self.field.shape

    def is_free(self, x, y):
        """A tile that can be stepped onto this step."""
        return self.field[x, y] == 0

    def neighbours(self, x, y):
        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height and self.is_free(nx, ny):
                yield nx, ny

    def target_search(self, targets):
        """Breadth-first search from the agent to the closest of `targets`.

        Returns (direction index + 1, distance), or (DIR_NONE, -1) when nothing
        is reachable. BFS is used rather than a distance heuristic because the
        board has dead ends, where Manhattan distance points into a wall.
        """
        if not targets:
            return DIR_NONE, -1
        goals = set(targets)
        start = self.pos
        if start in goals:
            return DIR_NONE, 0

        # Each frontier entry carries the first move that led to it, so the
        # answer falls out of the search without a parent table.
        queue = deque()
        seen = {start}
        for index, (dx, dy) in enumerate(DIRS):
            nxt = (start[0] + dx, start[1] + dy)
            if self._walkable(nxt):
                seen.add(nxt)
                queue.append((nxt, index, 1))

        while queue:
            (x, y), first_move, dist = queue.popleft()
            if (x, y) in goals:
                return first_move + 1, dist
            for nxt in self.neighbours(x, y):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, first_move, dist + 1))
        return DIR_NONE, -1

    def _walkable(self, tile):
        x, y = tile
        return 0 <= x < self.width and 0 <= y < self.height and self.is_free(x, y)


class Observation:
    """The features one call to `observe` produced."""

    __slots__ = ("move_status", "target_dir", "target_dist", "pos")

    def __init__(self, move_status, target_dir, target_dist, pos):
        self.move_status = move_status
        self.target_dir = target_dir
        self.target_dist = target_dist
        self.pos = pos


def observe(game_state):
    board = Board(game_state)
    x, y = board.pos

    move_status = tuple(1 if board._walkable((x + dx, y + dy)) else 0 for dx, dy in DIRS)
    target_dir, target_dist = board.target_search(board.coins)
    return Observation(move_status, target_dir, target_dist, board.pos)
