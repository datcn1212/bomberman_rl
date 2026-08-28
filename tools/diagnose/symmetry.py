"""How much of the table is the same situation seen from a different angle?

The arena's wall pattern is symmetric under the dihedral group D4 (four
rotations x two reflections), and the rules are too. But the encoding is written
in absolute directions - `move_status` is a fixed (UP, RIGHT, DOWN, LEFT) tuple
and `target_dir` is an absolute direction - so a board rotated by 90 degrees
produces a *different* table row describing a strategically identical situation.

This counts how many distinct rows collapse into how many D4 orbits, which is
the factor by which canonicalising the encoding would shrink the table and
multiply the experience per row.

Run from the repository root:
    python3 tools/diagnose.py symmetry experiments/p10_g99 --seeds 1 2 3 4 5
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from agent_code.tabular_q.model import (  # noqa: E402
    LAYOUT, RADIX_BOMB_OPT, RADIX_ESCAPE_DIR, RADIX_T_HERE, RADIX_TARGET_DIR,
    RADIX_TARGET_KIND, QModel)

# DIRS is (UP, RIGHT, DOWN, LEFT). A 90-degree clockwise rotation sends
# (dx, dy) -> (-dy, dx), which maps UP->RIGHT->DOWN->LEFT->UP, i.e. index i to
# index (i + 1) % 4. A mirror in the vertical axis sends (dx, dy) -> (-dx, dy),
# which fixes UP and DOWN and swaps RIGHT and LEFT.
ROT = [1, 2, 3, 0]
MIRROR = [0, 3, 2, 1]


def compose(p, q):
    """Permutation p applied after q."""
    return [p[q[i]] for i in range(4)]


def d4_permutations():
    """The eight direction permutations of the dihedral group."""
    perms = []
    rot = [0, 1, 2, 3]
    for _ in range(4):
        perms.append(list(rot))
        perms.append(compose(MIRROR, rot))
        rot = compose(ROT, rot)
    return perms


def decode(index):
    bomb_opt = index % RADIX_BOMB_OPT
    index //= RADIX_BOMB_OPT
    escape_dir = index % RADIX_ESCAPE_DIR
    index //= RADIX_ESCAPE_DIR
    target_kind = index % RADIX_TARGET_KIND
    index //= RADIX_TARGET_KIND
    target_dir = index % RADIX_TARGET_DIR
    index //= RADIX_TARGET_DIR
    t_here = index % RADIX_T_HERE
    move = index // RADIX_T_HERE
    status = []
    for _ in range(4):
        status.append(move % 3)
        move //= 3
    return tuple(reversed(status)), t_here, target_dir, target_kind, escape_dir, bomb_opt


def encode_parts(status, t_here, target_dir, target_kind, escape_dir, bomb_opt):
    move = 0
    for i in range(4):
        move = move * 3 + status[i]
    index = move
    index = index * RADIX_T_HERE + t_here
    index = index * RADIX_TARGET_DIR + target_dir
    index = index * RADIX_TARGET_KIND + target_kind
    index = index * RADIX_ESCAPE_DIR + escape_dir
    index = index * RADIX_BOMB_OPT + bomb_opt
    return index


def apply_perm(perm, index):
    """Relabel one row's directions by a D4 element."""
    status, t_here, target_dir, target_kind, escape_dir, bomb_opt = decode(index)
    # perm[i] is where direction i ends up, so the new tuple takes its value for
    # direction perm[i] from the old direction i.
    new_status = [0] * 4
    for i in range(4):
        new_status[perm[i]] = status[i]

    def move_dir(value):
        # 0 = none and 5 = here are direction-free and stay put.
        if value in (0, 5):
            return value
        return perm[value - 1] + 1

    return encode_parts(tuple(new_status), t_here, move_dir(target_dir),
                        target_kind, move_dir(escape_dir), bomb_opt)


def orbit(index, perms):
    return min(apply_perm(p, index) for p in perms)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("exp_dir")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = p.parse_args()

    perms = d4_permutations()
    assert len({tuple(x) for x in perms}) == 8, "D4 should have eight elements"

    print("| seed | rows | D4 orbits | rows per orbit | visits, median row | median orbit |")
    print("|---|---|---|---|---|---|")
    orbit_sizes = Counter()
    for seed in args.seeds:
        model = QModel.load(ROOT / args.exp_dir / ("seed%d" % seed) / "model.pkl")
        rows = list(model.q)
        orbits = {}
        for index in rows:
            orbits.setdefault(orbit(index, perms), []).append(index)
        visits_row = sorted(int(model.visits(i).sum()) for i in rows)
        visits_orbit = sorted(sum(int(model.visits(i).sum()) for i in group)
                              for group in orbits.values())
        for group in orbits.values():
            orbit_sizes[len(group)] += 1
        print("| %d | %d | %d | %.2f | %d | %d |"
              % (seed, len(rows), len(orbits), len(rows) / len(orbits),
                 visits_row[len(visits_row) // 2],
                 visits_orbit[len(visits_orbit) // 2]))

    print()
    print("Orbit size distribution (how many visited rows share one orbit):")
    for size in sorted(orbit_sizes):
        print("  %d row(s): %d orbits" % (size, orbit_sizes[size]))


if __name__ == "__main__":
    main()
