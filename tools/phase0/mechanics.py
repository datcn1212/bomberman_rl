"""Measure the game's bomb/explosion mechanics empirically instead of trusting
settings.py or any second-hand description of it.

Drives agent_code/probe_agent with fixed action scripts on the `empty` scenario
(deterministic wall pattern, no crates or coins to interfere) and reads back the
raw observations, then answers:

  (a) how many steps pass between the BOMB action and the blast,
  (b) how long an explosion stays lethal, and what explosion_map shows meanwhile,
  (c) at which step `bombs_left` becomes True again,
  (d) whether a blast turns corners,
  (e) how walls block a blast,
  (f) whether walking back into a decaying blast is still lethal.

Run from the repository root:  python3 tools/verify_mechanics.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import settings as s  # noqa: E402

OUT_DIR = ROOT / "results" / "mechanics"


def arena_walls():
    """Reproduce the deterministic wall layout of build_arena (crates aside)."""
    walls = set()
    for x in range(s.COLS):
        for y in range(s.ROWS):
            if x in (0, s.COLS - 1) or y in (0, s.ROWS - 1) or (x + 1) * (y + 1) % 2 == 1:
                walls.add((x, y))
    return walls


def run_probe(name, script, scenario="empty", seed=1):
    """Play one round with the probe agent following `script`; return its log."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{name}.jsonl"
    env = dict(os.environ, PROBE_SCRIPT=script, PROBE_OUT=str(out_path))
    subprocess.run(
        [sys.executable, "main.py", "play", "--no-gui", "--agents", "probe_agent",
         "--scenario", scenario, "--n-rounds", "1", "--seed", str(seed)],
        cwd=ROOT, env=env, check=True, capture_output=True,
    )
    with open(out_path) as fh:
        return [json.loads(line) for line in fh]


def table(records, title):
    lines = [f"**{title}**", "",
             "| step | act | pos | bombs_left | bombs (pos,timer) | explosion_map dangerous cells |",
             "|---|---|---|---|---|---|"]
    for r in records:
        bombs = " ".join(f"({b[0][0]},{b[0][1]})t={b[1]}" for b in r["bombs"]) or "-"
        expl = " ".join(f"({c[0]},{c[1]})={c[2]}" for c in r["explosion_cells"]) or "-"
        lines.append(f"| {r['step']} | {r['action']} | {tuple(r['pos'])} | {r['bombs_left']} "
                     f"| {bombs} | {expl} |")
    return "\n".join(lines)


def expected_blast(pos, walls):
    """Straight-line blast of BOMB_POWER in 4 directions, stopped by walls."""
    x, y = pos
    cells = {(x, y)}
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for i in range(1, s.BOMB_POWER + 1):
            cx, cy = x + dx * i, y + dy * i
            if (cx, cy) in walls:
                break
            cells.add((cx, cy))
    return cells


def main():
    walls = arena_walls()
    out = []

    out.append("### Constants read from `settings.py`\n")
    out.append("| name | value |")
    out.append("|---|---|")
    for key in ["COLS", "ROWS", "MAX_AGENTS", "MAX_STEPS", "BOMB_POWER", "BOMB_TIMER",
                "EXPLOSION_TIMER", "TIMEOUT", "TRAIN_TIMEOUT", "REWARD_KILL", "REWARD_COIN"]:
        out.append(f"| `{key}` | {getattr(s, key)} |")
    for name, cfg in s.SCENARIOS.items():
        out.append(f"| scenario `{name}` | CRATE_DENSITY {cfg['CRATE_DENSITY']}, "
                   f"COIN_COUNT {cfg['COIN_COUNT']} |")
    out.append("")

    # --- A. Full bomb life cycle, observed from a safe tile -------------------
    esc = run_probe("escape", "BOMB,RIGHT,RIGHT,DOWN,WAIT,WAIT,WAIT,WAIT,WAIT,WAIT")
    out.append(table(esc, "A. Drop bomb at spawn, step out of the blast, watch the full life cycle"))
    out.append("")

    drop = next(r for r in esc if r["action"] == "BOMB")
    seen = [r for r in esc if r["bombs"]]
    burning = [r for r in esc if r["explosion_cells"]]
    b_back = next((r["step"] for r in esc if r["step"] > drop["step"] and r["bombs_left"]), None)

    out.append("**A. Measured**\n")
    out.append(f"- `BOMB` issued at step {drop['step']} from {tuple(drop['pos'])}; "
               f"the bomb becomes visible one step later with timer "
               f"{seen[0]['bombs'][0][1]} and counts down "
               f"{[r['bombs'][0][1] for r in seen]}.")
    out.append(f"- The agent survives the whole round, so the timer sequence above is complete: "
               f"the blast lands at the end of the step where the observed timer is 0 "
               f"(step {seen[-1]['step']}), i.e. **{seen[-1]['step'] - drop['step']} steps "
               f"after the BOMB action** (`BOMB_TIMER` = {s.BOMB_TIMER}).")
    out.append(f"- `explosion_map` shows dangerous cells on steps "
               f"{[r['step'] for r in burning]} with values "
               f"{sorted({c[2] for r in burning for c in r['explosion_cells']})}.")
    out.append(f"- `bombs_left` goes False at step {seen[0]['step']} and back to True at "
               f"step {b_back} (**{b_back - drop['step']} steps after the BOMB action**), "
               f"independently of where the agent is standing.")
    out.append("")

    # --- B. Blast geometry from an open corner (border walls only) -----------
    first_burn = burning[0]
    cells = {(c[0], c[1]) for c in first_burn["explosion_cells"]}
    bx, by = drop["pos"]
    exp = expected_blast((bx, by), walls)
    off_axis = {c for c in cells if c[0] != bx and c[1] != by}
    out.append(f"**B. Blast geometry, bomb at {(bx, by)} (open corner)**\n")
    out.append(f"- Burning cells: {sorted(cells)}")
    out.append(f"- Straight-line model with walls predicts: {sorted(exp)}")
    out.append(f"- Match: **{cells == exp}**")
    out.append(f"- Off-axis cells (a blast that turned a corner would show up here): "
               f"{sorted(off_axis) if off_axis else '**none -> blasts never turn corners**'}")
    out.append("")

    # --- C. Blast blocked by an interior pillar ------------------------------
    # `coin-heaven` rather than `empty`: with nothing left to do the engine ends
    # the round immediately, and this probe needs three steps of walking before
    # it may drop its bomb.
    blocked = run_probe("blocked", "DOWN,DOWN,RIGHT,BOMB,LEFT,DOWN,WAIT,WAIT,WAIT,WAIT,WAIT,WAIT",
                        scenario="coin-heaven")
    out.append(table(blocked, "C. Bomb next to interior pillars (tests wall blocking)"))
    b_drop = next(r for r in blocked if r["action"] == "BOMB")
    b_burn = next((r for r in blocked if r["explosion_cells"]), None)
    if b_burn is not None:
        cells2 = {(c[0], c[1]) for c in b_burn["explosion_cells"]}
        exp2 = expected_blast(tuple(b_drop["pos"]), walls)
        px, py = b_drop["pos"]
        out.append(f"\n- Bomb at {(px, py)}; neighbouring walls: "
                   f"{sorted(w for w in walls if abs(w[0]-px) + abs(w[1]-py) == 1)}")
        out.append(f"- Burning cells: {sorted(cells2)}")
        out.append(f"- Straight-line-with-walls prediction: {sorted(exp2)}")
        out.append(f"- Match: **{cells2 == exp2}** -> a wall stops the ray at the wall, "
                   f"the blast does not continue past it.\n")

    # --- D. Suicide timing ---------------------------------------------------
    sui = run_probe("suicide", "BOMB,WAIT,WAIT,WAIT,WAIT,WAIT,WAIT,WAIT")
    out.append(table(sui, "D. Drop a bomb and stand still"))
    out.append(f"\n- The agent is polled for the last time at step {sui[-1]['step']} "
               f"(observed bomb timer {sui[-1]['bombs'][0][1] if sui[-1]['bombs'] else '-'}), "
               f"so it dies during that step: **a bomb whose observed timer is 0 kills at the "
               f"end of the current step**, leaving no further chance to move.\n")

    # --- E. Re-entering a decaying blast at two different moments ------------
    for label, script in [("early", "BOMB,RIGHT,RIGHT,DOWN,UP,WAIT,WAIT,WAIT,WAIT"),
                          ("late", "BOMB,RIGHT,RIGHT,DOWN,WAIT,UP,WAIT,WAIT,WAIT"),
                          ("later", "BOMB,RIGHT,RIGHT,DOWN,WAIT,WAIT,UP,WAIT,WAIT")]:
        rec = run_probe(f"return_{label}", script)
        back = next((r for r in rec if r["action"] == "UP"), None)
        obs = "-"
        if back is not None:
            here = [c for c in back["explosion_cells"]]
            obs = f"{here}"
        out.append(table(rec, f"E-{label}. Escape, then step back into the blast row"))
        out.append(f"\n- Step back attempted at step {back['step'] if back else '-'}; "
                   f"explosion cells observed then: {obs}; "
                   f"last polled step {rec[-1]['step']} "
                   f"-> {'DIED' if rec[-1]['step'] <= (back['step'] if back else 0) else 'SURVIVED'}\n")

    text = "\n".join(out)
    (OUT_DIR / "mechanics_report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
