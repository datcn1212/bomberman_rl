"""Diagnostic agent: plays a fixed script and records what it observes.

Not a learner. It exists so that the game's rules can be measured from the
outside -- bomb timings, blast geometry, event delivery -- rather than inferred
from settings.py or from someone's description of it.

Driven by two environment variables:
  PROBE_SCRIPT  comma separated actions, e.g. "BOMB,RIGHT,RIGHT,DOWN,WAIT"
  PROBE_OUT     absolute path of the JSONL file to append observations to
"""

import json
import os

import numpy as np


def setup(self):
    script = os.environ.get("PROBE_SCRIPT", "")
    self.script = [a.strip() for a in script.split(",") if a.strip()]
    self.out_path = os.environ.get("PROBE_OUT")
    self.cursor = 0
    if self.out_path and os.path.isfile(self.out_path):
        os.remove(self.out_path)


def act(self, game_state):
    action = self.script[self.cursor] if self.cursor < len(self.script) else "WAIT"
    self.cursor += 1

    explosion = game_state["explosion_map"]
    cells = [[int(x), int(y), int(explosion[x, y])]
             for x, y in zip(*np.nonzero(explosion))]
    record = {
        "step": int(game_state["step"]),
        "action": action,
        "pos": [int(v) for v in game_state["self"][3]],
        "bombs_left": bool(game_state["self"][2]),
        "bombs": [[[int(v) for v in pos], int(timer)] for pos, timer in game_state["bombs"]],
        "explosion_cells": cells,
        "n_coins": len(game_state["coins"]),
    }
    if self.out_path:
        with open(self.out_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")
    return action
