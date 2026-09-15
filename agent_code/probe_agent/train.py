"""Records how the framework delivers training callbacks.

A learner has to know exactly which transitions it is handed and how often,
because a transition counted twice is a reward counted twice. This logs every
call so the delivery pattern can be read off instead of assumed.

  PROBE_EVENTS  absolute path of the JSONL file to append callbacks to
"""

import json
import os


def setup_training(self):
    self.events_path = os.environ.get("PROBE_EVENTS")
    if self.events_path and os.path.isfile(self.events_path):
        os.remove(self.events_path)


def _log(self, record):
    if self.events_path:
        with open(self.events_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    _log(self, {
        "callback": "game_events_occurred",
        "old_step": old_game_state["step"] if old_game_state else None,
        "new_step": new_game_state["step"] if new_game_state else None,
        "action": self_action,
        "events": list(events),
    })


def end_of_round(self, last_game_state, last_action, events):
    _log(self, {
        "callback": "end_of_round",
        "old_step": last_game_state["step"] if last_game_state else None,
        "new_step": None,
        "action": last_action,
        "events": list(events),
    })
