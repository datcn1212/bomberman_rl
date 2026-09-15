"""Measure how the framework delivers training callbacks.

A learner consumes one transition per callback, so the exact delivery pattern is
part of its correctness, not a detail. Two questions decide how train.py must be
written:

  (a) when the agent survives to the end of a round, is the final step handed
      over once or twice (game_events_occurred *and* end_of_round)?
  (b) when the agent dies, which callback carries the fatal events?

Run from the repository root:  python3 tools/phase0/events.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "results" / "mechanics"


def run(name, script, scenario="empty", seed=1):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    events_path = OUT_DIR / f"events_{name}.jsonl"
    env = dict(os.environ,
               PROBE_SCRIPT=script,
               PROBE_OUT=str(OUT_DIR / f"probe_{name}.jsonl"),
               PROBE_EVENTS=str(events_path))
    subprocess.run(
        [sys.executable, "main.py", "play", "--no-gui", "--agents", "probe_agent",
         "--train", "1", "--scenario", scenario, "--n-rounds", "1", "--seed", str(seed)],
        cwd=ROOT, env=env, check=True, capture_output=True,
    )
    with open(events_path) as fh:
        return [json.loads(line) for line in fh]


def render(records, title):
    lines = [f"**{title}**", "",
             "| callback | old_step | new_step | action | events |",
             "|---|---|---|---|---|"]
    for r in records:
        lines.append("| `%s` | %s | %s | %s | %s |"
                     % (r["callback"], r["old_step"], r["new_step"],
                        r["action"], ", ".join(r["events"]) or "-"))
    return "\n".join(lines)


def main():
    out = ["### Training callback delivery\n"]

    survive = run("survive", "BOMB,RIGHT,RIGHT,DOWN,WAIT,WAIT,WAIT,WAIT,WAIT,WAIT")
    out.append(render(survive, "A. Agent survives to the end of the round"))
    steps = [r["old_step"] for r in survive]
    last_step = steps[-1]
    duplicated = steps.count(last_step)
    out.append("")
    out.append("- transitions delivered: %d, distinct `old_step` values: %d"
               % (len(survive), len(set(steps))))
    out.append("- the final step (%s) appears **%d times**%s"
               % (last_step, duplicated,
                  " -- once via `game_events_occurred` and once via `end_of_round`"
                  if duplicated > 1 else ""))
    if duplicated > 1:
        pair = [r for r in survive if r["old_step"] == last_step]
        same = pair[0]["events"] == pair[1]["events"]
        out.append("- both carry the same events: **%s** (%s vs %s)"
                   % (same, pair[0]["events"], pair[1]["events"]))
        out.append("- consequence: a learner that updates in both callbacks counts "
                   "the last transition of every surviving round **twice**.")
    out.append("")

    die = run("suicide", "BOMB,WAIT,WAIT,WAIT,WAIT,WAIT")
    out.append(render(die, "B. Agent blows itself up"))
    steps_d = [r["old_step"] for r in die]
    out.append("")
    out.append("- transitions delivered: %d, distinct `old_step` values: %d"
               % (len(die), len(set(steps_d))))
    fatal = [r for r in die if "KILLED_SELF" in r["events"] or "GOT_KILLED" in r["events"]]
    out.append("- fatal events arrive in: %s"
               % (", ".join(sorted({r["callback"] for r in fatal})) or "no callback"))
    out.append("- consequence: the transition into death is only visible through "
               "`end_of_round`, so a learner that only handles "
               "`game_events_occurred` never learns that dying is bad.")
    out.append("")

    text = "\n".join(out)
    (OUT_DIR / "events_report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
