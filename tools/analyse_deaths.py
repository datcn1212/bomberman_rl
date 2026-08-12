"""Group the steps that lead up to each death and count the patterns.

Inspecting single steps answers "did the agent do something sensible here";
it cannot answer "how did it get into a situation where nothing was sensible".
This walks whole traces backwards from each death and classifies the cause, so
the dominant failure mode is a number rather than an impression.

    python3 tools/analyse_deaths.py --model experiments/s2_curriculum/seed1/model.pkl \\
        --scenario classic --rounds 40
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import settings as s  # noqa: E402

SAFE_BUCKET = 4          # Observation.t_here value meaning "no danger scheduled"
BOMB_USEFUL_TRAPPED = 3  # features.BOMB_USEFUL_TRAPPED


def collect(model_path, scenario, rounds, opponents, seed):
    trace = Path(tempfile.mkdtemp()) / "trace.jsonl"
    cfg = Path(tempfile.mkdtemp()) / "cfg.json"
    cfg.write_text(json.dumps({"model_path": str(Path(model_path).resolve())}))
    cmd = [sys.executable, "main.py", "play", "--no-gui",
           "--agents", "q_bomber", *opponents,
           "--scenario", scenario, "--n-rounds", str(rounds), "--seed", str(seed)]
    env = dict(os.environ, QB_CONFIG=str(cfg), QB_TRACE=str(trace))
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-3000:])
    with open(trace) as fh:
        return [json.loads(line) for line in fh]


def split_rounds(records):
    rounds, current = [], []
    for record in records:
        if current and record["step"] <= current[-1]["step"]:
            rounds.append(current)
            current = []
        current.append(record)
    if current:
        rounds.append(current)
    return rounds


def classify(round_records):
    """Why did this round end? Read backwards from the last decision."""
    last = round_records[-1]
    if last["step"] >= s.MAX_STEPS:
        return "reached step cap alive"
    if last["n_coins"] == 0 and last["n_crates"] == 0:
        return "board cleared"

    # The agent stopped being polled, so it died on its last decision.
    if last["t_here"] == SAFE_BUCKET:
        # It believed it was safe: the danger appeared within one step, which
        # can only come from a bomb dropped by somebody else this step.
        return "died on a tile it read as safe (opponent bomb)"

    tail = round_records[-8:]
    dropped = [r for r in tail if r["action"] == "BOMB"]
    if dropped and dropped[-1]["bomb_opt"] == BOMB_USEFUL_TRAPPED:
        return "bombed with no escape, then died"
    if dropped:
        followed = sum(1 for r in tail
                       if r["t_here"] < SAFE_BUCKET and r["escape_dir"] != 0
                       and _matches(r["escape_dir"], r["action"]))
        in_danger = sum(1 for r in tail if r["t_here"] < SAFE_BUCKET)
        if in_danger and followed < in_danger:
            return "own bomb, ignored the escape route"
        return "own bomb, followed escape route and still died"
    if any(r["escape_dir"] == 0 and r["t_here"] < SAFE_BUCKET for r in tail):
        return "trapped by someone else's bomb, no escape existed"
    return "in danger from elsewhere, did not escape"


DIRECTION_ACTION = {1: "UP", 2: "RIGHT", 3: "DOWN", 4: "LEFT", 5: "WAIT"}


def _matches(escape_dir, action):
    return DIRECTION_ACTION.get(escape_dir) == action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--scenario", default="classic")
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--opponents", nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=9001)
    parser.add_argument("--show", type=int, default=2,
                        help="print this many full death sequences")
    args = parser.parse_args()

    records = collect(args.model, args.scenario, args.rounds, args.opponents, args.seed)
    rounds = split_rounds(records)
    causes = Counter(classify(r) for r in rounds)

    print("%d rounds traced from %s" % (len(rounds), args.model))
    for cause, count in causes.most_common():
        print("  %5.1f%%  %-50s (%d)" % (100.0 * count / len(rounds), cause, count))

    shown = 0
    for round_records in rounds:
        cause = classify(round_records)
        if cause.startswith(("own bomb", "bombed")) and shown < args.show:
            shown += 1
            print("\n--- %s (round of %d steps) ---" % (cause, len(round_records)))
            for r in round_records[-10:]:
                print("  s%-4d pos%-9s t_here=%-2d escape=%-2d bomb_opt=%d "
                      "target=%d/%s -> %s"
                      % (r["step"], tuple(r["pos"]), r["t_here"], r["escape_dir"],
                         r["bomb_opt"], r["target_dir"], r["target_dist"], r["action"]))


if __name__ == "__main__":
    main()
