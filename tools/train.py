"""Training driver: run a curriculum for several seeds in parallel, then
evaluate under the standard protocol.

A phase is `scenario:episodes[@opponent+opponent]`, and phases chain by
warm-starting each one from the model the previous phase produced:

    python3 tools/train.py --exp-id p1_minimal --phases coin-heaven:2000 --seeds 1 2 3

The multiprocessing ban applies to the submitted agent, not to this harness -
the agent itself never forks.

Everything needed to reproduce a run lands in experiments/<exp_id>/seed<k>/:
the config of each phase, the per-episode log, and the model.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.evaluate import (EVAL_SEEDS, config_env_var, evaluate,  # noqa: E402
                            format_metrics, register)

EXPERIMENTS = ROOT / "experiments"

# Disjoint from EVAL_SEEDS (9001..9030), so nothing is ever reported on an
# arena the agent trained on.
TRAIN_SEED_BASE = 1000


def code_fingerprint(agent):
    """Commit the agent code was at, with a +dirty marker.

    Written into every phase config so two runs can't be compared without it
    being visible that different code produced them.
    """
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--",
                                "agent_code/%s" % agent], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"
    return head + ("+dirty" if dirty else "")


def parse_phase(spec):
    """`scenario:episodes[@opp+opp]` -> (scenario, episodes, [opponents])."""
    head, _, opponents = spec.partition("@")
    scenario, _, episodes = head.partition(":")
    if not episodes:
        raise ValueError("phase %r needs an episode count, e.g. classic:4000" % spec)
    return scenario, int(episodes), [o for o in opponents.split("+") if o]


def run_seed(job):
    (exp_id, seed_index, phases, overrides, phase_overrides, full_rounds,
     quiet, args_agent, fingerprint) = job
    seed_dir = EXPERIMENTS / exp_id / ("seed%d" % seed_index)
    seed_dir.mkdir(parents=True, exist_ok=True)

    previous = None
    model_path = None
    for phase_index, (scenario, episodes, opponents) in enumerate(phases):
        # checkpoint per phase, not per seed, so overwriting one doesn't lose
        # whether the skill survived the next phase
        model_path = seed_dir / ("model_phase%d.pkl" % phase_index)

        cfg = dict(overrides)
        if phase_index < len(phase_overrides):
            cfg.update(phase_overrides[phase_index])
        cfg["n_episodes"] = episodes
        cfg["model_path"] = str(model_path)
        cfg["log_path"] = str(seed_dir / ("phase%d_%s.csv" % (phase_index, scenario)))
        cfg["continue_from"] = str(previous) if previous else None
        cfg["seed"] = TRAIN_SEED_BASE + seed_index * 97 + phase_index
        cfg["git_commit"] = fingerprint
        config_path = seed_dir / ("phase%d_config.json" % phase_index)
        config_path.write_text(json.dumps(cfg, indent=2, sort_keys=True))

        cmd = [sys.executable, "main.py", "play", "--no-gui",
               "--agents", args_agent, *opponents,
               "--train", "1",
               "--scenario", scenario,
               "--n-rounds", str(episodes),
               "--seed", str(TRAIN_SEED_BASE + seed_index)]
        if full_rounds:
            # otherwise a reckless policy's rounds end at 30 steps, a passive
            # one's at 400, and training skews toward whichever ends later
            cmd.append("--continue-without-training")

        # Seeded opponent during training too, so a run repeats. Deriving
        # OPP_SEED from the training seed means the ten seeds face ten
        # different but reproducible opponents, not the same one ten times.
        env = dict(os.environ, **{config_env_var(args_agent): str(config_path)},
                   OPP_SEED=str(TRAIN_SEED_BASE + seed_index))

        started = time.time()
        proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError("phase %d of %s seed %d failed:\n%s\n%s"
                               % (phase_index, exp_id, seed_index,
                                  proc.stdout[-2000:], proc.stderr[-4000:]))
        if not quiet:
            print("  %s seed%d phase%d (%s x%d) %.0fs"
                  % (exp_id, seed_index, phase_index, scenario, episodes,
                     time.time() - started), flush=True)
        previous = model_path

    final = seed_dir / "model.pkl"
    shutil.copyfile(model_path, final)
    return str(final)


def _eval_config(exp_id, seed, model_path, overrides):
    """Config used at evaluation time.

    Learning parameters don't matter here (act() is greedy when self.train is
    False), but the overrides are still carried over: some of them change what
    the agent *observes*, and evaluating a model on a different observation
    than it trained on measures the wrong thing without any error.
    """
    seed_dir = EXPERIMENTS / exp_id / ("seed%d" % seed)
    cfg = dict(overrides)
    cfg["model_path"] = str(model_path)
    cfg["continue_from"] = None
    cfg["log_path"] = None
    path = seed_dir / "eval_config.json"
    path.write_text(json.dumps(cfg, indent=2, sort_keys=True))
    return str(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--agent", default="tabular_q")
    parser.add_argument("--phases", nargs="+", required=True,
                        help="scenario:episodes[@opp+opp], applied in order")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--overrides", default="{}", help="JSON of config keys")
    parser.add_argument("--phase-overrides", default="[]",
                        help="JSON list, one object per phase, on top of --overrides. "
                             "Mostly for eps_start: a warm-started phase that "
                             "restarts exploration at 1.0 throws the model away.")
    parser.add_argument("--eval-scenario", default=None)
    parser.add_argument("--eval-opponents", nargs="*", default=[])
    parser.add_argument("--eval-mode", default="fast", choices=["fast", "exact"])
    parser.add_argument("--eval-rounds", type=int, default=20)
    parser.add_argument("--eval-seeds", type=int, default=len(EVAL_SEEDS))
    parser.add_argument("--full-rounds", action="store_true",
                        help="don't cut rounds short when the agent dies")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    phases = [parse_phase(p) for p in args.phases]
    overrides = json.loads(args.overrides)
    phase_overrides = json.loads(args.phase_overrides)
    print("exp %s | phases %s | seeds %s | overrides %s | per-phase %s"
          % (args.exp_id, args.phases, args.seeds, overrides, phase_overrides),
          flush=True)

    fingerprint = code_fingerprint(args.agent)
    print("agent code at %s" % fingerprint, flush=True)

    jobs = [(args.exp_id, seed, phases, overrides, phase_overrides,
             args.full_rounds, False, args.agent, fingerprint)
            for seed in args.seeds]
    started = time.time()
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        models = list(pool.map(run_seed, jobs))
    print("training wall time %.0fs" % (time.time() - started), flush=True)

    eval_scenario = args.eval_scenario or phases[-1][0]
    results = []
    for seed, model_path in zip(args.seeds, models):
        tag = "%s_seed%d" % (args.exp_id, seed)
        metrics = evaluate(
            args.agent, args.eval_opponents, eval_scenario, mode=args.eval_mode,
            seeds=EVAL_SEEDS[:args.eval_seeds], n_rounds=args.eval_rounds, tag=tag,
            extra_env={config_env_var(args.agent):
                       _eval_config(args.exp_id, seed, model_path, overrides)},
            workers=args.eval_workers)
        print(format_metrics(metrics), flush=True)
        register(metrics, exp_id=tag, note=args.note)
        results.append(metrics)

    print("\n=== %s across %d seeds ===" % (args.exp_id, len(results)))
    for key in ("mean_score", "mean_coins", "mean_kills", "suicide_rate",
                "mean_bombs", "mean_crates", "win_rate", "survival_rate", "mean_steps"):
        values = [m[key] for m in results]
        mean = sum(values) / len(values)
        spread = max(values) - min(values)
        print("  %-14s %.3f   (per seed: %s, spread %.3f)"
              % (key, mean, ", ".join("%.3f" % v for v in values), spread))


if __name__ == "__main__":
    main()
