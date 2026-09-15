"""Training driver: runs a curriculum of phases for several seeds in parallel,
then evaluates the result under the standard protocol.

A phase is written `scenario:episodes[@opponent+opponent]`, and phases chain by
warm-starting each one from the model the previous phase left behind:

    python3 tools/train.py --exp-id p1_minimal \\
        --phases coin-heaven:2000 --seeds 1 2 3

Training may use several processes (the ban on multiprocessing applies to the
submitted agent, not to the experiment harness); the agent itself never does.

Everything a run needs to be reproduced is written to
experiments/<exp_id>/seed<k>/: the exact config of each phase, the per-episode
training log, and the model.
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


def code_fingerprint(agent):
    """Commit of the agent code, plus a dirty marker.

    Stamped into every phase config so that two runs can never be compared
    without it being visible that they were produced by different code.
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
# Training seeds are drawn from a range disjoint from EVAL_SEEDS (9001..9030),
# so no result is ever reported on an arena the agent trained on.
TRAIN_SEED_BASE = 1000


def parse_phase(spec):
    """`scenario:episodes[@opponent+opponent]` -> (scenario, episodes, [opponents])."""
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
        # One checkpoint per phase rather than one per seed: overwriting a single
        # file makes the intermediate model unavailable, and "did the skill
        # survive the next phase" is exactly the question a curriculum raises.
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
            # Without this the engine ends the round the moment our agent dies,
            # so a policy that dies early produces a 30-step episode while a
            # passive one produces 400 steps. The learner then sees far more
            # passive transitions than bombing ones, which is self-reinforcing.
            cmd.append("--continue-without-training")
        # Training faces a seeded opponent too, so a run repeats. OPP_SEED is
        # derived from the training seed, so each of the ten seeds meets a
        # different but reproducible opponent rather than all ten meeting the
        # same one. Ignored by the stock `rule_based_agent`.
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
        # Each phase continues the model the previous one produced.
        previous = model_path
    final = seed_dir / "model.pkl"
    shutil.copyfile(model_path, final)
    return str(final)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--agent", default="tabular_q")
    parser.add_argument("--phases", nargs="+", required=True,
                        help="scenario:episodes[@opp+opp], applied in order")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--overrides", default="{}",
                        help="JSON of config.py keys to override")
    parser.add_argument("--phase-overrides", default="[]",
                        help="JSON list, one object per phase, applied on top of "
                             "--overrides. Mainly for eps_start: a warm-started "
                             "phase that restarts exploration at 1.0 throws away "
                             "the model it was given.")
    parser.add_argument("--eval-scenario", default=None)
    parser.add_argument("--eval-opponents", nargs="*", default=[])
    parser.add_argument("--eval-mode", default="fast", choices=["fast", "exact"])
    parser.add_argument("--eval-rounds", type=int, default=20)
    parser.add_argument("--eval-seeds", type=int, default=len(EVAL_SEEDS))
    parser.add_argument("--full-rounds", action="store_true",
                        help="pass --continue-without-training so rounds are not "
                             "cut short when our agent dies")
    parser.add_argument("--workers", type=int, default=3,
                        help="training processes; one per seed removes a whole "
                             "scheduling round, and single-threaded jobs tolerate "
                             "mild oversubscription")
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


def _eval_config(exp_id, seed, model_path, overrides):
    """Config used at evaluation time.

    Learning parameters are irrelevant here because act() is greedy when
    self.train is False, but the overrides are still carried over: some of them
    (use_opponents, for one) change what the agent *observes*, and evaluating a
    model with a different observation than it trained on would silently measure
    the wrong thing.
    """
    seed_dir = EXPERIMENTS / exp_id / ("seed%d" % seed)
    cfg = dict(overrides)
    cfg["model_path"] = str(model_path)
    cfg["continue_from"] = None
    cfg["log_path"] = None
    path = seed_dir / "eval_config.json"
    path.write_text(json.dumps(cfg, indent=2, sort_keys=True))
    return str(path)


if __name__ == "__main__":
    main()
