"""Evaluation harness: runs an agent under a fixed protocol and turns the
framework's stats JSON into the metrics the experiment log reports.

Two modes, as described in the runbook:

FAST  -- one process per seed playing `n_rounds` rounds. Cheap, used for the
         inner loop (hyperparameter search, ablations). `by_agent` only holds
         totals over the rounds, so this mode yields means but no per-round
         distribution and no exact win rate.

EXACT -- one process per seed playing a single round, so every stats file
         describes exactly one round. Expensive (process startup dominates) but
         gives per-round distributions, an exact win rate, and survival, because
         `by_round[round]["steps"]` can be compared against the agent's own step
         count.

Both modes always play with `--train 0`, which makes the framework set
`continue_without_training`, so rounds are not cut short when our agent dies.
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
REGISTRY = ROOT / "experiments" / "registry.csv"

# Fixed evaluation seeds. Training never uses these, so every reported number is
# measured on arenas the agent was not trained on. Identical for every version
# so that comparisons across experiments are fair.
EVAL_SEEDS = list(range(9001, 9031))

# Upper bound on one `main.py play` call. Generous: the slowest observed block
# is ~120 s per process.
PLAY_TIMEOUT = 900


def _play(args):
    """Run one `main.py play` process and return the parsed stats dict."""
    tag, agents, scenario, n_rounds, seed, train_flag, extra_env = args
    stats_path = RESULTS / "eval" / f"{tag}.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "main.py", "play", "--no-gui",
           "--agents", *agents,
           "--scenario", scenario,
           "--n-rounds", str(n_rounds),
           "--seed", str(seed),
           "--train", str(train_flag),
           "--save-stats", str(stats_path)]
    env = dict(os.environ, **(extra_env or {}))
    # A hard timeout, because a child that dies or wedges otherwise blocks the
    # pool for ever: one segfaulting worker once stalled a run for three hours
    # with no output and no error.
    try:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True,
                              text=True, timeout=PLAY_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise RuntimeError("eval run %s exceeded %ds and was killed"
                           % (tag, PLAY_TIMEOUT))
    if proc.returncode != 0:
        raise RuntimeError(f"eval run {tag} failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}")
    with open(stats_path) as fh:
        return json.loads(fh.read())


def _agent_key(stats, agent_name, index=0):
    """Resolve the stats key for an agent, accounting for the `_0`/`_1` suffix
    the framework adds when the same code runs several times."""
    keys = list(stats["by_agent"].keys())
    if agent_name in keys:
        return agent_name
    suffixed = sorted(k for k in keys if k.startswith(agent_name + "_"))
    if suffixed:
        return suffixed[index]
    raise KeyError(f"{agent_name} not among {keys}")


def _mean_std(values):
    if not values:
        return float("nan"), float("nan")
    m = sum(values) / len(values)
    if len(values) == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return m, math.sqrt(var)


def evaluate(agent, opponents, scenario, mode="fast", seeds=None, n_rounds=30,
             tag="eval", extra_env=None, workers=4):
    """Evaluate `agent` against `opponents`; return a metrics dict."""
    seeds = list(seeds if seeds is not None else EVAL_SEEDS)
    agents = [agent] + list(opponents)
    rounds_per_call = 1 if mode == "exact" else n_rounds

    jobs = [(f"{tag}_{mode}_s{seed}", agents, scenario, rounds_per_call, seed, 0, extra_env)
            for seed in seeds]
    t0 = time.time()
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            all_stats = list(pool.map(_play, jobs))
    else:
        all_stats = [_play(j) for j in jobs]
    wall = time.time() - t0

    per_seed_score, per_seed_coins, per_seed_kills = [], [], []
    per_seed_suicide, per_seed_margin = [], []
    per_seed_bombs, per_seed_crates = [], []
    wins, draws, played = 0, 0, 0
    survived, total_invalid, total_steps, total_time = 0, 0, 0, 0.0
    round_steps = []

    for stats in all_stats:
        me_key = _agent_key(stats, agent)
        me = stats["by_agent"][me_key]
        rounds = me.get("rounds", 1) or 1
        opp_keys = [k for k in stats["by_agent"] if k != me_key]
        opp_scores = [stats["by_agent"][k].get("score", 0) / (stats["by_agent"][k].get("rounds", 1) or 1)
                      for k in opp_keys]

        my_mean_score = me.get("score", 0) / rounds
        per_seed_score.append(my_mean_score)
        per_seed_coins.append(me.get("coins", 0) / rounds)
        per_seed_kills.append(me.get("kills", 0) / rounds)
        per_seed_suicide.append(me.get("suicides", 0) / rounds)
        per_seed_margin.append(my_mean_score - (max(opp_scores) if opp_scores else 0.0))
        # Bombs and crates separate a genuinely safe policy from a merely
        # passive one: never bombing also scores zero suicides.
        per_seed_bombs.append(me.get("bombs", 0) / rounds)
        per_seed_crates.append(me.get("crates", 0) / rounds)

        total_invalid += me.get("invalid", 0)
        total_steps += me.get("steps", 0)
        total_time += me.get("time", 0.0)

        for rid, rstat in stats["by_round"].items():
            round_steps.append(rstat["steps"])

        if mode == "exact":
            played += 1
            best_opp = max(opp_scores) if opp_scores else -1
            if my_mean_score > best_opp:
                wins += 1
            elif opp_scores and my_mean_score == best_opp:
                draws += 1
            # Surviving means being polled on every step of the round AND not
            # having blown itself up. The step comparison alone is not enough:
            # in a solo game the round ends the moment the agent dies, so its
            # step count always equals the round length and every round would
            # look survived.
            only_round = next(iter(stats["by_round"].values()))
            if me.get("steps", 0) >= only_round["steps"] and not me.get("suicides", 0):
                survived += 1

    score_mean, score_std = _mean_std(per_seed_score)
    margin_mean, margin_std = _mean_std(per_seed_margin)
    coins_mean, coins_std = _mean_std(per_seed_coins)
    kills_mean, kills_std = _mean_std(per_seed_kills)
    sui_mean, sui_std = _mean_std(per_seed_suicide)
    bombs_mean, _ = _mean_std(per_seed_bombs)
    crates_mean, _ = _mean_std(per_seed_crates)
    steps_mean, _ = _mean_std([float(x) for x in round_steps])

    metrics = {
        "tag": tag, "mode": mode, "agent": agent,
        "opponents": "+".join(opponents) if opponents else "solo",
        "scenario": scenario, "seeds": len(seeds),
        "rounds_per_seed": rounds_per_call,
        "mean_score": score_mean, "score_std": score_std,
        "score_margin": margin_mean, "score_margin_std": margin_std,
        "mean_coins": coins_mean, "coins_std": coins_std,
        "mean_kills": kills_mean, "kills_std": kills_std,
        "suicide_rate": sui_mean, "suicide_std": sui_std,
        "mean_bombs": bombs_mean, "mean_crates": crates_mean,
        "mean_steps": steps_mean,
        "invalid_action_rate": (total_invalid / total_steps) if total_steps else float("nan"),
        "mean_think_time": (total_time / total_steps) if total_steps else float("nan"),
        "wall_seconds": wall,
    }
    if mode == "exact" and played:
        metrics["win_rate"] = wins / played
        metrics["draw_rate"] = draws / played
        metrics["survival_rate"] = survived / played
    else:
        metrics["win_rate"] = float("nan")
        metrics["draw_rate"] = float("nan")
        metrics["survival_rate"] = float("nan")
    return metrics


FIELDS = ["timestamp", "exp_id", "tag", "mode", "agent", "opponents", "scenario", "seeds",
          "rounds_per_seed", "mean_score", "score_std", "score_margin", "score_margin_std",
          "mean_coins", "coins_std", "mean_kills", "kills_std", "suicide_rate", "suicide_std",
          "mean_bombs", "mean_crates",
          "win_rate", "draw_rate", "survival_rate", "mean_steps", "invalid_action_rate",
          "mean_think_time", "wall_seconds", "note"]


def register(metrics, exp_id="", note=""):
    """Append one evaluation to experiments/registry.csv."""
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    row = {k: metrics.get(k, "") for k in FIELDS}
    row["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    row["exp_id"] = exp_id
    row["note"] = note
    # Appending under a header written by an older version of FIELDS silently
    # shifts every column added since, so the header is checked, not assumed.
    if REGISTRY.exists():
        with open(REGISTRY) as fh:
            existing = next(csv.reader(fh), [])
        if existing and existing != FIELDS:
            raise RuntimeError(
                "experiments/registry.csv was written with a different column set "
                "(%d columns, expected %d). Rewrite it with the current FIELDS "
                "before appending, or the columns will not line up."
                % (len(existing), len(FIELDS)))
    write_header = not REGISTRY.exists()
    with open(REGISTRY, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return row


def format_metrics(m):
    return (f"{m['agent']} vs {m['opponents']} @{m['scenario']} [{m['mode']}, "
            f"{m['seeds']} seeds x {m['rounds_per_seed']}]\n"
            f"  score      {m['mean_score']:.3f} +/- {m['score_std']:.3f}"
            f"   margin {m['score_margin']:+.3f} +/- {m['score_margin_std']:.3f}\n"
            f"  coins      {m['mean_coins']:.3f} +/- {m['coins_std']:.3f}"
            f"   kills {m['mean_kills']:.3f} +/- {m['kills_std']:.3f}\n"
            f"  suicide    {m['suicide_rate']:.3f} +/- {m['suicide_std']:.3f}"
            f"   win_rate {m['win_rate']:.3f}   survival {m['survival_rate']:.3f}\n"
            f"  bombs      {m['mean_bombs']:.2f}/round   crates {m['mean_crates']:.2f}/round\n"
            f"  steps      {m['mean_steps']:.1f}"
            f"   invalid {m['invalid_action_rate']:.4f}"
            f"   think {m['mean_think_time'] * 1000:.2f} ms"
            f"   wall {m['wall_seconds']:.1f}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", required=True)
    p.add_argument("--opponents", nargs="*", default=[])
    p.add_argument("--scenario", default="classic")
    p.add_argument("--mode", choices=["fast", "exact"], default="fast")
    p.add_argument("--n-rounds", type=int, default=30)
    p.add_argument("--seeds", type=int, default=len(EVAL_SEEDS))
    p.add_argument("--tag", default=None)
    p.add_argument("--exp-id", default="")
    p.add_argument("--note", default="")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    tag = args.tag or f"{args.agent}_{args.scenario}_{args.mode}"
    m = evaluate(args.agent, args.opponents, args.scenario, mode=args.mode,
                 seeds=EVAL_SEEDS[:args.seeds], n_rounds=args.n_rounds,
                 tag=tag, workers=args.workers)
    print(format_metrics(m))
    register(m, exp_id=args.exp_id, note=args.note)


if __name__ == "__main__":
    main()
