"""Runs an agent under a fixed protocol and turns the framework's stats JSON
into the numbers the experiment log reports.

Two modes:

FAST   one process per seed, n_rounds rounds each. Cheap, for sweeps and
       ablations. by_agent only has totals: means, no per-round distribution,
       no real win rate.
EXACT  one process per seed, one round each, so every stats file describes a
       single round. Slow (process startup dominates) but gives win rate and
       survival honestly.

Always played with --train 0 so the framework keeps rounds going after the
agent dies, instead of cutting them short.
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

# Each agent reads its run config from its own env var. Hardcoding one name
# here would mean silently evaluating a different agent on untouched defaults -
# no error, just a measurement of nothing. Unknown names fail loudly.
CONFIG_ENV_VAR = {
    "tabular_q": "TQ_CONFIG",
    "linear_q": "LQ_CONFIG",
}


def config_env_var(agent):
    try:
        return CONFIG_ENV_VAR[agent]
    except KeyError:
        raise KeyError("no config env var for agent %r; add it to "
                       "tools/evaluate.py:CONFIG_ENV_VAR" % agent) from None


# Never used for training, so every reported number comes from arenas the agent
# has not seen. Same list for every experiment, so runs stay comparable.
EVAL_SEEDS = list(range(9001, 9031))

# Cap on one play call, generous (slowest block seen is ~120s), just enough
# to stop a wedged child blocking the pool indefinitely.
PLAY_TIMEOUT = 900


def _play(args):
    """One `main.py play` process; returns the parsed stats dict."""
    tag, agents, scenario, n_rounds, seed, train_flag, extra_env = args
    stats_path = RESULTS / "eval" / ("%s.json" % tag)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, "main.py", "play", "--no-gui",
           "--agents", *agents,
           "--scenario", scenario,
           "--n-rounds", str(n_rounds),
           "--seed", str(seed),
           "--train", str(train_flag),
           "--save-stats", str(stats_path)]
    env = dict(os.environ, **(extra_env or {}))

    try:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True,
                              text=True, timeout=PLAY_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise RuntimeError("eval run %s exceeded %ds and was killed"
                           % (tag, PLAY_TIMEOUT))
    if proc.returncode != 0:
        raise RuntimeError("eval run %s failed:\n%s\n%s"
                           % (tag, proc.stdout[-2000:], proc.stderr[-4000:]))
    with open(stats_path) as fh:
        return json.load(fh)


def _agent_key(stats, agent_name, index=0):
    """Find the agent in by_agent; framework adds _0/_1 for repeated code."""
    keys = list(stats["by_agent"].keys())
    if agent_name in keys:
        return agent_name
    suffixed = sorted(k for k in keys if k.startswith(agent_name + "_"))
    if suffixed:
        return suffixed[index]
    raise KeyError("%s not among %s" % (agent_name, keys))


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
    """Play `agent` against `opponents` and return the metrics dict."""
    seeds = list(seeds if seeds is not None else EVAL_SEEDS)
    agents = [agent] + list(opponents)
    rounds_per_call = 1 if mode == "exact" else n_rounds

    # OPP_SEED = the arena seed. rule_based_seeded seeds `random` from it, so
    # each arena gets one fixed opponent behaviour and the run repeats exactly;
    # 30 arenas then sample 30 reproducible opponents instead of one. The stock
    # rule_based_agent ignores the variable, so setting it is harmless.
    jobs = [("%s_%s_s%d" % (tag, mode, seed), agents, scenario, rounds_per_call,
             seed, 0, dict(extra_env or {}, OPP_SEED=str(seed)))
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
        opp_scores = [stats["by_agent"][k].get("score", 0)
                      / (stats["by_agent"][k].get("rounds", 1) or 1)
                      for k in opp_keys]

        my_mean_score = me.get("score", 0) / rounds
        per_seed_score.append(my_mean_score)
        per_seed_coins.append(me.get("coins", 0) / rounds)
        per_seed_kills.append(me.get("kills", 0) / rounds)
        per_seed_suicide.append(me.get("suicides", 0) / rounds)
        per_seed_margin.append(my_mean_score - (max(opp_scores) if opp_scores else 0.0))
        # bombs/crates separate a safe policy from a merely passive one: never
        # bombing also gives a suicide rate of zero
        per_seed_bombs.append(me.get("bombs", 0) / rounds)
        per_seed_crates.append(me.get("crates", 0) / rounds)

        total_invalid += me.get("invalid", 0)
        total_steps += me.get("steps", 0)
        total_time += me.get("time", 0.0)
        for rstat in stats["by_round"].values():
            round_steps.append(rstat["steps"])

        if mode == "exact":
            played += 1
            best_opp = max(opp_scores) if opp_scores else -1
            if my_mean_score > best_opp:
                wins += 1
            elif opp_scores and my_mean_score == best_opp:
                draws += 1
            # survived = polled every step AND no suicide. Step count alone
            # isn't enough since solo, a round ends exactly when the agent dies
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


# Column order of experiments/registry.csv. Don't reorder, or old rows stop
# lining up with the header.
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

    # Check the header rather than assume it: appending under a header from an
    # older FIELDS would shift every column added since.
    if REGISTRY.exists():
        with open(REGISTRY) as fh:
            existing = next(csv.reader(fh), [])
        if existing and existing != FIELDS:
            raise RuntimeError(
                "registry.csv has a different column set (%d, expected %d); "
                "rewrite it before appending or the columns won't line up"
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

    tag = args.tag or "%s_%s_%s" % (args.agent, args.scenario, args.mode)
    m = evaluate(args.agent, args.opponents, args.scenario, mode=args.mode,
                 seeds=EVAL_SEEDS[:args.seeds], n_rounds=args.n_rounds,
                 tag=tag, workers=args.workers)
    print(format_metrics(m))
    register(m, exp_id=args.exp_id, note=args.note)


if __name__ == "__main__":
    main()
