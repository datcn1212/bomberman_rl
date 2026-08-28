# Tools

Nineteen scripts, but only four are used routinely. This table is the map; the
rest of this file explains the three groups.

| script | when you use it |
|---|---|
| `train.py` | **daily** - run a curriculum over several seeds, then evaluate |
| `evaluate.py` | **library** - every other script calls it; not run directly |
| `eval_existing.py` | **daily** - re-measure models already on disk |
| `audit_report.py` | **daily** - check every figure in the report against the registry |
| `verify_unchanged.py` | **before any code change** - prove a change alters nothing |
| `hpsearch.py` | occasionally - random search with successive halving |
| `inspect_policy.py` | when a result surprises you - decode the table into a policy |
| `analyse_bombs.py` | when a result surprises you - what the bombs actually achieved |
| `analyse_symmetry.py` | when a result surprises you - how much the table folds |
| `plot_curves.py` | occasionally - learning curves from the training logs |
| `select_final.py` | **shipping** - pick which seed to ship, on held-out arenas |
| `final_eval.py` | **shipping** - the decisive measurement of the shipped model |
| `gate_check.py` | **shipping** - EXACT mode, the only way to count win rate |
| `submission_check.py` | **shipping** - the assignment's hard constraints |
| `benchmark_latency.py` | **shipping** - worst-case decision time |
| `run_baselines.py` | reference - the four agents the framework ships with |
| `reference_winrate.py` | reference - those agents at our own sample size |
| `verify_mechanics.py` | **once, Phase 0** - bomb timing, blast shape |
| `verify_events.py` | **once, Phase 0** - how callbacks are delivered |

## The daily loop

```
train.py  ->  writes experiments/<exp>/seed<k>/ and appends to registry.csv
              |
              +-- eval_existing.py   re-measure without retraining
              +-- audit_report.py    confirm the report still matches the data
```

`train.py --exp-id NAME --phases classic:6000@rule_based_seeded --seeds 1 2 3`
is the whole interface. Everything else is a flag on that.

## Before changing agent code

`verify_unchanged.py` trains a few hundred episodes twice under two configs and
compares the tables entry by entry. Run it for any change described as a
refactor, an optimisation, or a new option that defaults to off. It runs solo,
because with opponents on the board no two runs repeat and the comparison cannot
conclude anything.

## Measuring against opponents

Use `rule_based_seeded` rather than `rule_based_agent`. It is the same agent with
its tie-breaks seeded from `OPP_SEED`, which the harness sets to the arena seed,
so a measurement repeats exactly. The stock agent draws its tie-breaks from an
unseeded `random` module, which is where this project's whole noise floor came
from. Keep the stock agent for the final check, since the tournament runs it.
