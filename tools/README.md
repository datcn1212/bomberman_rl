# Tools

Nine entry points. Four of them are what you use day to day; the rest are
grouped by job so that this folder shows one file per task rather than one per
script.

## Daily

| | |
|---|---|
| `train.py` | run a curriculum over several seeds, then evaluate |
| `eval_existing.py` | re-measure models already on disk, without retraining |
| `audit_report.py` | check every figure in the report against the registry |
| `verify_unchanged.py` | **before any agent-code change** - prove it alters nothing |

`train.py --exp-id NAME --phases classic:6000@rule_based_seeded --seeds 1 2 3`
is the whole interface. Everything else is a flag on it.

## Grouped

| | subcommands |
|---|---|
| `diagnose.py` | `policy` `bombs` `symmetry` `curves` |
| `ship.py` | `select` `final` `gate` `check` `latency` |
| `reference.py` | `baselines` `winrate` |

Run as `python3 tools/ship.py check`. Implementations live in the matching
folder and can be run directly too.

`diagnose` is not called `inspect`: a module named `inspect.py` on sys.path
shadows the standard library's, `dataclasses` imports that, and every tool here
breaks with an error that points nowhere near the cause.

## Library and one-offs

| | |
|---|---|
| `evaluate.py` | the measurement harness every other tool calls; not run directly |
| `hpsearch.py` | random search with successive halving |
| `phase0/` | `mechanics.py`, `events.py` - run once, before the agent existed |

## Two rules that the tools enforce

**Measure against `rule_based_seeded`, not `rule_based_agent`.** Same agent,
tie-breaks seeded from `OPP_SEED`, which `evaluate.py` sets to the arena seed
and `train.py` to the training seed. A measurement then repeats exactly. The
stock agent draws its tie-breaks from an unseeded `random` module - and one of
those draws sits inside the breadth-first search it runs every step - which is
where this project's entire noise floor came from. Keep the stock agent for the
final check, since the tournament runs it.

**Run `verify_unchanged.py` before trusting a refactor.** It trains a few
hundred episodes twice under two configs and compares the tables entry by entry.
It runs solo, because with unseeded opponents no two runs repeat and the
comparison cannot conclude anything.
