# Tools

Six scripts. `evaluate.py` is the library everything else calls; the rest are
run from the command line.

| | |
|---|---|
| `evaluate.py` | the measurement harness - protocol, metrics, registry. Not run directly |
| `train.py` | run a curriculum over several seeds, then evaluate |
| `eval_existing.py` | re-measure models already on disk, without retraining |
| `audit_report.py` | check every figure in report_tabular_q.md against the registry |
| `ship.py` | `choose`, `final`, `check` - see below |

```
python3 tools/train.py --exp-id NAME --phases classic:6000@rule_based_seeded --seeds 1 2 3
python3 tools/ship.py check --agent linear_q
```

Both agents work with all of these: pass `--agent linear_q`. The config
environment variable each agent reads is looked up in
`evaluate.py:CONFIG_ENV_VAR`, so an unregistered agent fails loudly instead of
being silently run on its defaults.

## ship.py

| | |
|---|---|
| `choose` | rank trained seeds on arenas 9101-9130, so the reported score comes from a block that had no say in the choice |
| `final` | the decisive evaluation: EXACT mode, 600 rounds, one process per round |
| `check` | the assignment's hard constraints, plus worst-case act() timing |

## Two rules worth knowing

**Measure against `rule_based_seeded`, not `rule_based_agent`.** Same agent,
but its tie-breaks are seeded from `OPP_SEED`, which `evaluate.py` sets to the
arena seed and `train.py` to the training seed, so a measurement repeats
exactly. The stock agent draws tie-breaks from an unseeded `random` - and one
of those draws sits inside the BFS it runs every step - which is where this
project's whole noise floor came from. Keep the stock agent for the final
check, because that is what the tournament runs.

**Don't name a tool after a stdlib module.** A file called `inspect.py` or
`select.py` on sys.path shadows the real one, `dataclasses` and
`concurrent.futures` import those, and every tool here breaks with an error
pointing nowhere near the cause. That cost a day once.
