# Bomberman RL, team JADA

Our code for the final project of Machine Learning Essentials (Summer Semester
2026, Heidelberg University). The game framework is a fork of
[ukoethe/bomberman_rl](https://github.com/ukoethe/bomberman_rl); our work is in
the agent directories below, `tools/` and `experiments/`.

**Additional libraries:** none. The agents import only `numpy` (provided by the
course Dockerfile), the Python standard library and the framework's own
modules.

## Agents

| Directory | What it is |
|---|---|
| `agent_code/tabular_q` | Tabular Q-learning, submitted to the tournament as `JADA` |
| `agent_code/linear_q` | Linear Q-learning over the same observation, our second model |
| `agent_code/tabular_q_agent`, `agent_code/linear_q_agent` | Early prototypes, kept for the comparison in the report |
| `agent_code/rule_based_seeded` | `rule_based_agent` with seeded tie-breaks, used only for evaluation |

Two earlier lines of work are on branches: `feature/datcn/SARSA` (SARSA on the
tabular encoding) and `dev/datcn/RF_fittedQ` (random-forest fitted Q, dropped).

Both shipped models (`model.pkl`) load with the defaults in their `config.py`,
which is what the tournament runs.

## Reproducing the results

```
# watch the submitted agent against three rule-based opponents
python3 main.py play --agents tabular_q rule_based_agent rule_based_agent rule_based_agent

# train the shipped configurations, ten seeds each
python3 tools/train.py --exp-id p26_anchor --seeds 1 2 3 4 5 6 7 8 9 10 \
    --phases classic:6000@rule_based_seeded+rule_based_seeded+rule_based_seeded
python3 tools/train.py --agent linear_q --exp-id lq_p6_budget12k \
    --seeds 1 2 3 4 5 6 7 8 9 10 --phases loot-crate:12000

# pick a seed on the selection arenas, measure it, check the hard constraints
python3 tools/ship.py choose --exp-id p26_anchor
python3 tools/ship.py final --agent tabular_q --reference
python3 tools/ship.py check --agent tabular_q
```

Every evaluation appends a row to `experiments/registry.csv`, and every trained
model keeps its configuration and git commit under
`experiments/<exp_id>/seed<k>/`. `tools/README.md` describes the tools.

## Logs

`report_tabular_q.md` and `report_linear_q.md` are our working logs, one entry
per experimental phase, and `tools/audit_report.py` re-derives the tabular
log's headline figures from the registry. They are not the project report,
which is submitted separately.
