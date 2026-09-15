# Measured results of javiuv's agents

Measurements of the two agents in `agent_code/tabular_q_agent/` and
`agent_code/linear_q_agent/`, whose code is by javiuv (branches
`feature/javiuv/tabular_q` and `feature/javiuv/linear_q`). javiuv's own
experiment outputs were not committed - his `.gitignore` excluded
`experiments/`, `results/` and `*.pkl` - so these were produced by datcn1212 on
2026-09-14 by running his code as committed.

## How they were produced

- Code: commit `54056c3`, no tracked file modified.
- Hyperparameters: his `config.yaml` - learning_rate 0.3, gamma 0.95, epsilon 0.4.
- Training: 10000 episodes per run, the value of `TOTAL_TRAINING_EPISODES` in his
  `experiment.py`, starting from an empty model each time:
  `python3 main.py play --no-gui --agents <agent> --train 1 --scenario <scenario> --n-rounds 10000`
- Each agent was trained twice, once on `coin-heaven` and once on `classic`, and
  evaluated on the scenario it was trained on.
- Evaluation: 300 rounds without `--train`, so his `setup()` sets epsilon to 0:
  `python3 main.py play --no-gui --agents <agent> [rule_based_agent x3] --scenario <scenario> --n-rounds 300 --save-stats <file>`
- Reference: `rule_based_agent` evaluated with the same commands.
- Boards and opponents are unseeded.

## Results, per round

| file | agent | setting | rounds | score | coins | self-kill | steps |
|---|---|---|---|---|---|---|---|
| `jav_lin_classic_four.json` | linear_q_agent | classic, vs 3 rule_based_agent | 300 | 0.030 | 0.030 | 0.003 | 137.0 |
| `jav_lin_classic_solo.json` | linear_q_agent | classic, solo | 300 | 0.000 | 0.000 | 0.000 | 400.0 |
| `jav_lin_coinheaven_solo.json` | linear_q_agent | coin-heaven, solo | 300 | 0.197 | 0.197 | 0.430 | 230.5 |
| `jav_tab_classic_four.json` | tabular_q_agent | classic, vs 3 rule_based_agent | 300 | 0.017 | 0.017 | 0.823 | 49.6 |
| `jav_tab_classic_solo.json` | tabular_q_agent | classic, solo | 300 | 0.000 | 0.000 | 0.643 | 146.0 |
| `jav_tab_coinheaven_solo.json` | tabular_q_agent | coin-heaven, solo | 300 | 50.000 | 50.000 | 0.000 | 124.4 |
| `ref_classic_four.json` | rule_based_agent | classic, 4 copies (mean) | 300 | 3.214 | 2.227 | 0.517 | 232.7 |
| `ref_classic_solo.json` | rule_based_agent | classic, solo | 300 | 8.507 | 8.507 | 0.000 | 398.4 |
| `ref_coinheaven_solo.json` | rule_based_agent | coin-heaven, solo | 300 | 50.000 | 50.000 | 0.000 | 124.4 |

## Notes

- The trained models committed in the two agent directories are the ones trained
  on `classic`. Both `coin-heaven` models were overwritten by the later `classic`
  runs; their results survive only in the JSON files here.
- The linear agent's weights after `classic` training have max |w| = 2.45e115
  (finite, not NaN). The configuration that produced them: learning_rate 0.3,
  eight raw features used directly as the feature vector, no bias term.
- The reference run agrees with the registry protocol used for
  `tabular_q`/`linear_q`: `rule_based_agent` scores 8.507 solo on `classic` here
  against 8.480 in `ref_rule_classic` (0.3% apart), and 3.214 on the four-agent
  board against 3.260 in `ref_4x_rulebased`, a gap inside the four-agent
  run-to-run standard deviation of 0.090 measured in `report_tabular_q.md`.
