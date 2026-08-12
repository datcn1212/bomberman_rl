# Tabular Q-learning for Bomberman - experiment log

This file is the working log of one model of the final project: a **tabular
value-based agent** trained with temporal-difference control. It records, in
order, what was tried, why it was tried, what was measured, and what the
measurement implies for the next step. Negative results are kept, because a
change that did not help is evidence about the problem.

The agent lives in `agent_code/tabular_q/`. Every experiment is reproducible
from the JSON config stored next to its model in `experiments/<exp_id>/seed<k>/`.

---

## 1. Approach

### 1.1 Why tabular

Bomberman is a Markov decision process with a discrete action set and, in its raw
form, an astronomically large state space: a 17x17 board with crates, coins,
bombs and up to four agents cannot be enumerated. A table cannot index the raw
state, so the whole design problem moves into the **encoding**: a map from the
game state onto a small set of integers that preserves the information an
optimal policy needs and discards the rest.

That constraint is the reason to start here rather than with a network. A table
has no representation error and no optimisation noise - if the agent fails, it
fails because the encoding threw away something it needed, or because the reward
did not express what we wanted, or because exploration never reached the
relevant states. Each of those is diagnosable. The same failure inside a neural
network is not. The table is therefore used as an instrument for understanding
the task, and the numbers it produces are a baseline every later model has to
beat.

### 1.2 MDP formulation

**Actions.** The six the framework defines, in the framework's own order:
`UP, RIGHT, DOWN, LEFT, WAIT, BOMB`. Direction index *i* in the feature code is
the same integer as move action *i*, which removes a whole class of off-by-one
bugs.

**States.** A tuple of small integers, encoded into one table row by mixed-radix
arithmetic. The tuple is the part of this project that actually evolves; its
Phase 1 version is deliberately minimal and grows only when a measurement shows
that something is missing. The radix of each component is recorded in
`model.LAYOUT`, stored inside the pickle, and checked on load - changing the
encoding while older models exist otherwise fails silently, because every lookup
misses, the agent still runs, and the scores merely get worse for no visible
reason.

**Rewards.** The game's own score (1 per coin, 5 per kill) is far too sparse to
learn from directly: an agent that never finds a coin sees a constant zero and
has no gradient to follow. So the reward is built from the framework's event
list, which fires on every step. All reward weights are configuration values, so
a reward change is an experiment with a config diff rather than a code edit.

**Update.** One-step temporal-difference control, off-policy (Q-learning):

```
Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a')  -  Q(s,a) ]
```

with the terminal transition using target `r` alone. `max` runs over the
*legal* actions only (see Phase 1.3). Exploration starts as epsilon-greedy with
epsilon decaying linearly from `eps_start` to `eps_end`.

### 1.3 Measurement protocol

Fixed from the start so that every number in this log is comparable.

| | |
|---|---|
| Training seeds | 1000 + k, k = 1..5 |
| Evaluation seeds | 9001..9030, **disjoint from training** |
| FAST mode | 30 seeds x 20 rounds in one process per seed; gives means, no per-round distribution |
| EXACT mode | one process per round, so survival and win rate are exact; expensive |
| Reported spread | standard deviation across seeds; a difference below ~2 standard errors is not treated as a difference |

Every configuration is run on **5 training seeds**. A single seed says nothing:
the variance between seeds of the same configuration is routinely as large as
the effect being measured, and the whole point of the log is to separate the two.
Head-to-head claims against `rule_based_agent` are made only from **600 rounds**
or more.

---

## 2. Phase 0 - verify the environment before building on it

**What and why.** Everything the agent believes about bombs, blasts and training
callbacks is a modelling assumption. If any of it is wrong the learner is
optimising the wrong problem, and the resulting bug is invisible: the agent
still plays, it just plays badly. So before any learning code was written, the
game was measured from the outside with a scripted, non-learning `probe_agent`
(`tools/verify_mechanics.py`, `tools/verify_events.py`), and the answers were
compared against a straight-line blast model.

### 2.1 Bomb and blast mechanics (measured)

| question | measurement |
|---|---|
| Delay from `BOMB` to the blast | The bomb appears one step later with timer 3 and counts down 3,2,1,0; it detonates at the end of the step where the observed timer is **0**, i.e. 4 steps after the action. |
| Lethal window | The detonation step **and the following step** (where `explosion_map` shows 1). Stepping into the tile two steps after detonation survives. Verified with three scripts that re-enter the blast row at different moments: re-entry at t+0 and t+1 died, at t+2 survived. |
| Blast shape | Exactly the straight-line model: up to `BOMB_POWER` = 3 tiles in each of the four directions, stopped at the first wall. Off-axis cells: **none**, so blasts never turn corners. Verified both from an open corner and next to interior pillars; predicted and observed cell sets matched exactly in both cases. |
| Bomb availability | `bombs_left` returns to True **7 steps** after the `BOMB` action, independently of where the agent stands. |
| Standing still after bombing | The agent is polled for the last time on the step where the timer reads 0, so a bomb with observed timer 0 kills at the end of the current step - there is no further chance to move. |

The practical consequence for the state encoding is that danger is not a
property of a tile but of a **(tile, step) pair**: a tile can be lethal now and
safe in two steps, or safe now and lethal in three. Phase 2 returns to this.

### 2.2 Training callback delivery (measured)

Two properties of the framework decide how the learner must be structured, and
both are easy to get wrong by assumption:

**(a) The last step of a surviving round is delivered twice.** Measured
directly: a round that ends at step 9 produced nine `game_events_occurred`
calls, the last with `old_step` = 9, and then an `end_of_round` call **also**
with `old_step` = 9, carrying `[WAITED, SURVIVED_ROUND]` where the first
carried `[WAITED]`. A learner that updates in both callbacks therefore applies
the final transition of every surviving round twice, and double-counts its
reward.

**(b) A fatal step is delivered only through `end_of_round`.** In the suicide
probe, `game_events_occurred` stopped at step 4 and the events
`[WAITED, BOMB_EXPLODED, KILLED_SELF, GOT_KILLED]` arrived at step 5 through
`end_of_round` alone. A learner that handles only `game_events_occurred` never
observes dying and cannot learn to avoid it.

**Design consequence.** Transitions are **staged rather than learned on
arrival**. A staged transition is flushed when the next one arrives;
`end_of_round` compares its step number with the staged one and either replaces
it (same number - it is the re-delivery, and the `end_of_round` version wins
because its event list is complete) or flushes it first (different number - the
agent died, and this is a genuinely new transition). This yields exactly one
update per step in both cases.

**Comment.** This phase produced no learning at all, and it is still the phase
with the best effort-to-value ratio in the log. Both findings are silent bugs:
neither crashes, neither shows up in a training curve, and both bias the value
function in a direction that looks like "the agent is just not learning well".

---
