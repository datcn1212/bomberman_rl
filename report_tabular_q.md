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

## 3. Phase 1 - the smallest agent that can solve Task 1

**What and why.** Task 1 is coin collection on a board with no crates and no
opponents (`coin-heaven`, 50 coins, 400 steps). The goal of this phase is not a
good agent but a *baseline whose failures are understandable*, so the state was
made as small as it can be while still being sufficient in principle:

| component | radix | meaning |
|---|---|---|
| `move_status` | 16 | each of the four neighbours free or blocked (1 bit each) |
| `target_dir` | 5 | first step of the shortest path to the nearest coin: none, UP, RIGHT, DOWN, LEFT |

That is **80 table rows**, of which 32 are reachable on this scenario. The
direction comes from a breadth-first search rather than from comparing
coordinates, because the board has dead ends where the straight-line direction
points into a wall.

Rewards: `+1` coin, `-0.5` invalid action, `-0.05` wait, `-0.01` per step,
`-5` for blowing oneself up. Exploration: epsilon-greedy, 1.0 to 0.05 linearly
over 2000 episodes. Step size: constant `alpha = 0.1`, `gamma = 0.9`.

### 3.1 The action set on Task 1

A smoke test showed the agent killing itself in **every one of the first five
episodes**: random exploration drops a bomb, and the state contains no
information about bombs at all, so the consequence of that action is not
representable - the states before dying are indistinguishable from safe ones.
Since `coin-heaven` has no crates and no opponents, a bomb has no possible
upside there, so Phase 1 masks `BOMB` out of the action set. The mask is removed
again in Phase 2, when the state gains the information needed to survive a bomb.

One detail this forces: the bootstrap `max` must run over **legal actions only**.
With `BOMB` masked its row stays at zero, and a plain `max` would bootstrap from
that zero whenever every legal action is worth less - an optimistic target
taken from an action the agent cannot even choose.

### 3.2 Result

`p1_const`: 5 training seeds, evaluated on 30 held-out seeds x 20 rounds.

| seed | coins / round | steps / round |
|---|---|---|
| 1 | 50.000 | 124.1 |
| 2 | 50.000 | 124.0 |
| 3 | **8.798** | **400.0** |
| 4 | 50.000 | 123.8 |
| 5 | 50.000 | 123.9 |

Four seeds are **perfect** - all 50 coins, every round, in about 124 steps. One
seed collects 8.8 coins and never finishes a round. The training logs give no
warning: at episode 2000 seed 3 looked exactly like seed 1 (135 steps, ~50
coins). The difference appears only under a **greedy** policy; during training
the residual 5% exploration was enough to hide it.

### 3.3 Diagnosis

A table can be read, so the failure was located rather than guessed at
(`tools/inspect_policy.py`). Across all 32 reachable states, seed 3 has
**exactly one** whose greedy action disagrees with the coin direction:

> **state 54** - a horizontal corridor (only LEFT and RIGHT free), nearest coin
> to the **LEFT** - greedy action **RIGHT**, margin **0.0112**.

In a corridor, walking away from the coin leads to another corridor tile with
the same encoding, so the greedy policy repeats the same mistake until the
corridor ends. One wrong row out of 32 costs 82% of the score.

The obvious reading is "seed 3 was unlucky". The Q values across seeds say
otherwise:

| seed | Q(54, LEFT) | Q(54, RIGHT) | margin | visits |
|---|---|---|---|---|
| 1 | 3.6360 | 3.0977 | +0.5382 | 77390 |
| 2 | 3.5032 | 3.0378 | +0.4654 | 76778 |
| 3 | 3.1361 | 3.1473 | **-0.0112** | 76488 |
| 4 | 3.6275 | 3.1710 | +0.4565 | 77321 |
| 5 | 3.5395 | 3.1932 | +0.3463 | 77402 |

Every seed updated this row about 77000 times, so this is not a data problem.
To see what those updates were doing, the Q row of state 54 was logged **once
per episode** (`trace_state` in the config) and the margin examined over the
last 500 episodes:

| seed | mean margin | sd | sign changes in 500 episodes | value at stop |
|---|---|---|---|---|
| 1 | +0.3554 | 0.1797 | 28 | +0.5382 |
| 2 | +0.3271 | 0.1703 | 20 | +0.4654 |
| 3 | +0.3461 | 0.2010 | 37 | **-0.0112** |
| 4 | +0.3675 | 0.1894 | 28 | +0.4565 |
| 5 | +0.3231 | 0.1815 | 36 | +0.3463 |

This is the actual finding, and it is not the one the score table suggested:

* **all five seeds learned the same correct preference** - the mean margin is
  +0.32 to +0.37 everywhere, seed 3 included;
* the noise on that estimate has sd ~0.18-0.20, **about half the signal**;
* the margin **crosses zero 20 to 37 times in the last 500 episodes of every
  seed**, including the four that scored 50.

So no seed converged. The estimate is a stationary random walk around the right
answer, and which policy comes out is decided by **which side of zero the walk
happens to be on when training stops**. Four seeds stopped on the lucky side.

This is textbook: a constant step size satisfies `sum(alpha) = inf` but not
`sum(alpha^2) < inf`, so the Robbins-Monro conditions for almost-sure
convergence do not hold. The practical form of that theorem here is that the
estimate keeps a permanent variance proportional to `alpha`, and **a decision
whose true margin is smaller than that noise is decided by chance**.

**Comment.** The instructive part is how invisible this is. The reward curve is
smooth, the training score is at its maximum, 96.8% of individual stopping times
in seed 3 would have produced a working policy, and the resulting agent still
loses 82% of its score. Any conclusion drawn from a single seed here would have
been wrong - in either direction.

---

## 4. Phase 2 - step size

**What and why.** Phase 1 produced a hypothesis, not a conclusion: the estimate
is a random walk because the step size is constant, and the policy is decided by
where the walk sits at the stopping time. That hypothesis makes two predictions
that point in opposite directions, so the experiment can actually fail:

* if the problem is *insufficient training*, then doubling the number of
  episodes should reduce the failure rate;
* if the problem is *non-convergence*, then doubling the number of episodes
  should change nothing at all, while a step size satisfying the Robbins-Monro
  conditions should fix it outright.

Three configurations, **10 seeds each** (10 rather than 5, because the quantity
being estimated is a failure *rate* and 5 seeds cannot distinguish 20% from 5%):

| id | step size | episodes |
|---|---|---|
| `p2_const2k` | constant `alpha = 0.1` | 2000 |
| `p2_const4k` | constant `alpha = 0.1` | 4000 |
| `p2_visit2k` | `alpha / (1 + n(s,a)/1000)` | 2000 |

The decaying schedule counts visits **per (state, action) pair**, not globally.
That matters here: the corridor rows are updated ~80000 times while the rarest
reachable rows are updated ~150 times, and a global schedule would either freeze
the rare rows before they had learned anything or leave the common ones noisy.

### 4.1 Result

| config | mean coins | seeds broken | which seeds | spread |
|---|---|---|---|---|
| `p2_const2k` | 41.65 | 2 / 10 | 3, 9 | 42.31 |
| `p2_const4k` | 48.72 | 1 / 10 | **4** | 12.82 |
| `p2_visit2k` | **50.00** | **0 / 10** | - | **0.000** |

Both predictions were tested and only one survived.

Doubling the training did **not** fix the problem. The count went from 2/10 to
1/10, which at n = 10 is not a distinguishable difference, but the informative
part is not the count: **the set of failing seeds is completely disjoint**.
Seeds 3 and 9, which failed at 2000 episodes, are both perfect at 4000; seed 4,
perfect at 2000, fails at 4000 - and on state 36, a three-way junction rather
than a corridor. More data did not make weak seeds stronger, it reshuffled which
seed was unlucky. That is what a random walk does and what an estimator
converging on insufficient data does not.

The decaying step size, in contrast, solves Task 1 **exactly**, on every seed:
50.000 coins with a between-seed spread of 0.000, and 123.9 steps per round.

### 4.2 The mechanism, checked directly

The per-episode trace of state 54 over the last 500 episodes, both schedules:

| | constant alpha | visit-decay alpha |
|---|---|---|
| mean margin (range over seeds) | +0.307 to +0.368 | +0.293 to +0.321 |
| sd of the margin | 0.170 to 0.208 | **0.020 to 0.041** |
| sign changes in 500 episodes | 20 to 40 | **0, in all 10 seeds** |

The mean is unchanged - both schedules learn the same preference - while the
noise drops by roughly a factor of six and the sign of the decision stops
flipping entirely. That is precisely the predicted mechanism, and it is
measured rather than inferred from the score.

### 4.3 Where the failures actually live

Reading the tables directly shows that neither failing seed had more than one
bad row, and that the two bad rows are the same situation on different axes:

| run | seed | bad state | situation | greedy | margin |
|---|---|---|---|---|---|
| 2000 ep | 3 | 54 | horizontal corridor, coin LEFT | RIGHT | -0.0112 |
| 2000 ep | 9 | 28 | vertical corridor, coin DOWN | UP | -0.0581 |
| 4000 ep | 4 | 36 | junction (U+R+D free), coin UP | RIGHT | - |

Grouping every reachable row of all 10 constant-alpha seeds by its local
geometry explains why those two:

| state group | rows | median margin | smallest margin |
|---|---|---|---|
| **straight corridor** (only U+D or only L+R free) | 40 | **0.371** | **0.011** |
| corner (2 open, perpendicular) | 80 | 0.879 | 0.312 |
| 3 open | 120 | 0.689 | 0.132 |
| 4 open | 40 | 0.570 | 0.208 |

Straight corridors carry roughly **half the decision margin** of any other
geometry and contain the smallest margin in the whole table - which is why two
of the three observed failures landed there, though seed 4's junction failure
shows the noise can flip any row that happens to be close. The reason is
aliasing in the minimal encoding: in a straight corridor, stepping away from the
target lands on a tile with the *identical* encoding - still a corridor, still
the same target direction - so the value of the wrong action bootstraps from
almost the same successor value as the right one. Everywhere else the wrong
actions lead somewhere visibly different. The state simply does not contain the
distance that would separate them.

So the noise was only half the story. The failure needs a small *true* margin
**and** a large estimation noise at the same time, and the state encoding
supplies the first while the step size supplies the second. Phase 2 removes the
noise; the narrow margin is a property of the encoding and is revisited later
with potential-based shaping.

**Decision.** `alpha_schedule = "visit"` from here on.

---

## 5. Phase 3 - bombs, and danger as a schedule

**What and why.** Task 2 adds crates, and crates can only be removed with
bombs - so the agent must use the one action that can kill it. Phase 1 already
showed why the minimal state cannot support that: with no representation of
bombs, the states before dying are indistinguishable from safe ones, and the
`-5` death penalty gets smeared over rows that are mostly fine.

The design decision that follows from the Phase 0 measurements is that **danger
is not a property of a tile but of a (tile, step) pair**. A tile can be lethal
now and safe in two steps, or safe now and lethal in three. A boolean "am I in
a blast radius" flag cannot express either, and an agent holding one has to
treat a tile it could safely cross as a wall.

So the features carry a schedule, `lethal[k][x, y]`, built from the measured
timing rules, and two searches run on top of it:

* **`escape_search`** - a breadth-first search over **(tile, step) pairs**
  rather than tiles, asking whether any sequence of moves survives the whole
  horizon. Because the search carries the step index, it will happily route a
  path *through* a tile that burns later, which is often the only way out of a
  corridor.
* **`target_search`** - the Phase 1 coin search, extended so that a tile next to
  a crate is also a goal (a crate cannot be walked onto, so adjacency is what
  "arriving" means), and with tiles that burn now or next step treated as walls.

State grows from 80 rows to **43740**:

| component | radix | meaning |
|---|---|---|
| `move_status` | 81 | each neighbour: blocked / free-safe / free-lethal |
| `t_here` | 5 | decisions until the current tile burns; 0 = it does not |
| `target_dir` | 6 | direction to the nearest coin or crate, or "already there" |
| `target_kind` | 3 | whether that target is a crate, a coin, or an opponent |
| `escape_dir` | 6 | first move of a surviving path, or "none", or "stay" |

`BOMB` is unmasked, and the reward gains `+0.3` per crate destroyed and `+0.1`
per coin revealed, counted **per occurrence** so that a bomb clearing four
crates is worth more than one clearing a single crate.

### 5.1 Result

`p3_danger`: `loot-crate`, 4000 episodes, 5 seeds, evaluated on 30 held-out
seeds x 20 rounds.

| metric | mean | per seed |
|---|---|---|
| coins | 20.21 | 22.0, 24.8, 10.2, 14.3, 29.7 |
| crates destroyed | 62.55 | 68.3, 73.4, 38.7, 50.5, 81.8 |
| bombs dropped | 25.13 | 26.4, 23.3, 14.2, 31.3, 30.4 |
| **suicide rate** | **0.030** | 0.040, 0.007, 0.010, 0.025, 0.068 |
| steps survived | 389.4 / 400 | |

The agent bombs, clears crates, collects the coins underneath, and **almost
never kills itself**. The danger model does its job. But the spread across seeds
is enormous: 10.2 to 29.7 coins, a factor of three.

### 5.2 Diagnosis: it is bomb *quality*, not bomb *quantity*

The obvious hypothesis is that the weak seeds are too timid. The correlations
say otherwise:

| | correlation with coins |
|---|---|
| crates destroyed | **+0.996** |
| bombs dropped | +0.540 |

Coins are almost a deterministic function of crates destroyed, but only weakly
related to how often the agent bombs. Seed 4 drops the **most** bombs of any
seed (31.3 per round) and finishes second worst (14.3 coins), because its yield
is 1.62 crates per bomb against seed 2's 3.15.

To see where that yield goes, every bomb the trained policies chose to drop was
audited: how many crates its blast would have covered, and whether an escape
route existed at that moment (`tools/analyse_bombs.py`, 1405 bombs).

| seed | bombs | hit 0 crates | hit 1 | hit 2+ | mean payload | no escape |
|---|---|---|---|---|---|---|
| 1 | 316 | 12% | 11% | 77% | 2.62 | **0** |
| 2 | 252 | **0%** | 8% | 92% | 3.16 | **0** |
| 3 | 104 | 3% | 6% | 91% | 3.40 | **0** |
| 4 | 351 | **28%** | 10% | 61% | 2.19 | **0** |
| 5 | 382 | 5% | 12% | 83% | 2.83 | **0** |
| **all** | 1405 | 11% | 10% | 78% | 2.73 | **0 (0.0%)** |

Two things stand out.

**The danger model is exactly right.** Not one bomb out of 1405 was dropped from
a position with no escape route. That is the direct measurement behind the 0.030
suicide rate, and it says the remaining suicides come from walking into danger,
not from self-trapping.

**The waste is real and seed-dependent.** Overall 11% of bombs hit nothing at
all, but the range runs from 0% (seed 2) to 28% (seed 4). The state cannot tell
these apart: `target_kind = crate` and `target_dir = here` are true both when
the blast would clear four crates and when it would clear none, because the
crate that made the agent "arrive" may sit diagonally, where no blast reaches.

The two weak seeds fail in **opposite** ways, which is why an aggregate like
"bombs per round" hides the problem:

* seed 4 bombs constantly and wastes 28% of it;
* seed 3 bombs well (3.40 crates per bomb, the best of any seed) but only 104
  times, a third of seed 5.

**Comment.** This phase moved the agent from "cannot play Task 2 at all" to
"plays it safely", and the safety half is solid. What is missing is not more
learning but a distinction the state cannot currently draw: whether dropping a
bomb *here* accomplishes anything. That is Phase 4.

---

## 6. Phase 4 - telling good bombs from useless ones (a failure)

**What and why.** Phase 3 measured that 11% of bombs destroyed nothing, and that
the state could not distinguish a blast covering four crates from one covering
none. The obvious fix is to put that distinction into the state. A new component
`bomb_opt` answers "what would dropping a bomb here achieve":

| value | meaning |
|---|---|
| `BOMB_NONE` | no bomb available |
| `BOMB_POINTLESS` | a bomb is available but its blast covers no crate |
| `BOMB_USEFUL` | the blast covers at least one crate and an escape route exists |
| `BOMB_TRAPPED` | the blast covers something, but nothing survives afterwards |

The escape question is answered against the **hypothetical** danger schedule
that already includes the bomb under consideration, which is the only way to
know whether the agent would still have a way out after dropping it.

Only the feature changed. Rewards, episodes, seeds and scenario are identical to
Phase 3, so the comparison is clean.

Adding the component also forced an engineering change. The table grew from
43740 to 174960 addressable rows, but measurement showed only ~630 of the 43740
were ever visited (1.4%) - most feature combinations are geometrically
impossible. The dense array was spending 98.6% of a 4.2 MB pickle on zeros and
would have spent 17 MB, so the table became a **sparse dictionary** keyed by row
index.

### 6.1 Result: substantially worse

| metric | Phase 3 | Phase 4 | change |
|---|---|---|---|
| coins | 20.21 | **6.29** | -69% |
| crates destroyed | 62.55 | 24.30 | -61% |
| bombs dropped | 25.13 | 33.46 | +33% |
| **suicide rate** | 0.030 | **0.263** | **x8.8** |
| steps survived | 389.4 | 304.1 | -22% |

Giving the agent exactly the information it was missing made it worse on every
metric that matters. Repeating the bomb audit shows how completely the intent
was inverted:

| | bombs | hit 0 crates | mean payload | no escape |
|---|---|---|---|---|
| Phase 3 | 1405 | **11%** | 2.73 | 0 |
| Phase 4 | 2134 | **77%** | 0.64 | 0 |

The feature introduced to *stop* pointless bombs made them seven times more
common.

### 6.2 Diagnosis

Two hypotheses were checked and rejected before the real one was found.

*Data dilution* - a finer state partition spreads the same experience over more
rows. Rejected by measurement: rows went from 628 to 750 (+19%, not the 4x a
naive count suggests, because most combinations are unreachable) and the
**median visits per row went up**, from 37 to 43.

*Latency* - the feature runs a second breadth-first search per step, and
evaluation enforces a 0.5 s timeout that training does not. Rejected: mean think
time is 0.13-0.22 ms, three orders of magnitude below the limit.

The answer came from grouping the learned table by `bomb_opt` and weighting each
row by how often it was actually visited, rather than counting rows - a
distinction that matters, because an unweighted count over rows gave a
completely misleading picture first time round:

| `bomb_opt` | share of visited time | share of that time where BOMB is greedy |
|---|---|---|
| `NONE` | 64.8% | **0.3%** |
| `POINTLESS` | 12.4% | **55.7%** |
| `USEFUL` | 20.2% | **51.0%** |
| `TRAPPED` | 2.5% | **0.0%** |

The agent bombs at essentially the **same rate whether the bomb is useful
(51.0%) or pointless (55.7%)**. The information is in the state and it is being
ignored. Meanwhile the other two categories are learned perfectly: it stops
trying to bomb without a bomb (0.3%, and the invalid-action rate collapses from
0.036 to 0.001), and it never bombs itself into a dead end (0.0%).

That pattern is the whole answer. The agent learned exactly those distinctions
the reward function **pays for**:

* `BOMB_NONE` is enforced by `INVALID_ACTION`, worth -0.5 **immediately**;
* `BOMB_TRAPPED` is enforced by death, worth -5;
* `BOMB_POINTLESS` has **no corresponding penalty at all** - a bomb that
  destroys nothing costs exactly the -0.01 of any other action.

The crate reward does exist, but it arrives **four steps after the decision**,
and in between the trajectories merge: once the bomb is on the ground,
`bomb_opt` reads `NONE` whether the bomb was well placed or not, and the escape
features describe the danger, not the payload. So the states following a good
bomb and a useless bomb are **indistinguishable**, and the delayed reward lands
where it can no longer be attributed to the choice that earned it. This is a
credit-assignment failure caused by aliasing *after* the decision, not before
it.

**Comment.** The instructive part is that the feature was not wrong - it is used
correctly in the two categories that carry an immediate consequence. What was
wrong was the assumption that giving the agent information is enough. A
distinction the agent can see but is never paid to act on is not a distinction
it will learn. Phase 5 attaches the missing consequence.

---

## 7. Phase 5 - paying for the distinction, and finding the real cause

**What and why.** Phase 4 diagnosed a missing immediate consequence for a
useless bomb, so one was added: `reward_bomb_wasted`, charged the moment a bomb
is dropped whose blast covers no crate. It is evaluated on the state the bomb
was dropped *from*, which puts it on the transition that made the decision.

The size of that penalty is itself a hypothesis, so instead of picking one value
it was run as a **dose-response**: three magnitudes, 5 seeds each, everything
else identical to Phase 4.

### 7.1 Result: the direction is right, the level is not

| penalty | coins | crates | bombs | suicide | wasted bombs |
|---|---|---|---|---|---|
| 0.0 (Phase 4) | 6.29 | 24.30 | 33.45 | 0.263 | 77% |
| -0.1 | 9.04 | 32.07 | 35.77 | 0.227 | |
| **-0.3** | **15.15** | **49.26** | 37.34 | **0.178** | **48%** |
| -0.6 | 14.24 | 48.07 | 21.55 | 0.264 | |

The response is monotone up to -0.3 and then turns over: at -0.6 the agent stops
bombing (37.3 to 21.6 per round) and the suicide rate climbs back. That is the
familiar shape of a penalty that cures the symptom by suppressing the behaviour.

But the honest comparison is not against Phase 4. It is against **Phase 3**,
which scored **20.21 coins at a 0.030 suicide rate with 11% wasted bombs** and
had neither the feature nor the penalty. Reporting "6.29 to 15.15, a 2.4x
improvement" would be arithmetically true and scientifically wrong.

### 7.2 The controlled experiment that should have come first

Between Phase 3 and Phase 4 **two** things changed: `bomb_opt` was added, *and*
the dense table was replaced by a sparse dictionary. Attributing the regression
to either one was not possible. That was an experimental design error, and it
was fixed by adding an ablation switch that holds `bomb_opt` at a constant, so
the encoding collapses to exactly the Phase 3 partition while every other line of
code, including the sparse table, stays in place.

| run | coins | suicide | crates |
|---|---|---|---|
| Phase 3 (dense table, no `bomb_opt`) | 20.213 | 0.030 | 62.549 |
| `p5_ablate_nobombopt` (**sparse** table, no `bomb_opt`) | **20.213** | **0.030** | **62.549** |

Identical to three decimals, and identical seed by seed (22.010, 24.750, 10.245,
14.332, 29.728). So the sparse table is exactly equivalent, and **`bomb_opt`
alone accounts for the entire regression**.

### 7.3 Why the feature hurts

With the ablation switch it is possible to log what `bomb_opt` *would* have been
at every step the working Phase 3 policy visits, without that value influencing
anything (11990 steps):

| situation | share of steps | `NONE` | `POINTLESS` | `USEFUL` | `TRAPPED` |
|---|---|---|---|---|---|
| **standing next to a crate** | 22.9% | 48.2% | **0.0%** | 51.8% | **0.0%** |
| travelling | 77.1% | 34.6% | 40.9% | 24.2% | 0.3% |

In the states where the bombing decision is actually made - standing next to a
crate - `bomb_opt` takes only two values, and both are already implied by the
rest of the state. An orthogonally adjacent crate is *always* inside the blast,
so "next to a crate" already means the bomb is useful; the only other case is
that no bomb is available, which follows from having bombed in the last seven
steps. **The feature carries no information where the decision is taken.**

Its variation lives entirely in the 77% of steps where the agent is travelling -
and there, bombing is not what the agent should be doing. So the component
splits the largest part of the state space along a dimension irrelevant to the
choice being made there:

| | rows for "next to a crate" | rows for "travelling" |
|---|---|---|
| without `bomb_opt` | 161 | 467 |
| with `bomb_opt` | 168 (**+4%**) | 582 (**+25%**) |

The fragmentation lands almost entirely on the **movement** policy - the part
that finds targets and dodges blasts - while the bombing policy it was meant to
improve gains almost nothing. That is the complete explanation for the pattern
seen in Phase 4: coins fall because pathing degrades, and the suicide rate
multiplies because blast avoidance now has to be relearned separately in each
fragment of an otherwise identical geometric situation.

**Decision.** `bomb_opt` is switched off (`use_bomb_opt = False`). The slot is
kept in `LAYOUT` at a constant so that earlier models stay loadable and the
component can be re-enabled if opponents later make self-trapping matter.

**Comment.** The lesson generalises past this one feature. A tabular agent pays
for every state component in *statistical resolution*, and that price is charged
across the whole state space while the benefit is collected only in the states
where the distinction changes the decision. A component is worth adding only
when those two sets overlap. Checking that overlap costs one logging run and
would have saved two full training rounds here.

---
