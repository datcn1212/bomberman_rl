# Tabular Q-learning - experiment log

Working log for the tabular agent in `agent_code/tabular_q/`. One entry per
phase: what changed, what it measured, what we decided. Negative results are
kept - a change that did nothing is still evidence.

Every run is reproducible from the JSON config stored next to its model in
`experiments/<exp_id>/seed<k>/`.

---

## Setup

**Actions** - the framework's six, in its order: `UP RIGHT DOWN LEFT WAIT BOMB`.
Direction index `i` equals move-action index `i` everywhere in the code.

**State** - a tuple of small integers, packed into one table row by mixed-radix
encoding. This is the part that evolves; it starts minimal and only grows when a
measurement shows something is missing. `model.LAYOUT` records the radix of each
component and is checked on load, because changing the encoding while old models
exist fails *silently*: every lookup misses, the agent still runs, scores just
get worse.

Current encoding (`bomb_opt` held constant since Phase 5):

| component | radix | meaning |
|---|---|---|
| `move_status` | 81 | each neighbour: blocked / free-safe / free-lethal |
| `t_here` | 5 | steps until this tile burns; 0 = it doesn't |
| `target_dir` | 6 | direction to nearest coin or crate, or "already there" |
| `target_kind` | 3 | is that target a crate, a coin, or an opponent |
| `escape_dir` | 6 | first move of a surviving path / none / stay put |

About 630-750 rows are ever visited, so the table is a **sparse dict**, not an
array (a dense array was 98.6% zeros).

**Rewards** - the game score (1/coin, 5/kill) is far too sparse to learn from, so
reward is built from the per-step event list. All weights live in `config.py`, so
a reward change is a config diff, not a code edit.

**Update** - one-step TD control, off-policy:

```
Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
```

Terminal transitions use target `r` alone. `max` runs over *legal* actions only.
Step size `alpha / (1 + n(s,a)/half_life)`, counted per (state, action) pair.

**Protocol** - training seeds `1000+k`; evaluation seeds `9001..9030`, disjoint.
FAST mode = 30 seeds x 20 rounds. Configs are compared **paired on the same
seeds**; a difference under ~2 standard errors is not called a difference. Most
comparisons use 10 seeds, because between-seed variance is as large as the
effects being measured.

---

## Phase summary

| # | change | result | verdict |
|---|---|---|---|
| 0 | measure the environment before building | found 2 silent framework bugs | **kept** |
| 1 | minimal state, Task 1 | 4/5 seeds perfect, 1 broken | diagnosed |
| 2 | visit-count step-size decay | 0/10 seeds broken, spread 0.000 | **kept** |
| 3 | danger as a schedule, bombs on | plays Task 2, suicide 0.030 | **kept** |
| 4 | `bomb_opt` feature | coins -69%, suicide x8.8 | **reverted** |
| 5 | wasted-bomb penalty + ablation | ablation proved `bomb_opt` was the cause | **feature off** |
| 6 | potential-based shaping | no effect, suicide x4 | **rejected** |
| 7 | 3x training budget | +7.9 coins (t=2.27); penalty falls to t=0.19 | **kept** |
| 8 | Max-Boltzmann exploration | nothing (t=-0.26 / +0.40) | **rejected** |
| 9 | curriculum loot-crate -> classic | nothing (t=+0.46) | **rejected** |
| 10 | gamma 0.9 -> 0.99 | classic score **x3.2** (t=5.24) | **kept** |
| 11 | gamma 0.995 / 0.999, Max-Boltzmann retest | 0.999 best; Boltzmann now helps | see below |
| 12 | D4 canonicalisation | ~3.9 rows per orbit, ~5x data per row | measuring |
| 13 | combine the marginal effects | leave-one-out | running |

---

## Phase 0 - measure the environment first

Drove a scripted, non-learning `probe_agent` and read the answers back
(`tools/verify_mechanics.py`, `tools/verify_events.py`).

Mechanics: bomb detonates 4 steps after `BOMB`; lethal on the detonation step
**and the next one** (verified by re-entering the blast at t+0/t+1/t+2 - died,
died, survived); blast is straight rays of 3, stopped at the first wall, **never
turns corners**; `bombs_left` returns after 7 steps.

Two framework behaviours that dictate the learner's structure:

1. **The last step of a surviving round is delivered twice** - once via
   `game_events_occurred`, once via `end_of_round` with `SURVIVED_ROUND` added.
   Updating in both double-counts it.
2. **A fatal step arrives only via `end_of_round`.** Handling only
   `game_events_occurred` means never observing death.

So transitions are **staged, not learned on arrival**: a staged transition is
flushed when the next arrives, and `end_of_round` either replaces it (same step =
re-delivery) or flushes it first (different step = the agent died). Exactly one
update per step either way.

> Both are silent bugs - no crash, nothing visible in a training curve. Cheapest
> phase in the log by value.

---

## Phase 1 - minimal state on Task 1

State: 4 neighbours free/blocked (16) x direction to nearest coin (5) = 80 rows.
Constant `alpha=0.1`. `coin-heaven`, 2000 episodes.

**Result:** 4 of 5 seeds perfect (50/50 coins, ~124 steps). Seed 3 got **8.8
coins**. Training logs looked identical - the failure only appears under a greedy
policy, because 5% exploration hid it during training.

**Diagnosis** (`tools/inspect_policy.py`): exactly **one** bad row - a horizontal
corridor with the coin to the LEFT where greedy went RIGHT, margin `-0.0112`. In
a corridor the wrong move lands on a tile with the same encoding, so it repeats.
One bad row out of 32 cost 82% of the score.

But logging that row's Q values every episode showed the real story:

| | constant alpha |
|---|---|
| mean margin, last 500 ep | +0.32 to +0.37 **in all 5 seeds** |
| sd of the margin | 0.17 to 0.21 (**half the signal**) |
| sign changes in 500 ep | 20 to 37, **in every seed** |

No seed converged. All five learned the same correct preference; the estimate is
a random walk around it, and which policy you get depends on **which side of zero
it sits on when training stops**. Constant alpha satisfies `sum(alpha)=inf` but
not `sum(alpha^2)<inf`, so Robbins-Monro does not hold.

---

## Phase 2 - step size

Designed so the hypothesis could fail: if the problem is *too little training*,
doubling episodes helps; if it is *non-convergence*, doubling changes nothing and
a decaying step size fixes it. 10 seeds each (a failure *rate* needs 10).

| config | coins | seeds broken | which |
|---|---|---|---|
| constant, 2000 ep | 41.65 | 2/10 | 3, 9 |
| constant, 4000 ep | 48.72 | 1/10 | **4** |
| `alpha/(1+n/1000)`, 2000 ep | **50.00** | **0/10** | - |

More training did **not** help: the failing seeds are a *disjoint set* (3 and 9
recover, 4 breaks - and at a junction, not a corridor). More data reshuffled who
was unlucky. That is a random walk, not a slow-converging estimator.

Mechanism confirmed directly - mean margin unchanged, **sd drops ~6x (0.17-0.21
-> 0.020-0.041), sign changes go to 0 in all 10 seeds**.

Where failures live: grouping every visited row by local geometry,

| geometry | rows | median margin |
|---|---|---|
| **straight corridor** | 40 | **0.371** |
| corner | 80 | 0.879 |
| 3 open | 120 | 0.689 |
| 4 open | 40 | 0.570 |

Straight corridors carry half the margin of anything else, because stepping *away*
from the target lands on an identically-encoded tile - the state has no distance,
so both actions bootstrap from nearly the same value.

**Decision:** `alpha_schedule = "visit"`.

---

## Phase 3 - danger as a schedule, bombs enabled

Key idea from Phase 0: **danger is a property of a (tile, step) pair, not a
tile.** A boolean "am I in a blast" flag forces the agent to treat a tile it
could safely cross as a wall.

So features carry `lethal[k][x,y]`, and two searches run on it:
- `escape_search` - BFS over **(tile, step)** pairs, so it will route *through* a
  tile that burns later - often the only way out of a corridor.
- `target_search` - coins plus "tile adjacent to a crate" as goals; tiles burning
  now or next step are treated as walls.

**Result** (`loot-crate`, 4000 ep, 5 seeds): 20.21 coins, 62.55 crates, **suicide
0.030**, survives 389/400 steps. But seed spread 10.2 - 29.7.

**Diagnosis** - it is bomb *quality*, not quantity:
`corr(crates, coins) = +0.996` but `corr(bombs, coins) = +0.540`. Seed 4 drops the
most bombs and scores second worst.

Auditing all 1405 bombs (`tools/analyse_bombs.py`):
- **0 of 1405** dropped with no escape route -> the danger model is exactly right,
  and the remaining suicides come from walking into danger, not self-trapping.
- **11% hit nothing**, ranging 0% (seed 2) to 28% (seed 4).

---

## Phase 4 - `bomb_opt` (failed)

Added a state component saying what a bomb here would achieve: none / pointless /
useful / trapped. Only the feature changed.

| | Phase 3 | Phase 4 |
|---|---|---|
| coins | 20.21 | **6.29** |
| suicide | 0.030 | **0.263** |
| bombs hitting nothing | 11% | **77%** |

Rejected two explanations by measurement first: *data dilution* (rows only +19%,
and median visits per row went **up**, 37 -> 43) and *latency* (0.13-0.22 ms,
three orders below the 0.5 s limit).

The answer came from weighting rows by **how often they are actually visited**
(counting rows gave a completely misleading picture first time):

| `bomb_opt` | share of time | BOMB is greedy |
|---|---|---|
| `NONE` | 64.8% | **0.3%** |
| `POINTLESS` | 12.4% | **55.7%** |
| `USEFUL` | 20.2% | **51.0%** |
| `TRAPPED` | 2.5% | **0.0%** |

The agent bombs at the same rate whether it is useful or pointless. It learned
exactly the distinctions the reward **pays for**: `NONE` is enforced by
`INVALID_ACTION` (-0.5, immediate), `TRAPPED` by death (-5), and `POINTLESS` by
nothing at all. The crate reward arrives 4 steps later, by which point the
trajectories have merged - once the bomb is down, `bomb_opt` reads `NONE` either
way. Credit-assignment failure from aliasing *after* the decision.

---

## Phase 5 - the ablation that settled it

Added `reward_bomb_wasted` as a dose-response (5 seeds each): 0.0 -> 6.29,
-0.1 -> 9.04, **-0.3 -> 15.15**, -0.6 -> 14.24 (bombing collapses, suicide
returns). Right direction, but still **below the Phase 3 baseline of 20.21**.

Between Phases 3 and 4, *two* things had changed: `bomb_opt` **and** dense ->
sparse table. That was a design error. Fixed with an ablation switch holding
`bomb_opt` constant, leaving every other line in place:

| | coins | suicide | crates |
|---|---|---|---|
| Phase 3 (dense) | 20.213 | 0.030 | 62.549 |
| sparse, no `bomb_opt` | **20.213** | **0.030** | **62.549** |

Identical seed by seed. Sparse table is exactly equivalent; **`bomb_opt` alone
caused the whole regression.**

**Why it hurts** - logging what `bomb_opt` *would* have been over 11990 steps of
the working policy:

| situation | share | `NONE` | `POINTLESS` | `USEFUL` | `TRAPPED` |
|---|---|---|---|---|---|
| **next to a crate** | 22.9% | 48.2% | **0.0%** | 51.8% | **0.0%** |
| travelling | 77.1% | 34.6% | 40.9% | 24.2% | 0.3% |

Where the bombing decision is made, `bomb_opt` takes only two values and both are
already implied (an orthogonally adjacent crate is *always* in the blast). Its
variation lives entirely in the 77% of steps where the agent should be moving,
not bombing - so it fragments the movement policy:

| | rows "next to a crate" | rows "travelling" |
|---|---|---|
| without | 161 | 467 |
| with | 168 (+4%) | 582 (**+25%**) |

**Decision:** `use_bomb_opt = False`. Slot kept in LAYOUT so old models still
load and it can be re-enabled if opponents make self-trapping matter.

> **General lesson:** a tabular agent pays for every state component in
> statistical resolution across the *whole* space, but collects the benefit only
> where the distinction changes the decision. Only add a component when those two
> sets overlap. Checking costs one logging run.

---

## Phase 6 - reward design (both rejected)

On the confirmed encoding, 5 seeds each:

| config | coins | suicide |
|---|---|---|
| baseline | 20.21 | 0.030 |
| + wasted-bomb penalty -0.3 | 28.85 | 0.062 |
| + potential shaping w=0.05 | 20.59 | 0.120 |
| + potential shaping w=0.2 | 22.46 | 0.119 |

**Potential-based shaping** (`F = gamma*Phi(s') - Phi(s)`, `Phi = -w*distance`,
`Phi(terminal)=0`, per Ng et al. 1999) **did nothing** and quadrupled suicide. The
theorem guarantees the optimal policy is unchanged; it promises nothing about
learning speed.

The penalty looked like a big win, so it went to 10 seeds before being believed.

---

## Phase 7 - training budget (and my Phase 6 result dying)

At 10 seeds the penalty shrank to **+4.07 +/- 2.40 coins, t = +1.70**, only 6/10
seeds better, with suicide moving *against* it. The 5-seed result was luck.

The 10-seed data exposed the real problem: baseline spans **10.2 to 31.5 coins**
across seeds. The training curve says why -

| episode block | coins/episode |
|---|---|
| 2000-2500 | 0.7 |
| 3000-3500 | 9.8 |
| 3500-4000 | **14.4** |

still climbing steeply at the stop. **Every reward comparison so far compared
half-trained agents.**

Re-run at 12000 episodes, 10 seeds:

| comparison (paired) | effect | t |
|---|---|---|
| budget 4000 -> 12000 | **+7.86 +/- 3.46 coins** | **+2.27** |
| wasted-bomb penalty, at 12000 ep | +0.70 +/- 3.67 | **+0.19** |

The penalty is **worth nothing at convergence** (5/10 seeds - chance). Its value
at 4000 episodes was purely an undertraining artifact.

**Decisions:** `reward_bomb_wasted = 0`; budget 12000 episodes.

> **General lesson:** settle the training budget *before* comparing rewards or
> hyperparameters. Done in that order, Phases 5-6 would have been one experiment.

---

## Phase 8 - exploration (rejected)

`loot-crate`, 12000 ep, 10 seeds:

| config | coins | suicide | t vs control |
|---|---|---|---|
| uniform, decay 8000 | 30.85 | 0.078 | - |
| uniform, decay 2000 | 33.96 | 0.040 | +1.13 |
| Max-Boltzmann, decay 8000 | 30.10 | 0.049 | **-0.26** |
| Max-Boltzmann, decay 2000 | 31.71 | 0.038 | **+0.40** |

**Max-Boltzmann gives nothing** on either schedule. What limits this agent is not
*which* exploratory action it picks but how long it spends exploring. Faster decay
is directionally right on every metric but only t=+1.13 - adopted as a default,
not claimed as proven.

**Reference points** (`rule_based_agent`, solo): `loot-crate` 43.08 coins /
107.3 crates; `classic` 8.48 of 9 / 116.9 crates.

---

## Phase 9 - `classic`, curriculum rejected

`classic` is the real Task 2 board: same crate density, 9 coins instead of 50.
Budget-matched arms, 12000 episodes each.

| config | coins | crates | bombs |
|---|---|---|---|
| `classic` only | 1.54 | 30.57 | 20.37 |
| `loot-crate` -> `classic` | 1.73 | 32.58 | 19.14 |
| *rule_based* | *8.48* | *116.9* | *37.9* |

**Curriculum does nothing** (+0.199 +/- 0.430, t=+0.46, 4/10). Both at ~18% of
reference, against 79% on `loot-crate`.

**Diagnosis:** coins are almost a deterministic function of crates destroyed (23%
of ~130 crates -> ~2.1 expected, 1.54 observed). The agent collects nearly every
coin it uncovers, so the whole gap is the crate rate. And on `loot-crate` the same
agent got 3.3 crates/bomb, matching rule_based, versus 1.50 here.

Logging every position explains it:

| | distinct tiles/round | bbox span |
|---|---|---|
| `loot-crate` | **104.4** | 25.0 |
| `classic` | **18.3** | 8.4 |

The agent sits in an ~18-tile pocket of a 17x17 board, re-bombing cleared ground.
Its target is 2 steps away in both cases - it never runs out of things to aim at,
it just never travels.

---

## Phase 10 - the discount factor

That points at a parameter untouched for nine phases: `gamma = 0.9` gives an
effective horizon of ~10 steps while crossing the board takes 20-30. **Half the
board is invisible to the value function.** On `loot-crate` this never showed,
because with 50 coins there is always a reward within 10 steps.

`classic`, 12000 ep, 10 seeds:

| gamma | coins | crates | t vs 0.9 |
|---|---|---|---|
| 0.90 | 1.54 | 30.57 | - |
| 0.95 | 2.70 | 45.65 | +1.81 (7/10) |
| **0.99** | **4.86** | **76.86** | **+5.24 (9/10)** |

**Largest single improvement in the log** - score x3.2, crates x2.5. Mechanism
confirmed by repeating the coverage measurement: tiles per round **18.3 -> 60.0**,
span 8.4 -> 18.5.

This also explains Phase 9: a curriculum transfers *skill*, but the deficiency was
the **horizon of the value function**, which is a property of gamma, not of where
the agent trained.

> Nine phases of feature and reward work moved the `classic` score less than one
> hyperparameter left at its initial guess the whole time.

---

## Phase 11 - pushing gamma, and re-testing a rejected idea

Phase 7 showed conclusions flip when a deeper parameter moves, so Max-Boltzmann
was re-tested at the new gamma rather than left rejected.

| config | coins | crates | suicide | t vs gamma 0.99 |
|---|---|---|---|---|
| gamma 0.99 | 4.86 | 76.86 | 0.055 | - |
| gamma 0.995 | 5.91 | 90.14 | 0.047 | +1.18 (8/10) |
| **gamma 0.999** | **6.26** | **94.15** | 0.044 | +1.75 (8/10) |
| gamma 0.99 + Max-Boltzmann | 6.18 | 90.55 | **0.014** | +1.42 (7/10) |

**Max-Boltzmann, rejected in Phase 8, now helps.** At gamma 0.9 on `loot-crate`
it scored t = -0.26; at gamma 0.99 on `classic` it gives t = +1.42 and cuts
suicide roughly 4x. Plausible reason: with a high discount, values propagate far
enough that `softmax(Q/tau)` has real signal to sample from, whereas at gamma 0.9
most Q values sit close together and the softmax degenerates towards uniform.

> **General lesson (second time):** a rejected idea stays rejected only for the
> setting it was tested in. Re-test the cheap ones after any change to a deeper
> parameter.

---

## Phase 12 - dihedral symmetry

The arena and the rules are symmetric under D4 (4 rotations x 2 reflections),
and every feature is relative to the agent - but the encoding uses **absolute**
directions, so the same situation rotated by 90 degrees lands on a different row.
Measured on a trained table (`tools/analyse_symmetry.py`):

| | value |
|---|---|
| visited rows | ~660 |
| D4 orbits | ~171 |
| **rows per orbit** | **3.87** |
| median visits per row | 98 |
| **median visits per orbit** | **506** |

So roughly 5x the experience per row is available for free. The group action was
verified before trusting the count: true orbit sizes come out as 1, 2, 4, 8 only,
as orbit-stabiliser requires.

Implementation: fold each state onto the **smallest index in its orbit**. The
subtlety is that this relabels directions, so the agent looks up a row written in
a canonical frame and the chosen action must be translated back - "UP" in the
canonical frame may be LEFT on the board. Getting that wrong does not crash and
does not look like a bug; the agent just runs the right policy in the wrong
frame. Tests cover the round trip and check that rotating the board rotates the
recovered action (`RIGHT` becomes `DOWN` after one clockwise turn).

With `use_symmetry = False` the frame is the identity and the index is the plain
encoding, so the two modes differ by exactly one lookup and the ablation is clean.

---

## Phase 13 - combining the marginal effects

gamma 0.999, Max-Boltzmann and symmetry each land at t = 1.2-1.8 alone:
individually inconclusive, all pointing the same way. Rather than tuning each
until it crosses a threshold, they are tested **together** against the gamma-0.99
baseline, with leave-one-out arms so the result is not credited to the wrong part.

*(running)*

---

## Settings currently in force

```
alpha 0.1, alpha_schedule "visit", alpha_half_life 1000
gamma 0.999            (config default still says 0.9 - passed via overrides)
exploration "epsilon", eps 1.0 -> 0.05 over 2000 episodes
use_bomb_opt False, use_symmetry (under test), reward_bomb_wasted 0, shaping_weight 0
rewards: coin +1, crate +0.3, coin_found +0.1, invalid -0.5,
         wait -0.05, step -0.01, killed_self -5, got_killed -5
budget: 12000 episodes on `classic`
```

## Rejected, with evidence

`bomb_opt` feature - potential-based shaping - wasted-bomb penalty -
Max-Boltzmann exploration - curriculum `loot-crate` -> `classic`.

## Open

- Phase 13 combination result, then fold the winner into the config defaults.
- Tasks 3-4: opponent features, training against `rule_based_agent`.
- Final: hyperparameter search, latency benchmark, submission checks, ship
  `model.pkl` next to the agent.
