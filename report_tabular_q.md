# Tabular Q-learning - experiment log

Working log for the agent in `agent_code/tabular_q/`. One entry per phase: what
changed, what it measured, what was decided. Negative results are kept, because
a change that did nothing is still evidence.

**Where this stands.** Twenty-five phases. The agent plays Task 2 well (8.88
coins solo) and is still **below the reference on the four-agent board**: 2.275
against `rule_based_agent` at 3.260. Every intervention that raised the
four-agent score so far did it by bombing *less*; nothing yet has made the agent
bomb *better*. That is the open problem.

**How to read it.** The phase table below is the index. Each phase states the one
thing it changed and the number that decided it. Every quoted figure is
re-derived from `experiments/registry.csv` by `tools/audit_report.py`, which
fails if a number here has gone stale or if a check cannot run.

---

## Setup

**Actions** - the framework's six, in its order: `UP RIGHT DOWN LEFT WAIT BOMB`.
Direction index `i` equals move-action index `i` throughout the agent.

**State** - a tuple of small integers packed into one table row by mixed-radix
encoding. `model.LAYOUT` records each component's radix and is checked on load,
because changing the encoding while old models exist fails *silently*: every
lookup misses, the agent still runs, the score just drops.

| component | radix | meaning |
|---|---|---|
| `move_status` | 81 | each neighbour: blocked / free-safe / free-lethal |
| `t_here` | 5 | steps until this tile burns; 0 = it does not |
| `target_dir` | 6 | direction to the nearest coin or crate, or "already there" |
| `target_kind` | 3 | is that target a crate or a coin |
| `escape_dir` | 6 | first move of a surviving path / none / stay put |

Three further slots (`bomb_opt`, `opponent`, `last_move`) are reserved by LAYOUT
and held constant - each was measured and rejected in Phases 5, 20 and 22. They
stay in the layout so that every model ever saved still loads; their producing
code was removed in Phase 25 and lives in git history.

About 630-750 rows are ever visited, so the table is a **sparse dict** rather
than an array, which would be 98.6% zeros.

**Rewards** - the game score (1/coin, 5/kill) is far too sparse to learn from, so
reward is assembled from the per-step event list. Every weight lives in
`config.py`, which makes a reward change a config diff rather than a code edit.

**Update** - one-step TD control, off-policy:

```
Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
```

Terminal transitions use target `r` alone, and `max` runs over *legal* actions
only. Step size `alpha / (1 + n(s,a)/half_life)`, counted per (state, action).

**Protocol** - training seeds `1000+k`; evaluation seeds `9001..9030`, disjoint.
FAST mode = 30 arenas x 20 rounds. Configurations are compared **paired on the
same seeds**, over 10 seeds, because between-seed variance is as large as the
effects being measured.

**Noise floor.** Phases 16-24 were measured against the stock
`rule_based_agent`, which draws its tie-breaks from an unseeded `random` module,
so two runs of an *identical* configuration do not repeat. Running one
configuration three times (`p19b`, `final_tabular_q`, `repeat_c`) gave 2.379,
2.221 and 2.224: a **run-to-run standard deviation of 0.090**.

The paired t statistic is invalid against that - the same difference of -0.16
gives t = -3.45 in one pairing and t = -0.70 in another - so four-agent effects
in those phases are judged by effect size against the spread, with a null
standard error of `0.090 * sqrt(2) = 0.127`.

| comparison | difference | z |
|---|---|---|
| Phase 16, opponents block moves | +0.424 | 3.3 |
| Phase 17a, all training with opponents | +0.435 | 3.4 |
| Phase 18, D4 symmetry | +0.456 | 3.6 |
| Phase 20, `bomb_safety` | -0.617 | -4.9 |
| Phase 21, best search candidate | +0.044 | 0.3 |

Solo comparisons are unaffected: with no opponents the runs are deterministic
and repeat exactly. **Phase 25 removes this noise at its source**, so
measurements from that phase on are exactly reproducible and the paired test is
valid again.

---

## Tooling

Nine entry points in `tools/`, mapped in `tools/README.md`. Four matter for
reading this log:

| | |
|---|---|
| `train.py` | runs a curriculum over several seeds, then evaluates |
| `evaluate.py` | the measurement harness; fixes the protocol above |
| `audit_report.py` | re-derives every figure quoted here from the registry |
| `verify_unchanged.py` | proves a change alters nothing, before it is trusted |

The last two exist because of specific failures: an observation cache believed
to be exact silently changed what the agent saw and cost every experiment after
it, and figures that were correct when written stopped being correct later.

---

## Phase summary

| # | change | result | verdict |
|---|---|---|---|
| 0 | measure the environment before building | 2 delivery rules that shape the learner | **kept** |
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
| 12 | D4 canonicalisation | **coins 5.91 -> 8.88 (t=4.96, 10/10)**, beats reference | **kept** |
| 13 | combine the marginal effects | **worse** (7.41 vs 8.88); optimum moved | **rejected** |
| 15 | gamma = 1 (undiscounted) | mean down, **variance x9** | **rejected** |
| 16 | Tasks 3-4: model opponents as obstacles | score +0.42, kills +0.04 (t=2.03) | **kept** |
| 17a | train all 12000 episodes with opponents | **coins +0.42 (t=2.96, 9/10)** | **kept** |
| 17b | death penalty -5 -> -15 | score -0.50, kills -0.03 | **rejected** |
| 18 | is D4 still needed with opponents? | score +0.46 (t=3.79), coins +0.48 (t=9.44) | **kept** |
| 19 | halve the training budget to 6000 | coins unchanged (t=0.30) | **kept** |
| 20 | `bomb_safety`: count escape routes | score -0.62 (t=-7.1), bombs +54% | **rejected** |
| 21 | random hyperparameter search, 13 candidates | nothing beats the current defaults | **kept as is** |
| 22 | four candidate extensions to state and schedule | three hurt, one neutral | **all rejected** |
| 23 | a bundled alternative configuration | 1v1 score 2.919 -> 3.727 | **see 24** |
| 24 | the same models on the four-agent board | the 1v1 gain is worth nothing there | **rejected** |
| 25 | seed the opponent; strip rejected code | measurements repeat exactly | **kept** |
| 26 | re-anchor on the current encoding | 2.210, and the seeded opponent is unbiased | **kept** |
| 27 | multi-step credit assignment (n-step, Q(lambda)) | n-step harmful, Q(lambda) neutral | **rejected** |

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

> Neither rule is visible from the outside: a learner that ignores them still
> runs and still produces a training curve, it just optimises a different problem.
> Cheapest phase in the log by value.

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

The answer comes from weighting rows by **how often they are actually visited**.
Weighting matters here: an unweighted count over rows describes the table, not
the policy, because the agent spends almost all of its time in a small minority
of rows.

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

Phase 4 changed two things at once: `bomb_opt` **and** the move from a dense
array to a sparse dict. Neither can be blamed until they are separated, so an
ablation switch holds `bomb_opt` at a constant, collapsing the encoding back to
the Phase 3 partition while every other line - including the sparse table -
stays in place:

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

## Phase 7 - training budget

At 10 seeds the penalty shrinks to **+4.07 +/- 2.40 coins, t = +1.70**, only 6/10
seeds better, with suicide moving *against* it. The effect does not survive the
larger sample, which is why the threshold in this log is 10 seeds and not 5.

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
> hyperparameters. A shaping term that measurably helps an undertrained agent can
> be worth nothing at convergence, so any reward comparison run before the budget
> is settled measures the budget, not the reward.

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

### Result - the largest and cleanest win in the log

`classic`, gamma 0.995, 12000 episodes, 10 seeds, paired:

| | coins | crates | suicide | seed range |
|---|---|---|---|---|
| symmetry off | 5.91 | 90.14 | 0.047 | 2.4 - 8.0 |
| **symmetry on** | **8.88** | **122.23** | **0.002** | **8.6 - 9.0** |
| *reference `rule_based_agent`* | *8.48* | *116.9* | *0.000* | |

| metric | paired diff | t | seeds better |
|---|---|---|---|
| coins | **+2.97 +/- 0.60** | **+4.96** | **10/10** |
| crates | +32.10 +/- 7.09 | +4.53 | 10/10 |
| suicide | -0.045 +/- 0.013 | -3.52 | 9/10 |

The board holds 9 coins and the agent collects **8.88**, with the best seeds at
8.997. It **beats `rule_based_agent` (8.48)** on `classic` solo.

The between-seed spread - the problem that had dominated every phase since
Phase 3 - collapses from 2.4-8.0 to **8.6-9.0**. That is the direct consequence
of the orbit measurement: each row now carries ~5x the experience, so rows stop
being decided by noise.

Side effects, all checked:

| | before | after |
|---|---|---|
| table rows | ~660 | **174** |
| pickle size | ~100 KB | **28 KB** |
| think time | 0.13-0.22 ms | 0.15-0.22 ms |

The extra work per step (eight relabellings and a `min`) costs nothing
measurable against the 0.5 s limit.

**Comment.** This was the one idea taken from the assignment's own list of
"successful strategies", and it is the only change in the log that improved
*every* seed. Worth noting why it works where added features failed: symmetry
does not add information, it removes a *distinction the encoding was making
without reason* - absolute direction. Phase 5's lesson was that a component
costs resolution everywhere and pays only where it changes the decision;
canonicalisation is the same trade run backwards, and it is free.

## Phase 13 - combining the marginal effects

gamma 0.999, Max-Boltzmann and symmetry each land at t = 1.2-1.8 alone:
individually inconclusive, all pointing the same way. Rather than tuning each
until it crosses a threshold, they are tested **together**, with leave-one-out
arms so the result is not credited to the wrong part.

**The combination is worse than its parts.** `classic`, 12000 ep, 10 seeds:

| config | coins | crates | suicide | seed range |
|---|---|---|---|---|
| **gamma 0.995 + symmetry** | **8.88** | 122.23 | 0.002 | **8.60 - 9.00** |
| gamma 0.999 + symmetry | 8.62 | 120.39 | 0.002 | 7.13 - 9.00 |
| gamma 1.0 + symmetry | 8.02 | 112.83 | 0.010 | 3.23 - 9.00 |
| gamma 0.999 + symmetry + Boltzmann | 7.41 | 104.91 | 0.004 | 2.00 - 9.00 |
| gamma 0.999 + Boltzmann, no symmetry | 6.39 | 96.45 | 0.014 | 3.21 - 8.77 |

Attribution, each change measured paired against the arm without it:

| change | effect | t | seeds better |
|---|---|---|---|
| add Max-Boltzmann | **-1.20 +/- 0.90** | -1.33 | 4/10 |
| add symmetry | +1.02 +/- 1.02 | +1.00 | 6/10 |
| gamma 0.995 -> 0.999 | **-0.26 +/- 0.20** | -1.34 | 2/10 |
| gamma 0.999 -> 1.0 | **-0.60 +/- 0.60** | -1.00 | 4/10 |

**The optimum of gamma moved once symmetry was switched on.** Without symmetry,
higher was strictly better (0.99 -> 0.995 -> 0.999, monotone). With symmetry,
0.995 is best and going further hurts. Max-Boltzmann, which had looked helpful
at gamma 0.99 *without* symmetry, is now the single most harmful component.

> **General lesson (third time):** an effect measured in one setting does not
> transfer to another. Every marginal gain in this log was measured before
> symmetry existed, and symmetry invalidated all of them. Combining separately
> measured improvements is not additive - it needs its own experiment.

**Decision.** Final configuration is **gamma 0.995, symmetry on, plain
epsilon-greedy** - the Phase 12 config.

---

## Phase 15 - gamma = 1

Legitimate in principle: the task is episodic and terminal transitions bootstrap
from nothing, so returns are finite. The concern was specific - **the state
carries no clock**, so the same row is visited at step 5 and at step 395, whose
true values differ by an entire episode. A discount hides that; gamma = 1 cannot.

The prediction was that this shows up as *variance*, not as a lower mean. It does:

| gamma | mean coins | between-seed sd |
|---|---|---|
| 0.995 | 8.88 | **0.19** |
| 0.999 | 8.62 | **0.56** |
| 1.0 | 8.02 | **1.76** |

The mean drops modestly; the standard deviation across seeds grows **nine-fold**,
with the worst seed at 3.23 of 9. Undiscounted returns make the agent's value
estimates depend on information it does not have.

---

## Phase 16 - Tasks 3-4, and modelling other agents

Against three `rule_based_agent`s the solo policy does not transfer: **8.88
coins solo becomes 1.27**, and the self-kill rate goes from 0.002 to 0.387.

The first modelling question is whether other agents should be treated as
obstacles at all. Two agents cannot share a tile, so a move into an occupied one
is invalid and the agent simply stays put - which matters most in the middle of
an escape. Opponents block only the **immediate** move: beyond that they have
moved too, and treating a body as a permanent wall would report traps that do
not exist. This changes no radix, so it costs no resolution.

10 seeds, paired, only `use_opponent_blocking` differs:

| | score | coins | kills | self-kill | crates |
|---|---|---|---|---|---|
| opponents invisible | 1.633 | 1.273 | 0.072 | 0.387 | 28.93 |
| **opponents block moves** | **2.056** | 1.499 | **0.112** | 0.479 | 32.02 |

| metric | paired diff | t | seeds better |
|---|---|---|---|
| score | +0.424 +/- 0.299 | +1.42 | 7/10 |
| **kills** | +0.040 +/- 0.020 | **+2.03** | 7/10 |
| self-kill | +0.091 +/- 0.053 | +1.72 | **2/10** |

The expected gain was fewer deaths - an escape route through another agent's
body is not an escape route. What the measurement shows instead is a **more
aggressive** agent: kills rise 55% and score rises, while the self-kill rate
rises too. Knowing where the opponents are makes bombing near them attractive,
and bombing near them is dangerous.

`score_margin` improves from -2.58 to -2.04, so the gap to the opponents closes
but does not disappear: they score about 4.1, we score 2.06.

### What actually limits Tasks 3-4

The dominant loss is **self-inflicted**: `suicide_rate` counts `KILLED_SELF`, and
it is 0.479 with opponents against 0.002 solo - with *fewer* bombs per round
(13.8 against 41). Each bomb is far more likely to be fatal.

The mechanism this points at: the escape route is computed at the moment the bomb
is dropped, assuming the corridors stay empty, and an opponent then steps into it.
Modelling opponents at the immediate step does not cover that.

Two independent candidates, one arm each against Phase 16.

**(a) How much of training has opponents present.** Phase 16 trains 6000
episodes solo then 6000 with three opponents; this arm uses all 12000 with
opponents.

| | score | coins | kills | self-kill |
|---|---|---|---|---|
| 6000 solo + 6000 with opponents | 2.056 | 1.499 | 0.112 | 0.479 |
| **all 12000 with opponents** | **2.492** | **1.913** | 0.116 | 0.471 |

| metric | paired diff | t | seeds better |
|---|---|---|---|
| score | +0.435 +/- 0.188 | **+2.32** | **9/10** |
| **coins** | +0.415 +/- 0.140 | **+2.96** | **9/10** |
| kills | +0.004 +/- 0.015 | +0.28 | 5/10 |
| self-kill | -0.008 +/- 0.032 | -0.25 | 6/10 |

The gain is **entirely in coins**, with kills and self-kills unmoved. Collecting
coins while three other agents contest the board is a distinct skill from
collecting them alone, and solo episodes do not teach it. It also rules out the
self-kill rate as the binding constraint: doubling opponent experience left it
at 0.47 while the score rose.

**(b) Pricing death higher.** `reward_killed_self` from -5 to -15:

| metric | paired diff vs Phase 16 | t |
|---|---|---|
| score | **-0.497 +/- 0.295** | -1.68 |
| kills | **-0.033 +/- 0.018** | -1.88 |
| self-kill | -0.047 +/- 0.036 | -1.31 |

Slightly safer, materially worse. This is the third time in this log that an
intervention aimed directly at dying less has bought safety by suppressing the
behaviour instead of improving it, after the wasted-bomb penalty at -0.6 and the
death penalty at -15. The agent does not learn to bomb *more safely*; it learns
to bomb *less*.

Context for why that trade never pays here: `rule_based_agent`, measured in the
same four-agent setting, kills itself in **50%** of rounds and still scores 3.26.
A high self-kill rate is a property of a crowded board, not a defect to fix.

**Decision.** Train entirely with opponents; keep `reward_killed_self` at -5.

---

## Phase 18 - is symmetry still needed once opponents are present?

D4 was measured on `classic` **solo** and switched on in every four-agent run
since. Whether it still pays there is a separate question, and there is reason to
doubt it: the state does not encode opponent positions, so situations that differ
only in where the opponents stand already share a row. Folding another four to
eight situations on top could over-aggregate.

Single variable against Phase 17a, 12000 episodes all with opponents, 10 seeds:

| | score | coins | kills | self-kill |
|---|---|---|---|---|
| **symmetry on** | **2.492** | **1.913** | 0.116 | 0.471 |
| symmetry off | 2.036 | 1.438 | 0.119 | 0.517 |

| metric | paired diff | t |
|---|---|---|
| score | +0.456 +/- 0.120 | **+3.79** |
| coins | +0.476 +/- 0.050 | **+9.44** |
| self-kill | -0.046 +/- 0.037 | -1.26 |

So the doubt was unfounded: folding opponent-varying situations together does not
over-aggregate. The gain is entirely in coins, with kills and self-kills unmoved -
the same shape as the Phase 17a gain, and consistent with the mechanism measured
on the solo board, where symmetry raises the experience per row about fivefold.

It is worth noting the size. On `classic` solo symmetry was worth +2.97 coins and
took the agent past `rule_based_agent`; here it is worth +0.48. Both are real and
both are significant, but the solo figure does not transfer as a magnitude - only
as a direction.

---

## Phase 19 - how much training is actually needed

The budget of 12000 episodes was fixed in Phase 7, when the table held ~660 rows.
Symmetry has since reduced it to 174, so each row now receives roughly four times
the experience and should converge sooner. The training curve agrees - coins per
episode is flat from about episode 4000 in every configuration - but a curve
measured at `epsilon = 0.05` can hide slow refinement that a greedy evaluation
would show, so it was measured directly.

Half the episodes, everything else identical, 10 seeds:

| | score | coins | kills | self-kill |
|---|---|---|---|---|
| 12000 episodes | 2.492 | 1.913 | 0.116 | 0.471 |
| **6000 episodes** | 2.379 | **1.895** | 0.097 | 0.447 |

| metric | paired diff | t |
|---|---|---|
| coins | +0.019 +/- 0.064 | **+0.30** |
| score | +0.113 +/- 0.078 | +1.45 |

Coins are unchanged. **Half the budget buys the same agent**, which halves the
cost of every experiment from here on - the point at which that matters most,
because the hyperparameter search is next.

---

## Phase 20 - can the agent learn to bomb safely rather than bomb less?

Three interventions aimed at dying less (wasted-bomb -0.6, death -15, Double Q)
all bought safety by suppressing bombing. The diagnosis was that the reward pays
for the distinction between a safe bomb and a reckless one while the state cannot
express it: `escape_dir` is binary, so a bomb with one way out and a bomb with
three look identical.

`bomb_safety` reuses the dormant `bomb_opt` slot - no radix added - to report how
many distinct escape routes a bomb here would leave: none / trapped / exactly one
/ two or more.

| | score | coins | kills | self-kill | bombs |
|---|---|---|---|---|---|
| control | **2.492** | **1.913** | 0.116 | 0.471 | 20.69 |
| + `bomb_safety` | 1.875 | 1.361 | 0.103 | 0.564 | **31.92** |

| metric | paired diff | t | seeds better |
|---|---|---|---|
| score | -0.617 +/- 0.087 | **-7.11** | 0/10 |
| coins | -0.553 +/- 0.060 | -9.19 | 0/10 |
| self-kill | +0.094 +/- 0.021 | +4.50 | 1/10 |
| bombs | **+11.23 +/- 1.28** | +8.76 | 10/10 |

The intent was selective bombing; the result is 54% **more** bombing, more deaths
and less score. Weighting the learned rows by visits shows why:

| `bomb_safety` reports | share of time | BOMB is greedy there |
|---|---|---|
| no bomb available | 77.7% | 0.1% |
| trapped | 0.9% | 0.6% |
| **exactly one route** | **0.8%** | 6.8% |
| two or more routes | 20.5% | **69.3%** |

The distinction the component was built to draw **barely occurs**: the "one route
only" case covers 0.8% of steps. On an open board a bomb almost always leaves two
or more ways out at the moment it is dropped, so in practice the component reports
"robust" whenever a bomb is available - and the agent reads that as permission.

The flaw is in the question, not the implementation. Escape routes are counted
against the *hypothetical schedule containing only this bomb*. The danger that
actually kills comes from bombs the three opponents drop **afterwards**, which no
static count at drop time can see. A measure of present freedom is being used as a
proxy for future risk, and with three other agents acting it is a poor one.

This is the third component placed in this slot to fail, after `bomb_opt` and an
earlier "does this bomb trap an opponent" variant. All three answer "what would
happen if I bombed here" by evaluating a static board, and all three are wrong for
the same reason.

**Decision.** Rejected; the slot stays dormant.

**Comment.** The honest reading of the original question - can the agent learn to
bomb safely instead of bombing less - is that this attempt does not answer it. The
information that would separate a safe bomb from a reckless one is a prediction
about what the opponents will do, and none of the features tried so far are
predictions; they are all measurements of the current board.

---

## Phase 21 - hyperparameter search

Every parameter had been examined individually across the log, but never jointly,
and the assignment treats hyperparameter optimisation as required work. Thirteen
candidates - the current configuration plus twelve random draws over nine
dimensions (`alpha`, `alpha_half_life`, `eps_end`, `eps_decay_episodes`, `gamma`
and four reward weights).

Two stages, because running every candidate at full protocol is not affordable:
**screen** on 3 seeds and 4000 episodes to rank, then **confirm** the best three
on 10 seeds and 6000 episodes. Nothing is concluded from the screen; it only
orders candidates. Values were drawn around the current setting rather than
around values tuned for some other configuration, because Phase 13 showed the
optimum moves when the surrounding configuration changes.

**Screen** - the current configuration came out on top of all thirteen (2.393),
ahead of the best random draw (2.247).

**Confirm** - 10 seeds, paired:

| config | score | vs default | t | seeds better |
|---|---|---|---|---|
| **current defaults** | 2.376 +/- 0.141 | - | - | - |
| best random draw | 2.420 +/- 0.075 | +0.044 +/- 0.132 | **+0.33** | 6/10 |
| second | 2.109 +/- 0.070 | -0.267 +/- 0.163 | -1.64 | 1/10 |

Coins, kills and self-kill rate are all inside noise as well (|t| <= 1.29).

**The search finds nothing better.** That is a result rather than a failure: the
configuration reached by reasoning phase by phase already sits at a local optimum
of this space. A random search pays off in proportion to how untuned its starting
point is, and here every one of these parameters had already been the subject of
its own experiment.

**Decision.** Keep the configuration unchanged, and fold it into `config.py` as
the defaults, since the tournament runs the agent with no config file.

---

## Phase 22 - four candidate extensions to the state and the schedule

Four additions the encoding had never carried, each a plausible answer to a
weakness the earlier phases exposed: one step of memory, an opponent-distance
band, a five-way bomb classification separating "reaches an opponent" from
"clears crates", and a three-phase curriculum. Each was tested as a single change
against the control, which is the current configuration measured three times
(2.275).

| arm | score | coins | kills | self-kill | z against the noise floor |
|---|---|---|---|---|---|
| **control** | **2.275** | **1.803** | 0.095 | 0.489 | - |
| curriculum, opponent phase matched | 2.102 | 1.612 | 0.098 | 0.492 | -1.35 |
| opponent distance (none/far/near) | 1.993 | 1.599 | 0.079 | 0.415 | -2.21 |
| `last_move`, one step of memory | 1.878 | 1.410 | 0.094 | 0.535 | -3.10 |
| five-way `bomb_hits` | 1.606 | 1.112 | 0.099 | 0.483 | -5.23 |

**The curriculum is neutral** (z = -1.35, inside the floor). A first attempt at it
looked catastrophic (-5.48) because holding the episode total fixed and splitting
it three ways left only 2000 opponent episodes against the control's 6000, and
Phase 17a had already shown opponent exposure to be worth more than that. Matching
the opponent phase at 6000 and prepending the warm-up removes the effect.

**The three state components all hurt**, and the reason is visible in the table:

| arm | rows | score |
|---|---|---|
| control | 336 | 2.275 |
| curriculum | 348 | 2.102 |
| opponent distance | 680 | 1.993 |
| `last_move` | 854 | 1.878 |
| `bomb_hits` | 758 | 1.606 |

Correlation between table size and score across these arms: **-0.80**. Median
visits per row falls from about 40 to 10-14.

This is the Phase 5 lesson again, now with a fifth and sixth instance: a state
component costs resolution across the whole space and pays only where it changes
the decision. Symmetry folding has compressed this table to ~336 rows, and every
new component undoes a share of exactly the compression that makes the table
learnable inside the episode budget.

**Comment.** The result says these components do not pay *here*, not that they are
bad features. Their cost scales with how tightly the table is already compressed
and how little data each row gets; a design with a larger table and a larger
budget could plausibly afford them. Rejecting them is a decision about this
regime, and it should be revisited if the budget or the encoding changes
substantially.

---

## Phase 23 - a bundled alternative configuration

Every phase so far changed one thing at a time, and the Phase 21 search found
nothing better nearby. That leaves a question a one-at-a-time protocol cannot
answer: several settings this log rejected individually were each measured in a
different surrounding configuration, so none of those measurements rules out the
*combination*. Potential-based shaping was rejected in Phase 6, the wasted-bomb
penalty in Phase 7, Max-Boltzmann exploration in Phase 8.

So the whole bundle was applied at once, against the tuned defaults:

| | bundle | this branch's defaults |
|---|---|---|
| `bomb_opt` | on (4 effective values) | off |
| `alpha` / `alpha_half_life` | 0.2 / 2000 | 0.1 / 1000 |
| `gamma` | 0.95 | 0.995 |
| `reward_survived` | **0.5** | 0.0 |
| `reward_trapped` | **-1.0** | 0 |
| `reward_bomb_no_escape` | **-1.0** | 0 |
| `reward_bomb_wasted` | -0.2 | 0 |
| `shaping_weight` | **0.15** | 0.0 |
| exploration | Max-Boltzmann | epsilon-greedy |
| curriculum | `loot-crate` -> `classic` -> `classic` + 1 opponent | `classic` throughout |

Ten seeds, evaluated 1v1 against `rule_based_agent` on `classic`:

| config (1v1, FAST 30x20) | score | crates |
|---|---|---|
| this branch's own defaults | 2.919 | ~21 |
| **the bundle** | **3.727** | **62.4** |

**The bundle wins by 0.81 and triples crate clearance.** That is the largest
single improvement in this log, and it is a genuine warning about the
one-at-a-time protocol: three of these settings were rejected on their own and
are carrying weight here together. Shaping and the wasted-bomb penalty both push
towards placing bombs that clear crates; on their own each was measured against a
configuration whose learning rate and discount could not exploit that, and the
crate count is where the difference shows.

**Comment.** A bundle that wins says nothing about which of its parts did the
work, and this one changes nine things at once. It should not be adopted as the
configuration until the parts have been separated - and Phase 24 shows there is a
prior question to settle first, because the whole of this gain sits in a
measurement that does not represent the tournament.

## Phase 24 - the same models on the four-agent board

Every figure in Phase 23 is a 1v1 score, and the tournament is played with four
agents. The ten models from the bundle were therefore re-measured on the four-agent
board under the protocol used for every other four-agent number here (30 arenas
x 20 rounds, three `rule_based_agent` opponents).

| configuration | 1v1 | four agents |
| --- | --- | --- |
| this branch's own defaults | 2.919 | 2.275 |
| the bundle | 3.727 | **2.264** |
| difference | **+0.808** | **-0.011** |

The whole benefit of the bundle is 1v1. On the four-agent board it is
0.09 of the null standard error, which is nothing.

**Comment.** This is a result about the measurement, not about the bundle. A 1v1
score separates agents that the tournament condition does not separate, so tuning
against it can buy improvements that do not exist where they matter. Every
conclusion in Phases 21-23 was reached on 1v1 numbers and inherits this caveat;
none of them is contradicted, but none of them is evidence of tournament
strength either.

The practical rule this branch now follows: **1v1 is a diagnostic, not a
selection criterion.** It is sensitive enough to localise where a difference
comes from - the crate count in Phase 23 is what made the bundle's mechanism
legible - and it must not be used to decide what gets shipped. Anything proposed
on a 1v1 result is re-measured on the four-agent board before it is adopted.

---

## Phase 25 - making the measurements repeatable, and shrinking the agent

Three problems had accumulated that were about the apparatus rather than the
agent: results that would not repeat, an agent file carrying code for features
that had already been rejected, and nineteen scripts in `tools/` with no map.

### 25.1 The noise had one cause, and it was fixable

The noise floor recorded in Setup was treated for eight phases as a property of
the game. It is not. Reading `rule_based_agent`:

```
np.random.seed()        line 69   <- no other np.random call anywhere
shuffle(neighbors)      line 45   <- inside look_for_targets, the BFS
shuffle(action_ideas)   line 140  <- fallback priority order
```

**Numpy is seeded but never used for a decision.** Every tie-break comes from the
`random` module, which nothing seeds - and the one at line 45 sits inside the
breadth-first search the opponent runs on *every step*, so it decides which of
several equally short paths the opponent takes. That is the whole noise floor.

`agent_code/rule_based_seeded/` is the same agent with `setup` seeding `random`
from `OPP_SEED`. `evaluate.py` sets it to the arena seed and `train.py` to the
training seed, so thirty arenas sample thirty reproducible opponents rather than
one - reproducible without fitting a single set of tie-breaks.

Measured on eight arenas, the same model, three repeats:

| opponent | run 1 | run 2 | verdict |
|---|---|---|---|
| `rule_based_seeded`, same OPP_SEED | 32 score, 255 crates | 32, 255 | **identical** |
| `rule_based_seeded`, different OPP_SEED | 32, 255 | different | behaviour still varies |
| `rule_based_agent` (stock) | 18, 204 | 29, 217 | **does not repeat** |

The stock agent is kept for the final check, because the tournament runs it.

Repeating a whole measurement three times - one model, ten arenas, eight rounds
each - puts a number on it:

| opponent | three repeats of one measurement | sd |
|---|---|---|
| `rule_based_seeded` | 0.0875, 0.0875, 0.0875 | **0.0000** |
| `rule_based_agent` | 0.1125, 0.2250, 0.1125 | 0.0650 |

**The measurement noise is gone, not reduced.** Pairing now removes the opponent
as well as the board, so the paired t test is a valid instrument again after
being abandoned in Phase 17.

### 25.2 The agent carried the features it had rejected

`features.py` was 473 lines, of which about a quarter implemented components that
were measured and rejected: `bomb_option` (Phase 5), `bomb_safety` and
`escape_routes` (Phase 20), `bomb_hits`, `opponent_state` and `last_move`
(Phase 22), and `_escape_exact` (a slower duplicate of the live escape search).
`survival_table` supported only those, so it went too.

| | before | after |
|---|---|---|
| `features.py` | 473 | 306 |
| agent total | 1294 | 1058 |

LAYOUT is unchanged, so **every model ever saved still loads**. The three slots
those features filled are now held at a constant, which is what they already
were with the flags off.

Verified rather than assumed: the pre-strip and post-strip code were trained on
identical seeds and the Q tables compared entry by entry - **identical on both
seeds**, 133 and 127 rows. The rejected code is recoverable from git history;
re-running one of those ablations means checking out the commit before this one.

### 25.3 Tools grouped by job

Nineteen scripts became nine entry points: four daily drivers, three
subcommand groups (`diagnose`, `ship`, `reference`), the harness library, and a
`phase0/` folder for the two scripts that ran once before the agent existed.
`tools/README.md` is the map.

One trap worth recording: the diagnostics group was first called `inspect`. A
module named `inspect.py` on `sys.path` shadows the standard library's, which
`dataclasses` imports, and *every* tool in the folder then fails with an error
that names neither file.

### 25.4 Two defects this surfaced

- **`verify_unchanged.py` could not conclude anything.** Its default trained
  against three unseeded opponents, so it reported BEHAVIOUR CHANGED when
  comparing a config against itself. Caught by running that control before
  trusting its verdict. It now runs solo by default.
- **The shipped `model.pkl` no longer loads.** It was trained under the six-slot
  layout; Phase 22 moved the encoding to eight. The load-time layout check
  catches it, so the failure is loud rather than silent, but the agent as
  committed cannot play. Listed under Open.

---

## Phase 26 - a clean anchor on the current encoding

Phase 25 left two things unsettled. The shipped model no longer loaded, because
the encoding had moved from six slots to eight in Phase 22 and it was never
retrained. And every number from Phase 16 on had been measured against the
unseeded opponent, so there was no anchor on the new protocol.

One run settles both: the configuration in `config.py` - no overrides, which is
what the tournament runs - trained for 6000 episodes on `classic` against three
seeded opponents, ten seeds. The same ten models were then measured twice.

### 26.1 The seeded opponent does not move the answer

| | score | coins | crates | self-kill |
|---|---|---|---|---|
| vs `rule_based_seeded` | 2.210 | 1.679 | 35.4 | 0.527 |
| vs `rule_based_agent` | 2.216 | 1.650 | 35.5 | 0.537 |

Paired over the ten models: mean difference **-0.006**, sd 0.141,
**t = -0.13**. No detectable bias on any measure.

This is the result that matters for the rest of the log: it means the seeded
opponent is a lower-noise substitute rather than a different opponent, so the
numbers from Phases 16-24 stay on the same scale as anything measured from here
on. Had it come out otherwise, twenty-four phases would have needed re-measuring.

### 26.2 The cleanup did not cost anything

| | score |
|---|---|
| old anchor, three pooled runs, six-slot layout | 2.275 |
| new anchor, eight-slot layout, code stripped | **2.210** |

A difference of -0.065, which is 0.72 of the old noise floor. Consistent with the
entry-by-entry table comparison in Phase 25.2: removing 236 lines and moving the
encoding changed nothing measurable.

### 26.3 The shipped model

`ship choose` ranked the ten seeds on arenas 9101-9130 and picked seed 6, which
scores **2.768** on 9001-9030 - the block that played no part in the choice, and
therefore the number to report. 358 rows, 59 KB. Submission checks **10/10**,
worst-case decision 0.16 ms against the 0.5 s budget.

### 26.4 Two more shadowed module names

Grouping the tools introduced a second collision of the kind Phase 25.3 records,
and it was worse than the first because it lied rather than crashed:
`tools/ship/select.py` shadowed the built-in `select`, which `concurrent.futures`
imports, so the latency tool failed and the submission check reported the agent's
worst-case think time as **infinite**. It read as a failed agent rather than a
broken tool. A third name, `tools/phase0/events.py`, shadowed the framework's own
`events` module.

`tools/check_shadowing.py` now fails if any module under `tools/` shadows
something importable. It asks `importlib.util.find_spec` from outside the
repository, because the obvious test - globbing the standard library for `*.py` -
misses exactly the case that caused the damage: `select` is a C extension.

---

## Phase 27 - multi-step credit assignment

Every intervention that has raised the four-agent score did it by bombing *less*;
nothing has made the agent bomb *better*. One explanation does not involve the
state at all. A bomb detonates four decisions after it is placed and its blast is
lethal on that decision and the next, so the reward it earns arrives four to five
steps later. One-step TD has to push that signal back through five separate table
updates, each damped by the step size. If that is the bottleneck, a longer backup
should fix it - and it would cost nothing in table size, which is the constraint
that Phase 22 measured at a correlation of -0.80 against score.

Two backups, five arms, all against the Phase 26 anchor on the same ten seeds.

### 27.1 n-step returns: harmful, and monotonically so

| n | score | vs anchor | paired t | bombs | crates | self-kill |
|---|---|---|---|---|---|---|
| 1 (anchor) | **2.210** | - | - | 17.9 | 35.4 | 0.527 |
| 3 | 1.677 | -0.533 | -1.83 | 11.5 | 27.5 | 0.536 |
| 5 | 0.985 | -1.226 | **-4.57** | 7.4 | 18.7 | 0.481 |
| 8 | 0.053 | -2.157 | **-12.87** | 1.5 | 4.2 | **0.940** |

At n = 8 the agent has almost stopped bombing - 1.5 bombs a round against 17.9 -
and still kills itself in 94% of rounds, which is to say that nearly every bomb it
does place is fatal.

The monotonicity is the diagnosis. Epsilon starts at 1.0 and decays over 2000 of
the 6000 episodes, so for the first third of training the actions *after* a bomb
is placed are close to random, and walking at random beside a live bomb kills.
With n = 1 that death reaches the BOMB action only through five layers of
bootstrapping, heavily damped. With n = 8 it lands directly in the return for
BOMB itself.

So the uncorrected n-step return teaches, accurately, that *bombing followed by
random behaviour is fatal*. That is true of the exploring policy and false of the
greedy one, and it does maximum damage to precisely the action the change was
meant to help.

### 27.2 Watkins's Q(lambda): neutral, and it explains 27.1

| lambda | score | vs anchor | paired t | bombs | coins | kills |
|---|---|---|---|---|---|---|
| - (anchor) | **2.210** | - | - | 17.9 | 1.679 | 0.106 |
| 0.8 | 2.159 | -0.052 | **-0.30** | 14.7 | 1.755 | 0.081 |
| 0.9 | 1.857 | -0.353 | -1.48 | 12.5 | 1.530 | 0.065 |

Q(lambda) carries the same idea as an n-step return and differs in one respect:
Watkins cuts the trace at any action that is not greedy. It does not collapse.
That isolates the off-policy bias, and not the length of the backup, as what
destroyed 27.1.

The cut turns out to adapt on its own. Logging the trace during training:

| episode | mean trace length | fraction cut |
|---|---|---|
| 500 | 1.38 | 0.71 |
| 1500 | 2.63 | 0.33 |
| 2500 | 7.98 | 0.04 |
| 3000 | 8.59 | 0.05 |

While the policy is still random the trace is cut almost every step and Q(lambda)
is one-step learning in disguise - which is exactly the protection n-step lacked.
Once epsilon settles, traces run about 8.6 steps, comfortably past the four to
five the bomb needs.

That table also settles how far the result reaches. The mechanism was genuinely
engaged over the last two thirds of training, at a length that covers the delay
in question, and it bought nothing (t = -0.30). This is a tested hypothesis, not
an untested one.

**Verdict: rejected. Credit assignment is not the bottleneck.** A correct fix and
an incorrect one were both applied to the delay between placing a bomb and being
paid for it, and neither moved the score. Whatever stops this agent from bombing
well, it is not that the reward arrives late.

### 27.3 A step size shared across a trace diverges

Both lambda arms first died with Q at NaN, in `greedy`, where an all-NaN row
leaves the argmax empty. The cause was in the new code rather than in the
algorithm: one step size, read from the pair being visited, was applied to every
pair in the trace. A pair visited ten thousand times - whose own step size has
long since decayed toward zero - kept taking full-sized steps borrowed from
whatever state had just been discovered, which is a direct breach of the
Robbins-Monro condition the visit schedule exists to satisfy. Peak |Q| at 1000
episodes was 47.8 against 7.45 for the control.

Giving each traced pair its own step size fixes it: 4.33 at the same point, and
flat at 4.44 by episode 2000. A regression test now asserts that a pair visited a
hundred times moves less than a tenth as far as a fresh one under the same TD
error.

Worth recording that the wrong choice shipped with a comment justifying it. The
comment was plausible; the reasoning was wrong; and the failure only surfaced
several thousand episodes into a run that had already cost an hour and a half.

---

## Final result

The shipped model is `experiments/final_tabular_q/seed10/model.pkl`, chosen on
arenas 9101-9130 and reported below on arenas 9001-9600, which took no part in
choosing it. **600 rounds per setting, EXACT mode**, so survival and win rate are
counted rather than inferred. `rule_based_agent` was measured in the identical
settings as the reference.

| setting | metric | tabular Q | `rule_based_agent` |
|---|---|---|---|
| **Task 2, solo** | coins (of 9) | 7.767 +/- 2.284 | **8.532 +/- 0.693** |
| | self-kill | **0.000** | **0.000** |
| | survival | **1.000** | **1.000** |
| **Task 4, 1v1** | score | 3.842 | **4.775** |
| | win rate | 0.333 | **0.508** |
| | survival | 0.487 | **0.565** |
| **Task 4, four agents** | score | 2.488 | **3.012** |
| | win rate | 0.153 | **0.195** |
| | self-kill | **0.447** | 0.555 |
| | survival | **0.437** | 0.373 |
| | crates | **39.76** | 30.24 |
| | coins | 1.955 | **2.203** |

Win-rate differences at n = 600:

| setting | difference | z |
|---|---|---|
| 1v1 | -0.175 +/- 0.028 | **-6.24** |
| four agents | -0.042 +/- 0.022 | -1.92 |

**The agent is behind the reference, and closest where it matters.** In the
four-agent setting - the tournament shape - the win-rate gap is 0.042 at
z = -1.92, which is not a clear separation. In 1v1 the gap is decisive.

Two things in the four-agent column deserve attention, because they point the
opposite way to the score:

* the agent **survives better than the reference** (0.437 against 0.373) and
  kills itself less (0.447 against 0.555), so it is not losing by recklessness;
* it destroys **31% more crates** (39.76 against 30.24) and still collects
  **fewer coins** (1.955 against 2.203).

Doing more work and converting less of it is a collection problem, not a
demolition problem: the agent opens crates and then fails to reach what it
uncovered before an opponent does. Every remaining phase attacked bombing, danger
and exploration; none attacked the route between a revealed coin and the agent
while three others contest it. That is where the next gain lives, and it is not
somewhere this log looked.

The solo column carries its own caveat: 7.767 here against 8.88 in Phase 12. That
is not a regression - Phase 12's model was trained solo, this one was trained
against three opponents, and specialising for the crowded board costs something
on the empty one.

---

## Settings currently in force

The `config.py` defaults, which is what the tournament runs, since it starts the
agent with no config file.

```
alpha 0.1, alpha_schedule "visit", alpha_half_life 1000
gamma 0.995
exploration "epsilon", eps 1.0 -> 0.05 over 2000 episodes
use_symmetry True, use_opponent_blocking True
reward_bomb_wasted 0, shaping_weight 0
rewards: coin +1, kill +5, crate +0.3, coin_found +0.1, invalid -0.5,
         wait -0.05, step -0.01, killed_self -5, got_killed -5
budget: 12000 episodes on `classic` with opponents
```

## Rejected, with evidence

Each was measured, not argued about. Phase in brackets.

| | |
|---|---|
| `bomb_opt` feature [4, 5] | coins -69%, suicide x8.8; ablation identified it as the cause |
| potential-based shaping [6] | no effect, suicide x4 |
| wasted-bomb penalty [6, 7] | no effect once the budget was raised (t=0.19) |
| Max-Boltzmann exploration [8, 11] | helps at gamma 0.99 without symmetry, harmful with it |
| curriculum `loot-crate` -> `classic` [9, 22] | t=+0.46; neutral again when re-tested properly |
| gamma above 0.995, and gamma = 1 [11, 15] | mean down, variance x9 |
| combining the marginal effects [13] | 7.41 against 8.88; the optimum had moved |
| death penalty -15 [17b] | score -0.50 |
| `bomb_safety` [20] | score -0.62, bombs +54% |
| one step of memory, opponent distance, five-way `bomb_hits` [22] | all hurt; table size correlates -0.80 with score |
| the Phase 23 bundle [23, 24] | wins 1v1 by 0.81, worth nothing at four agents |

## Open

1. **The agent is below the reference where it counts.** 2.275 on the four-agent
   board against `rule_based_agent` at 3.260; win rate 0.153 against 0.195.
2. **Nothing has made the agent bomb better.** Both interventions that raised the
   four-agent score did it by bombing less. That is the substantive problem, and
   it is where the remaining 1.0 point sits.
3. **A second model.** The submission needs at least two and this branch has one
   function approximator. Q against SARSA shares the table, the encoding and the
   features, so it may not count as two.
4. **Re-check against `settings.py`.** The course may change the mechanics up to
   seven days before the agent deadline, and every timing constant in the danger
   schedule depends on them.
