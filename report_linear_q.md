# Linear Q-learning - experiment log

Working log for the agent in `agent_code/linear_q/`, following the same
discipline as `report_tabular_q.md`: one entry per phase, what changed, what it
measured, what was decided. Negative results are kept.

**Where this stands.** Task 1 (coin-heaven, no bombs) is solved: masking plus
a per-(feature, action) decaying step size converge every seed to score 50.0.
Task 2 (loot-crate, bombs on) scores but does not survive. A baseline collapsed
systematically - 60% of seeds learned to bomb once and then do almost nothing,
immune to re-sweeping the step-size schedule, which only moved which seed
failed. Potential-based shaping toward escaping danger removed the collapse
entirely (0/10 seeds at `coins=0.000`), and its weight was then swept: too much
of it re-creates the collapse from the opposite direction, by suppressing
bombing itself. Doubling the budget to 12000 episodes raised score 83% and
that gain transfers to `classic`, but it bought score with lives - the agent
now dies in 94% of rounds, at step 56 of 400, against tabular_q's 3.0% suicide
on the same scenario. The open problem is no longer collapse or convergence; it
is that nothing so far has made this agent bomb *and* live. On tabular_q's own
tournament protocol it scores 0.130 against 2.210 (t = 12.88, 10 of 10 seeds),
so it is not the submitted agent.

---

## Setup

**Q(s, a) = phi(s) . w[:, a]** - linear function approximation, replacing the
tabular agent's dict-of-rows with a dense weight matrix shared across every
state. `agent_code/linear_q/features.py` is a byte-identical copy of
`tabular_q`'s (Board, danger schedule, target/escape search): that machinery
computes an observation, not a representation, so nothing about switching to a
linear model changes it. `agent_code/linear_q/` has no import from
`agent_code/tabular_q/` at runtime - only two tests cross-reference it, to
prove the reused D4 logic still agrees - so the directory stays submittable on
its own.

**phi(s)**, 33 dimensions: four neighbours one-hot (3 each = 12), steps-until-lethal
(5), target direction (6), target kind (2), escape direction (6), one
normalised target-distance scalar, one bias term. Three slots tabular_q's
`Observation` still carries - bomb_opt, opponent, last_move - are held constant
on this branch and left out of phi() rather than one-hot encoded uselessly.

**Update** - one-step semi-gradient TD:

```
delta = target - phi(s) . w[:, a]
w[:, a] <- w[:, a] + alpha * delta * phi(s)
```

`alpha_schedule` supports only `"constant"` for now. A tabular agent's
visit-count decay has no equivalent here - a weight is touched by every state
that shares an active feature, not by one state alone - so what a correct
schedule looks like is an open question, deferred until there is a converged
baseline to compare a schedule against.

**D4 symmetry** is reused exactly as tabular_q established it: `canonical_frame`
picks the same permutation `tabular_q.model.canonical` would (same tie-break,
cross-checked on 500 random boards, two independent tests), and phi()
vectorises the *relabelled* observation. The choice of frame only asks "which
of eight equivalent views is canonical" - representation-independent - so the
tested function is reused rather than re-derived.

**Protocol** - matches tabular_q: training seeds `1000+k`, evaluation seeds
`9001..9030` (FAST mode, 20 rounds), config fully described by one JSON file
via `LQ_CONFIG`. `tools/train.py` and `tools/eval_existing.py` were extended
with a `CONFIG_ENV_VAR` registry mapping agent name to its config env var
(`TQ_CONFIG` for tabular_q, `LQ_CONFIG` for linear_q) rather than hardcoding
one name, so a tool built for one agent does not silently ignore config for
another - a mismatch here would not error, it would just train every arm on
identical defaults.

---

## Phase summary

| # | change | result | verdict |
|---|---|---|---|
| 1a | interaction feature `target_blocked` | never activates; wrong diagnosis | **reverted** |
| 1b | mask blocked directions before selection | invalid_action_rate 0.0 across 9 runs | **kept** |
| 1c | does alpha=0.001 converge? 2000 -> 6000 episodes | weight_norm never settles; eval mean fell | **constant alpha rejected** |
| 2 | `alpha_schedule="visit"`, per-(feature,action) decay, half_life sweep | 1000: 50.0/50.0/50.0, spread 0.000 | **kept, new default** |
| 3a | mask BOMB while on cooldown (`bombs_left`) | proactive, before Task 2 exercised it | **kept** |
| 3b | Task 2 baseline: `loot-crate`, `allow_bomb: true`, half_life=1000, 10 seeds | 60% of seeds collapse to ~1 bomb then idle, coins exactly 0.000 | **open problem** |
| 3c | half_life resweep (10000, 50000) on the collapsing seeds | rescues some, breaks others that were fine; no value fixes all three | **not a half_life problem** |
| 4 | potential-based shaping toward escaping danger, weight=0.2, same 10 seeds | 0/10 seeds collapse (was 6/10); mean suicide 0.864 -> 0.443 | **kept** |
| 5 | sweep `escape_shaping_weight` (0.05, 0.1, 0.5, 1.0) against 0.0 and 0.2 | interior values all work; 1.0 re-collapses 6/10 by the opposite mechanism | **0.1 chosen** |
| 6 | is the 6000-episode budget still binding at the winning weight? | 12000 scores 1.83x more, and the gain transfers to `classic` | **budget was binding** |
| 7 | tabular_q's own protocol: four agents on `classic`, trained with opponents | 0.130 against tabular_q's 2.210, t = 12.88, 10/10 seeds | **not competitive** |

---

## Phase 1 - does semi-gradient TD converge on Task 1

Task 1 scope: `coin-heaven` (no crates, no opponents), `allow_bomb: false`, so
this tests pure navigation and the update rule before adding any risk from
bomb-safety reward - the same order tabular_q's own Phase 1/2 followed.

### 1.1 First attempt: a greedy policy that never terminates

A first alpha sweep (0.001, 0.01, 0.1; 3 seeds x 2000 episodes) looked fine
during training - all three seeds reached 42-50 coins per episode by the end,
with epsilon near its floor - but two of three seeds at alpha=0.01 collapsed to
0.78 coins under fully greedy evaluation on a *different* seed set, with
`invalid_action_rate = 0.997`. alpha=0.1 diverged outright (`max|Q|` past
float range, `greedy()` crashed on an all-NaN row).

Direct instrumentation (not inference) on the evaluation arena that produced
the 0.78 score: the agent reaches a tile where `move_status` reports the
neighbour below as `BLOCKED`, and `Q(DOWN)` is nonetheless the highest of five
legal actions. The chosen move is invalid, the position does not change, the
observation is therefore identical next step, and the identical choice repeats
for the remaining ~399 steps of the episode - a permanent loop invisible during
training only because epsilon-exploration knocked the policy out of it often
enough to average out.

**First diagnosis, and where it went wrong.** The initial read of that log
mapped `target_dir=2` to the direction DOWN. `target_dir` (values 1-4) actually
indexes `DIRS[value - 1]`, so `target_dir=2` is RIGHT, which was open - the
target was not pointing at the wall the agent walked into. The fix built on
that misreading, `target_blocked` (1.0 exactly when target_dir points at a
blocked neighbour), was added to phi() and trained for a full alpha sweep. Its
weight column stayed at exactly zero through 2000 episodes on every arm, and
the retrained models reproduced the earlier run bit-for-bit - same per-episode
CSV, same eval scores, same 0.78 collapse. That degree of identity was the
signal something was structurally wrong rather than merely unhelpful: reading
`target_search` (features.py) confirms it seeds its breadth-first search only
from walkable neighbours, so `target_dir` can *never* point at a blocked tile.
The feature's precondition is unreachable by construction, which is why
nothing about training could ever touch its weight.

**What the failure actually was.** Independent of the target: a direction whose
`move_status` is `BLOCKED` can still outscore open directions, because
`move_status` and `target_dir` are separate additive one-hot blocks and nothing
forces "blocked" to dominate their sum. A dedicated one-hot slot for "this
neighbour is blocked" exists per direction (part of the 12-wide move_status
block) and could in principle carry a strongly negative weight on its own, but
that weight competes additively against every other feature active in the same
state, for the same action column, and there is no guarantee it wins.

### 1.2 The fix: mask blocked directions at decision time

Rather than ask phi() to represent "blocked overrides everything" - which
degree of interaction a linear model without cross terms cannot guarantee -
`callbacks.act` now excludes a direction from the candidate set before
greedy/exploratory selection whenever `move_status` (in the same canonical
frame the Q-values are indexed by) marks it `BLOCKED`. This is the same
mechanism already used for `BOMB` when `allow_bomb` is off, run every step
instead of once at setup. Nothing about learning changes: Q(blocked direction)
is still computed and still updated by whatever transition actually visits it
- it is only removed from the decision. `observe_and_encode` was extended to
return the canonical-frame `move_status` alongside phi, since the raw
(unrotated) one does not index the same four directions the Q-values do.

Re-running the alpha sweep on the corrected code:

| alpha | score | steps | invalid_action_rate |
|---|---|---|---|
| 0.001 | **42.835** (47.3, 31.6, 49.6) | 149-398 | 0.0 |
| 0.01 | 3.112 (4.3, 4.3, 0.8) | 400 | 0.0 |
| 0.03 | 0.278 (0.15, 0.39, 0.29) | 400 | 0.0 |

`invalid_action_rate` is exactly zero on all nine runs (three alphas x three
seeds), confirmed from the registry rather than inferred from the score - the
earlier failure mode is eliminated, not merely reduced. Score now falls
monotonically as alpha grows: 0.001 converges cleanly within the 2000-episode
budget, 0.01 is still learning something (better than the near-random 0.28-0.39
of the untrained early episodes) but far from converged, 0.03 does not visibly
improve over random. This is the ordinary large-step-size instability of
semi-gradient TD, not a repeat of the masking bug.

**Decision:** alpha=0.001 for the remaining Task 1 work. `target_blocked` is
removed from phi() and from `FEATURE_VERSION` (bumped to 3, so no model saved
under the interaction-feature version can load silently).

### 1.3 What this phase actually cost, and what it says about the method

Two full alpha sweeps (3 seeds x 2000 episodes each, ~25-30 minutes per sweep)
were spent before the real fix landed: one on the original masking bug, one on
the wrong fix for it. The second sweep would have been avoidable by running the
same cheap check used before the third sweep - a 300-episode, single-seed,
under-a-minute run checking `invalid_action_rate` alone - before committing to
a full multi-seed run. That check is now the standing procedure before any
alpha sweep on this branch: confirm the specific failure being targeted is
gone on the cheapest run that can show it, only then pay for the full protocol.

The byte-for-byte identical retraining under `target_blocked` was itself the
useful signal, not noise to explain away: a fix that changes nothing
measurable after 2000 episodes of real training is not weak evidence, it is
evidence the change never engaged, and it earned direct inspection of the
weight column before either being trusted or discarded.

### 1.4 Constant alpha does not converge

The per-seed spread at 2000 episodes (31.6-49.6) was flagged as possibly a
budget problem. The training log itself already argued against that before any
further run was needed: `weight_norm` and in-training coins both plateau by
episode 500-1000 (coins at 50/50, `weight_norm` in a 21.5-24.3 band) and stay
there through episode 2000 - the update process reads as settled on the metric
that budget would fix.

The same three seeds, same alpha, continued to 6000 episodes, tested that
reading directly:

| | 2000 episodes | 6000 episodes |
|---|---|---|
| eval score, per seed | 47.3, 31.6, 49.6 | **50.0, 4.3, 42.3** |
| eval mean | 42.835 | **32.199** |
| invalid_action_rate | 0.0 | 0.0 |

Longer training did not converge toward a stable value - it moved every seed,
in both directions, and the mean fell. Masking still holds (invalid_action_rate
0.0 on all three), so this is not a return of Phase 1.1's bug.

`weight_norm` tracked past episode 2000 explains why: it never settles, it
keeps drifting inside roughly the same band (21.6-25.3) through episode 6000,
on all three seeds, while in-training coins stay maxed at 50/50 throughout
(one exception, seed 2 at episode ~3000, scored 4 coins for a single episode
then recovered next check). A constant step size has no mechanism to damp
that drift once near a optimum - each update is still full-sized - so the
model does not converge to a point, it wanders inside a region of good
policies indefinitely. In-training reward cannot see this, because it
saturates at the 50-coin ceiling as soon as the region is reached; only the
weight trajectory shows the wandering. Eval score on a fixed, different set of
30 arenas is sensitive to *which point in that wander* happened to be saved,
which is exactly why extending the same run moved every seed's score without
any of them settling.

This is the same failure tabular_q's own Phase 2 named for the tabular case -
"the estimate never converges and the policy is decided by where the random
walk sits when training stops" - reached independently here for a linear
model under a literally identical root cause: a step size with no decay.

**Decision:** `"constant"` alpha is rejected as the schedule to ship with.
Nothing here says alpha=0.001 the *value* was wrong - training-time
performance was excellent throughout - only that stopping at an arbitrary
episode count under a non-decaying rate reports whichever snapshot the drift
happened to land on, not a property of the model. A decaying schedule (the
open question in Settings/Open below) is now the next thing to build, not
another point on this sweep.

---

## Phase 2 - a decaying alpha schedule

### 2.1 Design

Tabular_q's own fix for the identical symptom (Phase 2 of that report) decays
step size by how many times a **state** has been visited:
`alpha / (1 + n(state, action) / half_life)`. That count has no direct
equivalent here - a weight `w[i, a]` is shared by every state with feature `i`
active, not owned by one state - so the same formula is applied at the grain
this representation actually has: `n` counts how many times **weight entry**
`w[i, a]` itself has moved, incremented in `LinearQModel.update` whenever
`phi[i] != 0` for the action being updated. Common features (a neighbour being
free, the bias term) accumulate a large `n` quickly and their columns settle;
rare ones keep a near-full step size for longer. A global, episode-indexed
decay was the other candidate raised when this was opened; it was not built,
because it would shrink every weight at the same rate regardless of how often
that specific weight is actually touched, the opposite of what a shared,
unevenly-visited weight matrix needs.

`effective_alpha(phi, action, cfg)` returns either the untouched scalar
(`"constant"`) or a `(PHI_DIM,)` array (`"visit"`); `update()`'s existing
`alpha * delta * phi` line needed no change, since numpy broadcasts a
per-entry array the same way it broadcasts a scalar.

**Verified before any training:** `"constant"` reproduces the exact table it
did before this change. Two seeds trained on identical configs (pre- and
post-change code, `alpha_schedule="constant"`), weight matrices compared entry
by entry - `max|diff| = 0` on both. 9 new unit tests, including the one
argument the design rests on: two features touched 200 and 2 times
respectively, under identical starting conditions, end up with different step
sizes - a global episode-based decay could not produce that.

### 2.2 What scale `half_life` needs to be at

Tabular_q's `half_life=1000` is calibrated for a **state** visit count. A
2000-episode probe run measured what a **weight-entry** count actually reaches
here: some (feature, action) entries pass 250,000 by episode 2000, two to three
orders of magnitude past what a single tabular state accumulates over an entire
budget, because one feature is shared by every state that has it active rather
than owned by one. Reusing `half_life=1000` unchanged would have been a guess
carried over from a different unit, not a measurement - it would crush the
common features' step size within the first few hundred episodes.

### 2.3 Half-life sweep

Same protocol as 1.4 (6000 episodes, 3 seeds, `coin-heaven`,
`allow_bomb: false`), so the result is directly comparable to constant alpha's
non-convergence:

| half_life | eval score, per seed | mean | spread | invalid_action_rate |
|---|---|---|---|---|
| constant (1.4, for reference) | 50.0, 4.3, 42.3 | 32.199 | 45.723 | 0.0 |
| 1000 | 50.0, 50.0, 50.0 | **50.000** | **0.000** | 0.0 |
| 10000 | 48.9, 49.3, 48.9 | 49.017 | 0.390 | 0.0 |
| 50000 | 49.5, 49.6, 49.7 | 49.600 | 0.245 | 0.0 |

All three collapse the 45.7-point spread constant alpha left to under 0.4, and
`half_life=1000` reaches the maximum score on every seed with zero spread. That
exactness was checked rather than assumed: `mean_steps` is not just similar but
bit-identical across the three independently-trained seeds (124.193 on all
three), on a fixed external evaluation set - three separately trained models
converging to indistinguishable behaviour on boards none of them trained on,
which reads as genuine convergence to a shared near-optimal policy rather than
three lucky coincidences.

**Decision:** `alpha_schedule="visit"`, `alpha_half_life=1000` - both promoted
to the `config.py` default, together with `alpha=0.001` (Phase 1.2's finding,
which had never actually been written into the default before now). Mirrors
tabular_q's own history exactly: `"visit"` only became *its* default once
Phase 2 measured that it was needed.

**Caveat that has to travel with this number.** 1000 was swept on Task 1 alone
- no crates, no opponents, `allow_bomb` off, a small and low-noise feature
distribution. A schedule this aggressive settles common features within the
first few hundred episodes; whether that is still safe once Task 2 exercises
bomb-related features, which start rare and stay rare until the policy learns
to use them, is untested. A feature that locks in early on too little bomb data
could freeze onto a bad estimate before it has been visited enough to trust.
Re-sweep `half_life` once Task 2 is exercised rather than carrying this value
forward unexamined.

---

## Phase 3 - Task 2: bombs, and a systematic collapse

### 3.1 Masking `BOMB` on cooldown, before it was needed

Phase 1 masked movement into a wall because phi() has no component that
guarantees a blocked direction scores lowest. `bombs_left` (whether the agent
currently has a bomb available) has exactly the same property - phi() encodes
no bomb-availability feature at all - so `Q(BOMB)` ranking highest while on
cooldown is exactly as unguaranteed as `Q(blocked direction)` was, and would
reproduce the same trap. Fixed in `callbacks.act` before any Task 2 training
ran, rather than waiting to rediscover it: `BOMB` is now excluded from the
candidate set whenever `game_state["self"][2]` is false, the same mechanism as
the wall mask, checked every step. 3 new unit tests; the existing
`allow_bomb: false` path is unaffected by construction (the new check is only
ever reached when `BOMB` is already in `self.legal`).

### 3.2 First probe: 99.95% suicide, and why

`loot-crate`, `allow_bomb: true`, otherwise Phase 2's settings unchanged, 1
seed, 2000 episodes (matching the cheap-probe-first discipline). Training
places a bomb in every single episode and dies in 1999 of 2000 (average 16.8
steps per episode). Fully greedy evaluation afterward is the opposite extreme:
0 bombs, 0 coins, 0 suicide, 400/400 steps - having died from every bomb it
ever placed during training, the model learned `Q(BOMB)` low enough that it
never bombs at all once exploration is turned off.

Investigated directly rather than assumed:

- `escape_search()` computes correctly. A hand-built state (agent standing on
  a bomb with a 3-step fuse, open board) returns a real escape direction, and
  `lethal_at` correctly marks steps 3-4 as the danger window.
- The trained model does not use it. Querying the same state's Q-values
  directly: the correct escape direction is UP, the model's greedy choice is
  LEFT, and the six Q-values are small and close together (0.008-0.148) -
  undertrained, not confidently wrong.
- Only 1 of 2000 training episodes ever survived a bombing. With almost no
  genuine escape-success experience to reinforce, there is essentially nothing
  for the escape-direction weights to learn from.

### 3.3 More episodes helps - for some seeds

Same protocol, 6000 episodes instead of 2000, 1 seed first (cheap check):
suicide fell from 99.95% (training-time) to 68.0% (eval), coins rose from 0 to
1.653, crates from ~2.7 to 10.18, `weight_norm` grew smoothly with no
divergence (0.01 to 3.57). The rare-experience hypothesis held up under a
direct test.

Confirmed on the full 3-seed protocol (`lq_p3_6000`) - and immediately
complicated:

| seed | suicide | coins | bombs | crates |
|---|---|---|---|---|
| 1 | 0.715 | 1.472 | 16.08 | 10.33 |
| **2** | **0.985** | **0.000** | 0.98 | 2.52 |
| 3 | 0.690 | 1.467 | 17.94 | 8.88 |

Seed 2 never escaped the pattern 3.2 found for the single probe seed: its
training log shows `killed_self=1` on essentially every one of the 6000
episodes, bombing exactly once and dying, from episode 1 through episode 6000,
while `weight_norm` still grows smoothly throughout (3.358 by the end) - the
weights are moving, but the policy is not. `invalid_action_rate` is 0.0 on all
three seeds, so this is not a return of 3.1 or Phase 1's masking gaps.

### 3.4 Re-sweeping half_life does not fix it - the failure moves

The caveat flagged at the end of Phase 2 (2.3) was exactly this: `half_life`
was only ever measured on Task 1's low-noise distribution, and an aggressive
schedule could freeze a rare, bomb-related weight before enough data exists to
trust it. Re-swept on the same three seeds, same protocol:

| half_life | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| | suicide / coins | suicide / coins | suicide / coins |
| 1000 | 0.715 / 1.47 | **0.985 / 0.00** | 0.690 / 1.47 |
| 10000 | 0.020 / 0.79 | 0.005 / 0.02 | **1.000 / 0.00** |
| 50000 | **0.702 / 0.99** | 0.015 / 0.86 | 0.005 / 0.02 |

No value rescues all three. Each one rescues whichever seed was failing at a
different value while breaking (or leaving passive) one that had been fine.
The two "rescued" seeds at 10000/50000 are not simply safer - they are close
to inert (crates 0.23, essentially one bomb and then idle for the rest of the
episode), which is a different flavour of the same underlying problem, not a
solution to it. This is evidence against "half_life was miscalibrated" as the
explanation: the failure follows the seed, not the parameter.

One coincidence surfaced and was checked before being trusted, per the house
rule this project has followed since Phase 1: `lq_p3_hl10000` seed 2 and
`lq_p3_hl50000` seed 3 produced numerically near-identical eval statistics.
Loading both models and comparing weights directly: `max|w diff| = 0.649` -
genuinely different models, not a duplicate run. Both happen to have converged
to the same qualitative class of policy (bomb once, then sit almost idle for
the rest of the episode) on the same fixed 30 evaluation arenas, which is
enough on its own to produce near-identical aggregate statistics from
different weights. Not a bug; recorded because the identical-numbers reflex
from Phase 1 says to check before moving on, not because it changed anything.

### 3.5 Ten seeds: this is systematic, not a small-sample accident

Three seeds could not distinguish "one unlucky draw" from "this fails most of
the time." Seven more seeds at `half_life=1000` (the current default),
identical protocol, combined with the three already run:

| | fraction | suicide | coins | bombs | crates |
|---|---|---|---|---|---|
| productive | 4/10 | 0.746 | 1.480 | 14.44 | 10.16 |
| collapsed | **6/10** | 0.942 | **0.000** | 0.98 | 2.75 |

The split is exact, not approximate: every collapsed seed's `mean_coins` is
`0.000` to three decimals, every productive seed's is `1.47-1.50` - two
distinct outcomes, nothing in between, across ten independently trained
models. **60% of training runs land in a degenerate local optimum**: bomb
approximately once, then do almost nothing for the rest of the episode. Seed 2
was not unlucky; it was the typical case.

This is now a confirmed, systematic finding rather than a hyperparameter
question. What produces the bimodal split - and what specifically distinguishes
the four seeds that escape it - is the open question the next phase has to
answer, not "which half_life."

---

## Phase 4 - potential-based shaping toward escaping danger

### 4.1 Design

Phase 3's diagnosis was specific: escape-success experience is rare (1 of
2000 early episodes), so the only signal against dying is the terminal -5,
propagated back through gamma across however many steps preceded it. A dense,
per-step signal for the same objective is exactly what potential-based
shaping is for, and `_potential()` already exists (Phase 1, for the target).

Added as a second, independent potential summed into the same function:
`Phi(s) = -shaping_weight * target_distance - escape_shaping_weight *
danger_urgency`. Ng et al.'s policy-invariance guarantee holds for any
`Phi(s)`, and a sum of two valid potentials is itself a valid potential, so
the target term needed no change and both call sites (mid-episode and
terminal) needed none either - `_potential()` already gets called in exactly
the two places Phase 1 established.

`danger_urgency` reuses `t_here`, an existing feature, rather than computing
anything new: `t_here` is 0 when the current tile never becomes lethal and
1-4 counting down to detonation, so a *smaller* nonzero `t_here` is worse -
the opposite polarity of `target_dist`. Remapped to `0 if t_here==0 else
(5 - t_here)` so the sign convention matches the target term throughout: a
larger potential is always better.

**Verified before training:** 7 unit tests pin the urgency mapping, the
independence of the two terms, and the magnitude of the shaping reward for the
exact scenario Phase 3 diagnosed (t_here 1 -> 0 gives a clear positive
reward, not a rounding-sized nudge). `escape_shaping_weight` defaults to 0.0;
an A/B on real training (two seeds, pre- and post-change code, default
config) gave `max|w diff| = 0` - the addition is inert until turned on.

Hand-computing the shaping reward at `weight=0.5` for a one-step escape gave
+2.0 - larger than `reward_coin` (1.0) and comparable to a large fraction of
`reward_killed_self` (-5.0). Too strong a value to guess; the weight needed
measuring, not assuming.

### 4.2 Cheap check first, on a seed that was already fine

2000 episodes, 1 seed, `escape_shaping_weight=0.2`: training-time suicide
99.15% against the no-shaping baseline's 99.95% - barely different, and by
itself uninformative, since Phase 3.3 already showed 2000 episodes is not
enough to see an effect either way.

Extended to 6000 episodes on the same seed, matching Phase 3.3's exact
comparison point:

| | no shaping (3.3) | escape_shaping_weight=0.2 |
|---|---|---|
| suicide | 68.0% | **45.3%** |
| coins | 1.653 | 1.447 |
| steps | 145.6 | **230.7** |
| bombs | 17.0 | 32.0 |

Suicide fell substantially and survival time grew by more than half. But this
seed was already in the *productive* cluster without shaping (Phase 3.5) - it
does not yet answer whether shaping rescues the 60% that collapsed to
`coins=0.000`, which is the actual question.

### 4.3 The same ten seeds Phase 3.5 measured

`escape_shaping_weight=0.2`, otherwise identical protocol (`loot-crate`, 6000
episodes, `half_life=1000`), the same ten training seeds that gave the 60%
collapse:

| | no shaping (3.5) | shaping, weight=0.2 |
|---|---|---|
| seeds with `coins = 0.000` | **6 / 10** | **0 / 10** |
| mean suicide rate | 0.864 | **0.443** |
| coins, collapsed cluster | 0.000 | n/a - no cluster |
| coins, range across all 10 | 0.000 or 1.47-1.50 | **1.14 - 1.55** |

Every seed that previously collapsed - 2, 5, 7, 8, 9, 10 - now lands in the
same band as the seeds that were already productive: `coins` 1.14-1.55 across
all ten, spread 0.402, no seed at zero. `invalid_action_rate` is 0.0 on all
ten (masking integrity unaffected).

One seed is a partial outlier worth naming rather than averaging away: seed 5
reaches the *highest* coin and crate count of all ten (1.545, 10.69) while
suiciding far more than the rest (0.795 against 0.38-0.45) and dropping far
fewer bombs (7.8 against 30-36) - a bomb-scarce, high-value-per-bomb, higher-risk
profile, not simply "worse". Whether that is a seed still finding its footing
or a genuinely different strategy this weight permits is not resolved here.

**Decision:** `escape_shaping_weight=0.2` kept, not yet promoted to the
`config.py` default - one weight value has been tested, not swept. Suicide at
44.3% mean is a large improvement over 86.4% but still far from tabular_q's
eventual `loot-crate` result (suicide 0.030, Phase 3 of that report); whether a
different weight closes more of that gap, or whether 6000 episodes is again
the binding constraint the way it was in Phase 3, is the next question, not
assumed either way.

---

## Phase 5 - sweeping the shaping weight

### 5.1 Why a sweep, and what was declared before running it

Phase 4 tested exactly one weight. The shaping reward for a one-step escape is
`4 * w`, so at 0.2 it is 0.8 - already comparable to `reward_coin` (1.0). That
is not a minor knob, and one sample of it is not a hyperparameter study, which
the project brief requires be taken seriously.

Swept 0.05, 0.1, 0.5, 1.0 against the two values already measured (0.0 from
Phase 3.5, 0.2 from Phase 4), 10 seeds each, `loot-crate`, 6000 episodes.

A specific prediction was recorded before the runs started: **a large enough
weight should suppress bombing itself**, because dropping a bomb creates the
very danger the potential penalises - the same degenerate outcome as Phase 3's
collapse, reached from the opposite direction.

### 5.2 Result

| weight | score | suicide | bombs | crates | steps | collapsed seeds |
|---|---|---|---|---|---|---|
| 0.0 | 0.592 | 0.864 | 6.36 | 5.71 | 65.6 | **6 / 10** |
| 0.05 | **1.387** | 0.698 | 11.47 | 10.00 | 138.3 | 0 |
| 0.1 | 1.370 | 0.619 | 10.95 | 9.79 | 168.4 | 0 |
| 0.2 | 1.321 | 0.443 | 30.88 | 8.08 | 234.3 | 0 |
| 0.5 | 1.271 | 0.429 | 32.17 | 7.86 | 239.3 | 0 |
| 1.0 | **0.047** | 0.619 | 16.66 | 3.44 | 155.8 | **6 / 10** |

The prediction held. At weight 1.0 crates fall from ~10 to 3.44, score
collapses to 0.047, and six of ten seeds are back at `coins = 0.000` - the same
failure Phase 3 measured with no shaping at all, produced here by too much of
it. Safety pressure large enough stops the agent creating the situation it is
being rewarded for surviving.

Everything between 0.05 and 0.5 is safe from collapse, and inside that band
there is a real trade: 0.05 scores highest and suicides most (0.698), 0.5
suicides least (0.429) and scores lowest. No weight is best on both axes.

### 5.3 Choosing, by a rule fixed in advance

The criterion was written down before the results were read, precisely so the
winner could not be picked after seeing which number looked good:

- primary: `mean_score` across the 10 seeds
- tie-break: within 0.05 mean_score, prefer the lower suicide rate
- boundary check: if the winner is the largest weight swept, the optimum is not
  bracketed, so extend the sweep rather than move on

0.05 and 0.1 tie on score (1.387 against 1.370, inside the 0.05 band), and 0.1
suicides less (0.619 against 0.698), so **0.1 wins**. Not at a boundary, so the
follow-up is the budget question rather than more weights.

**A flaw in that rule, recorded because it matters.** The stated justification
for `mean_score` was "score is the tournament objective". That is wrong as
written: the tournament is `classic` with three opponents, while every number
above is `loot-crate` solo, a training environment with a much denser reward.
Fixing the criterion in advance did its job - it removed hindsight from the
choice - but a criterion fixed in advance is only as good as the proxy it
names, and this one named the wrong board. Phase 6 was designed around that
gap rather than ignoring it.

Wall time: 1h 55m for the four new arms. Runtime rose monotonically with weight
(13, 30, 35, 37 minutes), which is itself a readout - a higher weight keeps the
agent alive longer, so each episode simulates more steps.

---

## Phase 6 - was 6000 episodes still the binding constraint?

### 6.1 The confound

Every weight in Phase 5 was compared at 6000 episodes, and Phase 3 measured
that budget alone moved suicide from 99.95% to 68%. If different weights
converge at different rates, the ranking at 6000 need not be the ranking at
convergence - which would make the sweep's conclusion a fact about the budget
rather than about the weight. Re-ran the winning weight (0.1) at 12000
episodes, same ten seeds, everything else identical.

### 6.2 Result: the budget was binding, and the strategy changed shape

| | 6000 ep | 12000 ep |
|---|---|---|
| score | 1.370 | **2.500** |
| suicide | 0.619 | **0.938** |
| bombs | 10.95 | 7.37 |
| crates | 9.79 | **14.43** |
| crates per bomb | 0.89 | **1.96** |
| steps survived | 168.4 | **56.3** |

Score up 83%, and the agent is more than twice as efficient per bomb - but it
now dies in almost every round, at step 56 of 400. It did not learn to bomb
*and* survive; it learned to bomb far better and stop caring about the aftermath.
`invalid_action_rate` is 0.0 on all ten and no seed is at `coins = 0.000`, so
this is a genuine strategy shift rather than any of the earlier failure modes.

### 6.3 A prediction that turned out to be wrong

Recorded before testing: this looked like a strategy that scores well only
because of the board it was measured on. `loot-crate` has 50 coins and dense
crates, so rushing in, bombing hard and dying at step 56 can still pay; on
`classic` (9 coins, the tournament board) forfeiting 344 of 400 steps should
cost more than the extra crates are worth.

Both models re-evaluated on `classic`, solo, 10 seeds:

| | score | suicide | crates | steps |
|---|---|---|---|---|
| 6000 ep | 0.253 | 0.499 | 8.74 | 212.2 |
| 12000 ep | **0.422** | 0.858 | 13.32 | 83.7 |

**The prediction was wrong.** The ordering is preserved - the 12000-episode
model scores 67% more on `classic` too, despite surviving a quarter as long.
The reasoning missed that on `classic` the coins are *under* the crates, so
crate throughput is the only route to score at all: surviving to step 212
without breaking crates (8.74 against 13.32) simply leaves nothing to collect.
Longevity is not itself worth points.

What this does *not* establish: both evaluations are solo. In a four-agent
round, dying at step 84 hands the remaining ~316 steps to three opponents who
are still collecting, and that opportunity cost has not been measured. The
falsified prediction stands as falsified on what was tested; the multi-agent
case is untested, not quietly reserved as a rescue.

### 6.4 Harness limit found along the way

The first `classic` evaluation crashed: `eval run lq_p5_w01_seed7 exceeded 900s
and was killed` (`PLAY_TIMEOUT` in `tools/evaluate.py`). The cause is the
surviving seeds - a model that lasts 400 steps across 20 rounds on 30 arenas
exceeds the per-process budget, while a suicidal one never comes close. Re-run
at 10 rounds instead of 20, which fits; the registry's `rounds_per_seed` column
keeps the two sample sizes distinguishable. An earlier crash in the same phase
(`attempting to add a session with an ID that's already in use`, at
`--eval-workers 10`) was worked around with 5 workers; training had already
completed, so `tools/eval_existing.py` recovered it without retraining.

---

## Phase 7 - tabular_q's protocol: four agents on `classic`

### 7.1 Why

Every number above is solo, and the tournament is four agents on `classic`.
tabular_q's anchor (its Phase 26: 6000 episodes on `classic` against three
seeded opponents, ten seeds) had no linear_q counterpart, and 6.3 left the
multi-agent cost of dying early unpriced. Two measurements close both.

### 7.2 The solo-trained models, dropped onto the four-agent board

The Phase 6 models (weight 0.1, 12000 episodes, trained solo on `loot-crate`),
re-evaluated against three `rule_based_seeded` opponents on `classic`, 10 rounds
per arena: score 0.398, coins 0.330, kills 0.014, self-kill rate 0.894.

### 7.3 Trained with opponents, on tabular_q's exact protocol

`classic:6000@rule_based_seeded+rule_based_seeded+rule_based_seeded`, the same
ten seeds, weight 0.1, evaluated on the four-agent board:

| same protocol, 10 seeds | tabular_q (Phase 26 anchor) | linear_q |
|---|---|---|
| score | **2.210** | 0.130 |
| self-kill rate | 0.527 | 0.888 |
| bombs per round | 17.9 | 4.4 |
| crates per round | 35.4 | 4.9 |
| steps survived | 344 | 327 |

Paired over the same ten seeds, tabular_q is ahead by 2.080 (sd 0.511),
t = 12.88, and ahead on all ten seeds individually.

The two agents survive about equally long - 344 steps against 327 - so the gap
is not that linear_q dies sooner. It barely acts: 4.4 bombs and 4.9 crates per
round against 17.9 and 35.4. On `classic` every coin starts under a crate, so a
policy that does not break crates cannot score however long it lives.

Training with opponents made this agent *worse* (0.130) than training it solo
and dropping it in (0.398), the opposite of tabular_q's own Phase 17a (+0.42).
The direction is not attributable to either factor: the solo-trained models had
12000 episodes and these had 6000, so budget and training opponents differ at
once.

Training took 13114 s for ten seeds - about 25x the per-episode cost of the solo
run in Phase 6, since every step simulates four agents and the agent now
survives most of the round.

**Decision:** not competitive on the tournament protocol. tabular_q is the
submitted agent; this branch's value is that it shares the observation code,
reward design and harness with tabular_q, so its failure modes are attributable
to the representation.

---

## Settings currently in force

```
alpha 0.001, alpha_schedule "visit", alpha_half_life 1000
gamma 0.995
exploration "epsilon", eps 1.0 -> 0.05 over 2000 episodes
use_symmetry True, use_opponent_blocking True
allow_bomb True (Task 2, Phase 3)
shaping_weight 0.0, escape_shaping_weight 0.1 (Phase 5 sweep winner; still not
        a config default - Phase 6 showed the sweep that chose it was itself
        budget-limited, so the value is provisional)
rewards: identical to tabular_q's defaults at the point this branch forked
budget: 12000 episodes on `loot-crate` (Phase 6: 6000 was binding); Task 1 on
        `coin-heaven` at 6000
```

## Rejected, with evidence

`target_blocked` interaction feature - structurally unreachable, reverted in
favour of masking at decision time (1.1-1.2).

`alpha_schedule "constant"` as a final choice - `weight_norm` never settles
(1.4), so a run's eval score reports an arbitrary snapshot of an unconverged
drift, not a stable policy. Superseded by `"visit"` (2.1-2.3).

A global, episode-indexed decay (alternative design for Phase 2) - not built.
It would shrink every weight at the same rate regardless of how often that
specific weight is touched, which is the wrong axis for a matrix where some
entries are updated hundreds of thousands of times more than others (2.2).

`half_life` re-sweeping as *the* fix for Task 2's collapse (3.4) - 10000 and
50000 each rescue a different seed than 1000 does while leaving another
seed passive or broken. The failure follows the seed, not the schedule value.

`escape_shaping_weight` at 1.0 or above (5.2) - large enough safety pressure
stops the agent bombing at all, re-creating Phase 3's collapse (6/10 seeds at
`coins = 0.000`, score 0.047) from the opposite direction. The usable band is
roughly 0.05 to 0.5.

## Open

1. **Nothing yet makes this agent bomb *and* live.** Every intervention so far
   trades one against the other: shaping weight up buys survival and loses
   score (5.2), budget up buys score and loses survival (6.2). The best score
   available (12000 episodes, weight 0.1) suicides in 94% of rounds against
   tabular_q's 3.0%. `reward_survived` is still 0.0 and untested - it is the
   one lever aimed directly at this axis rather than at either side of the
   trade.
2. **The weight sweep is budget-limited and should be redone at 12000.** Phase
   5 compared weights at 6000 episodes; Phase 6 then showed 6000 was binding
   (6.1-6.2), so the ranking that chose 0.1 may not survive at the longer
   budget. No weight has been measured at 12000 except 0.1.
3. **Why training with opponents hurt is unresolved.** 0.130 trained with
   opponents against 0.398 trained solo (7.2-7.3), but the budgets differ too
   (6000 against 12000), so the direction is not attributed to either.
4. **Seed 5's profile at weight 0.2 is unexplained.** Highest coins and crates
   of all ten, far fewer bombs, and the highest suicide rate by a wide margin
   (4.3) - a different strategy or an unsettled outlier, not distinguished yet.
5. **`tools/verify_unchanged.py` is still hardcoded to `tabular_q`.** Every
   pre/post-change comparison on this branch (Phases 2.1, 4.1) has used a
   one-off script instead, because this tool cannot target `linear_q` yet -
   the same gap `train.py` and `eval_existing.py` had before Phase 1, not yet
   closed here.
