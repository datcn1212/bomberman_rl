# Linear Q-learning - experiment log

Working log for the agent in `agent_code/linear_q/`, following the same
discipline as `report_tabular_q.md`: one entry per phase, what changed, what it
measured, what was decided. Negative results are kept.

**Where this stands.** Two phases. The masking fix stops the agent walking
into walls (invalid_action_rate 0.0 throughout). A per-(feature, action)
decaying step size then fixed the non-convergence a constant rate left behind:
50.0/50.0/50.0 on Task 1, spread 0.000, against constant alpha's 45.7-point
spread on the same protocol. That value is validated on Task 1 only - Task 2
(bombs) is the next open question, not a settled one.

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

## Settings currently in force

```
alpha 0.001, alpha_schedule "visit", alpha_half_life 1000
gamma 0.995
exploration "epsilon", eps 1.0 -> 0.05 over 2000 episodes
use_symmetry True, use_opponent_blocking True
allow_bomb False (Task 1 scope only)
rewards: identical to tabular_q's defaults at the point this branch forked
budget: 6000 episodes on `coin-heaven` (2000 was Phase 1's; 1.4/2.3 showed a
        constant rate needed the longer budget just to reveal it does not
        converge - the decaying schedule reaches its result well inside it)
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

## Open

1. **`half_life=1000` is untested past Task 1.** It was swept on a small,
   low-noise feature distribution (2.3) and settles common features within a
   few hundred episodes; whether that is still safe once bomb-related features
   - rare until the policy learns to use them - are exercised is the open
   question, not assumed either way. Re-sweep once Task 2 is running, not
   before.
2. **Task 2 and beyond are untested.** `allow_bomb` is still off; bomb-safety
   reward and the resulting risk of Q divergence (the deadly triad -
   bootstrapping, off-policy, function approximation, all three present here)
   have not been exercised yet.
3. **No comparison to tabular_q on any shared protocol yet.** Everything here
   is Task 1 solo; the four-agent anchor tabular_q uses (Phase 26, 2.210) has
   no linear_q counterpart.
4. **`tools/verify_unchanged.py` is still hardcoded to `tabular_q`.** Phase
   2.1's pre/post-change comparison was done with a one-off script instead,
   because this tool cannot target `linear_q` yet - the same gap `train.py`
   and `eval_existing.py` had before Phase 1, not yet closed here.
