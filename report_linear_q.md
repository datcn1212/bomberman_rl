# Linear Q-learning - experiment log

Working log for the agent in `agent_code/linear_q/`, following the same
discipline as `report_tabular_q.md`: one entry per phase, what changed, what it
measured, what was decided. Negative results are kept.

**Where this stands.** One phase. The agent converges on Task 1 with a masking
fix in place; the interaction-feature fix tried first did not work, and both
attempts are recorded because the failure is the more useful half of the
story.

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

---

## Settings currently in force

```
alpha 0.001, alpha_schedule "constant"
gamma 0.995
exploration "epsilon", eps 1.0 -> 0.05 over 2000 episodes
use_symmetry True, use_opponent_blocking True
allow_bomb False (Task 1 scope only)
rewards: identical to tabular_q's defaults at the point this branch forked
budget: 2000 episodes on `coin-heaven`
```

## Rejected, with evidence

`target_blocked` interaction feature - structurally unreachable, reverted in
favour of masking at decision time (1.1-1.2).

## Open

1. **Not yet converged at any tested alpha over 2000 episodes.** 0.001 is best
   so far but its own per-seed spread (31.6 to 49.6) suggests it may not be
   settled either; a longer budget or a decaying schedule has not been tried.
2. **`alpha_schedule` has only `"constant"`.** What a visit-based or
   feature-based decay should look like for a shared weight matrix is
   unresolved, deferred until 0.001's convergence ceiling is known.
3. **Task 2 and beyond are untested.** `allow_bomb` is still off; bomb-safety
   reward and the resulting risk of Q divergence (the deadly triad -
   bootstrapping, off-policy, function approximation, all three present here)
   have not been exercised yet.
4. **No comparison to tabular_q on any shared protocol yet.** Everything here
   is Task 1 solo; the four-agent anchor tabular_q uses (Phase 26, 2.210) has
   no linear_q counterpart.
