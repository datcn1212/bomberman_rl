# SARSA - experiment log

Working log for the second model: the same tabular agent with an **on-policy**
bootstrap target instead of Q-learning's off-policy one. Everything else -
state encoding, rewards, step size, exploration, symmetry - is inherited
unchanged from the tabular Q branch, so every comparison here isolates the
target and nothing else.

The agent lives in `agent_code/tabular_q/`; the target is a config option.

---

## Why this is a separate model, not a variant

Q-learning bootstraps from `max_a' Q(s', a')`: the value of the *best* action
available next, regardless of what the agent will actually do. SARSA bootstraps
from `Q(s', a')` where `a'` is the action the behaviour policy actually takes,
exploration included.

```
Q-learning   Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a')  - Q(s,a) ]
SARSA        Q(s,a) <- Q(s,a) + alpha * [ r + gamma *      Q(s',a')    - Q(s,a) ]
```

The textbook consequence is that SARSA learns the value of the policy it is
following rather than of the greedy policy, so it prices in its own mistakes.
In a domain where exploration can be fatal - the cliff-walking example, and
Bomberman is a cliff with a timer - that usually produces a more cautious agent.

Whether caution is what this agent needs is the question this branch answers, and
the answer is not assumed from anywhere else.

### Implementation

The existing staged-transition design supplies `a'` for free. When
`game_events_occurred` fires for step *t*, the transition waiting to be applied
is the one that *ended* in `s_t`, and the call's own `self_action` is exactly the
action taken there. No extra delay is needed.

Two details that would fail silently if wrong:

* **Frame.** With D4 canonicalisation the successor's row is written in the
  successor's frame, so `a'` must be translated into *that* frame, not the frame
  the transition started from. The transition therefore carries `next_perm`.
* **Terminal.** The last transition bootstraps from nothing, so it needs no `a'`.
  The one staged before it does, and `end_of_round` supplies `last_action`.

### Verification before any experiment

Adding a second target must not disturb the first, or every SARSA-versus-Q
comparison is measuring two things at once.

| check | result |
|---|---|
| `target = "q"` reproduces the pre-change code exactly | **140 rows, largest Q difference 0** |
| `target = "sarsa"` actually changes behaviour | differs, as it must |
| learned values are pessimistic relative to Q | median max-Q **-1.01** against **+2.37** |

The first was run by checking the previous commit out into a `git worktree` and
training the same solo configuration under both, then comparing tables entry by
entry. Solo, because `rule_based_agent` is unseeded and runs with opponents do
not repeat.

The third is the signature that says the target is doing its job: values under an
exploring policy must sit below greedy values. They are also **negative**, which
is informative in itself - under the behaviour policy the expected return on this
board really is negative, because the agent dies in roughly half of all rounds and
carries a -5 for it. Q-learning reports positive values for the same states
because it assumes greedy play from the next step onward.

---

## Protocol

Identical to the tabular Q branch, including the measured noise floor:
`rule_based_agent` re-seeds from system entropy, so two runs of one configuration
differ with a **run-to-run standard deviation of 0.090** on the four-agent score
(null standard error 0.128 for a single comparison). Four-agent effects are judged
against that, not against a paired t. Solo runs are deterministic and repeat
exactly.

---

## Phase S1 - SARSA on the four-agent board

Everything at the tabular Q defaults - gamma 0.995, D4 symmetry, opponents block
the immediate move, 6000 episodes all against three `rule_based_agent` - with the
target as the only change. 10 seeds.

The Q-learning baseline is pooled over the **three** repeat runs of that
configuration, so it is not itself a lucky draw.

| target | score | coins | kills | self-kill | bombs |
|---|---|---|---|---|---|
| **Q-learning** (3 runs pooled) | **2.275** | **1.803** | 0.095 | 0.489 | 19.74 |
| SARSA | 1.717 | 1.141 | 0.115 | 0.555 | 19.50 |

**SARSA minus Q: -0.558 score, z = -4.36** against the noise floor. Clearly worse.

The shape of the loss is specific. Bombing is unchanged (19.50 against 19.74), so
SARSA has not become the timid agent the theory would predict; kills are slightly
*up* (0.115 against 0.095). The whole deficit is **coins**: 1.141 against 1.803,
a 37% drop.

Whether that deficit is a property of the target or of the state it reads is not
settled here: everything was measured under one configuration. Phase S2 changes
the configuration and re-runs the comparison.

---

## Phase S2 - coin-first targeting, on both targets

Phase S1 localised the deficit to coins, so the next question was about the rule
that decides where the agent walks. `target_search` returned the nearest of
{coin, crate}. Logging the feature tuple of every step over 30 four-agent rounds
showed what that costs: of the 2083 steps where a coin was on the board, a
**crate** was the target in **1855 of them (89.1%)**, because crates are dense
enough on `classic` that one is almost always adjacent - median distance 1.

`coin_priority` searches coins first and falls back to crates only when no coin
is reachable at all, so a distant coin outranks an adjacent crate. Added as a
config flag and proven inert while off (`tools/verify_unchanged.py`, identical
tables on two seeds; the same tool reports DIFFERS with the flag on, so it can
detect the change it is asked about).

Both targets, 10 seeds, against the existing controls at the same protocol:

| | control | coin-first | change |
|---|---|---|---|
| Q, score | 2.275 | **2.707** | +0.432 (3.4 SE) |
| SARSA, score | 1.717 | **2.343** | +0.626 (4.9 SE) |

Both gains are several times the noise floor. **The stated prediction still
failed.** It was: coins up, crates roughly flat. What happened instead:

| | Q control | Q coin-first |
|---|---|---|
| coins | 1.80 | 2.32 |
| crates | 37.6 | **25.1** |
| bombs | 19.7 | **12.7** |
| self-kill | 0.489 | 0.592 |

Crates fell by a third and bombs by 36%. The agent did not get better at
collecting what it uncovered; it stopped uncovering and went to fetch what was
already lying around. On `classic` there are nine coins and almost all start
under crates, so collecting *more* of them while breaking *fewer* crates means
the coins were revealed by somebody else - the three opponents.

### S2.1 The gain is parasitic, and it does not survive a different field

If that reading is right, the gain should disappear against an opponent that
never bombs. `peaceful_agent` moves at random and never places a bomb, so the
only crates that break are the ones our own agent breaks. Both controls and both
coin-first models, same 600-round protocol:

| model | vs three `rule_based_agent` | vs three `peaceful_agent` |
|---|---|---|
| Q control | 2.275 | **17.780** |
| Q coin-first | 2.707 | **7.780** |
| SARSA control | 1.717 | 6.062 |
| SARSA coin-first | 2.343 | 4.111 |

The ordering reverses completely. Coin-first wins by 0.43 where it was trained
and loses by **10.0** where nobody else opens the board.

The per-seed table shows this is not a shifted trade-off but a **fragility**:

| model | per-seed score vs `peaceful_agent` |
|---|---|
| Q control | 16.8 17.0 17.5 17.8 17.8 18.1 18.1 18.2 18.2 18.3 |
| Q coin-first | 0.5 0.6 3.5 3.6 9.8 10.5 12.2 12.2 12.4 12.4 |

Every control seed lands in a 1.5-point band. Coin-first is bimodal, and its two
worst seeds drop bombs to **2.3 and 3.1 per round** against the control's 38.2 -
they survive all 400 steps and do essentially nothing, because the policy waits
for coins that never appear.

**Verdict: rejected**, in spite of a gain several times the noise floor in the
setting it was measured in. Three reasons, in order of weight:

1. It buys its gain by **suppressing bombing**, which is the fourth time in this
   project that an intervention hit its target that way rather than by improving
   the behaviour.
2. The gain is **contingent on the opponents doing the crate work**. A tournament
   field is unknown, so this is a bet on the field rather than a skill.
3. It makes training **unreliable**: 4 of 10 seeds learned the dependence so
   strongly that they nearly stopped bombing altogether.

`coin_priority` stays in the code, defaulted off and proven inert, because the
measurement is worth keeping reproducible.

### S2.2 What this says about the target

The Q-SARSA gap narrows under the new rule - from 0.558 (4.4 SE) to 0.364
(2.8 SE) - and SARSA gained more from the change than Q did (+0.626 against
+0.432). The sign does not change: Q is still ahead in both configurations.

So Phase S1's conclusion survives, with its scope now measured rather than
assumed: **SARSA is worse here, and how much worse depends on the state
representation.** It is not a constant of the two targets.

---

## Open

- The agent is still below the reference where it counts: 2.275 on the
  four-agent board against `rule_based_agent` at 3.260.
- Both interventions that raised the four-agent score so far did it by bombing
  less. Nothing yet has made the agent bomb *better*.
- The 89% figure bounds the target-rule divergence from above ("a coin is on the
  board" is not "a coin is reachable"); the exact rate is unmeasured.
- No second model. SARSA and Q share one function approximator, one state
  encoding and one feature set, so they may not count as two.
