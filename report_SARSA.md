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

*(next: the same comparison on the solo board, where the runs are deterministic
and the opponent noise is absent)*
