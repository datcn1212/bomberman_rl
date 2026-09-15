### Training callback delivery

**A. Agent survives to the end of the round**

| callback | old_step | new_step | action | events |
|---|---|---|---|---|
| `game_events_occurred` | 1 | 1 | BOMB | BOMB_DROPPED |
| `game_events_occurred` | 2 | 2 | RIGHT | MOVED_RIGHT |
| `game_events_occurred` | 3 | 3 | RIGHT | MOVED_RIGHT |
| `game_events_occurred` | 4 | 4 | DOWN | MOVED_DOWN |
| `game_events_occurred` | 5 | 5 | WAIT | WAITED, BOMB_EXPLODED |
| `game_events_occurred` | 6 | 6 | WAIT | WAITED |
| `game_events_occurred` | 7 | 7 | WAIT | WAITED |
| `game_events_occurred` | 8 | 8 | WAIT | WAITED |
| `game_events_occurred` | 9 | 9 | WAIT | WAITED |
| `end_of_round` | 9 | None | WAIT | WAITED, SURVIVED_ROUND |

- transitions delivered: 10, distinct `old_step` values: 9
- the final step (9) appears **2 times** -- once via `game_events_occurred` and once via `end_of_round`
- both carry the same events: **False** (['WAITED'] vs ['WAITED', 'SURVIVED_ROUND'])
- consequence: a learner that updates in both callbacks counts the last transition of every surviving round **twice**.

**B. Agent blows itself up**

| callback | old_step | new_step | action | events |
|---|---|---|---|---|
| `game_events_occurred` | 1 | 1 | BOMB | BOMB_DROPPED |
| `game_events_occurred` | 2 | 2 | WAIT | WAITED |
| `game_events_occurred` | 3 | 3 | WAIT | WAITED |
| `game_events_occurred` | 4 | 4 | WAIT | WAITED |
| `end_of_round` | 5 | None | WAIT | WAITED, BOMB_EXPLODED, KILLED_SELF, GOT_KILLED |

- transitions delivered: 5, distinct `old_step` values: 5
- fatal events arrive in: end_of_round
- consequence: the transition into death is only visible through `end_of_round`, so a learner that only handles `game_events_occurred` never learns that dying is bad.
