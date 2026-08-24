from collections import namedtuple, deque

from typing import List

import events as e
from .callbacks import state_to_features

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    # self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.round_rewards = 0
    self.reward_history = []


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    rewards = reward_from_events(self, old_game_state, new_game_state, events)
    transition = Transition(state_to_features(old_game_state), 
        self_action, 
        state_to_features(new_game_state), 
        rewards
    )
    self.round_rewards += rewards
    # self.transitions.append(transition)
    self.model.update(transition)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    rewards = reward_from_events(self, last_game_state, last_game_state, events)
    transition = Transition(
        state_to_features(last_game_state), 
        last_action, 
        None, 
        rewards
    )
    
    # self.transitions.append(transition)
    self.model.update(transition)
    
    # self.transitions.clear()
    self.round_rewards += rewards
    self.reward_history.append(self.round_rewards)

    with open("reward_progress.txt", "a") as f:
        f.write(f"{self.round_rewards}\n")
    self.round_rewards = 0

    self.model.save("q_table.pkl")


def reward_from_events(self, old_game_state: dict, new_game_state: dict, events: List[str]) -> int:
    """
    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 500,  
        e.KILLED_OPPONENT: 50,
        e.KILLED_SELF: -50,  
        e.GOT_KILLED: -50, 
        e.INVALID_ACTION: -100,  
        e.BOMB_DROPPED: -100, 
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    old_x, old_y = old_game_state["self"][3]
    new_x, new_y = new_game_state["self"][3]

    old_coins = old_game_state["coins"]
    new_coins = new_game_state["coins"]

    # --- Incentive 1: go near the coins ---
    # If there are coins in the map and we haven't collected one in this turn...
    if old_coins and e.COIN_COLLECTED not in events:
        # Nearest coin from previous step
        old_closest = min(
            old_coins, key=lambda c: abs(c[0] - old_x) + abs(c[1] - old_y)
        )
        old_distance = abs(old_closest[0] - old_x) + abs(
            old_closest[1] - old_y
        )

        # Nearest coin in this step
        new_closest = min(
            new_coins, key=lambda c: abs(c[0] - new_x) + abs(c[1] - new_y)
        )
        new_distance = abs(new_closest[0] - new_x) + abs(
            new_closest[1] - new_y
        )

        # Check if the agent got closer
        if new_distance < old_distance:
            reward_sum += 50  
        elif new_distance > old_distance:
            reward_sum -= 100

    # --- Incentive 2: avoid danger ---
    # TODO (still working on coin chasing)

    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
