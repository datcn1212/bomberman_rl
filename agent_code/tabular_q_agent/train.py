from collections import namedtuple

from typing import List
import yaml

import events as e
from .callbacks import state_to_features

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

OPP_MOVEMENTS = {
    'UP': 'DOWN',
    'DOWN': 'UP',
    'LEFT': 'RIGHT',
    'RIGHT': 'LEFT',
    'WAIT': 'WAIT',
}

def setup_training(self):
    """
    Initialise self for training purpose.
    """

    with open('config.yaml', 'r') as file:
        cfg = yaml.safe_load(file)
    
    self.q_file = cfg["save_q_table"]
    
    self.round_rewards = 0
    self.reward_history = []
    self.last_action = None

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
    rewards = reward_from_events(self, self_action, old_game_state, new_game_state, events)
    transition = Transition(state_to_features(old_game_state), 
        self_action, 
        state_to_features(new_game_state), 
        rewards
    )
    self.round_rewards += rewards
    # self.transitions.append(transition)
    self.model.update(transition)
    self.last_action = self_action


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    rewards = reward_from_events(self, last_action, last_game_state, last_game_state, events)
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
    self.last_action = None

    self.model.save(self.q_file)


def reward_from_events(self, self_action: str, old_game_state: dict, new_game_state: dict, events: List[str]) -> int:
    """
    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 200,  
        e.KILLED_OPPONENT: 50,
        e.KILLED_SELF: -200,  
        e.GOT_KILLED: -50, 
        e.INVALID_ACTION: -100,  
        e.BOMB_DROPPED: -10, 
        e.CRATE_DESTROYED: 200
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
        else:
            reward_sum -= 2

    old_x, old_y = old_game_state["self"][3]
    new_x, new_y = new_game_state["self"][3]

    old_coins = old_game_state["coins"]
    new_coins = new_game_state["coins"]

    field = old_game_state["field"]
    bombs = old_game_state["bombs"]

    # --- Incentive 1: go near the coins ---
    # If there are coins in the map and we haven't collected one in this turn...
    if old_coins and e.COIN_COLLECTED not in events:
        # Nearest coin from previous step
        sorted_old = sorted(old_coins, key=lambda c: (c[0], c[1]))
        old_closest = min(
            sorted_old, key=lambda c: abs(c[0] - old_x) + abs(c[1] - old_y)
        )
        old_distance = abs(old_closest[0] - old_x) + abs(
            old_closest[1] - old_y
        )

        # Nearest coin in this step
        sorted_new = sorted(new_coins, key=lambda c: (c[0], c[1]))
        new_closest = min(
            sorted_new, key=lambda c: abs(c[0] - new_x) + abs(c[1] - new_y)
        )
        new_distance = abs(new_closest[0] - new_x) + abs(
            new_closest[1] - new_y
        )

        # Check if the agent got closer
        if new_distance < old_distance:
            reward_sum += 50  
        elif new_distance > old_distance:
            reward_sum -= 10

    if bombs or e.BOMB_DROPPED in events:
        all_bombs = list(bombs)
        if e.BOMB_DROPPED in events and not any(b[0] == (old_x, old_y) for b in all_bombs):
            all_bombs.append(((old_x, old_y), 4)) # Simulamos nuestra bomba con timer alto

        old_in_danger = is_position_in_danger(old_x, old_y, all_bombs, field)
        new_in_danger = is_position_in_danger(new_x, new_y, all_bombs, field)

        if old_in_danger and not new_in_danger:
            reward_sum += 80  
        elif old_in_danger and new_in_danger:
            reward_sum -= 10
            
    # --- Incentive 3: avoid loops ---
    if self.last_action is not None:
        if self.last_action == OPP_MOVEMENTS.get(self_action):
            reward_sum -= 10

    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def is_position_in_danger(x, y, bombs, field) -> bool:
    for (bx, by), timer in bombs:
        if bx == x and abs(by - y) <= 3:
            step = 1 if by > y else -1
            if all(field[x, check_y] == 0 for check_y in range(y + step, by, step)):
                return True
        elif by == y and abs(bx - x) <= 3:
            step = 1 if bx > x else -1
            if all(field[check_x, y] == 0 for check_x in range(x + step, bx, step)):
                return True
    return False