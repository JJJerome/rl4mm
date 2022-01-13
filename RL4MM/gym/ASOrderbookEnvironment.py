import gym
import numpy as np

from copy import deepcopy
from gym.spaces import Box, Discrete, Tuple
from math import sqrt

from RL4MM.rewards.RewardFunctions import Action, RewardFunction, State, PnL


# Coefficients from the original Avellaneda-Stoikov paper. TODO: make the framework flexible enough to permit RARL.
DRIFT = 0.0
VOLATILITY = 2.0
RATE_OF_ARRIVAL = 140
FILL_EXPONENT = 1.5
MAX_INVENTORY = 10
INITIAL_CASH = 100
INITIAL_INVENTORY = 0
INITIAL_STOCK_PRICE = 100


class ASOrderbookEnvironment(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, episode_length: float = 1.0, n_steps: int = 200, reward_function: RewardFunction = PnL):
        super(ASOrderbookEnvironment, self).__init__()
        self.episode_length = episode_length
        self.n_steps = n_steps
        self.reward_function = reward_function

        self.action_space = Box(low=0.0, shape=(2, 1), dtype=np.float32)  # agent chooses spread on bid and ask
        # observation space is (stock price, cash, inventory, time)
        self.observation_space = Tuple(Box(low=0.0, shape=(2, 1)), Discrete(MAX_INVENTORY), Discrete(n_steps))
        self.state = (INITIAL_STOCK_PRICE, INITIAL_CASH, INITIAL_INVENTORY, 0.0)
        self.dt = self.episode_length / self.n_steps

    def reset(self):
        self.state = (INITIAL_STOCK_PRICE, INITIAL_CASH, INITIAL_INVENTORY, 0.0)

    def step(self, action: Action):
        next_state = self._get_next_state(action)
        reward = self.reward_function.calculate(self.state, action, next_state)
        self.state = next_state
        done = self.state[3] == self.episode_length
        return self.state, reward, done, {}

    def _get_next_state(self, action: Action) -> State:
        next_state = deepcopy(self.state)
        next_state[0] += DRIFT * self.dt + VOLATILITY * sqrt(self.dt) * np.random.normal()
        next_state[3] += self.dt
        fill_prob_bid, fill_prob_ask = RATE_OF_ARRIVAL * np.exp(tuple(-FILL_EXPONENT * a for a in action)) * self.dt
        unif_bid, unif_ask = np.random.random(2)
        if unif_bid > fill_prob_bid and unif_ask > fill_prob_ask:  # neither the agent's bid nor their ask is filled
            pass
        if unif_bid < fill_prob_bid and unif_ask > fill_prob_ask:  # only bid filled
            # Note that market order gets filled THEN asset midprice changes
            next_state[1] -= self.state[0] - action[0]
            next_state[2] += 1
        if unif_bid > fill_prob_bid and unif_ask < fill_prob_ask:  # only ask filled
            next_state[1] += self.state[0] + action[1]
            next_state[2] -= 1
        if unif_bid < fill_prob_bid and unif_ask < fill_prob_ask:  # both bid and ask filled
            next_state[1] += action[0] + action[1]
