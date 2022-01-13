import gym
from gym.spaces import Box, Discrete, Tuple

from math import sqrt

import numpy as np

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

    def __init__(self, episode_length: float = 1.0, n_steps: int = 200):
        super(ASOrderbookEnvironment, self).__init__()
        self.episode_length = episode_length
        self.n_steps = n_steps

        self.action_space = Box(low=0.0, shape=(2, 1), dtype=np.float32)  # agent chooses spread on bid and ask
        # observation space is (stock price, cash, inventory, time)
        self.observation_space = Tuple(Box(low=0.0, shape=(2, 1)), Discrete(MAX_INVENTORY), Discrete(n_steps))
        self.initial_cash = self.current_cash = INITIAL_CASH
        self.initial_inventory = self.current_inventory = INITIAL_INVENTORY
        self.initial_stock_price = self.stock_price = INITIAL_STOCK_PRICE
        self.step = 0.0
        self.dt = self.episode_length / self.n_steps

    def reset(self):
        self.current_cash = self.initial_cash
        self.current_inventory = self.initial_inventory
        self.stock_price = self.initial_stock_price
        self.step = 0.0

    def step(self, action):
        noise = np.random.normal()
        self.stock_price += DRIFT * self.dt + VOLATILITY * sqrt(self.dt) * noise
        fill_prob_bid, fill_prob_ask = RATE_OF_ARRIVAL * np.exp(tuple(-FILL_EXPONENT * a for a in action)) * self.dt
        unif_bid, unif_ask = np.random.random(2)
        if unif_bid > fill_prob_bid and unif_ask > fill_prob_ask:  # neither the agent's bid or their ask is filled
            pass
        if unif_bid < fill_prob_bid and unif_ask > fill_prob_ask:  # only bid filled
            self.current_cash -= self.stock_price - action[0]
            self.current_inventory += 1
        if unif_bid > fill_prob_bid and unif_ask < fill_prob_ask:  # only ask filled
            self.current_cash += self.stock_price + action[1]
            self.current_inventory -= 1
        if unif_bid < fill_prob_bid and unif_ask < fill_prob_ask:  # both bid and ask filled
            self.current_cash += action[0]+action[1]
