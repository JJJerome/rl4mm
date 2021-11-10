from datetime import datetime, timedelta
from typing import List

import gym
import numpy as np
import pandas as pd

from gym.spaces import Discrete, Tuple
from gym.utils import seeding

from RL4MM.features.Feature import Feature
from RL4MM.simulator.OrderbookSimulator import OrderbookSimulator, ResultsDict


class OrderbookEnvironment(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        simulator: OrderbookSimulator,
        start_date: datetime,
        end_date: datetime,
        step_size: timedelta,
        features: List[Feature],
        num_steps: int = 10,
        initial_portfolio: dict = None,
    ):
        super(OrderbookEnvironment, self).__init__()
        self.simulator = simulator
        self.start_date = start_date
        self.end_date = end_date
        self.step_size = step_size
        self.features = features
        self.num_steps = num_steps
        self.initial_portfolio = initial_portfolio or {"cash": 1000, "stock": 0}
        self.portfolio = self.initial_portfolio
        self.now = start_date
        self.underlying_state = self.simulator.simulate_step(start_date=self.now, end_date=self.now)[2]

        # Actions can be (0,0), (0,1), (1,0), (1,1)
        self.action_space = Tuple(Discrete(2), Discrete(2))
        # Observation spaces are given by the features used
        self.observation_space = Tuple(Discrete(len(feature.feature_space)) for feature in self.features)
        self.reset()

    def seed(self, seed=42):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _take_action(self, action):
        pass

    def step(self, action):
        observation, reward, done, info = 1, 1, 1, 1
        return observation, reward, done, info

    def reset(self):
        self.portfolio = self.initial_portfolio
        max_step = int((self.end_date - self.start_date) / self.step_size) - self.num_steps
        self.now = self.start_date + np.random.randint(low=0, high=max_step) * self.step_size
        self.underlying_state = self.simulator.simulate_step(start_date=self.now, end_date=self.now)[2]
        return

    def get_features_from_underlying(self, results: ResultsDict):
        pass

    def get_imbalance(self):
        book = self.underlying_state["orderbook"]
        imbalance = book["bid_size_0"] / (book["bid_size_0"] + book["ask_size_0"])
        quantised_imbalance = 0
