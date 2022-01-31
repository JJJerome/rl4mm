import gym

import numpy as np

from RL4MM.agents.Agent import Agent
from RL4MM.base import State, Action


class RandomAgent(Agent):
    def __init__(self, env: gym.Env):
        self.env = env

    def get_action(self, state: State) -> Action:
        return self.env.action_space.sample()


class FixedSpreadAgent(Agent):
    def __init__(self, half_spread: float = 1.0, offset: float = 0.0):
        self.half_spread = half_spread
        self.offset = offset

    def get_action(self, state: State):
        return np.array([self.half_spread - self.offset, self.half_spread + self.offset])
