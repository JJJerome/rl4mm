import gym
import numpy as np

from RL4MM.agents.Agent import Agent


class RandomAgent(Agent):
    def __init__(self, env: gym.Env, seed: int = None):
        self.action_space = env.action_space
        self.action_space.seed(seed)

    def get_action(self, state: np.ndarray) -> np.ndarray:
        return self.action_space.sample()


class FixedActionAgent(Agent):
    def __init__(self, fixed_action: np.ndarray):
        self.fixed_action = fixed_action

    def get_action(self, state: np.ndarray) -> np.ndarray:
        return self.fixed_action


class HumanAgent(Agent):
    def get_action(self, state: np.ndarray):
        action_0 = float(input(f"Current state is {state}. How large do you want to set action[0]? "))
        action_1 = float(input(f"Current state is {state}. How large do you want to set action[1]? "))
        action_2 = float(input(f"Current state is {state}. How large do you want to set action[2]? "))
        action_3 = float(input(f"Current state is {state}. How large do you want to set action[3]? "))
        action_4 = float(input(f"Current state is {state}. How large do you want to set action[4]? "))
        return np.array([action_0, action_1, action_2, action_3, action_4])
