from RL4MM.agents.Agent import Agent

from stable_baselines3.common.base_class import BaseAlgorithm

from RL4MM.base import State, Action


class SBAgent(Agent):
    def __init__(self, model: BaseAlgorithm):
        self.model = model

    def get_action(self, state: State) -> Action:
        return self.model.predict(state)[0]

    def train(self, total_timesteps: int = 100000):
        self.model.learn(total_timesteps=total_timesteps)
