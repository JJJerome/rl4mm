from rl4mm.utils.rllib.custom_policy import CustomPolicy
from ray.rllib.agents.trainer import Trainer
from rl4mm.agents.baseline_agents import RandomAgent, FixedActionAgent, Teradactyl
import numpy as np


class FixedActionAgentPolicy(CustomPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent = FixedActionAgent(
            np.array(
                [
                    self.config["model"]["custom_model_config"]["a_1"],
                    self.config["model"]["custom_model_config"]["a_2"],
                    self.config["model"]["custom_model_config"]["b_1"],
                    self.config["model"]["custom_model_config"]["b_2"],
                    self.config["model"]["custom_model_config"]["threshold"],
                ]
            )
        )


class FixedActionAgentWrapper(Trainer):
    def get_default_policy_class(self, config):
        return FixedActionAgentPolicy


# --------------------------------------------------------------------------------------------------


class RandomAgentPolicy(CustomPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent = RandomAgent()


class RandomAgentWrapper(Trainer):
    def get_default_policy_class(self, config):
        return RandomAgentPolicy


# --------------------------------------------------------------------------------------------------


class TeradactylPolicy(CustomPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent = Teradactyl(
            default_kappa=self.config["model"]["custom_model_config"]["default_kappa"],
            default_omega=self.config["model"]["custom_model_config"]["default_omega"],
            max_kappa=self.config["model"]["custom_model_config"]["max_kappa"],
            max_inventory=self.config["model"]["custom_model_config"]["max_inventory"],
            exponent=self.config["model"]["custom_model_config"]["exponent"],
        )


class TeradactylWrapper(Trainer):
    def get_default_policy_class(self, config):
        return TeradactylPolicy
