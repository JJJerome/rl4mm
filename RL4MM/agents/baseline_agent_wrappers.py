from RL4MM.agents.custom_policy import CostumPolicy
from ray.rllib.agents.trainer import Trainer
from RL4MM.agents.baseline_agents import (
    RandomAgent, 
    FixedActionAgent, 
    TeradactylAgent, 
    ContinuousTeradactyl
    )
import numpy as np

class FixedActionAgentPolicy(CostumPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent = FixedActionAgent(
            np.array([
                self.config['model']["custom_model_config"]['a_1'],
                self.config['model']["custom_model_config"]['a_2'],
                self.config['model']["custom_model_config"]['b_1'],
                self.config['model']["custom_model_config"]['b_2'],
                self.config['model']["custom_model_config"]['threshold'],
            ])
        )

class FixedActionAgentWrapper(Trainer):
    def get_default_policy_class(self, config):
        return FixedActionAgentPolicy

#--------------------------------------------------------------------------------------------------

class RandomAgentPolicy(CostumPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent = RandomAgent()

class RandomAgentWrapper(Trainer):
    def get_default_policy_class(self, config):
        return RandomAgentPolicy

#--------------------------------------------------------------------------------------------------

class TeradactylAgentPolicy(CostumPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent =  TeradactylAgent(
            kappa=self.config['model']["custom_model_config"]['kappa'], 
            default_a=self.config['model']["custom_model_config"]['default_a'], 
            default_b=self.config['model']["custom_model_config"]['default_b'],
            max_inventory=self.config['model']["custom_model_config"]['max_inventory'],
            )

class TeradactylAgentWrapper(Trainer):
    def get_default_policy_class(self, config):
        return TeradactylAgentPolicy

#--------------------------------------------------------------------------------------------------

class ContinuousTeradactylPolicy(CostumPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent = ContinuousTeradactyl(
            default_kappa=self.config['model']["custom_model_config"]['default_kappa'], 
            default_omega=self.config['model']["custom_model_config"]['default_omega'], 
            max_kappa=self.config['model']["custom_model_config"]['max_kappa'],
            max_inventory=self.config['model']["custom_model_config"]['max_inventory'],
            )

class ContinuousTeradactylWrapper(Trainer):
    def get_default_policy_class(self, config):
        return ContinuousTeradactylPolicy
