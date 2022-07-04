import numpy as np
from copy import deepcopy

from RL4MM.agents.baseline_agents import FixedActionAgent, ContinuousTeradactyl

a_range = [1.01,2,5]
b_range = [1.01,2,5]

max_inv = 1000
max_kappa = 20

def get_env_configs_and_agents(env_config:dict):

    env_configs = [env_config]
    agents = list()

    for a in a_range:
        for b in b_range:

            if a == 1 and b == 1:
                continue

            omega = ContinuousTeradactyl.calculate_omega(a, b)
            kappa = ContinuousTeradactyl.calculate_kappa(a, b)

            agents.append(ContinuousTeradactyl(max_inventory=max_inv, 
                                               default_kappa=omega,
                                               default_omega=kappa,
                                               max_kappa=max_kappa))

    return env_configs, agents
