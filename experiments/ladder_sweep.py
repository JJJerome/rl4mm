from copy import deepcopy

import numpy as np

from RL4MM.agents.baseline_agents import FixedActionAgent, ContinuousTeradactyl

# max_inventory = 10000
inner_quotes = [-2,-1,0,1,2,3]
n_levels = 10

def get_env_configs_and_agents(env_config:dict):
    env_configs = list()
    agents = [FixedActionAgent(np.array([1, 1, 1, 1]))]
    env_conf = deepcopy(env_config)
    for inner in inner_quotes:
        env_conf["min_quote_level"] = inner
        env_conf["max_quote_level"] = inner + n_levels
        env_configs.append(env_conf)
    return env_configs, agents
