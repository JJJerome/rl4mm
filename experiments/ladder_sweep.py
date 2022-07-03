from copy import deepcopy

import numpy as np

from RL4MM.agents.baseline_agents import FixedActionAgent

INNER_QUOTES = [-2, -1, 0, 1, 2, 3]
N_LEVELS = 10

def get_env_configs_and_agents(env_config:dict):
    env_configs = list()
    agents = [FixedActionAgent(np.array([1, 1, 1, 1]))]
    for inner in INNER_QUOTES:
        env_conf = deepcopy(env_config)
        env_conf["min_quote_level"] = inner
        env_conf["max_quote_level"] = inner + N_LEVELS
        env_configs.append(env_conf)
    return env_configs, agents
