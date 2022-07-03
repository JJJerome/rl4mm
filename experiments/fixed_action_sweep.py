import numpy as np

from RL4MM.agents.baseline_agents import FixedActionAgent, ContinuousTeradactyl

##############################################################################
# NEW TEST SWEEP
##############################################################################

default_alphas = [1,2,5,10]
default_betas = [1,2,5,10]

def get_env_configs_and_agents(env_config:dict):
    agents = list()
    for default_alpha in default_alphas:
        for default_beta in default_betas:
            agents.append(FixedActionAgent(np.array([default_alpha,
                                                     default_beta,
                                                     default_alpha,
                                                     default_beta])))
    return [env_config], agents
