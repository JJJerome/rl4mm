import numpy as np

from RL4MM.agents.baseline_agents import FixedActionAgent, ContinuousTeradactyl

agents = list()

##############################################################################
# THIS WAS THE OLD SWEEP
##############################################################################

# for default_alpha in [1,2,5,10]:
    # for default_beta in [1,2,5,10]:
        # for max_inventory in [100,500,1000]:
            # agents.append(FixedActionAgent(np.array([default_alpha,
                                                     # default_beta,
                                                     # default_alpha,
                                                     # default_beta, 
                                                     # max_inventory])))

##############################################################################
# NEW TEST SWEEP
##############################################################################

default_alphas = [1,2,5,10]
default_betas = [1,2,5,10]
max_inventories = [100,1000,10000]

def get_env_configs_and_agents(env_config:dict):
    for default_alpha in default_alphas:
        for default_beta in default_betas:
            for max_inventory in max_inventories:
                agents.append(FixedActionAgent(np.array([default_alpha,
                                                         default_beta,
                                                         default_alpha,
                                                         default_beta,
                                                         max_inventory])))
    return [env_config], agents
