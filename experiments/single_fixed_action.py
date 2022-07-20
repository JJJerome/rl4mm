import numpy as np

from RL4MM.agents.baseline_agents import FixedActionAgent

##############################################################################
# NEW TEST SWEEP
##############################################################################

DEFAULT_ALPHAS = [1]
DEFAULT_BETAS = [1]

def get_env_configs_and_agents(env_config:dict):
    agents = list()
    for default_alpha in DEFAULT_ALPHAS:
        for default_beta in DEFAULT_BETAS:
            agents.append(FixedActionAgent(np.array([default_alpha,
                                                     default_beta,
                                                     default_alpha,
                                                     default_beta])))
    env_config['per_step_reward_function'] = 'RS'
    return [env_config], agents
