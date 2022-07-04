import numpy as np

from RL4MM.agents.baseline_agents import FixedActionAgent

##############################################################################
# NEW TEST SWEEP
##############################################################################

DEFAULT_ALPHAS = [1, 2, 5, 10]
DEFAULT_BETAS = [1, 2, 5, 10]


def get_env_configs_and_agents(env_config:dict):
    agents = list()
    for default_alpha in DEFAULT_ALPHAS:
        for default_beta in DEFAULT_BETAS:
            if default_alpha == 1 and default_beta == 1:
                print("Skipping alpha=beta=1 parameter combination as it is a ladder strategy")
                continue
            agents.append(FixedActionAgent(np.array([default_alpha,
                                                     default_beta,
                                                     default_alpha,
                                                     default_beta])))
    return [env_config], agents
