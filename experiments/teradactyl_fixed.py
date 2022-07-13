min_quote_range = [0]  # [1, 0, -1]
max_quote_range = [10]  # [10, 15, 20]
#max_inv_range = [1000]  # [1000]
default_omega = 0.480
default_kappa = 7.494
max_kappa = 63.879
exponent = 3.927

from RL4MM.agents.baseline_agents import ContinuousTeradactyl

max_inv = 1000
#max_kappa = 20


def get_env_configs_and_agents(env_config:dict):
    env_configs = [env_config]
    agents = list()

    #for default_omega in default_omega_range:
    #    for default_kappa in default_kappa_range:
    agents.append(ContinuousTeradactyl(max_inventory=max_inv,
                                       default_kappa=default_kappa,
                                       default_omega=default_omega,
                                       max_kappa=max_kappa,
                                       exponent=exponent 
                                       ))

    return env_configs, agents
