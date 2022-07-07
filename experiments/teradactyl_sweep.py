min_quote_range = [0]  # [1, 0, -1]
max_quote_range = [10]  # [10, 15, 20]
max_inv_range = [500]  # [1000]
default_omega_range = [0.1,0.2,0.3]
default_kappa_range = [5, 10]

from RL4MM.agents.baseline_agents import ContinuousTeradactyl

max_inv = 500
max_kappa = 20


def get_env_configs_and_agents(env_config:dict):
    env_configs = [env_config]
    agents = list()

    for default_omega in default_omega_range:
        for default_kappa in default_kappa_range:
            agents.append(ContinuousTeradactyl(max_inventory=max_inv,
                                               default_kappa=default_kappa,
                                               default_omega=default_omega,
                                               max_kappa=default_kappa*3))

    return env_configs, agents