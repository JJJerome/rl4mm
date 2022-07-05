min_quote_range = [0]  # [1, 0, -1]
max_quote_range = [10]  # [10, 15, 20]
default_omega_range = [0.1, 0.3]  # to reduce. Choose one
default_kappa_range = [6]  # maybe to reduce. Choose one

from RL4MM.agents.baseline_agents import ContinuousTeradactyl


kappa_scaling_range = [1,5,10]
max_inv_range = [200,1000]


def get_env_configs_and_agents(env_config:dict):
    env_configs = [env_config]
    agents = list()

    for default_omega in default_omega_range:
        for default_kappa in default_kappa_range:
            for kappa_scaling in kappa_scaling_range:
                for max_inv in max_inv_range:
                    agents.append(ContinuousTeradactyl(max_inventory=max_inv,
                                                       default_kappa=default_kappa,
                                                       default_omega=default_omega,
                                                       max_kappa=default_kappa*kappa_scaling))

    return env_configs, agents


