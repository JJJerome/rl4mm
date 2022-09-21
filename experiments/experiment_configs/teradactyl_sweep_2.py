min_quote_range = [0]  # [1, 0, -1]
max_quote_range = [10]  # [10, 15, 20]
default_omega_range = [0.2, 0.3]  # to reduce. Choose one
default_kappa_range = [6]  # maybe to reduce. Choose one

from rl4mm.agents.baseline_agents import Teradactyl


kappa_scaling_range = [10]
max_inv_range = [300]
exponent_range = [1.5, 2]


def get_env_configs_and_agents(env_config: dict):
    env_configs = [env_config]
    agents = list()

    for default_omega in default_omega_range:
        for default_kappa in default_kappa_range:
            for kappa_scaling in kappa_scaling_range:
                for max_inv in max_inv_range:
                    for exponent in exponent_range:
                        agents.append(
                            Teradactyl(
                                max_inventory=max_inv,
                                default_kappa=default_kappa,
                                default_omega=default_omega,
                                max_kappa=default_kappa * kappa_scaling,
                                exponent=exponent,
                            ),
                        )
    return env_configs, agents
