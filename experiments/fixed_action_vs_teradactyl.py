import numpy as np

from rl4mm.agents.baseline_agents import FixedActionAgent, ContinuousTeradactyl

max_inventory = 1000

default_alpha = 2
default_beta = 5
default_omega = ContinuousTeradactyl.calculate_omega(default_alpha,default_beta)
default_kappa = ContinuousTeradactyl.calculate_kappa(default_alpha,default_beta)

agents = list()

agents.append(FixedActionAgent(np.array([default_alpha,default_beta,default_alpha,default_beta, max_inventory])))
for default_kappa in [7,20]:
    for default_omega in [0.2,0.4]:
        for max_kappa in [40, 100]:
            agents.append(ContinuousTeradactyl(max_inventory, default_kappa, default_omega, max_kappa))

