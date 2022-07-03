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

for default_alpha in [1,2,5,10]:
    for default_beta in [1,2,5,10]:
        for max_inventory in [1000,5000]:
            agents.append(FixedActionAgent(np.array([default_alpha,
                                                     default_beta,
                                                     default_alpha,
                                                     default_beta, 
                                                     max_inventory])))
