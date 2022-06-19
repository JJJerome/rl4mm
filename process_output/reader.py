from ray.rllib.offline.json_reader import JsonReader
from ray.rllib.offline.wis_estimator import WeightedImportanceSamplingEstimator

import pandas as pd
import matplotlib.pyplot as plt

reader = JsonReader("./output-2022-06-17_01-32-59_worker-1_0.json")

dict_lst = []

for _ in range(1000):
    batch = reader.next()
    for episode in batch.split_by_episode():
        for o, a, o_, r in  zip(episode['obs'],episode['actions'], episode['new_obs'], episode['rewards']):

            # print([o, a, o_, r])

            d = dict()

            for i,obs in enumerate(o):
                d['obs%d' % i] = obs

            for i,act in enumerate(a):
                d['act%d' % i] = act

            # d = dict(obs=o,actions=a,new_obs=o_,rewards=r)  

            dict_lst.append(d) 

###############################################################################
# Uniform 
# ACTION_1 = np.array([1, 1, 1, 1])
###############################################################################
# More orders towards best prices
# ACTION_2 = np.array([1, 2, 1, 2])
###############################################################################

df = pd.DataFrame.from_dict(dict_lst)

# observations:
# [Spread(), MidpriceMove(), Volatility(), Inventory(), TimeRemaining(), MicroPrice()]

obs_names = ['spread', 'midprice_move', 'volatility', 'inventory', 'time_remaining', 'micro_price']

# actions:
# check order of bid and ask betas
# bid_beta1 and ask_beta2 go negative -- why??
action_names = ['bid_beta1', 'bid_beta2', 'ask_beta1', 'ask_beta2', 'clear_inventory']

# tmp = df.iloc[0:100]

tmp = df.iloc[0:5000]

tmp.columns = obs_names + action_names

###############################################################################
# Plotting
###############################################################################

fig, (ax1,ax2) = plt.subplots(1,2)

scatter1 = ax1.scatter(tmp.bid_beta1, tmp.bid_beta2, c=tmp.inventory)
ax1.set_xlabel('bid_beta1')
ax1.set_ylabel('bid_beta2')

legend1 = ax1.legend(*scatter1.legend_elements(num=5),
                    loc="upper right", title="Inventory")
ax1.add_artist(legend1)

scatter2 = ax2.scatter(tmp.ask_beta1, tmp.ask_beta2, c=tmp.inventory)
ax2.set_xlabel('ask_beta1')
ax2.set_ylabel('ask_beta2')

legend2 = ax2.legend(*scatter2.legend_elements(num=5),
                    loc="upper right", title="Inventory")
ax2.add_artist(legend2)

plt.show()
