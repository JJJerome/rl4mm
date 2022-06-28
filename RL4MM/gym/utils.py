from typing import Dict

import gym
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
import concurrent.futures
import copy
from datetime import timedelta

from RL4MM.agents.Agent import Agent
from RL4MM.simulation.OrderbookSimulator import OrderbookSimulator
from RL4MM.gym.HistoricalOrderbookEnvironment import HistoricalOrderbookEnvironment
from RL4MM.utils.utils import get_date_time
from RL4MM.rewards.RewardFunctions import InventoryAdjustedPnL, PnL


def get_reward_function(reward_function: str, inventory_aversion: float = 0.1):
    if reward_function == "AD":  # asymmetrically dampened
        return InventoryAdjustedPnL(inventory_aversion=inventory_aversion, asymmetrically_dampened=True)
    elif reward_function == "SD":  # symmetrically dampened
        return InventoryAdjustedPnL(inventory_aversion=inventory_aversion, asymmetrically_dampened=False)
    elif reward_function == "PnL":
        return PnL()
    else:
        raise NotImplementedError("You must specify one of 'AS', 'SD' or 'PnL'")

def env_creator(env_config):
    obs = OrderbookSimulator(ticker=env_config["ticker"], n_levels=env_config["n_levels"])
    return HistoricalOrderbookEnvironment(
        ticker=env_config["ticker"],
        episode_length=timedelta(minutes=env_config["episode_length"]),
        simulator=obs,
        quote_levels=10,
        min_date=get_date_time(env_config["min_date"]),  # datetime
        max_date=get_date_time(env_config["max_date"]),  # datetime
        step_size=timedelta(seconds=env_config["step_size"]),
        initial_portfolio=env_config["initial_portfolio"],  #: dict = None
        per_step_reward_function=get_reward_function(env_config["per_step_reward_function"]),
        terminal_reward_function=get_reward_function(env_config["terminal_reward_function"]),
        market_order_clearing=env_config["market_order_clearing"],
    )


def generate_trajectory(agent: Agent, env: gym.Env):
    observations = []
    rewards = []
    actions = []
    infos = []
    obs = env.reset()  # type:ignore
    observations.append(obs)
    while True:
        action = agent.get_action(obs)  # type:ignore
        obs, reward, done, info = env.step(action)
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        infos.append(info)
        if done:
            break
    return observations, actions, rewards, infos


def get_episode_summary_dict(agent, env_config, n_iterations, PARALLEL_FLAG=True):

    if PARALLEL_FLAG:

        # create list of agents and environments for the 
        agent_lst = [copy.deepcopy(agent) for _ in range(n_iterations)]
        env_lst = [env_creator(env_config) for _ in range(n_iterations)] 
        ret = get_episode_summary_dict_PARALLEL(agent_lst, env_lst)

    else:
    
        ret = get_episode_summary_dict_NONPARALLEL(agent, env_creator(env_config), n_iterations)

    return ret


def get_episode_summary_dict_NONPARALLEL(agent: Agent, env: gym.Env, n_iterations: int = 100):
    episode_mean_dict: Dict = {"rewards": [], "actions": [], "inventory": [], "spread": []}
    for _ in tqdm(range(n_iterations), desc="Simulating trajectories"):
        _, actions, rewards, infos = generate_trajectory(agent=agent, env=env)
        episode_mean_dict["rewards"].append(np.mean(rewards))
        episode_mean_dict["actions"].append(np.mean(np.array(actions), axis=0)[:-1])
        episode_mean_dict["inventory"].append(np.mean([info["inventory"] for info in infos]))
        episode_mean_dict["spread"].append(np.mean([info["spread"] for info in infos]))
    return episode_mean_dict


def process_parallel_results(results):
    """ 

    results is a list of length n_iterations
   
    each element is list with elements:

    0: observations
    1: actions
    2: rewards
    3: infos

    infos is a tuple of dictionaries with keys and values e.g.,

    {'price': 2729950.0,
     'inventory': -76.0,
     'spread': 100.0,
     'bid_action': (array([3, 1]),),
     'ask_action': (array([3, 1]),),
     'market_order_action': (array([10]),)}])

    """

    episode_mean_dict: Dict = {"rewards": [], "actions": [], "inventory": [], "spread": []}

    for _, actions, rewards, infos in results:
        episode_mean_dict["rewards"].append(np.mean(rewards))
        episode_mean_dict["actions"].append(np.mean(np.array(actions), axis=0)[:-1])
        episode_mean_dict["inventory"].append(np.mean([info["inventory"] for info in infos]))
        episode_mean_dict["spread"].append(np.mean([info["spread"] for info in infos]))

    return episode_mean_dict

def get_episode_summary_dict_PARALLEL(agent_lst, env_lst):

    assert len(agent_lst) == len(env_lst)
    l = len(agent_lst) 
    print(f"In get_episode_summary_dict_PARALLEL, running for {l} iterations")

    with concurrent.futures.ThreadPoolExecutor() as executor:

        # futures = {executor.submit(f, arg): arg for arg in args_list}

        futures = [executor.submit(generate_trajectory, *arg) for arg in zip(agent_lst, env_lst)]
        results = []
       
        with tqdm(total=l) as pbar:
            for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())
                    pbar.update(1)

    episode_mean_dict = process_parallel_results(results)

    return episode_mean_dict

###############################################################################

def plot_reward_distributions_OLD(agent: Agent, env: gym.Env, n_iterations: int = 100):
    sns.set()
    episode_mean_dict = get_episode_summary_dict(agent, env, n_iterations)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 6))

    plt.suptitle(f"{env.ticker} {agent.get_name()}")

    ax1.hist(episode_mean_dict["rewards"], bins=20)
    for action_loc in [0, 1, 2, 3]:
        ax2.hist(np.array(episode_mean_dict["actions"])[action_loc, :], bins=5, label="action " + str(action_loc))
        ax2.legend()
    ax3.hist(episode_mean_dict["inventory"], bins=20)
    ax4.hist(episode_mean_dict["spread"], bins=20)
    ax1.title.set_text("Mean rewards")
    ax2.title.set_text("Mean action")
    ax3.title.set_text("Mean inventory")
    ax4.title.set_text("Mean spread")
    fig.tight_layout()
    # plt.show()

    fig.savefig(f'{get_output_prefix(agent,env)}.pdf')


###############################################################################

def get_output_prefix(ticker, min_date, max_date, agent_name):
    env_str = f'{ticker}_{min_date}_{max_date}'
    return agent_name + '_' + env_str

def plot_reward_distributions(ticker, min_date, max_date, agent_name, episode_mean_dict):
    sns.set()
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2, figsize=(10, 6))

    plt.suptitle(f"{ticker} {agent_name}")

    ax1.hist(episode_mean_dict["rewards"], bins=20)
    ax1.title.set_text("Mean rewards")

    for action_loc, ax in zip([0, 1, 2, 3],[ax3,ax4,ax5,ax6]):
        ax.hist(np.array(episode_mean_dict["actions"])[action_loc, :], bins=5, label="action " + str(action_loc))
        ax.legend()

    ax3.title.set_text("Mean action - bid 1")
    ax4.title.set_text("Mean action - bid 2")
    ax5.title.set_text("Mean action - ask 1")
    ax6.title.set_text("Mean action - ask 2")

    ax7.hist(episode_mean_dict["inventory"], bins=20)
    ax8.hist(episode_mean_dict["spread"], bins=20)
    ax7.title.set_text("Mean inventory")
    ax8.title.set_text("Mean spread")

    fig.tight_layout()
    # plt.show()

    fname = get_output_prefix(ticker, min_date, max_date, agent_name)

    fig.savefig(f'{fname}.pdf')
    plt.close(fig)
