from typing import Dict, List

import gym
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
import concurrent.futures
import copy
from datetime import timedelta
import json
from numpyencoder import NumpyEncoder


from RL4MM.agents.Agent import Agent
from RL4MM.database.HistoricalDatabase import HistoricalDatabase
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


def env_creator(env_config, database: HistoricalDatabase = HistoricalDatabase()):

    episode_length = timedelta(minutes=env_config["episode_length"])

    orderbook_simulator = OrderbookSimulator(
        ticker=env_config["ticker"], n_levels=env_config["n_levels"], episode_length=episode_length, database=database
    )

    return HistoricalOrderbookEnvironment(
        ticker=env_config["ticker"],
        episode_length=episode_length,
        simulator=orderbook_simulator,
        min_date=get_date_time(env_config["min_date"]),  # datetime
        max_date=get_date_time(env_config["max_date"]),  # datetime
        step_size=timedelta(seconds=env_config["step_size"]),
        initial_portfolio=env_config["initial_portfolio"],  #: dict = None
        per_step_reward_function=get_reward_function(env_config["per_step_reward_function"]),
        terminal_reward_function=get_reward_function(env_config["terminal_reward_function"]),
        market_order_clearing=env_config["market_order_clearing"],
        market_order_fraction_of_inventory=env_config["market_order_fraction_of_inventory"],
        min_quote_level=env_config["min_quote_level"],
        max_quote_level=env_config["max_quote_level"],
        enter_spread=env_config["enter_spread"],
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
    return {"observations": observations, "actions": actions, "rewards": rewards, "infos": infos}


def get_episode_summary_dict(
    agent, env_config, n_iterations, PARALLEL_FLAG=True, databases: List[HistoricalDatabase] = None
):

    if databases is None:
        databases = [HistoricalDatabase() for _ in range(n_iterations)]

    if PARALLEL_FLAG:
        # create list of agents and environments for the
        agent_lst = [copy.deepcopy(agent) for _ in range(n_iterations)]
        env_lst = [env_creator(env_config, databases[i]) for i in range(n_iterations)]
        ret = get_episode_summary_dict_PARALLEL(agent_lst, env_lst)

    else:

        ret = get_episode_summary_dict_NONPARALLEL(agent, env_creator(env_config, databases[0]), n_iterations)

    return ret


def get_episode_summary_dict_NONPARALLEL(agent: Agent, env: gym.Env, n_iterations: int = 100):
    episode_mean_dict: Dict = {"equity_curves": [], "rewards": [], "actions": [], "inventory": [], "spread": []}
    for _ in tqdm(range(n_iterations), desc="Simulating trajectories"):
        d = generate_trajectory(agent=agent, env=env)
        episode_mean_dict["equity_curves"].append(d["rewards"])
        episode_mean_dict["rewards"].append(np.mean(d["rewards"]))
        episode_mean_dict["actions"].append(np.mean(np.array(d["actions"]), axis=0)[:-1])
        episode_mean_dict["inventory"].append(np.mean([info["inventory"] for info in d["infos"]]))
        episode_mean_dict["spread"].append(np.mean([info["market_spread"] for info in d["infos"]]))
    return episode_mean_dict


def process_parallel_results(results):
    """

    results is a list of length n_iterations

    each element is dictionary with keys:

    observations
    actions
    rewards
    infos

    infos is a tuple of dictionaries with keys and values e.g.,

    {'asset_price': 2729950.0,
     'inventory': -76.0,
     'market_spread': 100.0,
     'agent_weighted_spread':,
     'midprice_offset':,
     'bid_action': (array([3, 1]),),
     'ask_action': (array([3, 1]),),
     'market_order_count': 0,
     'market_order_total_volume'
     }


    """

    episode_mean_dict: Dict = {"equity_curves": [], "rewards": [], "actions": [], "inventory": [], "spread": []}

    for d in results:
        episode_mean_dict["equity_curves"].append(d["rewards"])
        episode_mean_dict["rewards"].append(np.mean(d["rewards"]))
        episode_mean_dict["actions"].append(np.mean(np.array(d["actions"]), axis=0)[:-1])
        episode_mean_dict["inventory"].append(np.mean([info["inventory"] for info in d["infos"]]))
        episode_mean_dict["spread"].append(np.mean([info["market_spread"] for info in d["infos"]]))

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
    # episode_mean_dict = None

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

    fig.savefig(f"{get_output_prefix(agent,env)}.pdf")


###############################################################################


def get_output_prefix(
    ticker, min_date, max_date, agent_name, episode_length, min_quote_level, max_quote_level, enter_spread
):
    env_str = f"{ticker}_{min_date}_{max_date}_el_{episode_length}_minq_{min_quote_level}_maxq_{max_quote_level}_es_{enter_spread}"
    return agent_name + "_" + env_str


def plot_reward_distributions(
    ticker,
    min_date,
    max_date,
    agent_name,
    episode_length,
    step_size,
    market_order_clearing,
    market_order_fraction_of_inventory,
    min_quote_level,
    max_quote_level,
    enter_spread,
    episode_mean_dict,
):
    sns.set()

    fig = plt.figure(constrained_layout=True, figsize=(12, 6))
    ax_dict = fig.subplot_mosaic(
        """
        AA
        AA
        BC
        DE
        FG
        HI
        """
    )

    plt.suptitle(
        f"{ticker} {agent_name} EL: {episode_length} SS: {step_size} mind: {min_date} maxd: {max_date} "
        + f"moc: {market_order_clearing} mofi: {market_order_fraction_of_inventory} \n minq: {min_quote_level} "
        + f"maxq: {max_quote_level} ES: {enter_spread}"
    )

    ###########################################################################
    # Plot equity curves
    ###########################################################################

    tmp = episode_mean_dict["equity_curves"]
    df = pd.DataFrame(tmp).transpose()
    df.cumsum().plot(ax=ax_dict["A"])
    ax_dict["A"].get_legend().remove()

    ###########################################################################
    # Plot rewards histogram
    ###########################################################################

    ax_dict["B"].hist(episode_mean_dict["rewards"], bins=20)
    ax_dict["B"].title.set_text("Mean rewards")

    ###########################################################################
    # Plot rewards summary table
    ###########################################################################

    rewards = episode_mean_dict["rewards"]
    df = pd.DataFrame(rewards).describe()
    df = np.round(df)
    df = df.astype(int)

    table = ax_dict["C"].table(
        cellText=df.values,
        rowLabels=df.index,
        # colLabels=df.columns,
        loc="center",
    )

    table.set_fontsize(6.5)
    table.scale(0.5, 1.1)

    ax_dict["C"].set_axis_off()

    ###########################################################################
    # Plot actions
    ###########################################################################

    for action_loc, ax in zip([0, 1, 2, 3], [ax_dict[p] for p in ["D", "E", "F", "G"]]):
        ax.hist(np.array(episode_mean_dict["actions"])[action_loc, :], bins=5, label="action " + str(action_loc))
        ax.legend()

    ax_dict["D"].title.set_text("Mean action - bid 1")
    ax_dict["E"].title.set_text("Mean action - bid 2")
    ax_dict["F"].title.set_text("Mean action - ask 1")
    ax_dict["G"].title.set_text("Mean action - ask 2")

    ###########################################################################
    # Plot inventory and Spread
    ###########################################################################

    ax_dict["H"].hist(episode_mean_dict["inventory"], bins=20)
    ax_dict["I"].hist(episode_mean_dict["spread"], bins=20)
    ax_dict["H"].title.set_text("Mean inventory")
    ax_dict["I"].title.set_text("Mean spread")

    ###########################################################################
    # Write output
    ###########################################################################

    # fig.tight_layout()
    # plt.show()

    fname = get_output_prefix(
        ticker, min_date, max_date, agent_name, episode_length, min_quote_level, max_quote_level, enter_spread
    )

    # Write plot to pdf
    fig.savefig(f"{fname}.pdf")
    plt.close(fig)

    # Write data to json
    with open(f"{fname}.json", "w") as outfile:
        json.dump(episode_mean_dict, outfile, cls=NumpyEncoder)

    # return rewards
