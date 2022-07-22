import argparse
import os
import copy
import numpy as np
import importlib

from RL4MM.database.HistoricalDatabase import HistoricalDatabase

from RL4MM.gym.utils import plot_reward_distributions, get_episode_summary_dict
# from RL4MM.utils.utils import save_best_checkpoint_path, get_timedelta_from_clock_time

from main_helper import add_env_args, get_env_configs

from RL4MM.gym.order_tracking.InfoCalculators import SimpleInfoCalculator

experiment_list = [
    "ladder_sweep",
    "fixed_action_sweep",
    "fixed_action_vs_teradactyl",
    "teradactyl_sweep",
    "teradactyl_sweep_small",
    "teradactyl_fixed",
]

def plot_reward_distributions_wrapper(env_config, 
                                      agent, 
                                      episode_summary_dict,
                                      experiment):

    plot_reward_distributions(
        ticker=env_config["ticker"],
        min_date=env_config["min_date"],
        max_date=env_config["max_date"],
        agent_name=agent.get_name(),
        episode_length=env_config["episode_length"],
        step_size=env_config["step_size"],
        market_order_clearing=env_config["market_order_clearing"],
        min_quote_level=env_config["min_quote_level"],
        max_quote_level=env_config["max_quote_level"],
        enter_spread=env_config["enter_spread"],
        episode_summary_dict=emd1,
        output_dir=args["output"],
        experiment_name=experiment,
    )

def add_rule_based_main_args(parser):
    """
    These args are only currently used in this script
    """
    parser.add_argument("-nt", "--n_trajectories", default=10, help="Number of trajectories to use.", type=int)
    parser.add_argument("-ex", "--experiment", default="fixed_action_sweep", help="The experiment to run.", type=str)
    parser.add_argument("-par", "--parallel", action="store_true", default=False, help="Run in parallel or not.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")

    # add env args
    add_env_args(parser)
    # add args specific to this script
    add_rule_based_main_args(parser)

    args = vars(parser.parse_args())

    env_config, _ = get_env_configs(args)

    #######################################################################
    # Add in info calculator (off by default for speed)
    #######################################################################
    tmp = SimpleInfoCalculator(market_order_fraction_of_inventory=0, 
                               enter_spread=args["enter_spread"], 
                               concentration=args["concentration"]),
    env_config["info_calculator"] = tmp

    experiment = args["experiment"]
    module = importlib.import_module(f"experiments." + args["experiment"])
    get_env_configs_and_agents = getattr(module, "get_env_configs_and_agents")
    env_configs, agents = get_env_configs_and_agents(env_config)

    if args["multiple_databases"]:
        databases = [HistoricalDatabase() for _ in range(args["n_trajectories"])]
    else:
        database = HistoricalDatabase()
        databases = [database for _ in range(args["n_trajectories"])]

    for agent in agents:
        for env_config in env_configs:
            emd1 = get_episode_summary_dict(
                agent, env_config, args["n_trajectories"], PARALLEL_FLAG=args["parallel"], databases=databases
            )
            plot_reward_distributions_wrapper(env_config, agent, emd1, experiment)
