import argparse
import os
import copy
import numpy as np
import importlib

from RL4MM.database.HistoricalDatabase import HistoricalDatabase

from RL4MM.gym.utils import plot_reward_distributions, get_episode_summary_dict
from RL4MM.utils.utils import save_best_checkpoint_path, get_timedelta_from_clock_time

from main_helper import add_env_args

experiment_list = [
    "ladder_sweep",
    "fixed_action_sweep",
    "fixed_action_vs_teradactyl",
    "teradactyl_sweep",
    "teradactyl_sweep_small",
    "teradactyl_fixed",
]

def get_configs(args):
    # ray.init()
    env_config = {
        "ticker": args["ticker"],
        "min_date": args["min_date"],
        "max_date": args["max_date"],
        "min_start_timedelta": get_timedelta_from_clock_time(args["min_start_time"]),
        "max_end_timedelta": get_timedelta_from_clock_time(args["min_start_time"]),
        "step_size": args["step_size"],
        "episode_length": args["episode_length"],
        "n_levels": args["n_levels"],
        "features": args["features"],
        "max_inventory": args["max_inventory"],
        "normalisation_on": args["normalisation_on"],
        "initial_portfolio": args["initial_portfolio"],
        "per_step_reward_function": args["per_step_reward_function"],
        "terminal_reward_function": args["terminal_reward_function"],
        "market_order_clearing": args["market_order_clearing"],
        "market_order_fraction_of_inventory": 0.0,
        "min_quote_level": args["min_quote_level"],
        "max_quote_level": args["max_quote_level"],
        "enter_spread": args["enter_spread"],
        "concentration": args["concentration"],
        "features": args["features"],
        "normalisation_on": args["normalisation_on"],
        "max_inventory": args["max_inventory"],
    }

    eval_env_config = copy.deepcopy(env_config)
    eval_env_config["min_date"] = args["min_date_eval"]
    eval_env_config["max_date"] = args["max_date_eval"]
    eval_env_config["per_step_reward_function"] = args["eval_per_step_reward_function"]
    eval_env_config["terminal_reward_function"] = args["terminal_reward_function"]

    return env_config, eval_env_config



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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser = add_env_args(parser)
    args = vars(parser.parse_args())

    # if args["concentration"] == 0:
        # args["concentration"] = None

    env_config, _ = get_configs(args)

    experiment = args["experiment"]
    module = importlib.import_module(f"experiments." + args["experiment"])
    get_env_configs_and_agents = getattr(module, "get_env_configs_and_agents")
    env_configs, agents = get_env_configs_and_agents(env_config)

    if args["multiple_databases"]:
        databases = [HistoricalDatabase() for _ in range(args["n_iterations"])]
    else:
        database = HistoricalDatabase()
        databases = [database for _ in range(args["n_iterations"])]

    for agent in agents:
        for env_config in env_configs:
            emd1 = get_episode_summary_dict(
                agent, env_config, args["n_iterations"], PARALLEL_FLAG=args["parallel"], databases=databases
            )
            plot_reward_distributions_wrapper(env_config, agent, emd1, experiment)
