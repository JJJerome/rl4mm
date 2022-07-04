import argparse
import os
import copy
import numpy as np
import importlib

from RL4MM.database.HistoricalDatabase import HistoricalDatabase
from RL4MM.database.PostgresEngine import MAX_POOL_SIZE
from RL4MM.gym.utils import env_creator
from RL4MM.gym.utils import generate_trajectory, plot_reward_distributions, get_episode_summary_dict
from RL4MM.agents.baseline_agents import RandomAgent, FixedActionAgent, TeradactylAgent, ContinuousTeradactyl
from RL4MM.utils.utils import boolean_string

from experiments.fixed_action_vs_teradactyl import agents
from experiments.teradactyl_sweep import (
    a_range,
    b_range,
    min_quote_range,
    max_quote_range,
    max_inv_range,
    default_omega_range,
    kappa_range,
)

experiment_list = ["ladder_sweep", "fixed_action_sweep", "fixed_action_vs_teradactyl", "teradactly_sweep"]


def get_configs(args):
    # ray.init()
    env_config = {
        "ticker": args["ticker"],
        "min_date": args["min_date"],
        "max_date": args["max_date"],
        "step_size": args["step_size"],
        "episode_length": args["episode_length"],
        "n_levels": args["n_levels"],
        "initial_portfolio": args["initial_portfolio"],
        "per_step_reward_function": args["per_step_reward_function"],
        "terminal_reward_function": args["terminal_reward_function"],
        "market_order_clearing": args["market_order_clearing"],
        "market_order_fraction_of_inventory": None,
        "min_quote_level": args["min_quote_level"],
        "max_quote_level": args["max_quote_level"],
        "enter_spread": args["enter_spread"],
        "concentration": args["concentration"],
    }

    eval_env_config = copy.deepcopy(env_config)
    eval_env_config["min_date"] = args["min_date_eval"]
    eval_env_config["max_date"] = args["max_date_eval"]
    eval_env_config["per_step_reward_function"] = args["eval_per_step_reward_function"]
    eval_env_config["terminal_reward_function"] = args["terminal_reward_function"]

    # register_env("HistoricalOrderbookEnvironment", env_creator)

    # config = {
    # "env": "HistoricalOrderbookEnvironment",
    # "num_gpus": args["num_gpus"],
    # "num_workers": args["num_workers"],
    # "framework": args["framework"],
    # "callbacks": Custom_Callbacks,
    # "rollout_fragment_length": args["rollout_fragment_length"],
    # "lambda": args["lambda"],
    # "lr": args["learning_rate"],
    # "gamma": args["discount_factor"],
    # "train_batch_size": args["train_batch_size"],
    # "model": {
    # "fcnet_hiddens": [256, 256],
    # "fcnet_activation": "tanh",  # torch.nn.Sigmoid,
    # "use_lstm": args["lstm"],
    # },
    # "output": args["output"],
    # "output_max_file_size": args["output_max_file_size"],
    # "env_config": env_config,
    # "evaluation_interval": 3,  # Run one evaluation step on every 3rd `Trainer.train()` call.
    # "evaluation_num_workers": args["num_workers_eval"],
    # "evaluation_parallel_to_training": True,
    # "evaluation_duration": "auto",
    # "evaluation_config": {"env_config": eval_env_config},
    # "recreate_failed_workers": False, #True,
    # "disable_env_checking":True,
    # }

    # tensorboard_logdir = args["tensorboard_logdir"]
    # if not os.path.exists(tensorboard_logdir):
    # os.makedirs(tensorboard_logdir)

    # analysis = tune.run(
    # "PPO",
    # stop={"training_iteration": args["iterations"]},
    # config=config,
    # local_dir=tensorboard_logdir,
    # checkpoint_at_end=True,
    # )
    # best_checkpoint = analysis.get_trial_checkpoints_paths(
    # trial=analysis.get_best_trial("episode_reward_mean"), metric="episode_reward_mean"
    # )
    # path_to_save_dir = args["output"] or "/home/ray"
    # print(best_checkpoint)
    # save_best_checkpoint_path(path_to_save_dir, best_checkpoint[0][0])

    # print(env_config)

    return env_config, eval_env_config


def parse_args():
    # -------------------- Training Args ----------------------
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-ni", "--n_iterations", default=10, help="Training iterations.", type=int)
    # -------------------- Training env Args ---------------------------
    parser.add_argument("-mind", "--min_date", default="2018-02-20", help="Train data start date.", type=str)
    parser.add_argument("-maxd", "--max_date", default="2018-02-20", help="Train data end date.", type=str)
    parser.add_argument("-t", "--ticker", default="SPY", help="Specify stock ticker.", type=str)
    parser.add_argument("-el", "--episode_length", default=60, help="Episode length (minutes).", type=int)
    parser.add_argument("-ip", "--initial_portfolio", default=None, help="Initial portfolio.", type=dict)
    parser.add_argument("-sz", "--step_size", default=5, help="Step size in seconds.", type=int)
    parser.add_argument("-nl", "--n_levels", default=200, help="Number of orderbook levels.", type=int)
    parser.add_argument(
        "-psr",
        "--per_step_reward_function",
        default="PnL",
        choices=["AD", "SD", "PnL"],
        help="Per step reward function: asymmetrically dampened (SD), asymmetrically dampened (AD), PnL (PnL).",
        type=str,
    )
    parser.add_argument(
        "-tr",
        "--terminal_reward_function",
        default="PnL",
        choices=["AD", "SD", "PnL"],
        help="Terminal reward function: asymmetrically dampened (SD), asymmetrically dampened (AD), PnL (PnL).",
        type=str,
    )
    parser.add_argument(
        "-moc", "--market_order_clearing", action="store_true", default=False, help="Market order clearing."
    )
    parser.add_argument("-minq", "--min_quote_level", default=0, help="minimum quote level from best price.", type=int)
    parser.add_argument("-maxq", "--max_quote_level", default=10, help="maximum quote level from best price.", type=int)
    parser.add_argument(
        "-es",
        "--enter_spread",
        default=False,
        help="Bool for whether best quote is the midprice. Otherwise it is the best bid/best ask price",
        type=bool,
    )
    parser.add_argument("-con", "--concentration", default=0, help="Concentration of the order distributor.", type=int)
    parser.add_argument("-par", "--parallel", action="store_true", default=False, help="Run in parallel or not.")
    # ------------------ Eval env args -------------------------------
    parser.add_argument("-minde", "--min_date_eval", default="2019-01-03", help="Evaluation data start date.", type=str)
    parser.add_argument("-maxde", "--max_date_eval", default="2019-01-03", help="Evaluation data end date.", type=str)
    parser.add_argument(
        "-epsr",
        "--eval_per_step_reward_function",
        default="PnL",
        choices=["AD", "SD", "PnL"],
        help="Eval per step reward function: asymmetrically dampened (SD), asymmetrically dampened (AD), PnL (PnL).",
        type=str,
    )
    parser.add_argument(
        "-etr",
        "--eval_terminal_reward_function",
        default="PnL",
        choices=["AD", "SD", "PnL"],
        help="Eval terminal reward function: symmetrically dampened (SD), asymmetrically dampened (AD), PnL (PnL).",
        type=str,
    )
    parser.add_argument("-o", "--output", default="/home/data/", help="Directory to save episode data to.", type=str)
    parser.add_argument("-ex", "--experiment", default="fixed_action_sweep", help="The experiment to run.", type=str)
    parser.add_argument(
        "-md", "--multiple_databases", action="store_true", default=False, help="Run using multiple databases."
    )
    # -------------------------------------------------
    args = vars(parser.parse_args())
    if args["concentration"] == 0:
        args["concentration"] = None
    return args


def import_get_env_configs_and_agents(experiment_name: str):
    if experiment_name == "ladder_sweep":
        from experiments.ladder_sweep import get_env_configs_and_agents
    elif experiment_name == "fixed_action_sweep":
        from experiments.fixed_action_sweep import get_env_configs_and_agents
    elif experiment_name == "fixed_action_vs_teradactyl":
        from experiments.fixed_action_vs_teradactyl import get_env_configs_and_agents
    elif experiment_name == "teradactly_sweep":
        from experiments.teradactyl_sweep import get_env_configs_and_agents
    else:
        raise NotImplementedError()


if __name__ == "__main__":

    args = parse_args()
    env_config, _ = get_configs(args)
    experiment = args["experiment"]
    assert experiment in experiment_list, f"Experiment name {experiment} not in list of experiments {experiment_list}."
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
            )
