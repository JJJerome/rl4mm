import argparse
import os
import copy

from main import env_creator

from RL4MM.gym.utils import generate_trajectory, plot_reward_distributions

from RL4MM.agents.baseline_agents import RandomAgent

# from datetime import timedelta
# import ray
# from ray import tune
# from ray.tune.registry import register_env

# from RL4MM.gym.HistoricalOrderbookEnvironment import HistoricalOrderbookEnvironment
# from RL4MM.rewards.RewardFunctions import InventoryAdjustedPnL, PnL
# from RL4MM.utils.custom_metrics_callback import Custom_Callbacks
# from RL4MM.simulation.OrderbookSimulator import OrderbookSimulator
# from RL4MM.utils.utils import get_date_time, save_best_checkpoint_path
from RL4MM.utils.utils import boolean_string


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
    parser.add_argument("-g", "--num_gpus", default=1, help="Number of GPUs to use during training.", type=int)
    parser.add_argument("-nw", "--num_workers", default=5, help="Number of workers to use during training.", type=int)
    parser.add_argument(
        "-nwe", "--num_workers_eval", default=1, help="Number of workers used during evaluation.", type=int
    )
    parser.add_argument("-fw", "--framework", default="torch", help="Framework, torch or tf.", type=str)
    parser.add_argument("-l", "--lstm", default=False, help="LSTM on/off.", type=boolean_string)
    parser.add_argument("-i", "--iterations", default=1000, help="Training iterations.", type=int)
    parser.add_argument("-la", "--lambda", default=1.0, help="Training iterations.", type=float)
    parser.add_argument(
        "-rfl",
        "--rollout_fragment_length",
        default=3600,
        help="Rollout fragment length, collected per worker..",
        type=int,
    )
    parser.add_argument("-mp", "--model_path", default=None, help="Path to existing model.", type=str)
    parser.add_argument("-lr", "--learning_rate", default=0.0001, help="Learning rate.", type=float)
    parser.add_argument("-df", "--discount_factor", default=0.99, help="Discount factor gamma of the MDP.", type=float)
    parser.add_argument(
        "-tb", "--train_batch_size", default=3600, help="The size of the training batch used for updates.", type=int
    )
    parser.add_argument(
        "-tbd",
        "--tensorboard_logdir",
        default="/home/data/tensorboard",
        help="Directory to save tensorboard logs to.",
        type=str,
    )
    # -------------------- Generating a dataset of eval episodes
    parser.add_argument("-o", "--output", default=None, help="Directory to save episode data to.", type=str)
    parser.add_argument(
        "-omfs",
        "--output_max_file_size",
        default=5000000,
        help="Max size of json file that transitions are saved to.",
        type=int,
    )
    # -------------------- Training env Args ---------------------------
    parser.add_argument("-mind", "--min_date", default="2018-02-20", help="Train data start date.", type=str)
    parser.add_argument("-maxd", "--max_date", default="2018-02-20", help="Train data end date.", type=str)
    parser.add_argument("-t", "--ticker", default="SPY", help="Specify stock ticker.", type=str)
    parser.add_argument("-el", "--episode_length", default=60, help="Episode length (minutes).", type=int)
    parser.add_argument("-ip", "--initial_portfolio", default=None, help="Initial portfolio.", type=dict)
    parser.add_argument("-sz", "--step_size", default=1, help="Step size in seconds.", type=int)
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
        "-moc", "--market_order_clearing", default=True, help="Market order clearing on/off.", type=boolean_string
    )

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
    # -------------------------------------------------
    args = vars(parser.parse_args())
    return args

if __name__ == "__main__":

    
    args = parse_args()

    # env_config, eval_env_config = get_configs(args)

    env_config, _ = get_configs(args)
    env = env_creator(env_config)

    agent = RandomAgent(env)

    # obs, acts, rews, infs = generate_trajectory(agent, env)

    plot_reward_distributions(agent, env, n_iterations=5)