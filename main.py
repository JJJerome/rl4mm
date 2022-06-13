from RL4MM.gym.HistoricalOrderbookEnvironment import HistoricalOrderbookEnvironment

# from RL4MM.gym.example_env import Example_v0

from RL4MM.simulation.OrderbookSimulator import OrderbookSimulator
from RL4MM.utils.utils import custom_logger
from RL4MM.features.Features import Feature
from RL4MM.utils.utils import get_date_time
from ray.tune.registry import register_env
from datetime import datetime, timedelta
from gym.spaces import Discrete, Tuple
from ray.rllib.agents import ppo
import argparse, gym, ray
from typing import List
import os


def env_creator(env_config):
    obs = OrderbookSimulator(ticker=env_config["ticker"], n_levels=env_config["n_levels"])
    return HistoricalOrderbookEnvironment(
        episode_length=timedelta(minutes=10),
        simulator=obs,  # OrderbookSimulator
        min_date=get_date_time(env_config["min_date"]),  # datetime
        max_date=get_date_time(env_config["max_date"]),  # datetime
        step_size=timedelta(seconds=env_config["step_size"]),
        initial_portfolio=env_config["initial_portfolio"],  #: dict = None
    )


def main(args):
    ray.init()
    config = {
        "num_gpus": args["num_gpus"],
        "num_workers": args["num_workers"],
        "framework": args["framework"],
        "model": {
            "fcnet_hiddens": [64, 64],
            "fcnet_activation": "relu",
            "use_lstm": args["lstm"],
        },
        "evaluation_num_workers": args["num_workers_eval"],
        "env_config": {
            "ticker": args["ticker"],
            "min_date": args["min_date"],
            "max_date": args["max_date"],
            "step_size": args["step_size"],
            "episode_length": args["episode_length"],
            "n_levels": args["n_levels"],
            "initial_portfolio": args["initial_portfolio"],
        },
    }
    register_env("HistoricalOrderbookEnvironment", env_creator)
    trainer = ppo.PPOTrainer(
        env="HistoricalOrderbookEnvironment", config=config, logger_creator=custom_logger(prefix=args["ticker"])
    )

    # -------------------- Train Agent ---------------------------
    for _ in range(args["iterations"]):
        print(trainer.train())

    # -------------------- Eval Agent ----------------------------
    trainer.evaluate()


if __name__ == "__main__":
    # -------------------- Training Args ----------------------
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-g", "--num_gpus", default="0", help="Number of GPUs to use during training.", type=int)
    parser.add_argument("-nw", "--num_workers", default="1", help="Number of wokers to use during training.", type=int)
    parser.add_argument(
        "-nwe", "--num_workers_eval", default="1", help="Number of wokers used during evaluation.", type=int
    )
    parser.add_argument("-fw", "--framework", default="torch", help="Framework, torch or tf.", type=str)
    parser.add_argument("-l", "--lstm", default=False, help="LSTM on/off.", type=bool)
    parser.add_argument("-i", "--iterations", default="10", help="Training iterations.", type=int)
    # -------------------- Env Args ---------------------------
    parser.add_argument("-mind", "--min_date", default="2019,1,2", help="Data start date.", type=str)
    parser.add_argument("-maxd", "--max_date", default="2019,1,2", help="Data end date.", type=str)
    parser.add_argument("-t", "--ticker", default="MSFT", help="Specify stock ticker.", type=str)
    parser.add_argument("-el", "--episode_length", default="10", help="Episode length (minutes).", type=int)
    parser.add_argument("-ip", "--initial_portfolio", default=None, help="Initial portfolio.", type=dict)
    parser.add_argument("-sz", "--step_size", default="1", help="Step size in seconds.", type=int)
    parser.add_argument("-nl", "--n_levels", default="200", help="Number of levels.", type=int)
    # -------------------------------------------------
    args = vars(parser.parse_args())
    # -------------------  Run ------------------------
    main(args)
