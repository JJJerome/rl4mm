import argparse
from datetime import timedelta
import ray
from ray.tune.registry import register_env
from ray.rllib.agents import ppo
import torch
from RL4MM.gym.HistoricalOrderbookEnvironment import HistoricalOrderbookEnvironment
from RL4MM.rewards.RewardFunctions import RewardFunction, InventoryAdjustedPnL
from RL4MM.utils.custom_metrics_callback import Custom_Callbacks

from RL4MM.simulation.OrderbookSimulator import OrderbookSimulator
from RL4MM.utils.utils import custom_logger
from RL4MM.utils.utils import get_date_time
from RL4MM.utils.utils import boolean_string

def env_creator(env_config):
    obs = OrderbookSimulator(ticker=env_config["ticker"], n_levels=env_config["n_levels"])
    return HistoricalOrderbookEnvironment(
        episode_length=timedelta(minutes=env_config["episode_length"]),
        simulator = obs,
        quote_levels=10, 
        min_date  = get_date_time(env_config['min_date']),  # datetime
        max_date  = get_date_time(env_config['max_date']),  # datetime
        step_size=timedelta(seconds = env_config['step_size']),
        initial_portfolio = env_config['initial_portfolio'], #: dict = None
        per_step_reward_function=InventoryAdjustedPnL(
            inventory_aversion=10 ** (-4), asymmetrically_dampened=env_config['asymmetrically_dampened']
        ),
        terminal_reward_function=InventoryAdjustedPnL(
            inventory_aversion=0.1, asymmetrically_dampened=env_config['asymmetrically_dampened']
        ),
        market_order_clearing=env_config['market_order_clearing'],
    ) 

def main(args):
    ray.init()
    config = {
        "num_gpus": args["num_gpus"],
        "num_workers": args["num_workers"],
        "framework": args["framework"],
        "callbacks": Custom_Callbacks,
        "model": {
            "fcnet_hiddens": [256, 256],
            "fcnet_activation":"tanh",#torch.nn.Sigmoid,
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
            "asymmetrically_dampened": args["asymmetrically_dampened"],
            "market_order_clearing": args["market_order_clearing"],
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
    parser.add_argument("-g", "--num_gpus", default="1", help="Number of GPUs to use during training.", type=int)
    parser.add_argument("-nw", "--num_workers", default="5", help="Number of workers to use during training.", type=int)
    parser.add_argument(
        "-nwe", "--num_workers_eval", default="1", help="Number of workers used during evaluation.", type=int
    )
    parser.add_argument("-fw", "--framework", default="torch", help="Framework, torch or tf.", type=str)
    parser.add_argument("-l", "--lstm", default=False, help="LSTM on/off.", type=boolean_string)
    parser.add_argument("-i", "--iterations", default="200", help="Training iterations.", type=int)
    # -------------------- Env Args ---------------------------
    parser.add_argument("-mind", "--min_date", default="2019,1,2", help="Data start date.", type=str)
    parser.add_argument("-maxd", "--max_date", default="2019,1,2", help="Data end date.", type=str)
    parser.add_argument("-t", "--ticker", default="MSFT", help="Specify stock ticker.", type=str)
    parser.add_argument("-el", "--episode_length", default="1", help="Episode length (minutes).", type=int)
    parser.add_argument("-ip", "--initial_portfolio", default=None, help="Initial portfolio.", type=dict)
    parser.add_argument("-sz", "--step_size", default="1", help="Step size in seconds.", type=int)
    parser.add_argument("-nl", "--n_levels", default="200", help="Number of levels.", type=int)
    parser.add_argument("-a", "--asymmetrically_dampened", default=True, help="AD on/off.", type=boolean_string)
    parser.add_argument("-moc", "--market_order_clearing", default=True, help="Market order clearing on/off.", type=boolean_string)
    # -------------------------------------------------
    args = vars(parser.parse_args())
    # -------------------  Run ------------------------
    main(args)
