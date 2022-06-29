import argparse
import os
from datetime import timedelta
import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.registry import register_env
import random

from RL4MM.gym.HistoricalOrderbookEnvironment import HistoricalOrderbookEnvironment
from RL4MM.rewards.RewardFunctions import InventoryAdjustedPnL, PnL
from RL4MM.utils.custom_metrics_callback import Custom_Callbacks
import copy
from RL4MM.simulation.OrderbookSimulator import OrderbookSimulator
from RL4MM.utils.utils import get_date_time, save_best_checkpoint_path
from RL4MM.utils.utils import boolean_string

from RL4MM.features.Features import (
    Feature,
    Spread,
    MicroPrice,
    InternalState,
    MidpriceMove,
    Volatility,
    Inventory,
    TimeRemaining,
    MidPrice,
)

def get_reward_function(reward_function: str, inventory_aversion: float = 1.0):
    if reward_function == "AD":  # asymmetrically dampened
        return InventoryAdjustedPnL(inventory_aversion=inventory_aversion, asymmetrically_dampened=True)
    elif reward_function == "SD":  # symmetrically dampened
        return InventoryAdjustedPnL(inventory_aversion=inventory_aversion, asymmetrically_dampened=False)
    elif reward_function == "PnL":
        return PnL()
    else:
        raise NotImplementedError("You must specify one of 'AS', 'SD' or 'PnL'")

def env_creator(env_config):
    print(env_config)
    obs = OrderbookSimulator(ticker=env_config["ticker"], n_levels=env_config["n_levels"])
    if env_config["features"] == "agent_state":
        features = [
                Inventory(
                    max_value=env_config["max_inventory"], 
                    normalisation_on=env_config["normalisation_on"]
                    )
                ] 
    elif env_config["features"] == "full_state":
        features = [
                Spread(
                    min_value=-1e4 if env_config["normalisation_on"] else 0, # If feature is a zscore
                    normalisation_on=env_config["normalisation_on"]
                    ), 
                MidpriceMove(normalisation_on=env_config["normalisation_on"]), 
                Volatility(
                    min_value=-1e4 if env_config["normalisation_on"] else 0, # If feature is a zscore
                    normalisation_on=env_config["normalisation_on"]
                    ), 
                Inventory(
                    max_value=env_config["max_inventory"], 
                    normalisation_on=env_config["normalisation_on"]
                    ), 
                TimeRemaining(), 
                MicroPrice(
                    min_value=-1e4 if env_config["normalisation_on"] else 0, # If feature is a zscore 
                    normalisation_on=env_config["normalisation_on"]
                    )
                ] 
    return HistoricalOrderbookEnvironment(
        ticker=env_config["ticker"],
        episode_length=timedelta(minutes=env_config["episode_length"]),
        simulator=obs,
        features=features,
        quote_levels=10,
        max_inventory=env_config["max_inventory"],
        min_date=get_date_time(env_config["min_date"]),  # datetime
        max_date=get_date_time(env_config["max_date"]),  # datetime
        step_size=timedelta(seconds=env_config["step_size"]),
        initial_portfolio=env_config["initial_portfolio"],  #: dict = None
        per_step_reward_function=get_reward_function(env_config["per_step_reward_function"]),
        terminal_reward_function=get_reward_function(env_config["terminal_reward_function"]),
        market_order_clearing=env_config["market_order_clearing"],
    )


def main(args):
    # Postprocess the perturbed config to ensure it's still valid
    def explore(config):
        # ensure we collect enough timesteps to do sgd
        if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
            config["train_batch_size"] = config["sgd_minibatch_size"] * 2
        # ensure we run at least one sgd iter
        if config["num_sgd_iter"] < 1:
            config["num_sgd_iter"] = 1
        return config

    pbt = PopulationBasedTraining(
        time_attr="time_total_s",
        perturbation_interval=120,
        resample_probability=0.25,
        # Specifies the mutations of these hyperparams
        hyperparam_mutations={
            #"lambda": lambda: random.uniform(0.9, 1.0),
            #"clip_param": lambda: random.uniform(0.01, 0.5),
            #"lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            "num_sgd_iter": lambda: random.randint(1, 30),
            "sgd_minibatch_size": lambda: random.randint(128, 16384),
            "train_batch_size": lambda: random.randint(2000, 160000),
            #"rollout_fragment_length": lambda: random.randint(200, 3600),
        },
        custom_explore_fn=explore,
    )

    ray.init(ignore_reinit_error=True, num_cpus=args["num_workers"]+2)
    env_config = {
        "ticker": args["ticker"],
        "min_date": args["min_date"],
        "max_date": args["max_date"],
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
        "inc_prev_action_in_obs": args["inc_prev_action_in_obs"],
    }

    eval_env_config = copy.deepcopy(env_config)
    eval_env_config["min_date"] = args["min_date_eval"]
    eval_env_config["max_date"] = args["max_date_eval"]
    #eval_env_config["per_step_reward_function"] = (args["eval_per_step_reward_function"],)
    eval_env_config["per_step_reward_function"] = args["eval_per_step_reward_function"]
    eval_env_config["terminal_reward_function"] = args["terminal_reward_function"]
    eval_env_config["per_step_reward_function"] = 'PnL'
    eval_env_config["terminal_reward_function"] = 'PnL'

    register_env("HistoricalOrderbookEnvironment", env_creator)

    config = {
        "env": "HistoricalOrderbookEnvironment",
        "num_gpus": args["num_gpus"],
        "num_workers": args["num_workers"],
        "framework": args["framework"],
        "callbacks": Custom_Callbacks,
        "sgd_minibatch_size":100,
        "num_sgd_iter":10,
        #"num_sgd_iter": tune.choice([10, 20, 30]),
        #"sgd_minibatch_size": tune.choice([128, 512, 2048]),
        "train_batch_size": 20000, #tune.choice([10000, 20000, 40000]),
        "num_cpus_per_worker": 1,
        "lambda": args["lambda"],
        "lr": args["learning_rate"],
        "gamma": args["discount_factor"],
        "model": {
            "fcnet_hiddens": [512, 256],
            "fcnet_activation": "tanh",  # torch.nn.Sigmoid,
            #"use_lstm": args["lstm"],
            #"lstm_use_prev_action": True,
        },
        "output": args["output"],
        "output_max_file_size": args["output_max_file_size"],
        "env_config": env_config,
        "evaluation_interval": 3,  # Run one evaluation step on every 3rd `Trainer.train()` call.
        "evaluation_num_workers": args["num_workers_eval"],
        "evaluation_parallel_to_training": True,
        "evaluation_duration": "auto",
        "evaluation_config": {"env_config": eval_env_config},
        "rollout_fragment_length": args["rollout_fragment_length"], #tune.choice([1800, 3600]), 
        # ---------------------------------------------
        #"recreate_failed_workers": False, # Get an error for some reason when this is enabled.
        #"disable_env_checking": True,
    }
    
    tensorboard_logdir = args["tensorboard_logdir"]+f"{args['per_step_reward_function']}_{args['features']}_moc_{args['market_order_clearing']}"
    if not os.path.exists(tensorboard_logdir):
        os.makedirs(tensorboard_logdir)

    analysis = tune.run(
        "PPO",
        #scheduler=pbt,
        num_samples=1,#8,
        metric="episode_reward_mean",
        mode="max",
        stop={"training_iteration": args["iterations"]},
        config=config,
        local_dir=tensorboard_logdir,
        checkpoint_at_end=True,
    )
    best_checkpoint = analysis.get_trial_checkpoints_paths(
        trial=analysis.get_best_trial("episode_reward_mean"), metric="episode_reward_mean"
    )
    path_to_save_dir = args["output"] or "/home/ray"
    print(best_checkpoint)
    save_best_checkpoint_path(path_to_save_dir, best_checkpoint[0][0])

if __name__ == "__main__":
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
    parser.add_argument(
        "-f", 
        "--features", 
        default="full_state", 
        choices=["agent_state", "full_state"],
        help="Agent state only or full state.", 
        type=str
        )
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
        default="./ray_results/tensorboard/",
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
    parser.add_argument("-ia", "--inc_prev_action_in_obs", default=True, help="Include prev action in obs.", type=bool)
    parser.add_argument("-n", "--normalisation_on", default=True, help="Normalise features.", type=bool)
    parser.add_argument("-mind", "--min_date", default="2018-02-20", help="Train data start date.", type=str)
    parser.add_argument("-maxd", "--max_date", default="2018-03-05", help="Train data end date.", type=str)
    parser.add_argument("-el", "--episode_length", default=60, help="Episode length (minutes).", type=int)
    parser.add_argument("-ip", "--initial_portfolio", default=None, help="Initial portfolio.", type=dict)
    parser.add_argument("-nl", "--n_levels", default=200, help="Number of orderbook levels.", type=int)
    parser.add_argument("-sz", "--step_size", default=1, help="Step size in seconds.", type=int)
    parser.add_argument("-t", "--ticker", default="SPY", help="Specify stock ticker.", type=str)
    parser.add_argument("-mi", "--max_inventory", default=100000, help="Maximum (absolute) inventory.", type=int)
    parser.add_argument(
        "-psr",
        "--per_step_reward_function",
        default="AD",
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
    parser.add_argument("-minde", "--min_date_eval", default="2018-03-07", help="Evaluation data start date.", type=str)
    parser.add_argument("-maxde", "--max_date_eval", default="2018-03-14", help="Evaluation data end date.", type=str)
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
    # -------------------  Run ------------------------
    main(args)
