import argparse
import os

# from datetime import timedelta
import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.registry import register_env
import random

# from RL4MM.gym.HistoricalOrderbookEnvironment import HistoricalOrderbookEnvironment
from RL4MM.gym.utils import env_creator

# from RL4MM.rewards.RewardFunctions import InventoryAdjustedPnL, PnL
from RL4MM.utils.custom_metrics_callback import Custom_Callbacks
import copy

# from RL4MM.simulation.OrderbookSimulator import OrderbookSimulator
from RL4MM.utils.utils import save_best_checkpoint_path, get_timedelta_from_clock_time
from RL4MM.utils.utils import boolean_string


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
            # "lambda": lambda: random.uniform(0.9, 1.0),
            # "clip_param": lambda: random.uniform(0.01, 0.5),
            # "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            "num_sgd_iter": lambda: random.randint(1, 30),
            "sgd_minibatch_size": lambda: random.randint(128, 16384),
            "train_batch_size": lambda: random.randint(2000, 160000),
            # "rollout_fragment_length": lambda: random.randint(200, 3600),
        },
        custom_explore_fn=explore,
    )

    ray.init(ignore_reinit_error=True, num_cpus=args["num_workers"] + 2)
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
        "market_order_fraction_of_inventory": args["market_order_fraction_of_inventory"],
        "min_quote_level": args["min_quote_level"],
        "max_quote_level": args["max_quote_level"],
        "min_start_timedelta": get_timedelta_from_clock_time(args["min_start_time"]),
        "max_end_timedelta": get_timedelta_from_clock_time(args["min_start_time"]),
        "enter_spread": args["enter_spread"],
        "inc_prev_action_in_obs": args["inc_prev_action_in_obs"],
        "concentration": args["concentration"],
    }

    eval_env_config = copy.deepcopy(env_config)
    eval_env_config["min_date"] = args["min_date_eval"]
    eval_env_config["max_date"] = args["max_date_eval"]
    # eval_env_config["per_step_reward_function"] = (args["eval_per_step_reward_function"],)
    eval_env_config["per_step_reward_function"] = args["eval_per_step_reward_function"]
    eval_env_config["terminal_reward_function"] = args["terminal_reward_function"]
    eval_env_config["per_step_reward_function"] = "PnL"
    eval_env_config["terminal_reward_function"] = "PnL"

    register_env("HistoricalOrderbookEnvironment", env_creator)

    config = {
        "env": "HistoricalOrderbookEnvironment",
        "num_gpus": args["num_gpus"],
        "num_workers": args["num_workers"],
        "framework": args["framework"],
        "callbacks": Custom_Callbacks,
        "num_cpus_per_worker": 1,
        "lambda": args["lambda"],
        "lr": args["learning_rate"],
        "gamma": args["discount_factor"],
        "model": {
            "fcnet_hiddens": [512, 256],
            "fcnet_activation": "tanh",  # torch.nn.Sigmoid,
            # "use_lstm": args["lstm"],
            # "lstm_use_prev_action": True,
        },
        "output": args["output"],
        "output_max_file_size": args["output_max_file_size"],
        "env_config": env_config,
        "evaluation_interval": 1,  # Run one evaluation step on every 3rd `Trainer.train()` call.
        "evaluation_num_workers": args["num_workers_eval"],
        "evaluation_parallel_to_training": True,
        "evaluation_duration": "auto",
        "evaluation_config": {"env_config": eval_env_config, "explore": False},
        "rollout_fragment_length": args["rollout_fragment_length"],  # tune.choice([1800, 3600]),
        # ---------------------------------------------
        # --------------- Tuning: ---------------------
        "rollout_fragment_length": args["rollout_fragment_length"], #tune.choice([1800, 3600]),  # args["rollout_fragment_length"],
        "num_sgd_iter": 20, # tune.choice([10, 20, 30]),
        "sgd_minibatch_size": 512, # tune.choice([128, 512, 2048]),
        "train_batch_size": args["train_batch_size"], # tune.choice([10000, 20000, 40000]),
        # "recreate_failed_workers": False, # Get an error for some reason when this is enabled.
        # "disable_env_checking": True,
    }

    tensorboard_logdir = (
        args["tensorboard_logdir"]
        + f"{args['ticker']}/"
        + f"{args['per_step_reward_function']}/"
        + f"concentration_{args['concentration']}/"
        + f"{args['features']}/"
        + f"normalisation_on_{args['normalisation_on']}/"
        + f"moc_{args['market_order_clearing']}/"
    )
    if not os.path.exists(tensorboard_logdir):
        os.makedirs(tensorboard_logdir)
    #import ray.rllib.agents.ppo as ppo
    #trainer = ppo.PPOTrainer(config=config)
    #print(trainer.train())
    analysis = tune.run(
        "PPO",
        # scheduler=pbt,
        num_samples=1,  # 8,
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
        type=str,
    )
    parser.add_argument("-la", "--lambda", default=1.0, help="Lambda for PBT.", type=float)
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
    parser.add_argument("-ia", "--inc_prev_action_in_obs", default=True, help="Include prev action in obs.", type=boolean_string)
    parser.add_argument("-n", "--normalisation_on", default=True, help="Normalise features.", type=boolean_string)
    parser.add_argument("-mind", "--min_date", default="2022-03-01", help="Train data start date.", type=str)
    parser.add_argument("-maxd", "--max_date", default="2022-03-14", help="Train data end date.", type=str)
    parser.add_argument("-el", "--episode_length", default=60, help="Episode length (minutes).", type=int)
    parser.add_argument("-ip", "--initial_portfolio", default=None, help="Initial portfolio.", type=dict)
    parser.add_argument("-nl", "--n_levels", default=50, help="Number of orderbook levels.", type=int)
    parser.add_argument("-sz", "--step_size", default=5, help="Step size in seconds.", type=int)
    parser.add_argument("-t", "--ticker", default="IBM", help="Specify stock ticker.", type=str)
    parser.add_argument("-mi", "--max_inventory", default=1000, help="Maximum (absolute) inventory.", type=int)
    parser.add_argument("-c", "--concentration", default=10.0, help="Concentration param for beta dist.", type=float)
    parser.add_argument(
        "-psr",
        "--per_step_reward_function",
        default="AD",
        choices=["AD", "SD", "PnL"],
        help="Per step reward function: asymmetrically dampened (SD), asymmetrically dampened (AD), PnL (PnL).",
        type=str,
    )
    parser.add_argument(
        "-moc", 
        "--market_order_clearing", 
        #action="store_true", 
        default=False, 
        help="Market order clearing on/off.", 
        type=boolean_string
    )
    parser.add_argument(
        "-mofi",
        "--market_order_fraction_of_inventory",
        default=0.0,
        help="Market order fraction of inventory.",
        type=float,
    )
    parser.add_argument("-minq", "--min_quote_level", default=0, help="minimum quote level from best price.", type=int)
    parser.add_argument("-maxq", "--max_quote_level", default=10, help="maximum quote level from best price.", type=int)
    parser.add_argument(
        "-tr",
        "--terminal_reward_function",
        default="PnL",
        choices=["AD", "SD", "PnL"],
        help="Terminal reward function: asymmetrically dampened (SD), asymmetrically dampened (AD), PnL (PnL).",
        type=str,
    )
    parser.add_argument(
        "-es",
        "--enter_spread",
        default=False,
        help="Bool for whether best quote is the midprice. Otherwise it is the best bid/best ask price",
        type=boolean_string,
    )
    parser.add_argument(
        "-min_st",
        "--min_start_time",
        default="1000",
        help="The minimum start time for an episode written in HHMM format.",
        type=str,
    )
    parser.add_argument(
        "-max_et",
        "--max_end_time",
        default="1530",
        help="The maximum end time for an episode written in HHMM format.",
        type=str,
    )
    # ------------------ Eval env args -------------------------------
    parser.add_argument("-minde", "--min_date_eval", default="2022-03-15", help="Evaluation data start date.", type=str)
    parser.add_argument("-maxde", "--max_date_eval", default="2022-03-30", help="Evaluation data end date.", type=str)
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
