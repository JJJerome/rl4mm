from RL4MM.utils.custom_metrics_callback import Custom_Callbacks
from ray.tune.schedulers import PopulationBasedTraining
from RL4MM.agents.baseline_agent_wrappers import (
    FixedActionAgentWrapper,
    TeradactylAgentWrapper,
    ContinuousTeradactylWrapper,
)
from RL4MM.utils.utils import boolean_string
from ray.tune.registry import register_env
from RL4MM.gym.utils import env_creator
from ray import tune
import numpy as np
import argparse
import random
import copy
import ray
import os

from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.suggest import ConcurrencyLimiter



def main(args):
    ray.init(ignore_reinit_error=True, num_cpus=args["num_workers"] + args["num_workers_eval"])
    register_env("HistoricalOrderbookEnvironment", env_creator)
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
        "inc_prev_action_in_obs": args["inc_prev_action_in_obs"],
        "concentration": None,
        "min_quote_level": args["min_quote_level"],
        "max_quote_level": args["max_quote_level"],
        "enter_spread": args["enter_spread"],
    }

    eval_env_config = copy.deepcopy(env_config)
    eval_env_config["min_date"] = args["min_date_eval"]
    eval_env_config["max_date"] = args["max_date_eval"]
    eval_env_config["per_step_reward_function"] = "PnL"
    eval_env_config["terminal_reward_function"] = "PnL"

    if args["rule_based_agent"] == "fixed":
        custom_model_config = {
            "a_1": tune.uniform(1, 10),
            "a_2": tune.uniform(1, 10),
            "b_1": tune.uniform(1, 10),
            "b_2": tune.uniform(1, 10),
            "threshold": tune.uniform(500, 1000),
        }
        rule_based_agent = FixedActionAgentWrapper
    elif args["rule_based_agent"] == "teradactyl":
        custom_model_config = {
            "kappa": tune.uniform(10.0, 10.0),
            "default_a": tune.uniform(3.0, 4.0),
            "default_b": tune.uniform(1.0, 2.0),
            "max_inventory": args["max_inventory"],
        }
        rule_based_agent = TeradactylAgentWrapper
    elif args["rule_based_agent"] == "continuous_teradactyl":
        custom_model_config = {
            "default_kappa": tune.uniform(3.0, 15.0),
            "default_omega": tune.uniform(0.1, 0.5),
            "max_kappa": tune.uniform(10.0, 100.0),
            "max_inventory": args["max_inventory"],
        }
        """
        space = {
            "model":{
                "custom_model_config": {
                    "default_kappa": (3.0, 15.0),
                    "default_omega": (0.1, 0.5),
                    "max_kappa": (10.0, 100.0),
                }
            }
        }
        """
        rule_based_agent = ContinuousTeradactylWrapper
    else:
        raise Exception(f"{args['rule_based_agent']} wrapper not implemented.")


    config = {
        "env": "HistoricalOrderbookEnvironment",
        # -----------------
        "simple_optimizer": True,
        "_fake_gpus": 0,
        "num_workers": 0,
        "train_batch_size": 0,
        "rollout_fragment_length": 3600,
        "timesteps_per_iteration": 0,
        # -----------------
        "framework": args["framework"],
        "num_cpus_per_worker": 1,
        "model": {"custom_model_config": custom_model_config},
        "env_config": env_config,
        "evaluation_interval": 1,  
        "evaluation_num_workers": args["num_workers_eval"],
        "evaluation_parallel_to_training": True,
        "evaluation_duration": "auto",
        "evaluation_config": {"env_config": eval_env_config},
    }

    # ---------------- For testing.... ----------------------
    # Uncomment for basic testing
    # print(rule_based_agent(config=config).train())
    # print(FixedActionAgentWrapper(config=config).evaluate())
    # -------------------------------------------------------
    tensorboard_logdir = f"{args['tensorboard_logdir']}{args['experiment']}/{args['per_step_reward_function']}"
    """
    analysis = tune.run(
        rule_based_agent,
        name=args["ticker"],
        search_alg=search_alg,
        scheduler=scheduler,
        metric="episode_reward_mean", 
        mode="max",
        num_samples=10,
        stop={"training_iteration": args["iterations"]},
        config=config,
        local_dir=tensorboard_logdir,
        checkpoint_at_end=True,
    )
    """
    algo = BayesOptSearch(
        utility_kwargs={"kind": "ucb", "kappa": 2.5, "xi": 0.0},
        random_search_steps=5, 
    )
    algo = ConcurrencyLimiter(algo, max_concurrent=10)
    scheduler = AsyncHyperBandScheduler()
    analysis = tune.run(
        #easy_objective,
        rule_based_agent,
        #name="my_exp",
        name=args["ticker"],
        #metric="mean_loss",
        metric="episode_reward_mean", 
        #mode="min",
        mode="max",
        search_alg=algo,
        scheduler=scheduler,
        num_samples=1000, #if args.smoke_test else 1000,
        config=config,
        #config={
        #    "steps": 100,
        #    "width": tune.uniform(0, 20),
        #    "height": tune.uniform(-100, 100),
        #},
        local_dir=tensorboard_logdir,
        checkpoint_at_end=True,
        resume="AUTO",
        stop={"training_iteration":1},# args["iterations"]},
    )
    print("Best hyperparameters found were: ", analysis.best_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    # -------------------- Agent ----------------------
    parser.add_argument(
        "-rba",
        "--rule_based_agent",
        default="continuous_teradactyl",
        choices=["fixed", "teradactyl", "continuous_teradactyl"],
        help="Specify rule based agent.",
        type=str,
    )
    # -------------------- Training Args ----------------------
    parser.add_argument("-nw", "--num_workers", default=20, help="Number of workers to use during training.", type=int)
    parser.add_argument(
        "-nwe", "--num_workers_eval", default=1, help="Number of workers used during evaluation.", type=int
    )
    parser.add_argument("-fw", "--framework", default="torch", help="Framework, torch or tf.", type=str)
    parser.add_argument("-i", "--iterations", default=1000, help="Training iterations.", type=int)
    parser.add_argument(
        "-f",
        "--features",
        default="full_state",
        choices=["agent_state", "full_state"],
        help="Agent state only or full state.",
        type=str,
    )
    parser.add_argument(
        "-rfl",
        "--rollout_fragment_length",
        default=3600,
        help="Rollout fragment length, collected per worker..",
        type=int,
    )
    parser.add_argument(
        "-tbd",
        "--tensorboard_logdir",
        default="./ray_results/tensorboard/",
        help="Directory to save tensorboard logs to.",
        type=str,
    )
    parser.add_argument("-ex", "--experiment", default="bayesopt", help="The experiment to run", type=str)
    # -------------------- Training env Args ---------------------------
    parser.add_argument("-ia", "--inc_prev_action_in_obs", default=True, help="Include prev action in obs.", type=bool)
    parser.add_argument("-n", "--normalisation_on", default=False, help="Normalise features.", type=bool)
    parser.add_argument("-mind", "--min_date", default="2018-02-20", help="Train data start date.", type=str)
    parser.add_argument("-maxd", "--max_date", default="2018-03-05", help="Train data end date.", type=str)
    parser.add_argument("-el", "--episode_length", default=60, help="Episode length (minutes).", type=int)
    parser.add_argument("-ip", "--initial_portfolio", default=None, help="Initial portfolio.", type=dict)
    parser.add_argument("-nl", "--n_levels", default=50, help="Number of orderbook levels.", type=int)
    parser.add_argument("-sz", "--step_size", default=5, help="Step size in seconds.", type=int)
    parser.add_argument("-t", "--ticker", default="SPY", help="Specify stock ticker.", type=str)
    parser.add_argument("-mi", "--max_inventory", default=10000, help="Maximum (absolute) inventory.", type=int)
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
        "-moc", "--market_order_clearing", default=False, help="Market order clearing on/off.", type=boolean_string
    )
    parser.add_argument(
        "-mofi",
        "--market_order_fraction_of_inventory",
        default=None,
        help="Market order fraction of inventory.",
        type=float,
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
