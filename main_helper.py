import os
import sys
import copy

from RL4MM.utils.utils import boolean_string
from RL4MM.utils.utils import get_timedelta_from_clock_time
from RL4MM.utils.custom_metrics_callback import Custom_Callbacks

from ray import tune

from RL4MM.agents.baseline_agent_wrappers import (
    FixedActionAgentWrapper,
    TeradactylAgentWrapper,
    ContinuousTeradactylWrapper,
)


def add_ray_args(parser):

    # -------------------- Workers, GPUs, CPUs ----------------------
    parser.add_argument("-g", "--num_gpus", default=1, help="Number of GPUs to use during training.", type=int)
    parser.add_argument("-nw", "--num_workers", default=5, help="Number of workers to use during training.", type=int)
    parser.add_argument("-nepw", "--num_envs_per_worker", default=1, help="Number of envs per worker.", type=int)
    parser.add_argument(
        "-nwe", "--num_workers_eval", default=1, help="Number of workers used during evaluation.", type=int
    )

    # -------------------- Training Args ----------------------
    parser.add_argument("-i", "--iterations", default=1000, help="Training iterations.", type=int)
    parser.add_argument("-fw", "--framework", default="torch", help="Framework, torch or tf.", type=str)
    parser.add_argument("-mp", "--model_path", default=None, help="Path to existing model.", type=str)
    parser.add_argument(
        "-tbd",
        "--tensorboard_logdir",
        default="./ray_results/tensorboard/",
        help="Directory to save tensorboard logs to.",
        type=str,
    )
    parser.add_argument("-wb", "--wandb", default=False, help="Track on Weights and biases.", type=boolean_string)
    parser.add_argument(
        "-wbd",
        "--wandb_api_key_dir",
        default="/home/ray/.netrc",
        help="Path to weights and bias api key",
        type=str,
    )
    # -------------------- Hyperparameters
    parser.add_argument("-la", "--lambda", default=1.0, help="Lambda for GAE.", type=float)
    parser.add_argument("-lr", "--learning_rate", default=0.0001, help="Learning rate.", type=float)
    parser.add_argument("-df", "--discount_factor", default=0.99, help="Discount factor gamma of the MDP.", type=float)
    # Currently using tune to determine the following:
    # parser.add_argument(
    #    "-rfl",
    #    "--rollout_fragment_length",
    #    default=3600,
    #    help="Rollout fragment length, collected per worker..",
    #    type=int,
    # )
    # parser.add_argument(
    #    "-tb", "--train_batch_size", default=20000, help="The size of the training batch used for updates.", type=int
    # )

    # -------------------- Generating a dataset of eval episodes
    # parser.add_argument("-o", "--output", default=None, help="Directory to save episode data to.", type=str)
    parser.add_argument(
        "-omfs",
        "--output_max_file_size",
        default=5000000,
        help="Max size of json file that transitions are saved to.",
        type=int,
    )


###############################################################################
###############################################################################
###############################################################################
###############################################################################


def add_env_args(parser):
    # -------------------- Env Args ---------------------------
    parser.add_argument("-sz", "--step_size", default=0.1, help="Step size in seconds.", type=float)
    parser.add_argument("-t", "--ticker", default="IBM", help="Specify stock ticker.", type=str)
    parser.add_argument("-nl", "--n_levels", default=50, help="Number of orderbook levels.", type=int)
    parser.add_argument("-ic", "--initial_cash", default=1e12, help="Initial portfolio.", type=float)
    parser.add_argument("-ii", "--initial_inventory", default=0, help="Initial inventory.", type=int)
    parser.add_argument("-el", "--episode_length", default=30, help="Episode length (minutes).", type=int)
    parser.add_argument("-mi", "--max_inventory", default=1000000, help="Maximum (absolute) inventory.", type=int)
    parser.add_argument("-n", "--normalisation_on", default=False, help="Normalise features.", type=boolean_string)
    parser.add_argument("-c", "--concentration", default=None, help="Concentration param for beta dist.", type=float)
    parser.add_argument("-minq", "--min_quote_level", default=0, help="minimum quote level from best price.", type=int)
    parser.add_argument("-maxq", "--max_quote_level", default=10, help="maximum quote level from best price.", type=int)
    parser.add_argument(
        "-ia", "--inc_prev_action_in_obs", default=True, help="Include prev action in obs.", type=boolean_string
    )
    parser.add_argument(
        "-es",
        "--enter_spread",
        default=False,
        help="Bool for whether best quote is the midprice. Otherwise it is the best bid/best ask price",
        type=boolean_string,
    )

    parser.add_argument(
        "-f",
        "--features",
        default="full_state",
        choices=["agent_state", "full_state"],
        help="Agent state only or full state.",
        type=str,
    )

    ##############################

    parser.add_argument("-o", "--output", default="/home/data/", help="Directory to save episode data to.", type=str)
    parser.add_argument(
        "-md", "--multiple_databases", action="store_true", default=False, help="Run using multiple databases."
    )

    ###########################################################################
    # ---------------------- Date and Time ------------------------------------
    ###########################################################################

    parser.add_argument("-maxd", "--max_date", default="2022-03-14", help="Train data end date.", type=str)
    parser.add_argument("-mind", "--min_date", default="2022-03-01", help="Train data start date.", type=str)
    parser.add_argument("-maxde", "--max_date_eval", default="2022-03-30", help="Evaluation data end date.", type=str)
    parser.add_argument("-minde", "--min_date_eval", default="2022-03-15", help="Evaluation data start date.", type=str)
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

    ###########################################################################
    # ---------------------- Rewards ------------------------------------------
    ###########################################################################

    r_choices = [
        "AD",  # Asymmetrically Dampened
        "SD",  # Symmetrically Dampened
        "PnL",  # PnL
        "RS",  # RollingSharpe
    ]
    parser.add_argument(
        "-tr",
        "--terminal_reward_function",
        default="PnL",
        choices=r_choices,
        help="Terminal reward function: symm dampened (SD), asymm d.. (AD), PnL (PnL), RollingSharpe (RS)",
        type=str,
    )
    parser.add_argument(
        "-psr",
        "--per_step_reward_function",
        default="PnL",
        choices=r_choices,
        help="Per step reward function: asymmetrically dampened (SD), asymmetrically dampened (AD), PnL (PnL).",
        type=str,
    )
    parser.add_argument(
        "-epsr",
        "--eval_per_step_reward_function",
        default="PnL",
        choices=r_choices,
        help="Eval per step rewards.",
        type=str,
    )
    parser.add_argument(
        "-etr",
        "--eval_terminal_reward_function",
        default="PnL",
        choices=r_choices,
        help="Eval terminal reward function: symmetrically dampened (SD), asymmetrically dampened (AD), PnL (PnL).",
        type=str,
    )

    ###########################################################################
    # ---------------------- Market Order Clearing ----------------------------
    ###########################################################################

    parser.add_argument(
        "-moc",
        "--market_order_clearing",
        # action="store_true",
        default=False,
        help="Market order clearing on/off.",
        type=boolean_string,
    )
    parser.add_argument(
        "-mofi",
        "--market_order_fraction_of_inventory",
        default=0.0,
        help="Market order fraction of inventory.",
        type=float,
    )

    ###########################################################################
    # ---------------------- END OF ENV ARGS ----------------------------------
    ###########################################################################


###############################################################################
###############################################################################
###############################################################################
###############################################################################


def get_env_configs(args):
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
        "initial_cash": args["initial_cash"],
        "initial_inventory": args["initial_inventory"],
        "per_step_reward_function": args["per_step_reward_function"],
        "terminal_reward_function": args["terminal_reward_function"],
        "market_order_clearing": args["market_order_clearing"],
        "market_order_fraction_of_inventory": 0.0,
        "min_quote_level": args["min_quote_level"],
        "max_quote_level": args["max_quote_level"],
        "enter_spread": args["enter_spread"],
        "concentration": args["concentration"],
        "info_calculator": None,
        "inc_prev_action_in_obs": args["inc_prev_action_in_obs"],
    }

    eval_env_config = copy.deepcopy(env_config)
    eval_env_config["min_date"] = args["min_date_eval"]
    eval_env_config["max_date"] = args["max_date_eval"]
    eval_env_config["per_step_reward_function"] = args["eval_per_step_reward_function"]
    eval_env_config["terminal_reward_function"] = args["terminal_reward_function"]

    return env_config, eval_env_config


###############################################################################
###############################################################################
###############################################################################
###############################################################################


def get_ray_config(args, env_config, eval_env_config, name, cmc=None):

    if name == "main":

        ray_config = {
            # ---  CPUs, GPUs, Workers ---
            "num_cpus_per_worker": 1,
            "num_gpus": args["num_gpus"],
            "framework": args["framework"],
            "num_workers": args["num_workers"],
            "num_envs_per_worker": args["num_envs_per_worker"],
            # --- Env ---
            "env_config": env_config,
            "evaluation_duration": "auto",
            "callbacks": Custom_Callbacks,
            "evaluation_parallel_to_training": True,
            "env": "HistoricalOrderbookEnvironment",
            "evaluation_num_workers": args["num_workers_eval"],
            "evaluation_config": {"env_config": eval_env_config, "explore": False},
            "evaluation_interval": 1,  # Run one evaluation step every n `Trainer.train()` calls.
            # --- Hyperparams ---
            "lambda": args["lambda"],
            "lr": args["learning_rate"],
            "gamma": args["discount_factor"],
            "model": {
                "fcnet_activation": "tanh",
                "fcnet_hiddens": [512, 256],
            },
            # --- Directory to save dataset data to ---
            "output": args["output"],
            "output_max_file_size": args["output_max_file_size"],
            # --------------- Tuning: ---------------------
            "num_sgd_iter": tune.choice([10, 20, 30, 100, 500]),
            "sgd_minibatch_size": tune.choice([2**3, 2**5, 2**7, 2**9, 2**11]),
            "train_batch_size": tune.choice([2**11, 2**12, 2**13, 2**14, 2**15]),
            "rollout_fragment_length": tune.choice(
                [2**8, 2**9, 2**10, 2**12, 2**13, 2**14]
            ),  # args["rollout_fragment_length"],
            # "recreate_failed_workers": False, # Get an error for some reason when this is enabled.
            # "disable_env_checking": True,
            # 'seed':tune.choice(range(1000)),
        }
        if args["wandb"]:
            ray_config["wandb"] = {"project": "RL4MM", "api_key_file": args["wandb_api_key_dir"], "log_config": True}

    elif name == "tune_rule_based_agents":

        ray_config = {
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
            "model": {"custom_model_config": cmc},  # DIFFERENT
            "env_config": env_config,
            "evaluation_interval": 1,
            "evaluation_num_workers": args["num_workers_eval"],
            "evaluation_parallel_to_training": True,
            "evaluation_duration": "auto",
            "evaluation_config": {"env_config": eval_env_config},
        }

    return ray_config


###############################################################################
###############################################################################
###############################################################################
###############################################################################


def get_tensorboard_logdir(args, name):

    if name == "main":
        tensorboard_logdir = (
            args["tensorboard_logdir"]
            + f"{args['ticker']}/"
            + f"{args['per_step_reward_function']}/"
            + f"concentration_{args['concentration']}/"
            + f"{args['features']}/"
            + f"normalisation_on_{args['normalisation_on']}/"
            + f"moc_{args['market_order_clearing']}/"
        )
    elif name == "tune_rule_based_agents":
        tensorboard_logdir = (
            args["tensorboard_logdir"]
            + f"{args['ticker']}/"
            + f"{args['rule_based_agent']}/"
            + f"{args['per_step_reward_function']}/"
        )

    else:
        sys.exit(f"name: {name} not recognised in get_tensorboard_logdir")

    if not os.path.exists(tensorboard_logdir):
        os.makedirs(tensorboard_logdir)
    return tensorboard_logdir


###############################################################################
###############################################################################
###############################################################################
###############################################################################


def get_rule_based_agent_and_custom_model_config(args):

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
        # Automatically converted to space format within BayesOpt:
        custom_model_config = {
            "default_kappa": tune.uniform(3.0, 15.0),
            "default_omega": tune.uniform(0.1, 0.5),
            "max_kappa": tune.uniform(10.0, 100.0),
            "exponent": tune.uniform(1.0, 5.0),
            "max_inventory": args["max_inventory"],
        }
        rule_based_agent = ContinuousTeradactylWrapper
    else:
        raise Exception(f"{args['rule_based_agent']} wrapper not implemented.")

    return rule_based_agent, custom_model_config
