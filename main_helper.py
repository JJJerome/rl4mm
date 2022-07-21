import copy

from RL4MM.utils.utils import boolean_string
from RL4MM.utils.utils import get_timedelta_from_clock_time
from RL4MM.gym.order_tracking.InfoCalculators import SimpleInfoCalculator


def add_ray_args(parser):

    # -------------------- Workers, GPUs, CPUs ----------------------
    parser.add_argument("-g", "--num_gpus", default=0.2, help="Number of GPUs to use during training.", type=int)
    parser.add_argument("-nw", "--num_workers", default=5, help="Number of workers to use during training.", type=int)
    parser.add_argument(
        "-nwe", "--num_workers_eval", default=1, help="Number of workers used during evaluation.", type=int
    )

    # -------------------- Training Args ----------------------
    parser.add_argument("-i", "--iterations", default=1000, help="Training iterations.", type=int)
    parser.add_argument("-fw", "--framework", default="torch", help="Framework, torch or tf.", type=str)
    # parser.add_argument(
        # "-f",
        # "--features",
        # default="full_state",
        # choices=["agent_state", "full_state"],
        # help="Agent state only or full state.",
        # type=str,
    # )
    parser.add_argument("-mp", "--model_path", default=None, help="Path to existing model.", type=str)
    parser.add_argument(
        "-tbd",
        "--tensorboard_logdir",
        default="./ray_results/tensorboard/",
        help="Directory to save tensorboard logs to.",
        type=str,
    )

    # -------------------- Hyperparameters
    parser.add_argument("-la", "--lambda", default=1.0, help="Lambda for PBT.", type=float)
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
    parser.add_argument("-o", "--output", default=None, help="Directory to save episode data to.", type=str)
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
    parser.add_argument("-sz", "--step_size", default=5, help="Step size in seconds.", type=int)
    parser.add_argument("-t", "--ticker", default="IBM", help="Specify stock ticker.", type=str)
    parser.add_argument("-nl", "--n_levels", default=50, help="Number of orderbook levels.", type=int)

    default_ip = dict(inventory=0, cash=1e12)
    parser.add_argument("-ip", "--initial_portfolio", default=None, help="Initial portfolio.", type=dict)

    parser.add_argument("-el", "--episode_length", default=60, help="Episode length (minutes).", type=int)
    parser.add_argument("-mi", "--max_inventory", default=10000, help="Maximum (absolute) inventory.", type=int)
    parser.add_argument("-n", "--normalisation_on", default=True, help="Normalise features.", type=boolean_string)
    # parser.add_argument("-con", "--concentration", default=None, help="Concentration of the order distributor.", type=int)
    parser.add_argument("-c", "--concentration", default=10.0, help="Concentration param for beta dist.", type=float)
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
    ]
    parser.add_argument(
        "-tr",
        "--terminal_reward_function",
        default="PnL",
        choices=["AD", "SD", "PnL"],
        help="Terminal reward function: asymmetrically dampened (SD), asymmetrically dampened (AD), PnL (PnL).",
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
        "info_calculator": SimpleInfoCalculator(market_order_fraction_of_inventory=0, 
                                      enter_spread=args["enter_spread"], 
                                      concentration=args["concentration"])
    }

    eval_env_config = copy.deepcopy(env_config)
    eval_env_config["min_date"] = args["min_date_eval"]
    eval_env_config["max_date"] = args["max_date_eval"]
    eval_env_config["per_step_reward_function"] = args["eval_per_step_reward_function"]
    eval_env_config["terminal_reward_function"] = args["terminal_reward_function"]

    return env_config, eval_env_config
