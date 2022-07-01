import argparse
import os
import copy
import numpy as np

from RL4MM.database.HistoricalDatabase import HistoricalDatabase
from RL4MM.gym.utils import env_creator
from RL4MM.gym.utils import generate_trajectory, plot_reward_distributions, get_episode_summary_dict
from RL4MM.agents.baseline_agents import RandomAgent, FixedActionAgent, TeradactylAgent
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
        "market_order_fraction_of_inventory": args["market_order_fraction"],
        "min_quote_level": args["min_quote_level"],
        "max_quote_level": args["max_quote_level"],
        "enter_spread": args["enter_spread"],
    }

    eval_env_config = copy.deepcopy(env_config)
    eval_env_config["min_date"] = args["min_date_eval"]
    eval_env_config["max_date"] = args["max_date_eval"]
    eval_env_config["per_step_reward_function"] = args["eval_per_step_reward_function"]
    eval_env_config["terminal_reward_function"] = args["terminal_reward_function"]

    return env_config, eval_env_config


def parse_args():
    # -------------------- Training Args ----------------------
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-ni", "--n_iterations", default=1000, help="Training iterations.", type=int)
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
    parser.add_argument(
        "-mof", "--market_order_fraction", default=0.2, help="Market order clearing fraction of inventory.", type=float
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
    # parser.add_argument("-con", "--concentration", default=10, help="Concentration of the order distributor.", type=int)
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
    # -------------------------------------------------
    args = vars(parser.parse_args())
    return args


if __name__ == "__main__":

    args = parse_args()
    env_config, _ = get_configs(args)
    # env = env_creator(env_config)

    ###########################################################################
    # Random agent
    ###########################################################################
    # agent = RandomAgent(env)

    ###########################################################################
    # two (1,1) beta distributions and a 1000 max inventory limit
    ###########################################################################
    # agent = FixedActionAgent(np.array([1,1,1,1,1000]))

    ###########################################################################
    # Away from best prices
    ###########################################################################
    # agent = FixedActionAgent(np.array([10,1,10,1,1000]))

    ###########################################################################
    # Close to best prices
    ###########################################################################
    # agent = FixedActionAgent(np.array([1,10,1,10,1000]))

    ###########################################################################
    # FixedAction - sweep
    ###########################################################################
    # for a in [1, 5, 10]:
    # for b in [5,10,20]:
    # for max_inv in [10, 100, 1000]:
    # agent = FixedActionAgent(np.array([a,b,a,b,max_inv]))
    # plot_reward_distributions(agent, env, n_iterations=30)

    ###########################################################################
    # Teradactyl - sweep
    ###########################################################################
    # for max_inv in [10,100,500]:
    # for kappa in [5,10,20]:
    # agent = TeradactylAgent(max_inventory=max_inv, kappa=kappa)
    # # obs, acts, rews, infs = generate_trajectory(agent, env)
    # plot_reward_distributions(agent, env, n_iterations=50)

    ###########################################################################
    # Teradactyl - single run
    ###########################################################################

    # agent = TeradactylAgent(max_inventory=10, kappa=10)

    # No max_inventory so there will only be 4 actions
    # agent = TeradactylAgent(kappa=10)

    # n_iterations = 5

    # emd1 = get_episode_summary_dict(agent, env_config, n_iterations, PARALLEL_FLAG=True)
    # emd2 = get_episode_summary_dict(agent, env_config, n_iterations, PARALLEL_FLAG=False)

    # fname = 'Teradactyl_def_a_3_def_b_1_kappa_10_max_inv_None_SPY_2018-02-20_2018-02-20_10'

    # import json

    # with open(f'{fname}.json') as json_file:
    # data = json.load(json_file)
    # print(data)

    # plot_reward_distributions(ticker=env_config['ticker'],
    # min_date=env_config['min_date'],
    # max_date=env_config['min_date'],
    # agent_name=agent.get_name(),
    # episode_length=env_config['episode_length'],
    # episode_mean_dict=data)

    ###########################################################################
    # Teradactyl - sweep
    ###########################################################################

    for a in [1, 5]:
        for b in [1, 5]:
            for max_inv in [100, 500]:
                # for kappa in [10, 100]:
                agent = FixedActionAgent(np.array([a, b, a, b, max_inv]))
                # TeradactylAgent(default_a=a, default_b=b, max_inventory=max_inv, kappa=kappa)

                databases = [HistoricalDatabase() for _ in range(args["n_iterations"])]

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
                    market_order_fraction_of_inventory=env_config["market_order_fraction_of_inventory"],
                    min_quote_level=env_config["min_quote_level"],
                    max_quote_level=env_config["max_quote_level"],
                    enter_spread=env_config["enter_spread"],
                    episode_mean_dict=emd1,
                )
