# from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.tune.registry import register_env
from RL4MM.gym.utils import env_creator

from ray import tune
import argparse
import ray

from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.suggest import ConcurrencyLimiter

from main_helper import (
    add_env_args,
    add_ray_args,
    get_env_configs,
    get_ray_config,
    get_tensorboard_logdir,
    get_rule_based_agent_and_custom_model_config,
)


def main(args):

    # TODO: any reason to have a different number of cpus for main.py and
    # tune_rule_based_agents.py
    num_cpus = args["num_workers"] + args["num_workers_eval"]
    ray.init(ignore_reinit_error=True, num_cpus=num_cpus)

    register_env("HistoricalOrderbookEnvironment", env_creator)

    env_config, eval_env_config = get_env_configs(args)

    # eval_env_config["per_step_reward_function"] = "PnL"
    # eval_env_config["terminal_reward_function"] = "PnL"

    rba, cmc = get_rule_based_agent_and_custom_model_config(args)

    ray_config = get_ray_config(args, env_config, eval_env_config, "tune_rule_based_agents", cmc)

    tensorboard_logdir = get_tensorboard_logdir(args, "tune_rule_based_agents")

    # ---------------- For testing.... ----------------------
    # Uncomment for basic testing
    # print(rule_based_agent(config=config).train())
    # print(FixedActionAgentWrapper(config=config).evaluate())
    # -------------------------------------------------------

    algo = BayesOptSearch(
        utility_kwargs={"kind": "ucb", "kappa": 2.5, "xi": 0.0},
        random_search_steps=5,
    )
    algo = ConcurrencyLimiter(algo, max_concurrent=10)
    scheduler = AsyncHyperBandScheduler()
    callbacks = (
        [WandbLoggerCallback(api_key_file=ray_config["wandb_api_key_dir"], project="RL4MM")]
        if ray_config["wandb"] is not None
        else None
    )
    analysis = tune.run(
        rba,
        name=args["ticker"],
        metric="episode_reward_mean",
        mode="max",
        search_alg=algo,
        scheduler=scheduler,
        num_samples=1000,
        config=ray_config,
        local_dir=tensorboard_logdir,
        checkpoint_at_end=True,
        resume="AUTO",
        stop={"training_iteration": 60},  # args["iterations"]},
        callbacks=callbacks,
    )
    print("Best hyperparameters found were: ", analysis.best_config)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")

    add_env_args(parser)
    add_ray_args(parser)

    parser.add_argument(
        "-rba",
        "--rule_based_agent",
        default="continuous_teradactyl",
        choices=["fixed", "teradactyl", "continuous_teradactyl"],
        help="Specify rule based agent.",
        type=str,
    )
    # parser.add_argument("-ex", "--experiment", default="bayesopt", help="The experiment to run", type=str)

    args = vars(parser.parse_args())

    main(args)
