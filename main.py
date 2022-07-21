import argparse
import os

# from datetime import timedelta
import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.schedulers import ASHAScheduler
from ray.tune.registry import register_env
import random

from RL4MM.gym.utils import env_creator

from RL4MM.utils.custom_metrics_callback import Custom_Callbacks
import copy

from RL4MM.utils.utils import save_best_checkpoint_path, get_timedelta_from_clock_time
from RL4MM.utils.utils import boolean_string

from main_helper import add_env_args, add_ray_args, get_env_configs, get_ray_config

def main(args):

    ray.init(ignore_reinit_error=True, num_cpus=(args["num_workers"] + args["num_workers_eval"] + 1) * 4)

    env_config, eval_env_config = get_env_configs(args)

    register_env("HistoricalOrderbookEnvironment", env_creator)

    ray_config = get_ray_config(args, env_config)

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

    # import ray.rllib.agents.ppo as ppo
    # trainer = ppo.PPOTrainer(config=config)
    # print(trainer.train())

    analysis = tune.run(
        "PPO",
        num_samples=8,
        config=ray_config,
        checkpoint_at_end=True,
        local_dir=tensorboard_logdir,
        stop={"training_iteration": args["iterations"]},
        scheduler=ASHAScheduler(metric="episode_reward_mean", mode="max"),
    )

    # TODO: use eval_env_config !!

    best_checkpoint = analysis.get_trial_checkpoints_paths(
        trial=analysis.get_best_trial("episode_reward_mean"), metric="episode_reward_mean"
    )
    print(best_checkpoint)
    path_to_save_dir = args["output"] or "/home/ray"
    save_best_checkpoint_path(path_to_save_dir, best_checkpoint[0][0])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")

    add_env_args(parser)
    add_ray_args(parser)

    args = vars(parser.parse_args())

    main(args)
