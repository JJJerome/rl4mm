import argparse

import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.schedulers import ASHAScheduler
from ray.tune.registry import register_env
import random

from RL4MM.utils.utils import save_best_checkpoint_path,\
                              get_timedelta_from_clock_time
from RL4MM.gym.utils import env_creator

from main_helper import add_env_args,\
                        add_ray_args,\
                        get_env_configs,\
                        get_ray_config,\
                        get_tensorboard_logdir

def main(args):

    num_cpus = (args["num_workers"] + args["num_workers_eval"] + 1)

    ray.init(ignore_reinit_error=True, num_cpus=num_cpus)

    env_config, eval_env_config = get_env_configs(args)

    register_env("HistoricalOrderbookEnvironment", env_creator)

    ray_config = get_ray_config(args, env_config, eval_env_config, 'main')

    tensorboard_logdir = get_tensorboard_logdir(args, 'main')

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

    best_checkpoint = analysis.get_trial_checkpoints_paths(
        trial=analysis.get_best_trial("episode_reward_mean"), 
        metric="episode_reward_mean",
        mode="max"
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
