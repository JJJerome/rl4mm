"""Example of using RLlib's debug callbacks.
Here we use callbacks to track the average CartPole pole angle magnitude as a
custom metric.
"""

from typing import Dict
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy


class Custom_Callbacks(DefaultCallbacks):
    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Make sure this episode is ongoing.
        assert episode.length > 0, "ERROR: `on_episode_step()` callback should not be called right " "after env reset!"
        for key in episode._agent_to_last_info["agent0"]:
            episode.custom_metrics[key] = episode._agent_to_last_info["agent0"][key]

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Check if there are multiple episodes in a batch, i.e.
        # "batch_mode": "truncate_episodes".
        if worker.policy_config["batch_mode"] == "truncate_episodes":
            # Make sure this episode is really done.
            assert episode.batch_builder.policy_collectors["default_policy"].batches[-1]["dones"][-1], (
                "ERROR: `on_episode_end()` should only be called " "after episode is done!"
            )
        for key in episode._agent_to_last_info["agent0"]:
            episode.custom_metrics[key] = episode._agent_to_last_info["agent0"][key]

    def on_train_result(self, *, result: dict, **kwargs):
        result["callback_ok"] = True
