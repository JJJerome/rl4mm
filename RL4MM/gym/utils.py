from typing import Dict

import gym
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm

from RL4MM.agents.Agent import Agent


def generate_trajectory(agent: Agent, env: gym.Env):
    observations = []
    rewards = []
    actions = []
    infos = []
    obs = env.reset()  # type:ignore
    observations.append(obs)
    while True:
        action = agent.get_action(obs)  # type:ignore
        obs, reward, done, info = env.step(action)
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        infos.append(info)
        if done:
            break
    return observations, actions, rewards, infos


def get_episode_summary_dict(agent: Agent, env: gym.Env, n_iterations: int = 100):
    episode_mean_dict: Dict = {"rewards": [], "actions": [], "inventory": [], "spread": []}
    for _ in tqdm(range(n_iterations), desc="Simulating trajectories"):
        _, actions, rewards, infos = generate_trajectory(agent=agent, env=env)
        episode_mean_dict["rewards"].append(np.mean(rewards))
        episode_mean_dict["actions"].append(np.mean(np.array(actions), axis=0)[:-1])
        episode_mean_dict["inventory"].append(np.mean([info["inventory"] for info in infos]))
        episode_mean_dict["spread"].append(np.mean([info["spread"] for info in infos]))
    return episode_mean_dict


def get_output_prefix(agent: Agent, env: gym.Env):
    env_str = f'{env.ticker}_{str(env.min_date.date())}_{str(env.max_date.date())}'
    agent_str = type(agent).__name__
    return agent_str + '_' + env_str

def plot_reward_distributions(agent: Agent, env: gym.Env, n_iterations: int = 100):
    sns.set()
    episode_mean_dict = get_episode_summary_dict(agent, env, n_iterations)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 6))
    plt.suptitle("Distribution of features")
    ax1.hist(episode_mean_dict["rewards"], bins=20)
    for action_loc in [0, 1, 2, 3]:
        ax2.hist(np.array(episode_mean_dict["actions"])[action_loc, :], bins=5, label="action " + str(action_loc))
        ax2.legend()
    ax3.hist(episode_mean_dict["inventory"], bins=20)
    ax4.hist(episode_mean_dict["spread"], bins=20)
    ax1.title.set_text("Mean rewards")
    ax2.title.set_text("Mean action")
    ax3.title.set_text("Mean inventory")
    ax4.title.set_text("Mean spread")
    fig.tight_layout()
    # plt.show()

    fig.savefig(f'{get_output_prefix(agent,env)}.pdf')
