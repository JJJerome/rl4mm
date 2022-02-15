import gym
import numpy as np

from RL4MM.agents.Agent import Agent


def generate_trajectory(env: gym.Env, agent: Agent, seed: int = None):
    action_dimension = env.action_space.shape[0]
    observation_dimension = env.observation_space.shape[0]
    if seed is not None:
        np.random.seed(seed)
    obs = env.reset()
    observations = obs.reshape(1, observation_dimension)
    action: np.ndarray = agent.get_action(obs)
    actions = action.reshape(1, action_dimension)
    rewards = []
    while True:
        obs, reward, done, _ = env.step(action)
        observations = np.append(arr=observations, values=obs.reshape(1, observation_dimension), axis=0)
        rewards.append(reward)
        action = agent.get_action(obs)
        actions = np.append(arr=actions, values=action.reshape(1, action_dimension), axis=0)
        if done:
            break
    return observations, rewards, actions
