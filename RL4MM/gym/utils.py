import gym

from RL4MM.agents.Agent import Agent


def generate_trajectory(env: gym.Env, agent: Agent):
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
