import gym

from RL4MM.agents.Agent import Agent


def generate_trajectory(env: gym.Env, agent: Agent):
    observations = []
    rewards = []
    actions = []
    obs = env.reset()
    observations.append(obs)
    while True:
        action = agent.get_action(obs)
        obs, reward, done, _ = env.step(action)
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        if done:
            break
    rewards = np.array(rewards)
    return observations, actions, rewards
