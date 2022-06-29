import gym
import numpy as np

from RL4MM.agents.Agent import Agent


class RandomAgent(Agent):
    def __init__(self, env: gym.Env, seed: int = None):
        self.action_space = env.action_space
        self.action_space.seed(seed)

    def get_action(self, state: np.ndarray) -> np.ndarray:
        return self.action_space.sample()

    def get_name(self):
        return "RandomAgent"


class FixedActionAgent(Agent):
    def __init__(self, fixed_action: np.ndarray):
        self.fixed_action = fixed_action

    def get_action(self, state: np.ndarray) -> np.ndarray:
        return self.fixed_action

    def get_name(self):
        tmp = "_".join(map(str, self.fixed_action))
        return f"FixedAction_{tmp}"


class TeradactylAgent(Agent):
    def __init__(self, max_inventory=None, kappa=10, default_a=3, default_b=1):
        self.max_inventory = max_inventory
        self.kappa = kappa
        self.default_a = default_a
        self.default_b = default_b

    def get_action(self, state: np.ndarray) -> np.ndarray:

        ############################
        # self.features
        ############################
        # 0: Spread,
        # 1: MidpriceMove,
        # 2: Volatility,
        # 3: Inventory,
        # 4: TimeRemaining,
        # 5: MicroPrice
        ############################

        def get_alpha(omega, kappa):
            return (omega * (kappa - 2)) + 1

        def get_beta(omega, kappa):
            return (1 - omega) * (kappa - 2) + 1

        inventory = state[3]

        if inventory == 0:
            return np.array([self.default_a, self.default_b, self.default_a, self.default_b, self.max_inventory])
        else:

            omega_bid = 0.5 * (1 + (inventory / self.max_inventory))
            omega_ask = 0.5 * (1 - (inventory / self.max_inventory))

            alpha_bid = get_alpha(omega_bid, self.kappa)
            alpha_ask = get_alpha(omega_ask, self.kappa)

            beta_bid = get_beta(omega_bid, self.kappa)
            beta_ask = get_beta(omega_ask, self.kappa)

            tmp = np.array([alpha_bid, beta_bid, alpha_ask, beta_ask])
            if self.max_inventory is not None:
                tmp = np.append(tmp, self.max_inventory)
            # print("inventory:", inventory)
            # print(tmp)

            return tmp

    def get_name(self):
        return (
            f"Teradactyl_def_a_{self.default_a}_def_b_{self.default_b}_kappa_{self.kappa}_max_inv_{self.max_inventory}"
        )


class HumanAgent(Agent):
    def get_action(self, state: np.ndarray):
        action_0 = float(input(f"Current state is {state}. How large do you want to set action[0]? "))
        action_1 = float(input(f"Current state is {state}. How large do you want to set action[1]? "))
        action_2 = float(input(f"Current state is {state}. How large do you want to set action[2]? "))
        action_3 = float(input(f"Current state is {state}. How large do you want to set action[3]? "))
        action_4 = float(input(f"Current state is {state}. How large do you want to set action[4]? "))
        return np.array([action_0, action_1, action_2, action_3, action_4])
