import gym
import numpy as np

from rl4mm.agents.Agent import Agent

TICK_SIZE = 100


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


class Teradactyl(Agent):
    def __init__(
        self,
        max_inventory=None,
        default_kappa: float = 10.0,
        default_omega: float = 0.5,
        max_kappa: float = 10.0,
        exponent: float = 1.0,
        market_clearing: bool = False,
        inventory_index: int = 3,
    ):
        self.max_inventory = max_inventory
        self.default_kappa = default_kappa
        self.default_omega = default_omega
        self.max_kappa = max_kappa
        self.exponent = exponent
        self.market_clearing = market_clearing
        self.inventory_index = inventory_index
        self.eps = 0.00001  # np.finfo(float).eps

        if max_inventory is None:
            self.denom = 100
        else:
            self.denom = self.max_inventory

    def get_omega_bid_and_ask(self, inventory: int):
        if inventory >= 0:
            omega_bid = self.default_omega * (
                1 + (1 / self.default_omega - 1) * self.clamp_to_unit(inventory / self.denom) ** self.exponent
            )
            omega_ask = self.default_omega * (1 - self.clamp_to_unit(inventory / self.denom) ** self.exponent)
        else:
            omega_bid = self.default_omega * (1 - abs(self.clamp_to_unit(inventory / self.denom)) ** self.exponent)
            omega_ask = self.default_omega * (
                1 + (1 / self.default_omega - 1) * abs(self.clamp_to_unit(inventory / self.denom)) ** self.exponent
            )
        return omega_bid, omega_ask

    def get_kappa(self, inventory: int):
        return (self.max_kappa - self.default_kappa) * abs(
            inventory / self.max_inventory
        ) ** self.exponent + self.default_kappa

    def get_action(self, state: np.ndarray) -> np.ndarray:
        inventory = state[self.inventory_index]
        omega_bid, omega_ask = self.get_omega_bid_and_ask(inventory)
        kappa = self.get_kappa(inventory)
        alpha_bid = self.calculate_alpha(omega_bid, kappa)
        alpha_ask = self.calculate_alpha(omega_ask, kappa)
        beta_bid = self.calculate_beta(omega_bid, kappa)
        beta_ask = self.calculate_beta(omega_ask, kappa)
        tmp = np.array([alpha_bid, beta_bid, alpha_ask, beta_ask])
        if self.market_clearing is True:
            tmp = np.append(tmp, self.max_inventory * 2)
        return tmp

    def get_name(self):
        return (
            f"Teradactyl_def_omega_{self.default_omega}_def_kappa_{self.default_kappa}_"
            + f"max_inv_{self.max_inventory}_max_kappa_{self.max_kappa}_exponent_{self.exponent}"
        )

    @staticmethod
    def calculate_alpha(omega, kappa):
        return (omega * (kappa - 2)) + 1

    @staticmethod
    def calculate_beta(omega, kappa):
        return (1 - omega) * (kappa - 2) + 1

    def clamp_to_unit(self, x: float, strict_containment: bool = True):
        if strict_containment:
            return max(min(x, 1 - self.eps), -1 + self.eps)
        else:
            return max(min(x, 1), -1)

    @staticmethod
    def calculate_omega(alpha: float, beta: float):
        return (alpha - 1) / (alpha + beta - 2)

    @staticmethod
    def calculate_kappa(alpha: float, beta: float):
        return alpha + beta


class HumanAgent(Agent):
    def get_action(self, state: np.ndarray):
        action_0 = float(input(f"Current state is {state}. How large do you want to set action[0]? "))
        action_1 = float(input(f"Current state is {state}. How large do you want to set action[1]? "))
        action_2 = float(input(f"Current state is {state}. How large do you want to set action[2]? "))
        action_3 = float(input(f"Current state is {state}. How large do you want to set action[3]? "))
        action_4 = float(input(f"Current state is {state}. How large do you want to set action[4]? "))
        return np.array([action_0, action_1, action_2, action_3, action_4])
