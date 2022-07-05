import gym
import numpy as np

from RL4MM.agents.Agent import Agent

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


class TeradactylAgent(Agent):
    def __init__(self, max_inventory=None, kappa=10, default_a=3, default_b=1):
        self.max_inventory = max_inventory
        self.kappa = kappa
        self.default_a = default_a
        self.default_b = default_b

        # TODO: fix
        if max_inventory is None:
            self.denom = 100
        else:
            self.denom = self.max_inventory

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
            tmp = np.array([self.default_a, self.default_b, self.default_a, self.default_b])

        else:

            clamp_to_unit_interval = lambda x: max(min(x, 1), -1)

            omega_bid = 0.5 * (1 + clamp_to_unit_interval(inventory / self.denom))
            omega_ask = 0.5 * (1 - clamp_to_unit_interval(inventory / self.denom))

            alpha_bid = get_alpha(omega_bid, self.kappa)
            alpha_ask = get_alpha(omega_ask, self.kappa)

            beta_bid = get_beta(omega_bid, self.kappa)
            beta_ask = get_beta(omega_ask, self.kappa)

            tmp = np.array([alpha_bid, beta_bid, alpha_ask, beta_ask])

        if self.max_inventory is not None:
            tmp = np.append(tmp, self.max_inventory)

        return tmp

    def get_name(self):
        return (
            f"Teradactyl_def_a_{self.default_a}_def_b_{self.default_b}_kappa_{self.kappa}_max_inv_{self.max_inventory}"
        )


class ContinuousTeradactyl(Agent):
    def __init__(
        self,
        max_inventory=None,
        default_kappa: float = 10.0,
        default_omega: float = 0.5,
        max_kappa: float = 10.0,
        exponent: float = 1.0,
        market_clearing: bool = False,
    ):
        self.max_inventory = max_inventory
        self.default_kappa = default_kappa
        self.default_omega = default_omega
        self.max_kappa = max_kappa
        self.exponent = exponent
        self.market_clearing = market_clearing
        self.eps = 0.00001  # np.finfo(float).eps

        if max_inventory is None:
            self.denom = 100
        else:
            self.denom = self.max_inventory

    def get_omega_bid_and_ask(self, inv: int):
        if inv >= 0:
            omega_bid = self.default_omega * (
                1 + (1 / self.default_omega - 1) * self.clamp_to_unit(inv / self.denom) ** self.exponent
            )
            omega_ask = self.default_omega * (1 - self.clamp_to_unit(inv / self.denom) ** self.exponent)
        else:
            omega_bid = self.default_omega * (1 - abs(self.clamp_to_unit(inv / self.denom)) ** self.exponent)
            omega_ask = self.default_omega * (
                1 + (1 / self.default_omega - 1) * abs(self.clamp_to_unit(inv / self.denom)) ** self.exponent
            )
        return omega_bid, omega_ask

    def get_kappa(self, inventory: int):
        return (
            abs((self.max_kappa - self.default_kappa) * inventory / self.max_inventory) ** self.exponent
            + self.default_kappa
        )

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

        inventory = state[3]

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
        return f"ContinuousTeradactyl_def_omega_{self.default_omega}_def_kappa_{self.default_kappa}_max_inv_{self.max_inventory}_max_kappa_{self.max_kappa}"

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
