from typing import TypeVar

import gym
import numpy as np

from copy import deepcopy
from gym.spaces import Box, Discrete, Tuple
from math import sqrt

from RL4MM.rewards.RewardFunctions import RewardFunction, PnL
from RL4MM.base import State, Action

# Coefficients from the original Avellaneda-Stoikov paper. TODO: make the framework flexible enough to permit RARL.
DRIFT = 0.0
VOLATILITY = 2.0
RATE_OF_ARRIVAL = 140
FILL_EXPONENT = 1.5
MAX_INVENTORY = 100
INITIAL_CASH = 100.0
INITIAL_INVENTORY = 0
INITIAL_STOCK_PRICE = 100.0

AsState = TypeVar("AsState", bound=State)


class AvellanedaStoikovEnvironment(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        episode_length: float = 1.0,
        n_steps: int = 200,
        reward_function: RewardFunction = None,
        continuous_observation_space: bool = True,  # This permits us to use out of the box algos from Stable-baselines
    ):
        super(AvellanedaStoikovEnvironment, self).__init__()
        self.episode_length = episode_length
        self.n_steps = n_steps
        self.reward_function = reward_function or PnL()
        self.continuous_observation_space = continuous_observation_space

        self.action_space = Box(low=0.0, high=np.inf, shape=(2,))  # agent chooses spread on bid and ask
        # observation space is (stock price, cash, inventory, step_number)
        self.observation_space = Box(
            low=np.zeros(4), high=np.array([np.inf, np.inf, MAX_INVENTORY, n_steps]), dtype=np.float64
        )

        if not continuous_observation_space:
            self.observation_space = Tuple(
                (
                    Box(low=0.0, high=np.inf, shape=(1, 1)),
                    Box(low=0.0, high=np.inf, shape=(1, 1)),
                    Discrete(MAX_INVENTORY),
                    Discrete(n_steps),
                )
            )
        self.state = np.array([])
        self.dt = self.episode_length / self.n_steps

    def reset(self):
        self.state = np.array([INITIAL_STOCK_PRICE, INITIAL_CASH, INITIAL_INVENTORY, 0])
        return self._convert_internal_state_to_obs(self.state)

    def step(self, action: Action):
        next_state = self._get_next_state(action)
        reward = self.reward_function.calculate(self.state, action, next_state)
        self.state = next_state
        done = self.state[3] == self.n_steps
        return self._convert_internal_state_to_obs(self.state), reward, done, {}

    def render(self, mode="human"):
        pass

    def _get_next_state(self, action: Action) -> State:
        next_state = deepcopy(self.state)
        next_state[0] += DRIFT * self.dt + VOLATILITY * sqrt(self.dt) * np.random.normal()
        next_state[3] += 1
        fill_prob_bid, fill_prob_ask = RATE_OF_ARRIVAL * np.exp(-FILL_EXPONENT * action) * self.dt
        unif_bid, unif_ask = np.random.random(2)
        if unif_bid > fill_prob_bid and unif_ask > fill_prob_ask:  # neither the agent's bid nor their ask is filled
            pass
        if unif_bid < fill_prob_bid and unif_ask > fill_prob_ask:  # only bid filled
            # Note that market order gets filled THEN asset midprice changes
            next_state[1] -= self.state[0] - action[0]
            next_state[2] += 1
        if unif_bid > fill_prob_bid and unif_ask < fill_prob_ask:  # only ask filled
            next_state[1] += self.state[0] + action[1]
            next_state[2] -= 1
        if unif_bid < fill_prob_bid and unif_ask < fill_prob_ask:  # both bid and ask filled
            next_state[1] += action[0] + action[1]
        return next_state

    def _convert_internal_state_to_obs(self, state: np.ndarray):
        if self.continuous_observation_space:
            return state
        else:
            return (
                np.array([[state[0]]], dtype=np.float32),
                np.array([[state[1]]], dtype=np.float32),
                state[2],
                state[3],
            )
