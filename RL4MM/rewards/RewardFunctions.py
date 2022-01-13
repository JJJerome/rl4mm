import abc
import numpy as np
from typing import TypeVar

State = TypeVar("State", int, np.ndarray, tuple)
Action = TypeVar("Action", int, np.ndarray, tuple)


class RewardFunction(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def calculate(self, current_state: State, action: Action, next_state: State) -> float:
        pass


class PnL(RewardFunction):
    """A simple profit and loss reward function of the 'mark-to-market' value of the agent's portfolio."""

    def calculate(self, current_state, action, next_state) -> float:
        current_market_value = current_state[1] + current_state[0] * current_state[2]
        next_market_value = next_state[1] + next_state[0] * next_state[2]
        return next_market_value - current_market_value
