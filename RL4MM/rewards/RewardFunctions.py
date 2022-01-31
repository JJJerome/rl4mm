import abc

from pydantic import NonNegativeFloat, PositiveFloat, PositiveInt

from RL4MM.base import State, Action


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


class InventoryAdjustedPnL(RewardFunction):
    def __init__(
        self,
        per_step_inventory_aversion: NonNegativeFloat = 0.01,
        terminal_inventory_aversion: NonNegativeFloat = 0.0,
        n_steps: PositiveInt = 200,
        inventory_aversion_exponent: PositiveFloat = 2.0,
    ):
        self.per_step_inventory_aversion = per_step_inventory_aversion
        self.terminal_inventory_aversion = terminal_inventory_aversion
        self.pnl_reward = PnL()
        self.n_steps = n_steps
        self.inventory_aversion_exponent = inventory_aversion_exponent

    def calculate(self, current_state: State, action: Action, next_state: State) -> float:
        if current_state[3] == self.n_steps - 1:
            return (
                self.pnl_reward.calculate(current_state, action, next_state)
                - self.terminal_inventory_aversion * abs(next_state[2]) ** self.inventory_aversion_exponent
            )
        return (
            self.pnl_reward.calculate(current_state, action, next_state)
            - self.per_step_inventory_aversion * abs(current_state[2]) ** self.inventory_aversion_exponent
        )
