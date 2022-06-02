import abc

from RL4MM.features.Features import InternalState


class RewardFunction(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def calculate(self, current_state: InternalState, next_state: InternalState) -> float:
        pass


class PnL(RewardFunction):
    def calculate(self, current_state: InternalState, next_state: InternalState) -> float:
        current_value = current_state["cash"] + current_state["inventory"] * current_state["asset_price"]
        next_value = next_state["cash"] + next_state["inventory"] * next_state["asset_price"]
        return next_value - current_value


class InventoryAdjustedPnL(RewardFunction):
    def __init__(self, inventory_aversion: float):
        self.inventory_aversion = inventory_aversion
        self.pnl = PnL()

    def calculate(self, current_state: InternalState, next_state: InternalState) -> float:
        return self.pnl.calculate(current_state, next_state) - self.inventory_aversion * (next_state["inventory"]) ** 2
