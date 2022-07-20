import abc
import numpy as np

from RL4MM.features.Features import InternalState

###############################################################################

def get_sharpe(aum_array):
    
    # print("MIN:", np.min(aum_array))

    if np.min(aum_array) <= 0:
        raise Exception("AUM has gone non_positive")

    log_returns = np.diff(np.log(aum_array))
    simple_returns = np.exp(log_returns) - 1

    # print("STD RETURNS:", np.std(simple_returns))

    sharpe = np.mean(simple_returns)/np.std(simple_returns)

    print("Sharpe:", sharpe)

    return sharpe

###############################################################################

class RewardFunction(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def calculate(self, current_state: InternalState, next_state: InternalState) -> float:
        pass

class RollingSharpe(RewardFunction):
    def __init__(self, window_size: int):

        # the size of the window for the rolling mean and std
        self.window_size = window_size

        # counter for how many elements of window have been filled
        # self.n_filled = 0

        # create array of nans; this array will contain our array of aum values
        # we use nans (rather than say zeros) to drop them during the init period
        # when the array is first filling up
        self.aum_array = np.full(self.window_size, np.nan)

        # current index; incremented cyclically in update_data
        self.idx = 0

    def calculate_aum(self, internal_state:InternalState) -> float:
        return internal_state["cash"] + internal_state["asset_price"] * internal_state["inventory"]

    def update_aum_array(self, current_state: InternalState, next_state: InternalState):
        """
        current_state not currently used
        Don't need it? Or use it but only use in first period?
        """       
        # calculate new aum
        new_aum = self.calculate_aum(next_state) 

        # write this to self.data
        self.aum_array[self.idx] = new_aum

        # update self.idx ready for next time
        self.idx = (self.idx + 1) % self.window_size

    def calculate(self, current_state: InternalState, next_state: InternalState) -> float:
        
        # first update self.aum_array
        # self.update_aum_array(next_state)
        self.update_aum_array(current_state, next_state)

        # during init we want to ignore nans
        tmp = self.aum_array[~np.isnan(self.aum_array)]

        # return sharpe
        return get_sharpe(tmp)

class PnL(RewardFunction):
    def calculate(self, current_state: InternalState, next_state: InternalState) -> float:
        current_value = current_state["cash"] + current_state["inventory"] * current_state["asset_price"]
        next_value = next_state["cash"] + next_state["inventory"] * next_state["asset_price"]
        return next_value - current_value


class InventoryAdjustedPnL(RewardFunction):
    def __init__(self, inventory_aversion: float, asymmetrically_dampened: bool = False):
        self.inventory_aversion = inventory_aversion
        self.pnl = PnL()
        self.asymmetrically_dampened = asymmetrically_dampened

    def calculate(self, current_state: InternalState, next_state: InternalState) -> float:
        delta_midprice = next_state["asset_price"] - current_state["asset_price"]
        dampened_inventory_term = self.inventory_aversion * next_state["inventory"] * delta_midprice
        if self.asymmetrically_dampened:
            dampened_inventory_term = max(0, dampened_inventory_term)
        return self.pnl.calculate(current_state, next_state) - dampened_inventory_term
