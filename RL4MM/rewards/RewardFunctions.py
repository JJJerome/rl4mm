import sys
import abc
import numpy as np

from RL4MM.features.Features import InternalState

###############################################################################

def get_sharpe(aum_array):

    if np.min(aum_array) <= 0:
        raise Exception("AUM has gone non_positive")

    log_returns = np.diff(np.log(aum_array))
    simple_returns = np.exp(log_returns) - 1

    # ddof = 1 to get divisor n-1 in std
    # add sys.float_info.min to avoid e.g. 0/0 = inf
    sharpe = np.mean(simple_returns)/(np.std(simple_returns, ddof=1) + sys.float_info.min)

    return sharpe

###############################################################################

class RewardFunction(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def calculate(self, current_state: InternalState, next_state: InternalState) -> float:
        pass
    
    @abc.abstractmethod
    def reset(self):
        pass

class RollingSharpe(RewardFunction):
    def __init__(self, max_window_size: int=120, min_window_size: int=60):

        assert max_window_size >= min_window_size, "Error with window sizes"

        # the size of the window for the rolling mean and std
        self.max_window_size = max_window_size
        self.min_window_size = min_window_size

        # set self.n_filled and self.aum_array
        self.reset()

    def reset(self):
        # counter for how many elements of window have been filled
        # used to ensure we never compute Sharpes on tiny windows  
        # maxes out at self.max_window_size
        self.n_filled = 0

        # create array of nans; this array will contain our array of aum values
        # we use nans (rather than say zeros) to drop them during the init period
        # when the array is first filling up
        self.aum_array = np.full(self.max_window_size, np.nan)

    def calculate_aum(self, internal_state:InternalState) -> float:
        return internal_state["cash"] + internal_state["asset_price"] * internal_state["inventory"]

    def update_aum_array(self, current_state: InternalState, next_state: InternalState):
        """
        current_state not currently used
        Don't need it? Or use it but only use in first period?
        """       
        # calculate new aum
        new_aum = self.calculate_aum(next_state) 

        # self.aum array has the oldest aum in idx 0, newest in -1
        # first overwrite oldest
        self.aum_array[0] = new_aum
        # second cyclically shift to left, so newest is in -1
        self.aum_array = np.roll(self.aum_array, -1)

        # update self.n_filled, maxing out at self.max_window_size
        self.n_filled = min(self.n_filled+1, self.max_window_size)

    def calculate(self, current_state: InternalState, next_state: InternalState) -> float:

        # first update self.aum_array
        # self.update_aum_array(next_state)
        self.update_aum_array(current_state, next_state)

        # Not enough elements to compute a Sharpe
        if self.n_filled < self.min_window_size:
            return 0
        elif self.n_filled < self.max_window_size:
            # until self.n_filled is self.max_window size we need to drop nans
            return get_sharpe(self.aum_array[~np.isnan(self.aum_array)])
        else:
            return get_sharpe(self.aum_array)

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
