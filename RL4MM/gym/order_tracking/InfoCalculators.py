import abc
from typing import List, Optional

import numpy as np

from RL4MM.features.Features import InternalState
from RL4MM.gym.action_interpretation.OrderDistributors import BetaOrderDistributor, OrderDistributor
from RL4MM.orderbook.models import FillableOrder

TICK_SIZE = 100  # TODO: add auto adjustment for small tick stocks


class InfoCalculator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def calculate(self, filled_orders: List[FillableOrder], internal_state: InternalState, action: np.ndarray):
        pass


class SimpleInfoCalculator(InfoCalculator):
    def __init__(
        self,
        market_order_fraction_of_inventory: Optional[float] = None,
        enter_spread: bool = False,
        order_distributor: OrderDistributor = None,
    ):
        self.market_order_count = 0
        self.market_order_total_volume = 0
        self.market_order_fraction_of_inventory = market_order_fraction_of_inventory
        self.enter_spread = enter_spread
        self.order_distributor = order_distributor or BetaOrderDistributor()

    def calculate(self, filled_orders: List[FillableOrder], internal_state: InternalState, action: np.ndarray):
        info_dict = dict(
            asset_price=internal_state["asset_price"],
            inventory=internal_state["inventory"],
            market_spread=self.calculate_spread(internal_state),
            agent_weighted_spread=self.calculate_agent_weighted_spread_and_midprice_offset(internal_state, action)[0],
            midprice_offset=self.calculate_agent_weighted_spread_and_midprice_offset(internal_state, action)[1],
        )
        if len(action) == 2 or len(action) == 3:
            info_dict["bid_action"] = (action[[0]],)
            info_dict["ask_action"] = (action[[1]],)
        elif len(action) == 4 or len(action) == 5:
            info_dict["bid_action"] = (action[[0, 1]],)
            info_dict["ask_action"] = (action[[2, 3]],)
        else:
            raise NotImplementedError("Action dim should be 2, 3, 4, 5 based on current options.")

        if len(action) == 3 or len(action) == 5:
            info_dict["market_order_action"] = (action[[-1]],)
            if np.abs(internal_state["inventory"]) > action[-1]:
                self._update_market_order_count_and_volume(internal_state["inventory"])
        info_dict["market_order_count"] = self.market_order_count
        info_dict["market_order_total_volume"] = self.market_order_total_volume
        return info_dict

    def calculate_agent_weighted_spread_and_midprice_offset(self, internal_state: InternalState, action: np.ndarray):
        orders = self.order_distributor.convert_action(action)
        total_volume = self.order_distributor.active_volume
        n_levels = self.order_distributor.n_levels
        level_distances = np.array(range(n_levels))
        buy_centre_of_mass = np.dot(orders["buy"], level_distances) / total_volume
        sell_centre_of_mass = np.dot(orders["sell"], level_distances) / total_volume
        midprice_offset = (sell_centre_of_mass - buy_centre_of_mass) / 2
        if self.enter_spread:
            return (buy_centre_of_mass + sell_centre_of_mass), midprice_offset
        else:
            return (buy_centre_of_mass + sell_centre_of_mass + self.calculate_spread(internal_state)), midprice_offset

    def calculate_spread(self, internal_state: InternalState):
        last_snapshot = internal_state["book_snapshots"].iloc[-1]
        return (last_snapshot.sell_price_0 - last_snapshot.buy_price_0) / TICK_SIZE

    def _update_market_order_count_and_volume(self, inventory: int):
        self.market_order_count += 1
        self.market_order_total_volume += self.market_order_fraction_of_inventory * inventory
