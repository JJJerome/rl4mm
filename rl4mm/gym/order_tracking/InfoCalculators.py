import abc

import numpy as np

from rl4mm.features.Features import State
from rl4mm.gym.action_interpretation.OrderDistributors import BetaOrderDistributor, OrderDistributor

TICK_SIZE = 100  # TODO: add auto adjustment for small tick stocks


class InfoCalculator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def calculate(self, internal_state: State, action: np.ndarray):
        pass


class SimpleInfoCalculator(InfoCalculator):
    def __init__(
        self,
        market_order_fraction_of_inventory: float = 0.0,
        enter_spread: bool = False,
        order_distributor: OrderDistributor = None,
        concentration: float = None,
    ):
        self.market_order_count = 0
        self.market_order_total_volume = 0
        self.market_order_fraction_of_inventory = market_order_fraction_of_inventory
        self.enter_spread = enter_spread
        self.order_distributor = order_distributor or BetaOrderDistributor(concentration=concentration)

    def calculate(self, internal_state: State, action: np.ndarray):
        spreads_and_offsets = self.calculate_agent_spreads_and_midprice_offset(internal_state, action)
        info_dict = dict(
            asset_price=internal_state.price,
            inventory=internal_state.portfolio.inventory,
            cash=internal_state.portfolio.cash,
            aum=self.calculate_aum(internal_state),
            market_spread=internal_state.orderbook.spread,
            agent_spread=spreads_and_offsets[0],
            agent_weighted_spread=[1],
            midprice_offset=spreads_and_offsets[2],
            weighted_midprice_offset=spreads_and_offsets[3],
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
            if np.abs(internal_state.portfolio.inventory) > action[-1]:
                self._update_market_order_count_and_volume(internal_state.portfolio.inventory)
        info_dict["market_order_count"] = self.market_order_count
        info_dict["market_order_total_volume"] = self.market_order_total_volume
        return info_dict

    def calculate_agent_spreads_and_midprice_offset(self, internal_state: State, action: np.ndarray):
        orders = self.order_distributor.convert_action(action)
        total_volume = self.order_distributor.active_volume
        n_levels = self.order_distributor.quote_levels
        level_distances = np.array(range(n_levels))
        # calculate regular spread and midprice offset
        best_buy = np.sign(orders["buy"]).argmax()
        best_sell = np.sign(orders["sell"]).argmax()
        midprice_offset = (best_sell - best_buy) / 2
        spread = best_buy + best_sell
        # calculate weighted spread and midprice offset
        buy_centre_of_mass = np.dot(orders["buy"], level_distances) / total_volume
        sell_centre_of_mass = np.dot(orders["sell"], level_distances) / total_volume
        weighted_midprice_offset = (sell_centre_of_mass - buy_centre_of_mass) / 2
        weighted_spread = buy_centre_of_mass + sell_centre_of_mass
        if not self.enter_spread:
            midprice_offset += internal_state.orderbook.spread
            weighted_midprice_offset += internal_state.orderbook.spread
        return spread, weighted_spread, midprice_offset, weighted_midprice_offset

    def calculate_market_spread(self, internal_state: State):
        return internal_state.orderbook.spread

    def _update_market_order_count_and_volume(self, inventory: int):
        self.market_order_count += 1
        self.market_order_total_volume += np.round(np.abs(inventory) * self.market_order_fraction_of_inventory)

    def calculate_aum(self, internal_state: State) -> float:
        return internal_state.portfolio.cash + internal_state.price * internal_state.portfolio.inventory
