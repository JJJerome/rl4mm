import abc
from typing import List

import numpy as np

from RL4MM.features.Features import InternalState
from RL4MM.orderbook.models import FillableOrder


class InfoCalculator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def calculate(self, filled_orders: List[FillableOrder], internal_state: InternalState, action: np.ndarray):
        pass


class SimpleInfoCalculator(InfoCalculator):
    def calculate(self, filled_orders: List[FillableOrder], internal_state: InternalState, action: np.ndarray):
        last_snapshot = internal_state["book_snapshots"].iloc[-1]
        info_dict = dict(
            price=internal_state["asset_price"],
            inventory=internal_state["inventory"],
            spread=last_snapshot.sell_price_0 - last_snapshot.buy_price_0,
        )
        if len(action) == 2 or len(action) == 3:
            info_dict["bid_action"] = (action[[0]],)
            info_dict["ask_action"] = (action[[1]],)
        elif len(action) == 4 or len(action) == 5:
            info_dict["bid_action"] = (action[[0, 1]],)
            info_dict["ask_action"] = (action[[2, 3]],)
        else:
            raise NotImplementedError("Action dim should be 2, 3, 4, 5 based on current options.")

        if len(action) == 3:
            info_dict["market_order_action"] = (action[[2]],)
        elif len(action) == 5:
            info_dict["market_order_action"] = (action[[4]],)
        return info_dict
