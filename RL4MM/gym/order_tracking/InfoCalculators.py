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
            bid_action=action[0, 1],
            ask_action=action[1, 2],
        )
        return info_dict
