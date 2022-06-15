import abc
from typing import List

from RL4MM.features.Features import InternalState
from RL4MM.orderbook.models import Order


class OrderTracker(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def track_orders(self, filled_orders: List[Order], internal_state: InternalState):
        pass

    @abc.abstractmethod
    def reset(self):
        pass


class SimpleOrderTracker(OrderTracker):
    def __init__(self):
        self.filled_orders = []
        self.prices = []
        self.inventories = []
        self.spreads = []

    def track_orders(self, filled_orders: List[Order], internal_state: InternalState):
        self.filled_orders += filled_orders
        self.prices.append(internal_state["asset_price"])
        self.inventories.append(internal_state["inventory"])
        last_snapshot = internal_state["book_snapshots"].iloc[-1]
        self.spreads.append(last_snapshot.sell_price_0 - last_snapshot.buy_price_0)

    def reset(self):
        self.filled_orders = []
        self.prices = []
        self.inventories = []
