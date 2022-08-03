# from typing import Optional, List, Literal, Union
import sys

import numpy as np

if sys.version_info[0] == 3 and sys.version_info[1] >= 8:
    from typing import Optional, Literal, Union, TypedDict, Tuple, List
else:
    from typing import Optional, Union
    from typing_extensions import Literal, TypedDict

from dataclasses import dataclass
from datetime import datetime
from sortedcontainers.sorteddict import SortedDict


@dataclass
class Order:
    """Base class for Orders."""

    timestamp: datetime
    direction: Literal["buy", "sell"]
    ticker: str
    internal_id: Optional[int]
    external_id: Optional[int]
    is_external: bool

    def __lt__(self, other):
        return self.timestamp < other.timestamp


@dataclass
class MarketOrder(Order):
    volume: int
    price: Optional[int] = None


@dataclass
class LimitOrder(Order):
    price: int
    volume: int


@dataclass
class Deletion(Order):
    price: int
    volume: Optional[int]  # This is due to deletions for historical orders from the initial order book needing a volume


@dataclass
class Cancellation(Deletion):
    volume: int


FillableOrder = Union[MarketOrder, LimitOrder]
FilledOrderTuple = Tuple[List[FillableOrder], List[FillableOrder]]


@dataclass
class Orderbook:
    buy: SortedDict  # SortedDict does not currently support typing. Type is SortedDict[int, Deque[LimitOrder]].
    sell: SortedDict
    ticker: str
    tick_size: int

    @property
    def best_buy_price(self):
        return next(reversed(self.buy), 0)

    @property
    def best_sell_price(self):
        return next(iter(self.sell.keys()), np.infty)

    @property
    def best_buy_volume(self):
        return sum(order.volume for order in self.buy[self.best_buy_price])

    @property
    def best_sell_volume(self):
        return sum(order.volume for order in self.sell[self.best_sell_price])

    @property
    def midprice(self):
        return (self.best_sell_price + self.best_buy_price) / 2

    @property
    def imbalance(self):
        return (self.best_buy_volume - self.best_sell_volume) / (self.best_buy_volume + self.best_sell_volume)

    @property
    def microprice(self):
        return (1 + self.imbalance) / 2 * self.best_sell_price + (1 - self.imbalance) / 2 * self.best_buy_price


class OrderDict(TypedDict):
    timestamp: datetime
    price: Optional[int]
    volume: Optional[int]
    direction: Literal["buy", "sell"]
    ticker: str
    internal_id: Optional[int]
    external_id: Optional[int]
    is_external: bool


def get_best_sell_price(orderbook: Orderbook):
    return next(iter(orderbook.sell.keys()), np.infty)
