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
    buy: SortedDict  # SortedDict does not currently support typing. Type is SortedDict[int, Deque[Order]].
    sell: SortedDict
    ticker: str
    tick_size: int


class OrderDict(TypedDict):
    timestamp: datetime
    price: Optional[int]
    volume: Optional[int]
    direction: Literal["buy", "sell"]
    ticker: str
    internal_id: Optional[int]
    external_id: Optional[int]
    is_external: bool


def get_best_buy_price(orderbook: Orderbook):
    return next(reversed(orderbook.buy), 0)


def get_best_sell_price(orderbook: Orderbook):
    return next(iter(orderbook.sell.keys()), np.infty)