from typing import Literal, TypedDict, Optional

from dataclasses import dataclass
from datetime import datetime
from sortedcontainers.sorteddict import SortedDict


@dataclass
class Order:
    timestamp: datetime
    direction: Literal["bid", "ask"]
    ticker: str
    internal_id: Optional[int]
    external_id: Optional[int]
    is_external: bool


@dataclass
class MarketOrder(Order):
    volume: int


@dataclass
class LimitOrder(Order):
    price: float
    volume: int


@dataclass
class Deletion(Order):
    price: float
    volume: Optional[int]  # This is due to deletions for historical orders from the initial order book needing a volume


@dataclass
class Cancellation(Deletion):
    volume: int


class Orderbook(TypedDict):
    bid: SortedDict  # Note that SortedDict does not currently support typing. Type is SortedDict[float, Deque[Order]].
    ask: SortedDict
    ticker: str
