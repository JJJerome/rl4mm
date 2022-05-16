from typing import Literal, TypedDict, Optional

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from sortedcontainers.sorteddict import SortedDict


class OrderType(Enum):
    LIMIT = "limit"
    MARKET = "market"
    CANCELLATION = "cancellation"
    DELETION = "deletion"


@dataclass
class Order:
    timestamp: datetime
    price: Optional[float]
    volume: Optional[int]
    direction: Literal["bid", "ask"]
    type: OrderType
    ticker: str
    internal_id: Optional[int] = None
    external_id: Optional[int] = None
    participant_id: Optional[str] = None
    is_external: bool = True


class Orderbook(TypedDict):
    bid: SortedDict  # Note that SortedDict does not currently support typing. Type is SortedDict[float, Deque[Order]].
    ask: SortedDict
    ticker: str
