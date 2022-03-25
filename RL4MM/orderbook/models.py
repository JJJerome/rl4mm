from typing import Literal, TypedDict, Optional

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from sortedcontainers.sorteddict import SortedDict


class OrderType(Enum):
    SUBMISSION = "submission"
    CANCELLATION = "cancellation"
    DELETION = "deletion"
    EXECUTION = "execution_visible"


@dataclass
class Order:
    timestamp: datetime
    price: float
    volume: int
    direction: Literal["bid", "ask"]
    type: OrderType
    ticker: str
    internal_id: Optional[int] = None
    external_id: Optional[int] = None
    is_external: bool = True


class Orderbook(TypedDict):
    bid: SortedDict  # Note that SortedDict does not currently support typing. Type is SortedDict[float, Deque[Order]].
    ask: SortedDict
    ticker: str
