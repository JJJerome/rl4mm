from typing import Deque, Literal, OrderedDict, TypedDict, Optional

from enum import Enum

from dataclasses import dataclass
from datetime import datetime


class OrderType(Enum):
    SUBMISSION = "submission"
    CANCELLATION = "cancellation"
    DELETION = "deletion"
    EXECUTION = "execution_visible"


@dataclass
class Order:
    timestamp: datetime
    price: float
    size: int
    direction: Literal["bid", "ask"]
    type: OrderType
    ticker: str
    internal_id: Optional[int] = None
    external_id: Optional[int] = None
    is_external: bool = True


class Orderbook(TypedDict):
    bid: OrderedDict[float, Deque[Order]]
    ask: OrderedDict[float, Deque[Order]]
    ticker: str
