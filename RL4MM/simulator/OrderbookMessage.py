from typing import Dict, Any, Optional


from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class EventType(Enum):
    SUBMISSION = 1
    CANCELLATION = 2
    DELETION = 3
    EXECUTION_VISIBLE = 4
    EXECUTION_HIDDEN = 5
    CROSS_TRADE = 6
    TRADING_HALT = 7


@dataclass
class OrderbookMessage:
    _id: str
    datetime: datetime
    event_type: str
    ticker: str
    size: float
    price: float
    side: str
    queue_position: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "_id": self._id,
            "datetime": self.datetime,
            "event_type": self.event_type,
            "ticker": self.ticker,
            "size": self.size,
            "price": self.price,
            "side": self.side,
            "distance_to_fill": self.queue_position,
        }
