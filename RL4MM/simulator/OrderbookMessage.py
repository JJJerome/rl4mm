from typing import Dict, Any


from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class EventType(Enum):
    SUBMISSION = "submission"
    DELETION = "deletion"
    CANCELLATION = "cancellation"
    EXECUTION_VISIBLE = "execution_visible"
    EXECUTION_HIDDEN = "execution_hidden"
    TRADING_HALT = "trading_halt"


@dataclass
class OrderbookMessage:
    _id: str
    datetime: datetime
    event_type: EventType
    ticker: str
    size: float
    price: float
    side: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "_id": self._id,
            "datetime": self.datetime,
            "event_type": self.event_type,
            "ticker": self.ticker,
            "size": self.size,
            "price": self.price,
            "side": self.side,
        }
