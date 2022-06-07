from collections import deque
from datetime import datetime
from typing import Deque
import warnings

import pandas as pd

from RL4MM.database.HistoricalDatabase import HistoricalDatabase
from RL4MM.orderbook.create_order import create_order
from RL4MM.orderbook.models import Order, OrderDict
from RL4MM.simulation.OrderGenerator import OrderGenerator


class HistoricalOrderGenerator(OrderGenerator):
    name = "historical"

    def __init__(self, ticker: str = "MSFT", database: HistoricalDatabase = None):
        self.ticker = ticker
        self.database = database or HistoricalDatabase()
        self.exchange_name = "NASDAQ"  # Here, we are only using LOBSTER data for now

    def generate_orders(self, start_date: datetime, end_date: datetime) -> Deque[Order]:
        messages = self.database.get_messages(start_date, end_date, self.ticker)
        messages = self._remove_hidden_executions(messages)
        return deque(get_order_from_external_message(m) for m in messages.itertuples())

    @staticmethod
    def _remove_hidden_executions(messages: pd.DataFrame):
        if messages.empty:
            warnings.warn('DataFrame is empty.')
            return messages
        else:
            assert "cross_trade" not in messages.message_type.unique(), "Trying to step forward before initial cross-trade!"
            return messages[messages.message_type != "market_hidden"]


def get_order_from_external_message(message: pd.Series):
    order_dict = OrderDict(
        timestamp=message.timestamp,
        price=message.price,
        volume=message.volume,
        direction=message.direction,
        ticker=message.ticker,
        internal_id=None,
        external_id=message.external_id,
        is_external=True,
    )
    return create_order(order_type=message.message_type, order_dict=order_dict)
