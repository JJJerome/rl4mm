from collections import deque
from datetime import datetime

import pandas as pd

from RL4MM.database.HistoricalDatabase import HistoricalDatabase
from RL4MM.orderbook.models import Order, OrderType
from RL4MM.simulation.OrderGenerator import OrderGenerator


class HistoricalOrderGenerator(OrderGenerator):
    name = "historical"

    def __init__(self, ticker: str = "MSFT", database: HistoricalDatabase = None):
        self.ticker = ticker
        self.database = database or HistoricalDatabase()
        self.exchange_name = "NASDAQ"  # Here, we are only using LOBSTER data for now

    def generate_orders(self, start_date: datetime, end_date: datetime):
        messages = self.database.get_messages(start_date, end_date, self.exchange_name, self.ticker)
        messages = self._remove_hidden_executions(messages)  # Ignore hidden executions as they don't affect the book
        return deque(messages.apply(self._get_order_from_message, axis=1))

    @staticmethod
    def _remove_hidden_executions(messages: pd.DataFrame):
        return messages[~messages.message_type.isin(["execution_hidden", "cross_trade"])]

    @staticmethod
    def _get_order_from_message(message: pd.Series):
        return Order(
            timestamp=message.timestamp,
            price=message.price,
            volume=message.volume,
            direction=message.direction,
            type=OrderType(message.message_type),
            ticker=message.ticker,
            external_id=message.external_id,
        )
