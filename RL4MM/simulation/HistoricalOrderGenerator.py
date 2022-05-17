from collections import deque
from datetime import datetime
from typing import Deque

import pandas as pd

from RL4MM.database.HistoricalDatabase import HistoricalDatabase
from RL4MM.orderbook.models import Order, OrderType
from RL4MM.simulation.OrderGenerator import OrderGenerator

LOBSTER_MESSAGE_TYPE_DICT = {
    "submission": OrderType.LIMIT,
    "execution_visible": OrderType.MARKET,
    "cancellation": OrderType.CANCELLATION,
    "deletion": OrderType.DELETION,
}


class HistoricalOrderGenerator(OrderGenerator):
    name = "historical"

    def __init__(self, ticker: str = "MSFT", database: HistoricalDatabase = None):
        self.ticker = ticker
        self.database = database or HistoricalDatabase()
        self.exchange_name = "NASDAQ"  # Here, we are only using LOBSTER data for now

    def generate_orders(self, start_date: datetime, end_date: datetime) -> Deque[Order]:
        messages = self.database.get_messages(start_date, end_date, self.exchange_name, self.ticker)
        messages = self._remove_hidden_executions(messages)
        return deque(self._get_order_from_external_message(m) for m in messages.itertuples())

    @staticmethod
    def _remove_hidden_executions(messages: pd.DataFrame):
        return messages[messages.message_type != "execution_hidden"]

    def _get_order_from_external_message(self, message: pd.Series):
        return Order(
            timestamp=message.timestamp,
            price=message.price,
            volume=message.volume,
            direction=self._get_order_direction_from_lobster_message_direction(message.direction, message.message_type),
            type=self._get_order_type_from_lobster_message_type(message.message_type),
            ticker=message.ticker,
            external_id=message.external_id,
        )

    @staticmethod
    def _get_order_type_from_lobster_message_type(message_type: str) -> OrderType:
        return LOBSTER_MESSAGE_TYPE_DICT[message_type]

    @staticmethod
    def _get_order_direction_from_lobster_message_direction(direction: str, message_type: str):
        if message_type != "execution_visible":
            return direction
        elif direction == "ask":
            return "bid"
        elif direction == "bid":
            return "ask"
