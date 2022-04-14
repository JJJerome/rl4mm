from datetime import datetime

import pandas as pd

from RL4MM.database.HistoricalDatabase import HistoricalDatabase
from RL4MM.orderbook.models import Order, OrderType
from RL4MM.simulation.OrderGenerator import OrderGenerator


class HistoricalOrderGenerator(OrderGenerator):
    def __init__(self, ticker: str = "MSFT", database: HistoricalDatabase = None ):
        self.ticker = ticker
        self.database = database or HistoricalDatabase()
        self.exchange_name = "NASDAQ"  # Here, we are only using LOBSTER data for now

    def generate_messages(self, start_date: datetime, end_date: datetime):
        messages = self.database.get_messages(start_date, end_date, self.exchange_name, self.ticker)
        messages = self._remove_hidden_executions(messages)


    @staticmethod
    def _remove_hidden_executions(messages: pd.DataFrame):
        return messages[messages.message_type != "execution_hidden"]

    @staticmethod
    def _get_order_from_external_message(message: pd.Series):
        return Order(
            timestamp=message.timestamp,
            price=message.price,
            volume=message.volume,
            direction=message.direction,
            type=OrderType(message.message_type),
            ticker=message.ticker,
            external_id=message.external_id,
        )

