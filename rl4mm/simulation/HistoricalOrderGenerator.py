from collections import deque
from datetime import datetime
from typing import Deque
import warnings

import pandas as pd

from rl4mm.database.HistoricalDatabase import HistoricalDatabase
from rl4mm.orderbook.create_order import create_order
from rl4mm.orderbook.models import Order
from rl4mm.simulation.OrderGenerator import OrderGenerator


class HistoricalOrderGenerator(OrderGenerator):
    name = "historical"

    def __init__(
        self,
        ticker: str = "MSFT",
        database: HistoricalDatabase = None,
        preload_orders: bool = True,
    ):
        self.ticker = ticker
        self.database = database or HistoricalDatabase()
        self.preload_orders = preload_orders
        if self.preload_orders:
            self.episode_messages = pd.DataFrame([])
            self.start_of_episode = datetime.max
            self.end_of_episode = datetime.min
        self.exchange_name = "NASDAQ"  # Here, we are only using LOBSTER data for now

    def generate_orders(self, start_date: datetime, end_date: datetime) -> Deque[Order]:
        if self.preload_orders:
            assert (self.start_of_episode < self._get_mid_datetime(start_date, end_date)) and (
                self._get_mid_datetime(start_date, end_date) < self.end_of_episode
            ), f"Cannot generate orders between {start_date} and {end_date} as they have not been stored locally yet."
            messages = self.episode_messages[
                (self.episode_messages.timestamp > start_date) & (self.episode_messages.timestamp <= end_date)
            ]
        else:
            messages = self.database.get_messages(start_date, end_date, self.ticker)
            messages = self._process_messages_and_add_internal(messages)
        if len(messages) == 0:
            return deque()
        else:
            return deque(messages.internal_message)

    @staticmethod
    def _remove_hidden_executions(messages: pd.DataFrame):
        if messages.empty:
            warnings.warn("DataFrame is empty.")
            return messages
        else:
            assert (
                "cross_trade" not in messages.message_type.unique()
            ), "Trying to step forward before initial cross-trade!"
            return messages[messages.message_type != "market_hidden"]

    def preload_episode_orders(self, min_date: datetime, max_date: datetime):
        messages = self.database.get_messages(min_date, max_date, self.ticker)
        self.episode_messages = self._process_messages_and_add_internal(messages)
        self.start_of_episode = min_date
        self.end_of_episode = max_date

    @staticmethod
    def _get_mid_datetime(datetime_1: datetime, datetime_2: datetime):
        return (max(datetime_1, datetime_2) - min(datetime_1, datetime_2)) / 2 + min(datetime_1, datetime_2)

    def _process_messages_and_add_internal(self, messages: pd.DataFrame):
        messages = self._remove_hidden_executions(messages)  #
        internal_messages = messages.apply(get_order_from_external_message, axis=1).values
        if len(internal_messages) > 0:
            messages = messages.assign(internal_message=internal_messages)
        return messages


def get_order_from_external_message(message: pd.Series):
    return create_order(
        order_type=message.message_type,
        order_dict=dict(
            timestamp=message.timestamp,
            price=message.price,
            volume=message.volume,
            direction=message.direction,
            ticker=message.ticker,
            internal_id=None,
            external_id=message.external_id,
            is_external=True,
        ),
    )
