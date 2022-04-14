from datetime import datetime
from typing import List, TypedDict, Optional

import pandas as pd

from RL4MM.database.HistoricalDatabase import HistoricalDatabase
from RL4MM.orderbook.models import Order, OrderType, Orderbook
from RL4MM.orderbook.Exchange import Exchange

N_LEVELS = 10


class OutputDict(TypedDict):
    orderbook: Orderbook
    filled_orders: List[Order]


class HistoricalOrderbookSimulator:
    def __init__(
        self,
        ticker: str = "MSFT",
        exchange: Exchange = None,
        database: HistoricalDatabase = None,
    ) -> None:
        self.ticker = ticker
        self.exchange = exchange or Exchange(ticker=ticker)
        self.database = database or HistoricalDatabase()
        self.episode_messages: pd.DataFrame = None
        self.now_is: datetime = None
        assert self.exchange.name == "NASDAQ", "Currently the only exchange we can simulate is NASDAQ!"
        assert self.ticker == self.exchange.ticker

    def reset_episode(self, episode_start: datetime, episode_end: datetime):
        self.exchange.orderbook = self.exchange.get_empty_orderbook()
        self.episode_messages = self.database.get_messages(episode_start, episode_end, self.exchange.name, self.ticker)
        self.now_is = episode_start
        self.initialise_orderbook(start_date=episode_start)

    def forward_step(self, until: datetime, internal_orders: Optional[List[Order]] = None) -> OutputDict:
        assert (
            until > self.now_is
        ), f"The current time is {self.now_is.time()}, but we are trying to step forward in time until {until.time()}!"
        external_messages = self._get_messages_in_interval(start_date=self.now_is, end_date=until)
        external_messages = self._remove_hidden_executions(external_messages)
        external_orders = [self._get_order_from_external_message(message) for message in external_messages.itertuples()]
        orders = internal_orders or list()
        orders += external_orders
        filled_orders = []
        for order in orders:
            filled = self.exchange.process_order(order)
            filled_orders.append(filled)
        self.now_is = until
        return {"orderbook": self.exchange.orderbook, "filled_orders": filled_orders}

    def initialise_orderbook(self, start_date: datetime):
        start_series = self.database.get_last_snapshot(start_date, exchange=self.exchange.name, ticker=self.ticker)
        assert len(start_series) > 0, f"There is no data before the episode start time: {start_date}"
        initial_orders = self._get_initial_orders_from_series(start_series)

        return self.exchange.initialise_orderbook_from_orders(initial_orders)

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

    def _get_initial_orders_from_series(self, series: pd.DataFrame):
        initial_orders = []
        for direction in ["bid", "ask"]:
            for level in range(N_LEVELS):
                initial_orders.append(
                    Order(
                        timestamp=series.name,
                        price=series[f"{direction}_price_{level}"],
                        volume=series[f"{direction}_volume_{level}"],
                        direction=direction,
                        type=OrderType.SUBMISSION,
                        ticker=self.ticker,
                        internal_id=-1,
                    )
                )
        return initial_orders

    def _get_messages_in_interval(self, start_date: datetime, end_date: datetime):
        interval_mask = (self.episode_messages.timestamp > start_date) & (self.episode_messages.timestamp <= end_date)
        return self.episode_messages[interval_mask]

    @staticmethod
    def _remove_hidden_executions(messages: pd.DataFrame):
        return messages[messages.message_type != "execution_hidden"]
