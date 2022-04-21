from datetime import datetime
from typing import Deque, Dict, List, Optional, Tuple, TypedDict

import pandas as pd

from RL4MM.database.HistoricalDatabase import HistoricalDatabase
from RL4MM.orderbook.models import Order, OrderType, Orderbook
from RL4MM.orderbook.Exchange import Exchange
from RL4MM.simulation.HistoricalOrderGenerator import HistoricalOrderGenerator
from RL4MM.simulation.OrderGenerator import OrderGenerator

N_LEVELS = 200


class OutputDict(TypedDict):
    orderbook: Orderbook
    filled_orders: List[Order]


class OrderbookSimulator:
    def __init__(
        self,
        ticker: str = "MSFT",
        exchange: Exchange = None,
        order_generators: List[OrderGenerator] = None,
    ) -> None:
        self.exchange = exchange or Exchange(ticker)
        order_generators = order_generators or [HistoricalOrderGenerator(ticker)]
        self.order_generators = {gen.name: gen for gen in order_generators}
        self.now_is: datetime = None

    def reset_episode(self, start_date: datetime, start_book: Optional[Orderbook] = None):
        if not start_book:
            start_book = self.get_historical_start_book(start_date)
        self.exchange.orderbook = start_book
        self.now_is = start_date
        return start_book

    def forward_step(self, until: datetime, internal_orders: Optional[List[Order]] = None) -> OutputDict:
        assert (
            until > self.now_is
        ), f"The current time is {self.now_is.time()}, but we are trying to step forward in time until {until.time()}!"
        order_dict = {name: gen.generate_orders(self.now_is, until) for name, gen in self.order_generators.items()}
        external_orders = self._compress_order_dict(order_dict)
        orders = internal_orders or list()
        orders += external_orders
        filled_orders = []
        for order in orders:
            filled = self.exchange.process_order(order)
            filled_orders.append(filled)
        self.now_is = until
        return {"orderbook": self.exchange.orderbook, "filled_orders": filled_orders}

    def get_historical_start_book(self, start_date: datetime):
        hdb = HistoricalDatabase(n_levels=N_LEVELS)
        start_series = hdb.get_last_snapshot(start_date, exchange=self.exchange.name, ticker=self.exchange.ticker)
        assert len(start_series) > 0, f"There is no data before the episode start time: {start_date}"
        initial_orders = self._get_initial_orders_from_series(start_series)
        return self.exchange.get_initial_orderbook_from_orders(initial_orders)

    @staticmethod
    def _compress_order_dict(order_dict: Dict[int, Deque[Order]]) -> List:
        if len(order_dict) == 1:
            return list(list(order_dict.values())[0])
        else:
            orders = []
            while len(order_dict) > 0:
                next_order_key = min(order_dict, key=order_dict.get)
                orders.append(order_dict[next_order_key].popleft())
                if len(order_dict[next_order_key]) == 0:
                    order_dict.pop(next_order_key)
            return orders

    def _get_messages_in_interval(self, start_date: datetime, end_date: datetime):
        interval_mask = (self.episode_messages.timestamp > start_date) & (self.episode_messages.timestamp <= end_date)
        return self.episode_messages[interval_mask]

    @staticmethod
    def _remove_hidden_executions(messages: pd.DataFrame):
        return messages[messages.message_type != "execution_hidden"]

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
                        ticker=self.exchange.ticker,
                        internal_id=-1,
                    )
                )
        return initial_orders
