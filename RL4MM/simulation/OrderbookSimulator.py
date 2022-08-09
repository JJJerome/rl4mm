import sys
from collections import deque
from datetime import datetime, timedelta

if sys.version_info[0] == 3 and sys.version_info[1] >= 8:
    from typing import Deque, Dict, List, Optional, Literal, Callable
else:
    from typing import Deque, Dict, List, Optional, Callable
    from typing_extensions import Literal

import numpy as np
import pandas as pd

from RL4MM.database.HistoricalDatabase import HistoricalDatabase
from RL4MM.orderbook.models import Orderbook, Order, LimitOrder, FilledOrders
from RL4MM.orderbook.Exchange import Exchange
from RL4MM.simulation.HistoricalOrderGenerator import HistoricalOrderGenerator
from RL4MM.simulation.OrderGenerator import OrderGenerator


class OrderbookSimulator:
    def __init__(
        self,
        ticker: str = "MSFT",
        exchange: Exchange = None,
        order_generators: List[OrderGenerator] = None,
        n_levels: int = 50,
        database: HistoricalDatabase = None,
        preload_messages: bool = True,
        episode_length: timedelta = timedelta(minutes=30),
        warm_up: timedelta = timedelta(seconds=0),
    ) -> None:
        self.ticker = ticker
        self.exchange = exchange or Exchange(ticker)
        order_generators = order_generators or [HistoricalOrderGenerator(ticker, database, preload_messages)]
        self.order_generators = {gen.name: gen for gen in order_generators}
        self.now_is: datetime = datetime(2000, 1, 1)
        self.n_levels = n_levels
        self.database = database or HistoricalDatabase()
        self.preload_messages = preload_messages
        if preload_messages:
            assert episode_length is not None, "When saving messages locally, episode length must be pre-specified."
        self.episode_length = episode_length
        self.warm_up = warm_up
        # The following is for re-syncronisation with the historical data
        self.max_sell_price: int = 0
        self.min_buy_price: int = np.infty  # type:ignore
        self.initial_buy_price_range: int = np.infty  # type:ignore
        self.initial_sell_price_range: int = np.infty  # type:ignore
        self.outer_levels: float = 20 / self.n_levels

    def reset_episode(self, start_date: datetime, start_book: Optional[Orderbook] = None):
        if not start_book:
            start_book = self.get_historical_start_book(start_date)
        self.exchange.central_orderbook = start_book
        self.exchange.reset_internal_orderbook()
        self._reset_initial_price_ranges()
        self.now_is = start_date
        if self.preload_messages:
            for order_generator_name in self.order_generators.keys():
                self.order_generators[order_generator_name].preload_messages(
                    start_date, start_date + self.episode_length + self.warm_up
                )
        return start_book

    def forward_step(self, until: datetime, internal_orders: Optional[List[Order]] = None) -> FilledOrders:
        assert (
            until > self.now_is
        ), f"The current time is {self.now_is.time()}, but we are trying to step forward in time until {until.time()}!"
        order_dict = {name: gen.generate_orders(self.now_is, until) for name, gen in self.order_generators.items()}
        external_orders = self._compress_order_dict(order_dict)
        orders = internal_orders or list()
        orders += external_orders
        filled_internal_orders = []
        filled_external_orders = []
        for order in orders:
            filled = self.exchange.process_order(order)
            if filled:
                filled_internal_orders += filled.internal
                filled_external_orders += filled.external
        self.now_is = until
        if self._near_exiting_initial_price_range and self.now_is.microsecond == 0:
            self.update_outer_levels()
        return FilledOrders(internal=filled_internal_orders, external=filled_external_orders)

    def get_historical_start_book(self, start_date: datetime) -> Orderbook:
        start_series = self.database.get_last_snapshot(start_date, ticker=self.ticker)
        assert len(start_series) > 0, f"There is no data before the episode start time: {start_date}"
        assert start_date - start_series.name <= timedelta(
            days=1
        ), f"Attempting to get data from > a day ago (start_date: {start_date}; start_series.name: {start_series.name})"
        initial_orders = self._get_initial_orders_from_book(start_series)
        return self.exchange.get_initial_orderbook_from_orders(initial_orders)

    def _initial_prices_filter_function(self, direction: Literal["buy", "ask"], price: int) -> bool:
        if direction == "buy" and price < self.min_buy_price or direction == "sell" and price > self.max_sell_price:
            return True
        else:
            return False

    def update_outer_levels(self) -> None:
        orderbook_series = self.database.get_last_snapshot(self.now_is, ticker=self.ticker)
        orders_to_add = self._get_initial_orders_from_book(orderbook_series, self._initial_prices_filter_function)
        for order in orders_to_add:
            book_side = getattr(self.exchange.internal_orderbook, order.direction)
            assert order.price not in book_side.keys(), "Attempting to re-syncronise levels containing internal orders."
            book_side[order.price] = deque([order])
        self.min_buy_price = min(self.min_buy_price, self.exchange.orderbook_price_range[0])
        self.max_sell_price = max(self.max_sell_price, self.exchange.orderbook_price_range[1])

    @staticmethod
    def _compress_order_dict(order_dict: Dict[str, Deque[Order]]) -> List:
        if len(order_dict) == 1:
            return list(list(order_dict.values())[0])
        else:
            orders = []
            while len(order_dict) > 0:
                next_order_key = min(order_dict, key=order_dict.get)  # type: ignore
                orders.append(order_dict[next_order_key].popleft())
                if len(order_dict[next_order_key]) == 0:
                    order_dict.pop(next_order_key)
            return orders

    @staticmethod
    def _remove_hidden_executions(messages: pd.DataFrame):
        return messages[messages.message_type != "execution_hidden"]

    always_true_function: Callable = lambda direction, price: True

    def _get_initial_orders_from_book(self, series: pd.DataFrame, filter_function: Callable = always_true_function):
        initial_orders = []
        for direction in ["buy", "sell"]:
            for level in range(self.n_levels):
                if f"{direction}_volume_{level}" not in series:
                    continue
                if filter_function(direction, series[f"{direction}_price_{level}"]):
                    initial_orders.append(
                        LimitOrder(
                            timestamp=series.name,
                            price=series[f"{direction}_price_{level}"],
                            volume=series[f"{direction}_volume_{level}"],
                            direction=direction,  # type: ignore
                            ticker=self.ticker,
                            internal_id=-1,
                            external_id=None,
                            is_external=True,
                        )
                    )
        return initial_orders

    @property
    def _near_exiting_initial_price_range(self) -> bool:
        return (
            self.exchange.best_buy_price < self.min_buy_price + self.outer_levels * self.initial_buy_price_range
            or self.exchange.best_sell_price > self.max_sell_price - self.outer_levels * self.initial_sell_price_range
        )

    def _reset_initial_price_ranges(self):
        self.min_buy_price, self.max_sell_price = self.exchange.orderbook_price_range
        self.initial_buy_price_range = self.exchange.best_buy_price - self.min_buy_price
        self.initial_sell_price_range = self.max_sell_price - self.exchange.best_sell_price
        self.initial_buy_price_range = self.exchange.best_buy_price - self.min_buy_price
