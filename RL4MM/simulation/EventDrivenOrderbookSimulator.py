import sys
from collections import deque
from datetime import datetime, timedelta

from RL4MM.orderbook.create_order import create_order

if sys.version_info[0] == 3 and sys.version_info[1] >= 8:
    from typing import Deque, Dict, List, Optional, Literal, Callable, cast
else:
    from typing import Deque, Dict, List, Optional, Callable
    from typing_extensions import Literal

import numpy as np
import pandas as pd

from RL4MM.database.HistoricalDatabase import HistoricalDatabase
from RL4MM.orderbook.models import Orderbook, Order, LimitOrder, FilledOrders, OrderDict
from RL4MM.orderbook.Exchange import Exchange
from RL4MM.simulation.HistoricalOrderGenerator import HistoricalOrderGenerator
from RL4MM.simulation.OrderGenerator import OrderGenerator


class EventDrivenOrderbookSimulator:
    def __init__(
        self,
        ticker: str = "MSFT",
        exchange: Exchange = None,
        order_generator: OrderGenerator = None,
        n_levels: int = 50,
        database: HistoricalDatabase = None,
        preload_messages: bool = True,
        episode_length: timedelta = timedelta(minutes=30),
        warm_up: timedelta = timedelta(seconds=0),
        outer_levels: int = 20,
    ) -> None:
        self.ticker = ticker
        self.exchange = exchange or Exchange(ticker)
        order_generator = order_generator or HistoricalOrderGenerator(ticker, database, preload_messages)
        self.order_generators = {gen.name: gen for gen in order_generators}
        self.now_is: datetime = datetime(2000, 1, 1)
        self.n_levels = n_levels
        self.database = database or HistoricalDatabase()
        self.preload_messages = preload_messages
        if preload_messages:
            assert episode_length is not None, "When saving messages locally, episode length must be pre-specified."
        self.episode_length = episode_length
        self.warm_up = warm_up
        self.outer_levels = outer_levels
        # The following is for re-syncronisation with the historical data
        self.max_sell_price: int = 0
        self.min_buy_price: int = np.infty  # type:ignore
        self.initial_buy_price_range: int = np.infty  # type:ignore
        self.initial_sell_price_range: int = np.infty  # type:ignore

    def reset_episode(self, start_date: datetime, start_book: Optional[Orderbook] = None):
        if not start_book:
            start_book = self.get_historical_start_book(start_date)
        self.exchange.central_orderbook = start_book
        self.exchange.reset_internal_orderbook()
        self._reset_initial_price_ranges()
        assert start_date.microsecond == 0, "Episodes must be started on the second."
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
        initial_orders = self._get_initial_orders_from_snapshot(start_series)
        return self.exchange.get_initial_orderbook_from_orders(initial_orders)

    def _initial_prices_filter_function(self, direction: Literal["buy", "ask"], price: int) -> bool:
        if direction == "buy" and price < self.min_buy_price or direction == "sell" and price > self.max_sell_price:
            return True
        else:
            return False

    def update_outer_levels(self) -> None:
        """Update levels that had no initial orders at the start of the episode. If this is not done, the volume at
        these levels will be substantially lower than it should be. If agent orders exist at these levels, they will be
        cancelled and replaced at the back of the queue."""
        print(f"Updating outer levels. Current time is {self.now_is}.")
        orderbook_series = self.database.get_last_snapshot(self.now_is, ticker=self.ticker)
        orders_to_add = self._get_initial_orders_from_snapshot(orderbook_series, self._initial_prices_filter_function)
        internal_orders_to_replace = list()
        for order in orders_to_add:
            internal_book_side = getattr(self.exchange.internal_orderbook, order.direction)
            internal_orders_to_cancel = list()
            if order.price in internal_book_side.keys():
                print(
                    "Resynchronising levels containing internal orders. These internal orders will be cancelled and"
                    + " replaced at the back of the queue."
                )
                for internal_order in internal_book_side[order.price]:
                    order_dict = cast(OrderDict, internal_order.__dict__)
                    order_dict["timestamp"] = self.now_is
                    cancellation = create_order("cancellation", order_dict)
                    limit = create_order("limit", order_dict)
                    internal_orders_to_cancel.append(cancellation)
                    internal_orders_to_replace.append(limit)
            for cancellation in internal_orders_to_cancel:
                self.exchange.process_order(cancellation)
            getattr(self.exchange.central_orderbook, order.direction)[order.price] = deque([order])
            assert order.price not in internal_book_side.keys(), "Orders remaining in internal book when updating!"
        for order in internal_orders_to_replace:
            self.exchange.process_order(order)
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

    def _get_initial_orders_from_snapshot(self, series: pd.DataFrame, filter_function: Callable = always_true_function):
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
        outer_proportion = self.outer_levels / self.n_levels
        return (
            self.exchange.best_buy_price < self.min_buy_price + outer_proportion * self.initial_buy_price_range
            or self.exchange.best_sell_price > self.max_sell_price - outer_proportion * self.initial_sell_price_range
        )

    def _reset_initial_price_ranges(self):
        self.min_buy_price, self.max_sell_price = self.exchange.orderbook_price_range
        self.initial_buy_price_range = self.exchange.best_buy_price - self.min_buy_price
        self.initial_sell_price_range = self.max_sell_price - self.exchange.best_sell_price
