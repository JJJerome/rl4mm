from datetime import datetime, timedelta

from typing import Deque, Dict, List, Optional, Literal, Callable
import pandas as pd

from RL4MM.database.HistoricalDatabase import HistoricalDatabase
from RL4MM.orderbook.models import Orderbook, Order, LimitOrder, FillableOrder
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
    ) -> None:
        self.exchange = exchange or Exchange(ticker)
        order_generators = order_generators or [HistoricalOrderGenerator(ticker)]
        self.order_generators = {gen.name: gen for gen in order_generators}
        self.now_is: datetime = datetime(2000, 1, 1)
        self.n_levels = n_levels
        self.database = database or HistoricalDatabase()

    def reset_episode(self, start_date: datetime, start_book: Optional[Orderbook] = None):
        if not start_book:
            start_book = self.get_historical_start_book(start_date)
        self.exchange.central_orderbook = start_book
        self.exchange.reset_internal_orderbook()
        self.now_is = start_date
        return start_book

    def forward_step(self, until: datetime, internal_orders: Optional[List[Order]] = None) -> List[FillableOrder]:
        assert (
            until > self.now_is
        ), f"The current time is {self.now_is.time()}, but we are trying to step forward in time until {until.time()}!"
        order_dict = {name: gen.generate_orders(self.now_is, until) for name, gen in self.order_generators.items()}
        external_orders = self._compress_order_dict(order_dict)
        orders = internal_orders or list()
        orders += external_orders
        filled_internal_orders = []
        for order in orders:
            filled = self.exchange.process_order(order)
            if filled:
                filled_internal_orders += filled
        self.now_is = until
        return filled_internal_orders

    def get_historical_start_book(self, start_date: datetime):
        start_series = self.database.get_last_snapshot(start_date, ticker=self.ticker)
        assert len(start_series) > 0, f"There is no data before the episode start time: {start_date}"
        assert start_date - start_series.name <= timedelta(days=1), "Attempting to get data from more than a day ago"
        initial_orders = self._get_initial_orders_from_book(start_series)
        return self.exchange.get_initial_orderbook_from_orders(initial_orders)

    def update_outer_levels(self) -> None:
        min_initial_price, max_initial_price = self._get_initial_price_interval()
        orderbook_series = self.database.get_last_snapshot(self.now_is, ticker=self.ticker)

        def filter_function(direction: Literal["buy", "ask"], price: int):
            if direction == "buy" and price < min_initial_price or direction == "sell" and price > max_initial_price:
                return True
            else:
                return False

        orders_to_add = self._get_initial_orders_from_book(orderbook_series, filter_function)
        for order in orders_to_add:
            assert (
                order.price not in self.exchange.internal_orderbook[order.direction].keys()
            ), "Trying to reset levels with internal orders in"
            self.exchange.central_orderbook[order.direction][order.price] = order

    def _get_initial_price_interval(self) -> List[int]:
        price_interval = []
        for side in ["buy", "sell"]:
            half_book = self.exchange.central_orderbook[side]
            outer = min if side is "buy" else max
            price_interval.append(outer(level for level in half_book.keys() if half_book[level][0].internal_id == -1))
        return price_interval

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

    always_true_function: Callable = lambda x: True

    def _get_initial_orders_from_book(self, series: pd.DataFrame, filter_function: Callable = always_true_function):
        initial_orders = []
        for direction in ["buy", "sell"]:
            for level in range(self.n_levels):
                if f"{direction}_volume_{level}" in series and filter_function(direction, level):
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
