from datetime import datetime
from typing import Deque, Dict, List, Optional, TypedDict

import pandas as pd

from RL4MM.database.HistoricalDatabase import HistoricalDatabase
from RL4MM.orderbook.models import Orderbook, Order, LimitOrder
from RL4MM.orderbook.Exchange import Exchange
from RL4MM.simulation.HistoricalOrderGenerator import HistoricalOrderGenerator
from RL4MM.simulation.OrderGenerator import OrderGenerator


class OutputDict(TypedDict):
    orderbook: Orderbook
    filled_orders: List[LimitOrder]


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
            if filled:
                filled_orders += filled
        self.now_is = until
        return {"orderbook": self.exchange.orderbook, "filled_orders": filled_orders}

    def get_historical_start_book(self, start_date: datetime):
        start_series = self.database.get_last_snapshot(start_date, ticker=self.exchange.ticker)
        assert len(start_series) > 0, f"There is no data before the episode start time: {start_date}"
        initial_orders = self._get_initial_orders_from_start_book(start_series)
        return self.exchange.get_initial_orderbook_from_orders(initial_orders)

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

    def _get_initial_orders_from_start_book(self, series: pd.DataFrame):
        initial_orders = []
        for direction in ["buy", "sell"]:
            for level in range(self.n_levels):
                if series[f"{direction}_volume_{level}"] > 0:
                    initial_orders.append(
                        LimitOrder(
                            timestamp=series.name,
                            price=series[f"{direction}_price_{level}"],
                            volume=series[f"{direction}_volume_{level}"],
                            direction=direction,  # type: ignore
                            ticker=self.exchange.ticker,
                            internal_id=-1,
                            external_id=None,
                            is_external=False,
                        )
                    )
        return initial_orders

    @property
    def orderbook(self):
        return self.exchange.orderbook
