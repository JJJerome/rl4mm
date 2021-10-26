import abc
import ast
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import chain

from typing import Tuple, Dict, DefaultDict, List, Any

import pandas as pd

from RL4MM.database.HistoricalDatabase import HistoricalDatabase
from RL4MM.simulator import OrderbookMessage


@dataclass
class SimulatedTrade:
    datetime: datetime
    size: float
    price: float
    side: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "datetime": self.datetime,
            "size": self.size,
            "price": self.price,
            "side": self.side,
        }


class OrderbookSimulator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def simulate_trading(self, *kwargs) -> Tuple[SimulatedTrade, pd.DataFrame]:
        """Returns the quantity traded given an order start and end date, as well as the resulting orderbook state."""
        pass

    @staticmethod
    def _get_best_levels(orderbook: Dict[float, float], levels: int) -> Dict[float, float]:
        bids = sorted([price for price, size in orderbook.items() if size > 0])[-levels:][::-1]
        asks = sorted([price for price, size in orderbook.items() if size < 0])[:levels]
        return {price: orderbook[price] for price in bids + asks}

    @classmethod
    def orderbook_dict_to_df(
        cls, orderbooks: Dict[datetime, Dict[float, float]], levels: int, pre_clipped: bool = False
    ) -> pd.DataFrame:
        if not pre_clipped:
            orderbooks = {ts: cls._get_best_levels(book, levels) for ts, book in orderbooks.items()}
        orderbook_df = pd.DataFrame.from_dict(
            {ts: [item for pair in bid_ask.items() for item in pair] for ts, bid_ask in orderbooks.items()},
            orient="index",
            columns=[
                side + name + f"{i}" for side in ["bid_", "ask_"] for i in range(levels) for name in ["price_", "size_"]
            ],
        )
        orderbook_df.loc[:, [f"ask_size_{i}" for i in range(levels)]] *= -1
        return orderbook_df.sort_index()

    @classmethod
    def orderbook_dict_to_series(
        cls, orderbook: Dict[float, float], timestamp: datetime, levels: int, pre_sorted: bool = False
    ) -> pd.Series:
        if not pre_sorted:
            orderbook = cls._get_best_levels(orderbook, levels)
        orderbook_series = pd.Series(
            [item for pair in orderbook.items() for item in pair],
            name=timestamp,
            index=[
                side + name + f"{i}" for side in ["bid_", "ask_"] for i in range(levels) for name in ["price_", "size_"]
            ],
        )
        orderbook_series.loc[[f"ask_size_{i}" for i in range(levels)]] *= -1
        return orderbook_series

    @staticmethod
    def orderbook_series_to_dict(orderbook: pd.Series, levels: int) -> DefaultDict[float, float]:
        prices = [
            orderbook.loc[name]
            for name in [
                side + name + f"{i}" for side in ["bid_", "ask_"] for i in range(levels) for name in ["price_"]
            ]
        ]
        sizes = [orderbook.loc[quantity] for quantity in [f"bid_size_{i}" for i in range(levels)]] + [
            -orderbook.loc[quantity] for quantity in [f"ask_size_{i}" for i in range(levels)]
        ]
        return defaultdict(float, dict(zip(prices, sizes)))


class SimpleOrderbookSimulator:
    def __init__(
        self,
        start_date: datetime,
        end_date: datetime,
        exchange: str,
        ticker: str,
        levels: int = 10,
        database: HistoricalDatabase = None,
    ) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.exchange = exchange
        self.ticker = ticker
        self.levels = levels
        self.database = database or HistoricalDatabase()

    def simulate_trading(
        self,
        start_date: datetime,
        end_date: datetime,
        messages_to_fill,  # : List[OrderbookMessage],
        slippage: int = 0,
    ):  # -> Tuple[List[OrderbookMessage], List[SimulatedTrade], pd.DataFrame]:
        # TODO: Sort out typing
        """
        Order is placed on the touch

        Example: resting limit sell on best ask

        - Register queue position
        - Count market trades at order price

        - If next best ask > best ask, then:
            - Order is filled
            - State = next state + order quantity at resting next best ask

        - If next best ask == best ask, then:
            - If market trades at order price < queue position, then:
                - Order is not filled
                - State = next state + order quantity at order price
            - If market trades at order price >= queue position, then:
                - Order is filled
                - State = next state + order quantity at order price

        - If next best ask < best ask, then:
            - If market trades at order price < queue position, then:
                - Order is not filled
                - State = next state + order quantity at order price
            - If market trades at order price >= queue position, then:
                - Order is filled
                - State = next state + order quantity at best ask price
        """

        # TODO: replace all occurrences of "events" with "messages".
        filled_messages = list()
        slip_start = start_date + timedelta(microseconds=slippage)
        book = self.database.get_last_snapshot(slip_start, exchange=self.exchange, ticker=self.ticker)
        messages = self.database.get_events(slip_start, end_date, self.exchange, self.ticker).set_index("timestamp")
        assert sum(messages.exchange != self.exchange) == 0, f"Some messages do not come from {self.exchange}!"
        assert sum(messages.ticker != self.ticker) == 0, f"Some messages do not correspond to the ticker {self.ticker}!"
        book_df = self.convert_book_to_df(book, self.levels)
        for message in messages_to_fill:
            if not message.distance_to_fill:  # join the back of the queue
                message.distance_to_fill = book_df.loc[book_df.price == message.price, "size"][0] + message.size
        target_price_levels = [message.price for message in messages_to_fill]

        for message in messages.iterrows():
            new_message = message[1]
            if new_message.price not in book_df.price.values:
                pass
            elif new_message.event_type in ["deletion", "execution_visible", "cancellation"]:
                book_df.loc[book_df["price"] == new_message.price, "size"] -= new_message["size"]
                if new_message.price in target_price_levels:
                    if (
                        len(
                            messages[
                                (messages.external_id == new_message.external_id)
                                & (messages.event_type == "submission")
                            ]
                        )
                        == 0
                    ):
                        for message in messages_to_fill:
                            if message.price == new_message.price:
                                message.distance_to_fill -= new_message["size"]
            elif new_message.event_type == "submission":
                book_df.loc[book_df["price"] == new_message.price, "size"] += new_message["size"]
            elif new_message.event_type == "execution_hidden":
                pass
            else:
                raise NotImplementedError

            for message_to_fill in messages_to_fill:
                if message_to_fill.distance_to_fill <= 0:
                    filled_messages.append(message_to_fill)
                    messages_to_fill.remove(message_to_fill)
            if len(messages_to_fill) == 0:
                break
            target_price_levels = [message.price for message in messages_to_fill]

        terminal_book = self.database.get_last_snapshot(end_date, exchange=self.exchange, ticker=self.ticker)
        terminal_book_df = self.convert_book_to_df(terminal_book)
        for message in messages_to_fill + filled_messages:
            terminal_book_df.loc[terminal_book_df["price"] == message.price, "size"] += message.size

        return messages_to_fill, filled_messages, self.convert_df_to_book(book_df=terminal_book_df)

    @classmethod
    def _aggregate_orderbook_and_queue(
        cls, ts_idx: int, orderbooks, snapshots: pd.DataFrame, deltas: pd.DataFrame, end_date: datetime, levels: int
    ) -> None:
        orderbooks_to_add: Dict[datetime, Dict[float, float]] = {}
        ts = snapshots.index[ts_idx]
        next_ts = snapshots.index[ts_idx + 1] if ts_idx + 1 < len(snapshots.index) else end_date
        orderbooks_to_add[ts] = {
            float(k): float(v)
            for k, v in (
                list(snapshots.at[ts, "data"]["bid"].items())
                + [(price, -size) for price, size in snapshots.at[ts, "data"]["ask"].items()]
            )
        }
        last_timestamp = ts
        for timestamp, row in deltas.loc[ts:next_ts].iterrows():
            next_orderbook = deepcopy(orderbooks_to_add[last_timestamp])
            for price, size in row["data"]["bid"].items():
                next_orderbook[float(price)] = size
            for price, size in row["data"]["ask"].items():
                next_orderbook[float(price)] = -size
            orderbooks_to_add[timestamp] = next_orderbook
            last_timestamp = timestamp
        for ts, ob in orderbooks_to_add.items():
            orderbooks[ts] = cls._get_best_levels(ob, levels)

    @staticmethod
    def _get_book_snapshots(
        historical_db: HistoricalDatabase, start_date: datetime, end_date: datetime, exchange: str, ticker: str
    ) -> pd.DataFrame:
        snapshot = historical_db.get_last_snapshot(start_date, exchange, ticker)
        assert len(snapshot) == 1, f"No snapshot exists before {start_date}."
        return historical_db.get_book_snapshots(snapshot.at[0, "timestamp"], end_date, exchange, ticker).set_index(
            "timestamp"
        )

    @staticmethod
    def _get_book_deltas(
        historical_db: HistoricalDatabase, start_date: datetime, end_date: datetime, exchange: str, ticker: str
    ) -> pd.DataFrame:
        return historical_db.get_book_deltas(start_date, end_date, exchange, ticker).set_index("timestamp")

    @staticmethod
    def _get_events(
        historical_db: HistoricalDatabase, start_date: datetime, end_date: datetime, exchange: str, ticker: str
    ) -> pd.DataFrame:
        return historical_db.get_events(start_date, end_date, exchange, ticker).set_index("timestamp")

    # TODO: write test
    @staticmethod
    def convert_book_to_df(book: pd.Series, levels: int = 10) -> pd.DataFrame:
        price_cols = list(chain(*[("ask_price_{0},bid_price_{0}".format(i)).split(",") for i in range(levels)]))
        size_cols = list(chain(*[("ask_size_{0},bid_size_{0}".format(i)).split(",") for i in range(levels)]))
        index_cols = list(chain(*[("ask_{0},bid_{0}".format(i)).split(",") for i in range(levels)]))
        book_df = pd.DataFrame(book.loc[price_cols].values, index=index_cols, columns=["price"])
        book_df["size"] = book.loc[size_cols].values
        return book_df

    @staticmethod
    def convert_df_to_book(book_df: pd.DataFrame) -> pd.Series:
        book = pd.Series(dtype="float64")
        for _index in book_df.index:
            for col_name in ["price", "size"]:
                book.loc[_index[0:3] + "_" + col_name + "_" + _index[-1]] = book_df.loc[_index, col_name]
        return book
