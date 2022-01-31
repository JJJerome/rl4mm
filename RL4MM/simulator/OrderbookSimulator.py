import abc

import numpy as np

from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import chain

from typing import Any, Dict, List, Optional, TypedDict

import pandas as pd

from RL4MM.database.HistoricalDatabase import HistoricalDatabase


@dataclass
class OrderbookMessage:
    _id: str
    timestamp: datetime
    message_type: str
    ticker: str
    size: float
    price: float
    side: str
    queue_position: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "_id": self._id,
            "datetime": self.timestamp,
            "message_type": self.message_type,
            "ticker": self.ticker,
            "size": self.size,
            "price": self.price,
            "side": self.side,
            "distance_to_fill": self.queue_position,
        }


@dataclass
class Orderbook:
    datetime: datetime
    bids: Dict[float, int]
    asks: Dict[float, int]

    def to_series(self) -> Dict[str, Any]:
        return {
            "_id": self._id,
            "datetime": self.datetime,
            "message_type": self.message_type,
            "ticker": self.ticker,
            "size": self.size,
            "price": self.price,
            "side": self.side,
            "distance_to_fill": self.queue_position,
        }


class ResultsDict(TypedDict):
    messages_to_fill: List[OrderbookMessage]
    filled_messages: List[OrderbookMessage]
    orderbook: pd.DataFrame
    midprice_change: float


class OrderbookSimulator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def simulate_step(
        self,
        start_date: datetime,
        end_date: datetime,
        messages_to_fill: List[OrderbookMessage],
        start_book: Optional[pd.Series] = None,
    ) -> ResultsDict:
        """Returns the quantity traded given an order start and end date, as well as the resulting orderbook state."""
        pass


class HistoricalOrderbookSimulator(OrderbookSimulator):
    def __init__(
        self,
        exchange: str = "NASDAQ",
        ticker: str = "MSFT",
        slippage: int = 0,
        levels: int = 10,
        database: HistoricalDatabase = None,
        message_duration: int = 2,
    ) -> None:
        assert exchange == "NASDAQ", "Currently the only exchange we can simulate is NASDAQ!"
        self.exchange = exchange
        self.ticker = ticker
        self.slippage = slippage
        self.levels = levels
        self.database = database or HistoricalDatabase()
        self.message_duration = message_duration

    def simulate_step(
        self,
        start_date: datetime,
        end_date: datetime,
        messages_to_fill: List[OrderbookMessage] = list(),
        start_book: Optional[pd.Series] = None,
    ) -> ResultsDict:

        slip_start = start_date + timedelta(microseconds=self.slippage)
        if start_book is None:
            start_book = self.get_last_snapshot(slip_start)
        book = start_book
        initial_book_delta = self.get_book_delta(start_book, book)
        filled_messages = list()
        history = self.database.get_messages(slip_start, end_date, self.exchange, self.ticker)
        if len(history) == 0:
            end_book = self.database.get_last_snapshot(end_date, exchange=self.exchange, ticker=self.ticker)
            return messages_to_fill, filled_messages, end_book
        history.set_index("timestamp", inplace=True)
        assert sum(history.exchange != self.exchange) == 0, f"Some messages do not come from {self.exchange}!"
        assert sum(history.ticker != self.ticker) == 0, f"Some messages do not correspond to the ticker {self.ticker}!"
        book_df = self.convert_book_to_df(book, self.levels)
        messages_not_in_book = list()
        for message in messages_to_fill:
            if message.price not in book_df.price.values:
                messages_not_in_book.append(message)
        for message in messages_not_in_book:  # TODO: is this the correct way to deal with messages not in book?
            messages_to_fill.remove(message)
            print(f"removed {message} from messages to fill, since it does not appear in the L10 book")
        for message in messages_to_fill:
            if not message.queue_position:  # join the back of the queue
                message.queue_position = book_df.loc[book_df.price == message.price, "size"][0]
        target_price_levels = {m.price for m in messages_to_fill}
        queue_positions = self.get_best_queue_positions(messages_to_fill)

        for message in history.itertuples():
            if len(messages_to_fill) == 0:
                break
            if message.price not in target_price_levels:
                pass  # TODO: is there anything we can do here, to speed up?
            if message.message_type in ["deletion", "cancellation"]:
                queue_positions = self.update_queue_positions(message, messages_to_fill, history)
                continue
            if message.message_type == "execution_hidden":
                pass
            if message.message_type in ["cross_trade", "trading_halt"]:
                raise NotImplementedError
            if message.message_type == "execution_visible":
                if min(queue_positions.values()) > 0:
                    queue_positions = self.update_queue_positions(message, messages_to_fill, history)
                elif queue_positions["bid"] <= 0 and message.direction == "bid":
                    best_bid = self.get_best_prices(messages_to_fill)["bid"]
                    if message.price <= best_bid:
                        size_on_bid = -np.infty
                        for our_message in messages_to_fill:
                            if our_message.price == best_bid:
                                our_message.queue_position -= message.size
                                size_on_bid = max(size_on_bid, our_message.queue_position + our_message.size)
                                if our_message.queue_position + our_message.size <= 0:
                                    messages_to_fill.remove(our_message)
                                    filled_messages.append(our_message)
                        if size_on_bid <= 0:
                            queue_positions = self.update_queue_positions(
                                message, messages_to_fill, history, size_on_bid
                            )
                elif queue_positions["ask"] <= 0 and message.direction == "ask":
                    best_ask = self.get_best_prices(messages_to_fill)["ask"]
                    if message.price >= best_ask:
                        size_on_ask = -np.infty
                        for our_message in messages_to_fill:
                            if our_message.price == best_ask:
                                our_message.queue_position -= message.size
                                size_on_ask = max(size_on_ask, our_message.queue_position + our_message.size)
                                if our_message.queue_position + our_message.size <= 0:
                                    messages_to_fill.remove(our_message)
                                    filled_messages.append(our_message)
                        if size_on_ask <= 0:
                            queue_positions = self.update_queue_positions(
                                message, messages_to_fill, history, size_on_ask
                            )

        # Remove old messages
        old_messages = list()
        for message in messages_to_fill:
            message_age = end_date - message.timestamp
            window_size = end_date - start_date
            if message_age / window_size >= self.message_duration:
                old_messages.append(message)
        for old_message in old_messages:
            messages_to_fill.remove(old_message)

        # TODO: think about this...
        end_book = self.database.get_last_snapshot(end_date, exchange=self.exchange, ticker=self.ticker)
        terminal_book_df = self.convert_book_to_df(end_book)
        for message in messages_to_fill:
            if message.queue_position > 0:
                terminal_book_df.loc[terminal_book_df["price"] == message.price, "size"] += message.size
            elif message.queue_position + message.size > 0:
                terminal_book_df.loc[terminal_book_df["price"] == message.price, "size"] += (
                    message.queue_position + message.size
                )
            else:
                pass
        for message in filled_messages:
            terminal_book_df.loc[f"{message.side}_0", "size"] += message.size
        end_book = self.convert_df_to_book(book_df=terminal_book_df) + initial_book_delta

        results = ResultsDict(
            {
                "messages_to_fill": messages_to_fill,
                "filled_messages": filled_messages,
                "orderbook": end_book,
                "midprice_change": self.get_midprice_change(start_book, end_book),
            }
        )

        return results

    @classmethod
    def get_book_delta(cls, book_1: pd.Series, book_2: pd.Series):
        assert len(book_1) == len(book_2), "Books must be the same size"
        for i in range(len(book_1) // 4):
            for side in ["bid", "ask"]:
                assert book_1[f"{side}_price_{i}"] == book_2[f"{side}_price_{i}"], "Book price levels must be the same!"
        return book_2 - book_1

    def get_last_snapshot(self, timestamp: datetime):
        return self.database.get_last_snapshot(timestamp, exchange=self.exchange, ticker=self.ticker)

    @staticmethod
    def get_midprice_change(start_book: pd.DataFrame, end_book: pd.DataFrame):
        start_midprice = (start_book["ask_price_0"] + start_book["bid_price_0"]) / 2
        end_midprice = (end_book["ask_price_0"] + end_book["bid_price_0"]) / 2
        return end_midprice - start_midprice

    @classmethod
    def update_queue_positions(
        cls,
        incoming_message: OrderbookMessage,
        messages_to_fill: List,
        history: pd.DataFrame,
        message_size: Optional[int] = None,
    ):
        recent = sum((history.external_id == incoming_message.external_id) & (history.message_type == "submission"))
        if not recent:
            message_size = incoming_message.size if message_size is None else message_size
            for our_message in messages_to_fill:
                if our_message.price == incoming_message.price:
                    our_message.queue_position -= message_size
        return cls.get_best_queue_positions(messages_to_fill)

    @staticmethod
    def _get_book_snapshots(
        historical_db: HistoricalDatabase, start_date: datetime, end_date: datetime, exchange: str, ticker: str
    ) -> pd.DataFrame:
        snapshot = historical_db.get_last_snapshot(start_date, exchange, ticker)
        assert len(snapshot) == 1, f"No snapshot exists before {start_date}."
        return historical_db.get_book_snapshots(snapshot.at[0, "timestamp"], end_date, exchange, ticker).set_index(
            "timestamp"
        )

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

    @staticmethod
    def get_best_prices(messages: List) -> dict:
        best_prices = dict()
        try:
            best_prices["bid"] = max([message.price for message in messages if message.side == "bid"])
        except ValueError:
            best_prices["bid"] = -np.infty
        try:
            best_prices["ask"] = min([message.price for message in messages if message.side == "ask"])
        except ValueError:
            best_prices["ask"] = np.infty
        return best_prices

    @classmethod
    def get_best_queue_positions(cls, messages: List) -> dict:
        best_queue_positions = dict()
        best_prices = cls.get_best_prices(messages)
        try:
            best_queue_positions["bid"] = min(
                [
                    message.queue_position
                    for message in messages
                    if message.price == best_prices["bid"] and message.side == "bid"
                ]
            )
        except ValueError:
            best_queue_positions["bid"] = np.infty
        try:
            best_queue_positions["ask"] = min(
                [
                    message.queue_position
                    for message in messages
                    if message.price == best_prices["ask"] and message.side == "ask"
                ]
            )
        except ValueError:
            best_queue_positions["ask"] = np.infty
        return best_queue_positions
