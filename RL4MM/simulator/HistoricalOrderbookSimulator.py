from datetime import datetime, timedelta

from typing import List, Optional

import pandas as pd
from pydantic.schema import timedelta as pytimedelta

from RL4MM.database.HistoricalDatabase import HistoricalDatabase
from RL4MM.simulator.OrderbookSimulator import (
    OrderbookSimulator,
    OrderbookMessage,
    ResultsDict,
)


class HistoricalOrderbookSimulator(OrderbookSimulator):
    def __init__(
        self,
        exchange: str = "NASDAQ",
        ticker: str = "MSFT",
        levels: int = 10,
        step_size: pytimedelta = timedelta(seconds=1),
        database: HistoricalDatabase = None,
        slippage: pytimedelta = timedelta(microseconds=10),
        message_duration: int = 2,
    ) -> None:
        assert exchange == "NASDAQ", "Currently the only exchange we can simulate is NASDAQ!"
        self.exchange = exchange
        self.ticker = ticker
        self.levels = levels
        self.step_size = step_size
        self.database = database or HistoricalDatabase()
        self.slippage = slippage
        self.message_duration = message_duration
        self.external_messages: pd.DataFrame = None
        self.now_is: datetime = None
        self.current_book = None

    def reset_episode(self, episode_start: datetime, episode_end: datetime):
        self.external_messages = self.database.get_messages(episode_start, episode_end, self.exchange, self.ticker)
        self.now_is = episode_start
        start_book = self.database.get_messages(episode_start, episode_end, self.exchange, self.ticker)
        if len(start_book) > 0:
            self.current_book = start_book
        else:
            raise Exception(f"There is no data before {episode_start}")

    def simulate_step(
        self,
        messages_to_fill: List[OrderbookMessage] = list(),
        start_book: Optional[pd.Series] = None,
    ) -> ResultsDict:
        step_end = self.now_is + self.step_size
        step_mask = self.external_messages.timestamp > self.now_is & self.external_messages.timestamp < step_end
        if len(step_mask) == 0:
            end_book = self.database.get_last_snapshot(step_end, exchange=self.exchange, ticker=self.ticker)
            return ResultsDict.from_constituents(messages_to_fill, [], end_book, self._get_midprice_change(end_book))
        self.external_messages.set_index("timestamp", inplace=True)

        # TODO: CONTINUE FROM HERE
