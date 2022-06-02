import abc
import sys

if sys.version_info[0] == 3 and sys.version_info[1] >= 8:
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

import pandas as pd


class CannotUpdateError(Exception):
    pass


class InternalState(TypedDict):
    inventory: int
    cash: float
    asset_price: float
    book_snapshots: pd.DataFrame
    proportion_of_episode_remaining: float


class Feature(metaclass=abc.ABCMeta):
    """Book features calculate a feature from a dataframe of Orderbooks where the columns are "buy_price_0",
    "buy_volume_0", "buy_price_0", "buy_volume_0". These inputs are hard coded."""

    @property
    @abc.abstractmethod
    def max_value(self) -> float:
        pass

    @property
    @abc.abstractmethod
    def min_value(self) -> float:
        pass

    def calculate(self, internal_state: InternalState) -> float:
        return self.clamp(self._calculate(internal_state), min_value=self.min_value, max_value=self.max_value)

    @abc.abstractmethod
    def _calculate(self, internal_state: InternalState) -> float:
        pass

    @staticmethod
    def clamp(number: float, min_value: float, max_value: float):
        return max(min(number, max_value), min_value)


# Book features


class Spread(Feature):
    max_value = int(100 * 100)  # 100 ticks
    min_value = 0.0

    def __init__(self, min_value=0.0, max_value=int(100 * 100)):
        self.min_value = min_value
        self.max_value = max_value

    def _calculate(self, internal_state: InternalState) -> float:
        current_book = internal_state["book_snapshots"].iloc[-1]
        return current_book.sell_price_0 - current_book.buy_price_0


class MidpriceMove(Feature):
    max_value = 100 * 100  # 100 tick upward move
    min_value = -100 * 100  # 100 tick downward move

    def __init__(self, lookback_period: int = 1):
        self.lookback_period = lookback_period

    def _calculate(self, internal_state: InternalState):
        book_snapshots = internal_state["book_snapshots"]
        self.midprice_series = (book_snapshots.sell_price_0 + book_snapshots.buy_price_0) / 2
        self.midprice_series = self.midprice_series.iloc[range(-(1 + self.lookback_period), 0)]
        return self.midprice_series.iloc[-1] - self.midprice_series.iloc[-(1 + self.lookback_period)]


class PriceRange(Feature):
    max_value = int(100 * 100)  # 100 ticks
    min_value = 0

    def _calculate(self, internal_state: InternalState):
        max_price = internal_state["book_snapshots"].sell_price_0.max()
        min_price = internal_state["book_snapshots"].buy_price_0.min()
        price_range = max_price - min_price
        return price_range


class Volatility(Feature):
    min_value = 0
    max_value = (100 * 100) ** 2

    def _calculate(self, internal_state: InternalState):
        book_snapshots = internal_state["book_snapshots"]
        midprice_df = (book_snapshots.sell_price_0 + book_snapshots.buy_price_0) / 2
        returns_df = midprice_df.diff()
        return returns_df.var()


class MicroPrice(Feature):
    min_value = 0
    max_value = 1000 * 10000  # Here, we assume that stock prices are less than $1000

    def _calculate(self, internal_state: InternalState):
        current_book = internal_state["book_snapshots"].iloc[-1]
        return self.calculate_from_current_book(current_book)

    @staticmethod
    def calculate_from_current_book(current_book: pd.Series):
        sell_weighting = current_book.buy_volume_0 / (current_book.buy_volume_0 + current_book.sell_volume_0)
        return current_book.sell_price_0 * sell_weighting + current_book.buy_price_0 * (1 - sell_weighting)


# Agent features


class Inventory(Feature):
    min_value = 0
    max_value = 1000

    def _calculate(self, internal_state: InternalState) -> float:
        return internal_state["inventory"]


class TimeRemaining(Feature):
    min_value = 0
    max_value = 1.0

    def _calculate(self, internal_state: InternalState) -> float:
        return internal_state["proportion_of_episode_remaining"]
