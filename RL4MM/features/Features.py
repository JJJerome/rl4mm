from collections import deque
from scipy import stats
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
    def name(self):
        pass

    @property
    @abc.abstractmethod
    def max_value(self) -> float:
        pass

    @property
    @abc.abstractmethod
    def min_value(self) -> float:
        pass

    def calculate(self, internal_state: InternalState) -> float:
        pre_clamped_value = self._calculate(internal_state)
        value = self.clamp(pre_clamped_value, min_value=self.min_value, max_value=self.max_value)
        if value != pre_clamped_value:
            print(f"Clamping value of {self.name} from {pre_clamped_value} to {value}.")
        return self.normalise(value) if self.normalisation_on else value

    @abc.abstractmethod
    def _calculate(self, internal_state: InternalState) -> float:
        pass

    @abc.abstractmethod
    def normalise(self, value: float) -> float:
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    @staticmethod
    def clamp(number: float, min_value: float, max_value: float):
        return max(min(number, max_value), min_value)


# Book features
class Spread(Feature):
    name = "Spread"

    def __init__(
        self,
        min_value: float = 0,
        max_value: float = (100 * 100),  # 100 ticks
        maxlen: int = 100000,
        normalisation_on: bool = False,
    ):
        self._min_value = min_value
        self._max_value = max_value
        self.normalisation_on = normalisation_on
        if self.normalisation_on:
            self.history = deque(maxlen=maxlen)

    @property
    def max_value(self) -> float:
        return self._max_value

    @property
    def min_value(self) -> float:
        return self._min_value

    def reset(self):
        if self.normalisation_on:
            self.history.clear()

    def normalise(self, value: float) -> float:
        if len(self.history) == 0:
            # To prevent a Nan vlaue from being returned
            # if the queue is empty:
            self.history.append(value + 1e-06)
        self.history.append(value)
        return stats.zscore(self.history)[-1]

    def _calculate(self, internal_state: InternalState) -> float:
        current_book = internal_state["book_snapshots"].iloc[-1]
        return current_book.sell_price_0 - current_book.buy_price_0


class MidpriceMove(Feature):
    name = "MidpriceMove"

    def __init__(
        self,
        min_value: float = -100 * 100,  # 100 tick downward move
        max_value: float = 100 * 100,  # 100 tick upward move
        lookback_period: int = 10,
        maxlen: int = 100000,
        normalisation_on: bool = False,
    ):
        self.lookback_period = lookback_period
        self._min_value = min_value
        self._max_value = max_value
        self.normalisation_on = normalisation_on
        if normalisation_on:
            self.history = deque(maxlen=maxlen)

    @property
    def max_value(self) -> float:
        return self._max_value

    @property
    def min_value(self) -> float:
        return self._min_value

    def reset(self):
        if self.normalisation_on:
            self.history.clear()

    def normalise(self, value: float) -> float:
        if len(self.history) == 0:
            # To prevent a Nan vlaue from being returned
            # if the queue is empty:
            self.history.append(value + 1e-06)
        self.history.append(value)
        return stats.zscore(self.history)[-1]

    def _calculate(self, internal_state: InternalState):
        book_snapshots = internal_state["book_snapshots"]
        self.midprice_series = (book_snapshots.sell_price_0 + book_snapshots.buy_price_0) / 2
        self.midprice_series = self.midprice_series.iloc[range(-(1 + self.lookback_period), 0)]
        return self.midprice_series.iloc[-1] - self.midprice_series.iloc[-(1 + self.lookback_period)]


class PriceRange(Feature):
    name = "PriceRange"

    def __init__(
        self,
        min_value: float = 0,
        max_value: float = (100 * 100),  # 100 ticks
        maxlen: int = 100000,
        normalisation_on: bool = False,
    ):
        self._min_value = min_value
        self._max_value = max_value
        self.normalisation_on = normalisation_on
        if normalisation_on:
            self.history = deque(maxlen=maxlen)

    @property
    def max_value(self) -> float:
        return self._max_value

    @property
    def min_value(self) -> float:
        return self._min_value

    def reset(self):
        if self.normalisation_on:
            self.history.clear()

    def normalise(self, value: float) -> float:
        if len(self.history) == 0:
            # To prevent a Nan vlaue from being returned
            # if the queue is empty:
            self.history.append(value + 1e-06)
        self.history.append(value)
        return stats.zscore(self.history)[-1]

    def _calculate(self, internal_state: InternalState):
        max_price = internal_state["book_snapshots"].sell_price_0.max()
        min_price = internal_state["book_snapshots"].buy_price_0.min()
        price_range = max_price - min_price
        return price_range


class Volatility(Feature):
    name = "Volatility"

    def __init__(
        self,
        min_value: float = 0,
        max_value: float = ((100 * 100) ** 2),
        maxlen: int = 100000,
        normalisation_on: bool = False,
    ):
        self._min_value = min_value
        self._max_value = max_value
        self.normalisation_on = normalisation_on
        if normalisation_on:
            self.history = deque(maxlen=maxlen)

    @property
    def max_value(self) -> float:
        return self._max_value

    @property
    def min_value(self) -> float:
        return self._min_value

    def reset(self):
        if self.normalisation_on:
            self.history.clear()

    def normalise(self, value: float) -> float:
        if len(self.history) == 0:
            # To prevent a Nan vlaue from being returned
            # if the queue is empty:
            self.history.append(value + 1e-06)
        self.history.append(value)
        return stats.zscore(self.history)[-1]

    def _calculate(self, internal_state: InternalState):
        book_snapshots = internal_state["book_snapshots"]
        midprice_df = (book_snapshots.sell_price_0 + book_snapshots.buy_price_0) / 2
        returns_df = midprice_df.diff()
        return returns_df.var()


class MidPrice(Feature):
    name = "MidPrice"

    def __init__(
        self,
        min_value: float = 0,
        max_value: float = (1000 * 10000),  # Here, we assume that stock prices are less than $1000
        maxlen: int = 100000,
        normalisation_on: bool = False,
    ):
        self._min_value = min_value
        self._max_value = max_value
        self.normalisation_on = normalisation_on
        if normalisation_on:
            self.history = deque(maxlen=maxlen)

    @property
    def max_value(self) -> float:
        return self._max_value

    @property
    def min_value(self) -> float:
        return self._min_value

    def reset(self):
        if self.normalisation_on:
            self.history.clear()

    def normalise(self, value: float) -> float:
        if len(self.history) == 0:
            # To prevent a Nan vlaue from being returned
            # if the queue is empty:
            self.history.append(value + 1e-06)
        self.history.append(value)
        return stats.zscore(self.history)[-1]

    def _calculate(self, internal_state: InternalState):
        current_book = internal_state["book_snapshots"].iloc[-1]
        return self.calculate_from_current_book(current_book)

    @staticmethod
    def calculate_from_current_book(current_book: pd.Series):
        return (current_book.sell_price_0 + current_book.buy_price_0) / 2


class MicroPrice(Feature):
    name = "MicroPrice"

    def __init__(
        self,
        min_value: float = 0,
        max_value: float = (5_000 * 10_000),  # Here, we assume that stock prices are less than $5000
        maxlen: int = 100000,
        normalisation_on: bool = False,
    ):
        self._min_value = min_value
        self._max_value = max_value
        self.normalisation_on = normalisation_on
        if normalisation_on:
            self.history = deque(maxlen=maxlen)

    @property
    def max_value(self) -> float:
        return self._max_value

    @property
    def min_value(self) -> float:
        return self._min_value

    def reset(self):
        if self.normalisation_on:
            self.history.clear()

    def normalise(self, value: float) -> float:
        if len(self.history) == 0:
            # To prevent a Nan vlaue from being returned
            # if the queue is empty:
            self.history.append(value + 1e-06)
        self.history.append(value)
        return stats.zscore(self.history)[-1]

    def _calculate(self, internal_state: InternalState):
        current_book = internal_state["book_snapshots"].iloc[-1]
        return self.calculate_from_current_book(current_book)

    @staticmethod
    def calculate_from_current_book(current_book: pd.Series):
        sell_weighting = current_book.buy_volume_0 / (current_book.buy_volume_0 + current_book.sell_volume_0)
        return current_book.sell_price_0 * sell_weighting + current_book.buy_price_0 * (1 - sell_weighting)


# Agent features


class Inventory(Feature):
    name = "Inventory"

    def __init__(self, max_value: float = 1000.0, maxlen: int = 100000, normalisation_on: bool = False):
        self._max_value = max_value
        self._min_value = -max_value
        self.normalisation_on = normalisation_on
        if normalisation_on:
            self.history = deque(maxlen=maxlen)

    def reset(self):
        if self.normalisation_on:
            self.history.clear()

    @property
    def max_value(self) -> float:
        return self._max_value

    @property
    def min_value(self) -> float:
        return self._min_value

    def normalise(self, value: float) -> float:
        if len(self.history) == 0:
            # To prevent a Nan vlaue from being returned
            # if the queue is empty:
            self.history.append(value + 1e-06)
        self.history.append(value)
        return stats.zscore(self.history)[-1]

    def _calculate(self, internal_state: InternalState) -> float:
        return internal_state["inventory"]


class TimeRemaining(Feature):
    name = "TimeRemaining"
    min_value = 0
    max_value = 1.0

    def __init__(self):
        self.normalisation_on = False

    def reset(self):
        pass

    def normalise(self, value: float) -> float:
        pass

    def _calculate(self, internal_state: InternalState) -> float:
        return internal_state["proportion_of_episode_remaining"]
