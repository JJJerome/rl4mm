from collections import deque
from datetime import datetime, time, timedelta

import numpy as np
from scipy import stats
import abc
import sys

from RL4MM.orderbook.models import Orderbook, FilledOrderTuple

if sys.version_info[0] == 3 and sys.version_info[1] >= 8:
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

import pandas as pd

MIN_TRADING_TIME = time(10)  # We ignore the first half an hour of trading
MAX_TRADING_TIME = time(15, 30)  # We ignore the last half an hour of trading


class CannotUpdateError(Exception):
    pass


class InternalState(TypedDict):
    inventory: int
    cash: float
    asset_price: float
    book_snapshots: pd.DataFrame
    proportion_of_episode_remaining: float


class Feature(metaclass=abc.ABCMeta):
    def __init__(
        self,
        name: str,
        min_value: float,
        max_value: float,
        update_frequency: timedelta,
        min_updates: int,
        normalisation_on: bool,
        max_norm_len: int = 10000,
    ):
        self.name = name
        self.min_value = min_value
        self.max_value = max_value
        assert update_frequency <= timedelta(minutes=1), "HFT update frequency must be less than 1 minute."
        self.update_frequency = update_frequency
        self.min_updates = min_updates
        self.normalisation_on = normalisation_on
        self.max_norm_len = max_norm_len
        self.current_value = None
        if self.normalisation_on:
            self.history = deque(maxlen=max_norm_len)

    @property
    def window_size(self) -> timedelta:
        return self.min_updates * self.update_frequency

    def normalise(self, value: float) -> float:
        if len(self.history) == 0:
            # To prevent a Nan value from being returned
            # if the queue is empty:
            # TODO: tidy this
            self.history.append(value + 1e-06)
        self.history.append(value)
        return stats.zscore(self.history)[-1]

    @abc.abstractmethod
    def reset(self, orderbook: Orderbook):
        pass

    def update(self, now_is: datetime, filled_orders: FilledOrderTuple, current_orderbook: Orderbook) -> float:
        if timedelta(seconds=now_is.second, microseconds=now_is.microsecond) % self.update_frequency == 0:
            self._update(filled_orders, current_orderbook)
            value = self.clamp(self.current_value, min_value=self.min_value, max_value=self.max_value)
            if value != self.current_value:
                print(f"Clamping value of {self.name} from {self.current_value} to {value}.")
            return self.normalise(value) if self.normalisation_on else value

    @abc.abstractmethod
    def _update(self, filled_orders: FilledOrderTuple, current_orderbook: Orderbook) -> None:
        pass

    def _reset(self):
        self.current_value = None
        if self.normalisation_on:
            self.history.clear()

    @staticmethod
    def clamp(number: float, min_value: float, max_value: float):
        return max(min(number, max_value), min_value)


# Book features
class Spread(Feature):
    def __init__(
        self,
        name: str = "Spread",
        min_value: float = 0,
        max_value: float = (100 * 100),  # 100 ticks
        update_frequency: timedelta = timedelta(seconds=1),
        normalisation_on: bool = False,
        max_norm_len: int = 10000,
    ):
        super().__init__(name, min_value, max_value, update_frequency, 1, normalisation_on, max_norm_len)

    def reset(self, orderbook: Orderbook):
        super()._reset()

    def _update(self, filled_orders: FilledOrderTuple, current_orderbook: Orderbook) -> None:
        self.current_value = current_orderbook.best_sell_price - current_orderbook.best_buy_price


class MidpriceMove(Feature):
    """The midprice move is calculated as the difference between the midprice at time now_is - min_updates * update_freq
    and now_is."""
    def __init__(
        self,
        name: str = "MidpriceMove",
        min_value: float = -100 * 100,  # 100 tick downward move
        max_value: float = 100 * 100,  # 100 tick upward move
        update_frequency: timedelta = timedelta(seconds=1),
        min_updates: int = 10,  # Calculate the move in the midprice between 10 time periods ago and now
        normalisation_on: bool = False,
        max_norm_len: int = 100000,
    ):
        super().__init__(name, min_value, max_value, update_frequency, min_updates, normalisation_on, max_norm_len)
        self.midprices = deque(maxlen=self.min_updates)

    def reset(self, orderbook: Orderbook):
        super()._reset()
        self.midprices = deque(maxlen=self.min_updates)
        self.midprices.appendleft(orderbook.midprice)

    def _update(self, filled_orders: FilledOrderTuple, current_orderbook: Orderbook) -> None:
        self.midprices.appendleft(current_orderbook.midprice)
        self.current_value = self.midprices[0] - self.midprices[-1]


class PriceRange(Feature):
    name = "PriceRange"

    def __init__(
        self,
        name:str = "PriceRange",
        min_value: float = 0,
        max_value: float = (100 * 100),  # 100 ticks
        update_frequency: timedelta = timedelta(seconds=1),
        min_updates: int = 10,
        normalisation_on: bool = False,
        max_norm_len: int = 100000,
    ):
        super().__init__(name, min_value, max_value, update_frequency, min_updates, normalisation_on, max_norm_len)
        self.midprices = deque(maxlen=self.min_updates)

    def reset(self, orderbook: Orderbook):
        super()._reset()
        self.midprices = deque(maxlen=self.min_updates)
        self.midprices.appendleft(orderbook.midprice)

    def _update(self, filled_orders: FilledOrderTuple, current_orderbook: Orderbook) -> float:
        self.midprices.appendleft(current_orderbook.midprice)
        self.current_value = max(self.midprices) - min(self.midprices)  # TODO: speed this up if needed


class Volatility(Feature):
    """The volatility of the midprice series over a trailing window. We use the variance of percentage returns as
    opposed to the standard deviation of percentage returns as variance scales linearly with time and is therefore more
    reasonably a dimensionless attribute of the returns series. Furthermore, """
    def __init__(
        self,
        name: str = "Volatility",
        min_value: float = 0,
        max_value: float = ((100 * 100) ** 2),
        update_frequency: timedelta = timedelta(seconds=1),
        min_updates: int = 10,
        normalisation_on: bool = False,
        max_norm_len: int = 100000,
    ):
        super().__init__(name, min_value, max_value, update_frequency, min_updates, normalisation_on, max_norm_len)
        self.midprices = deque(maxlen=self.min_updates)

    def reset(self, orderbook: Orderbook):
        super()._reset()
        self.midprices = deque(maxlen=self.min_updates)
        self.midprices.appendleft(orderbook.midprice)

    def _update(self, filled_orders: FilledOrderTuple, current_orderbook: Orderbook) -> float:
        if len(self.midprices) < self.min_updates:
            return np.nan
        elif not self.current_value:
            pct_returns = np.diff(self.midprices)/self.midprices[:-1]
            self.current_value = np.std(pct_returns)
        else:
            oldest_midprice = self.midprices.pop()
            new_midprice = current_orderbook.midprice
            old_variance = self.current_value ** 2
            new_variance = old_variance - oldest_midprice ** 2 + new_midprice ** 2
            self.current_value = np.sqrt(new_variance)
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
        max_norm_len: int = 100000,
        normalisation_on: bool = False,
    ):
        self._min_value = min_value
        self._max_value = max_value
        self.normalisation_on = normalisation_on
        if normalisation_on:
            self.history = deque(maxlen=max_norm_len)

    @property
    def max_value(self) -> float:
        return self._max_value

    @property
    def min_value(self) -> float:
        return self._min_value

    def reset(self, orderbook: Orderbook):
        super()._reset()

    def normalise(self, value: float) -> float:
        if len(self.history) == 0:
            # To prevent a Nan vlaue from being returned
            # if the queue is empty:
            self.history.append(value + 1e-06)
        self.history.append(value)
        return stats.zscore(self.history)[-1]

    def _update(self, internal_state: InternalState):
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
        max_norm_len: int = 100000,
        normalisation_on: bool = False,
    ):
        self._min_value = min_value
        self._max_value = max_value
        self.normalisation_on = normalisation_on
        if normalisation_on:
            self.history = deque(maxlen=max_norm_len)

    @property
    def max_value(self) -> float:
        return self._max_value

    @property
    def min_value(self) -> float:
        return self._min_value

    def reset(self, orderbook: Orderbook):
        super()._reset()

    def normalise(self, value: float) -> float:
        if len(self.history) == 0:
            # To prevent a Nan vlaue from being returned
            # if the queue is empty:
            self.history.append(value + 1e-06)
        self.history.append(value)
        return stats.zscore(self.history)[-1]

    def _update(self, internal_state: InternalState):
        current_book = internal_state["book_snapshots"].iloc[-1]
        return self.calculate_from_current_book(current_book)

    @staticmethod
    def calculate_from_current_book(current_book: pd.Series):
        sell_weighting = current_book.buy_volume_0 / (current_book.buy_volume_0 + current_book.sell_volume_0)
        return current_book.sell_price_0 * sell_weighting + current_book.buy_price_0 * (1 - sell_weighting)


# Agent features


class Inventory(Feature):
    name = "Inventory"

    def __init__(self, max_value: float = 1000.0, max_norm_len: int = 100000, normalisation_on: bool = False):
        self._max_value = max_value
        self._min_value = -max_value
        self.normalisation_on = normalisation_on
        if normalisation_on:
            self.history = deque(maxlen=max_norm_len)

    def reset(self, orderbook: Orderbook):
        super()._reset()

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

    def _update(self, internal_state: InternalState) -> float:
        return internal_state["inventory"]


class TimeRemaining(Feature):
    name = "TimeRemaining"
    min_value = 0
    max_value = 1.0

    def __init__(self):
        self.normalisation_on = False

    def reset(self, orderbook: Orderbook):
        pass

    def normalise(self, value: float) -> float:
        pass

    def _update(self, internal_state: InternalState) -> float:
        return internal_state["proportion_of_episode_remaining"]


class TimeOfDay(Feature):
    name = "TimeOfDay"
    min_value = 0
    normalisation_on = False

    def __init__(self, n_buckets: int = 10):
        self.n_buckets = n_buckets
        self.min_time = datetime(1, 1, 1, 0, 0, 0)
        self.max_time = datetime(1, 1, 1, 0, 0, 0)
        self._max_value = n_buckets - 1
        self.bucket_size = None

    def reset(self, orderbook: Orderbook):
        super()._reset()
        start_timestamp = internal_state["book_snapshots"].iloc[-1].name
        self.min_time = datetime.combine(start_timestamp.date(), MIN_TRADING_TIME)
        self.max_time = datetime.combine(start_timestamp.date(), MAX_TRADING_TIME)
        self.bucket_size = (self.max_time - self.min_time) / self.n_buckets

    def normalise(self, value: float) -> float:
        pass

    def _update(self, internal_state: InternalState) -> float:
        current_time = internal_state["book_snapshots"].iloc[-1].name
        return (current_time - self.min_time) // self.bucket_size

    @property
    def max_value(self) -> float:
        return self._max_value
