from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, time, timedelta, date

import numpy as np
from scipy import stats
import abc

from RL4MM.orderbook.models import Orderbook, FilledOrders

from typing import Optional


MIN_TRADING_TIME = time(10)  # We ignore the first half an hour of trading
MAX_TRADING_TIME = time(15, 30)  # We ignore the last half an hour of trading


class CannotUpdateError(Exception):
    pass


@dataclass
class Portfolio:
    inventory: int
    cash: float


@dataclass
class State:
    filled_orders: FilledOrders
    orderbook: Orderbook
    price: float
    portfolio: Portfolio
    now_is: datetime


class Feature(metaclass=abc.ABCMeta):
    def __init__(
        self,
        name: str,
        min_value: float,
        max_value: float,
        update_frequency: timedelta,
        lookback_periods: int,
        normalisation_on: bool,
        max_norm_len: int = 10000,
    ):
        self.name = name
        self.min_value = min_value
        self.max_value = max_value
        assert update_frequency <= timedelta(minutes=1), "HFT update frequency must be less than 1 minute."
        self.update_frequency = update_frequency
        self.lookback_periods = lookback_periods
        self.normalisation_on = normalisation_on
        self.max_norm_len = max_norm_len
        self.current_value = 0.0
        self.first_usage_time = datetime.min
        if self.normalisation_on:
            self.history: deque = deque(maxlen=max_norm_len)

    @property
    def window_size(self) -> timedelta:
        return self.lookback_periods * self.update_frequency

    def normalise(self, value: float) -> float:
        if len(self.history) == 0:
            # To prevent a Nan value from being returned
            # if the queue is empty:
            # TODO: tidy this
            self.history.append(value + 1e-06)
        self.history.append(value)
        return stats.zscore(self.history)[-1]

    @abc.abstractmethod
    def reset(self, state: State, first_usage_time: Optional[datetime] = None):
        pass

    def update(self, state: State) -> None:
        if state.now_is >= self.first_usage_time and self._now_is_multiple_of_update_freq(state.now_is):
            self._update(state)
            value = self.clamp(self.current_value, min_value=self.min_value, max_value=self.max_value)
            if value != self.current_value:
                print(f"Clamping value of {self.name} from {self.current_value} to {value}.")
            self.current_value = self.normalise(value) if self.normalisation_on else value

    @abc.abstractmethod
    def _update(self, state: State) -> None:
        pass

    def _reset(self, state: State, first_usage_time: Optional[datetime] = None):
        self.first_usage_time = first_usage_time or datetime.min
        if self.normalisation_on:
            self.history.clear()
        self._update(state)

    @staticmethod
    def clamp(number: float, min_value: float, max_value: float):
        return max(min(number, max_value), min_value)

    def _now_is_multiple_of_update_freq(self, now_is: datetime):
        return timedelta(seconds=now_is.second, microseconds=now_is.microsecond) % self.update_frequency == timedelta(
            microseconds=0
        )


########################################################################################################################
#                                                   Book features                                                      #
########################################################################################################################


class Spread(Feature):
    def __init__(
        self,
        name: str = "Spread",
        min_value: float = 0,
        max_value: float = (50 * 100),  # 50 ticks
        update_frequency: timedelta = timedelta(seconds=0.1),
        normalisation_on: bool = False,
        max_norm_len: int = 10000,
    ):
        super().__init__(name, min_value, max_value, update_frequency, 0, normalisation_on, max_norm_len)

    def reset(self, state: State, first_usage_time: Optional[datetime] = None):
        super()._reset(state, first_usage_time)

    def _update(self, state: State) -> None:
        self.current_value = state.orderbook.spread


class BookImbalance(Feature):
    def __init__(
        self,
        update_frequency: timedelta = timedelta(seconds=0.1),
    ):
        super().__init__("BookImbalance", -1, 1, update_frequency, 0, False, 0)

    def reset(self, state: State, first_usage_time: Optional[datetime] = None):
        super()._reset(state, first_usage_time)

    def _update(self, state: State) -> None:
        self.current_value = state.orderbook.imbalance


class PriceMove(Feature):
    """The price move is calculated as the difference between the price at time now_is - min_updates * update_freq
    and now_is. Here, the price is given when calling update and could be defined as the midprice, the microprice, or
    any other sensible notion of price."""

    def __init__(
        self,
        name: str = "MidpriceMove",
        min_value: float = -100 * 100,  # 100 tick downward move
        max_value: float = 100 * 100,  # 100 tick upward move
        update_frequency: timedelta = timedelta(seconds=1),
        lookback_periods: int = 10,  # Calculate the move in the midprice between 10 time periods ago and now
        normalisation_on: bool = False,
        max_norm_len: int = 100_000,
    ):
        super().__init__(name, min_value, max_value, update_frequency, lookback_periods, normalisation_on, max_norm_len)
        self.prices: deque = deque(maxlen=self.lookback_periods + 1)

    def reset(self, state: State, first_usage_time: Optional[datetime] = None):
        self.prices = deque(maxlen=self.lookback_periods + 1)
        super()._reset(state, first_usage_time)

    def _update(self, state: State) -> None:
        self.prices.appendleft(state.price)
        self.current_value = self.prices[0] - self.prices[-1]


class PriceRange(Feature):
    name = "PriceRange"

    def __init__(
        self,
        name: str = "PriceRange",
        min_value: float = 0,
        max_value: float = (100 * 100),  # 100 ticks
        update_frequency: timedelta = timedelta(seconds=1),
        lookback_periods: int = 10,
        normalisation_on: bool = False,
        max_norm_len: int = 100_000,
    ):
        super().__init__(name, min_value, max_value, update_frequency, lookback_periods, normalisation_on, max_norm_len)
        self.prices: deque = deque(maxlen=self.lookback_periods + 1)

    def reset(self, state: State, first_usage_time: Optional[datetime] = None):
        self.prices = deque(maxlen=self.lookback_periods + 1)
        super()._reset(state, first_usage_time)

    def _update(self, state: State) -> None:
        self.prices.appendleft(state.price)
        self.current_value = max(self.prices) - min(self.prices)  # TODO: speed this up if needed


class Volatility(Feature):
    """The volatility of the midprice series over a trailing window. We use the variance of percentage returns as
    opposed to the standard deviation of percentage returns as variance scales linearly with time and is therefore more
    reasonably a dimensionless attribute of the returns series. Furthermore, we ignore the mean of the returns since
    they are too noisy an observation and a *much* larger number of returns is required for it to be useful."""

    def __init__(
        self,
        name: str = "Volatility",
        min_value: float = 0,
        max_value: float = 1.0,
        update_frequency: timedelta = timedelta(seconds=1),
        lookback_periods: int = 10,
        normalisation_on: bool = False,
        max_norm_len: int = 100_000,
    ):
        super().__init__(name, min_value, max_value, update_frequency, lookback_periods, normalisation_on, max_norm_len)
        self.prices: deque = deque(maxlen=self.lookback_periods + 1)

    def reset(self, state: State, first_usage_time: Optional[datetime] = None):
        self.prices = deque(maxlen=self.lookback_periods + 1)
        super()._reset(state, first_usage_time)

    def _update(self, state: State) -> None:
        if len(self.prices) < self.lookback_periods:
            self.prices.append(state.price)
            self.current_value = 0.0
        elif len(self.prices) == self.lookback_periods:
            self.prices.append(state.price)
            pct_returns = np.diff(np.array(self.prices)) / np.array(self.prices)[:1]
            self.current_value = sum(pct_returns**2) / len(pct_returns)
        else:
            assert len(self.prices) == self.lookback_periods + 1
            oldest_price = self.prices.popleft()
            oldest_pct_return = (self.prices[0] - oldest_price) / oldest_price
            new_price = state.price
            new_pct_return = (new_price - self.prices[-1]) / self.prices[-1]
            self.prices.append(new_price)
            sum_of_squares = self.current_value * self.lookback_periods - oldest_pct_return**2 + new_pct_return**2
            self.current_value = sum_of_squares / self.lookback_periods


class Price(Feature):
    def __init__(
        self,
        name: str = "Price",
        min_value: float = 0,
        max_value: float = (10_000 * 10_000),  # 10,000 dollars
        update_frequency: timedelta = timedelta(seconds=1),
        normalisation_on: bool = False,
        max_norm_len: int = 100_000,
    ):
        super().__init__(name, min_value, max_value, update_frequency, 0, normalisation_on, max_norm_len)

    def reset(self, state: State, first_usage_time: Optional[datetime] = None):
        super()._reset(state, first_usage_time)

    def _update(self, state: State) -> None:
        self.current_value = state.price


########################################################################################################################
#                                                Order Flow features                                                   #
########################################################################################################################


class TradeVolumeImbalance(Feature):
    """The trade volume imbalance is given by volume_buy_executions - volume_sell_executions / total_volume_executions
    over a given period."""

    def __init__(
        self,
        name: str = "TradeVolumeImbalance",
        update_frequency: timedelta = timedelta(seconds=0.1),
        lookback_periods: int = 600,
        track_internal: bool = False,
        normalisation_on: bool = False,
        max_norm_len: int = 100_000,
    ):
        super().__init__(name, -1.0, 1.0, update_frequency, lookback_periods, normalisation_on, max_norm_len)
        self.track_internal = track_internal
        self.volumes = dict(buy=deque(maxlen=self.lookback_periods + 1), sell=deque(maxlen=self.lookback_periods + 1))
        self.total_volume = 0
        self.volume_imbalance = 0

    def reset(self, state: State, first_usage_time: Optional[datetime] = None):
        self.volumes = dict(buy=deque(maxlen=self.lookback_periods + 1), sell=deque(maxlen=self.lookback_periods + 1))
        self.total_volume = 0
        self.volume_imbalance = 0
        super()._reset(state, first_usage_time)

    def _update(self, state: State) -> None:
        buy_volume = sum(order.volume for order in state.filled_orders.external if order.direction == "buy")
        sell_volume = sum(order.volume for order in state.filled_orders.external if order.direction == "sell")
        if self.track_internal:
            buy_volume += sum(order.volume for order in state.filled_orders.internal if order.direction == "buy")
            sell_volume += sum(order.volume for order in state.filled_orders.internal if order.direction == "sell")
        if len(self.volumes) <= self.lookback_periods:
            self._update_volumes(buy_volume, sell_volume)
            self.current_value = 0.0
        elif self.total_volume == 0:
            self._update_volumes(buy_volume, sell_volume)
            self.total_volume = sum(self.volumes["buy"]) + sum(self.volumes["sell"])
            self.volume_imbalance = sum(self.volumes["buy"]) - sum(self.volumes["sell"])
            self.current_value = self.volume_imbalance / self.total_volume
        else:
            oldest_volumes = {side: self.volumes[side].popleft() for side in ("buy", "sell")}
            self.total_volume -= oldest_volumes["buy"] + oldest_volumes["sell"]
            self.total_volume += buy_volume + sell_volume
            self.volume_imbalance -= oldest_volumes["buy"] - oldest_volumes["sell"]
            self.volume_imbalance += buy_volume + sell_volume
            self._update_volumes(buy_volume, sell_volume)
            self.current_value = self.volume_imbalance / self.total_volume

    def _update_volumes(self, buy_volume: int, sell_volume: int):
        self.volumes["buy"].append(buy_volume)
        self.volumes["sell"].append(sell_volume)


class TradeDirectionImbalance(Feature):
    """The trade direction imbalance is given by number_buy_executions - number_sell_executions / total_executions
    over a given period."""

    def __init__(
        self,
        name: str = "TradeImbalance",
        update_frequency: timedelta = timedelta(seconds=0.1),
        lookback_periods: int = 600,
        track_internal: bool = False,
        normalisation_on: bool = False,
        max_norm_len: int = 100_000,
    ):
        super().__init__(name, -1.0, 1.0, update_frequency, lookback_periods, normalisation_on, max_norm_len)
        self.track_internal = track_internal
        self.trades = dict(buy=deque(maxlen=self.lookback_periods + 1), sell=deque(maxlen=self.lookback_periods + 1))
        self.total_trades = 0
        self.direction_imbalance = 0

    def reset(self, state: State, first_usage_time: Optional[datetime] = None):
        self.trades = dict(buy=deque(maxlen=self.lookback_periods + 1), sell=deque(maxlen=self.lookback_periods + 1))
        self.total_trades = 0
        self.direction_imbalance = 0
        super()._reset(state, first_usage_time)

    def _update(self, state: State) -> None:
        num_buys = sum(1 for order in state.filled_orders.external if order.direction == "buy")
        num_sells = sum(1 for order in state.filled_orders.external if order.direction == "sell")
        if self.track_internal:
            num_buys += sum(1 for order in state.filled_orders.internal if order.direction == "buy")
            num_sells += sum(1 for order in state.filled_orders.internal if order.direction == "sell")
        if len(self.trades) <= self.lookback_periods:
            self._update_trades(num_buys, num_sells)
            self.current_value = 0.0
        elif self.total_trades == 0:
            self._update_trades(num_buys, num_sells)
            self.total_trades = sum(self.trades["buy"]) + sum(self.trades["sell"])
            self.direction_imbalance = sum(self.trades["buy"]) - sum(self.trades["sell"])
            self.current_value = self.direction_imbalance / self.total_trades
        else:
            oldest_trades = {side: self.trades[side].popleft() for side in ("buy", "sell")}
            self.total_trades -= oldest_trades["buy"] + oldest_trades["sell"]
            self.total_trades += num_buys + num_sells
            self.direction_imbalance -= oldest_trades["buy"] - oldest_trades["sell"]
            self.direction_imbalance += num_buys + num_sells
            self._update_trades(num_buys, num_sells)
            self.current_value = self.direction_imbalance / self.total_trades

    def _update_trades(self, num_buys: int, num_sells: int):
        self.trades["buy"].append(num_buys)
        self.trades["sell"].append(num_sells)


########################################################################################################################
#                                                  Agent features                                                      #
########################################################################################################################


class Inventory(Feature):
    def __init__(
        self,
        name: str = "Inventory",
        min_value: float = -1000000,
        max_value: float = 1000000,
        update_frequency: timedelta = timedelta(seconds=0.1),
        normalisation_on: bool = False,
        max_norm_len: int = 100_000,
    ):
        super().__init__(name, min_value, max_value, update_frequency, 0, normalisation_on, max_norm_len)

    def reset(self, state: State, first_usage_time: Optional[datetime] = None):
        super()._reset(state, first_usage_time)

    def _update(self, state: State) -> None:
        self.current_value = state.portfolio.inventory


class EpisodeProportion(Feature):
    def __init__(
        self,
        name: str = "EpisodeProportion",
        update_frequency: timedelta = timedelta(seconds=0.1),
        episode_length: timedelta = timedelta(minutes=60),
        normalisation_on: bool = False,
        max_norm_len: int = 100_000,
    ):
        super().__init__(name, 0.0, 1.0, update_frequency, 0, normalisation_on, max_norm_len)
        self.current_value: float = 0.0
        self.episode_length = episode_length
        self.step_size: float = update_frequency / episode_length

    def reset(self, state: State, first_usage_time: Optional[datetime] = None):
        super()._reset(state, first_usage_time)
        self.current_value = 0.0

    def _update(self, state: State) -> None:
        self.current_value += self.step_size
        self.current_value = np.round(self.current_value, 3)  # To avoid always clamping due to floating point summation


class TimeOfDay(Feature):
    def __init__(
        self,
        name: str = "TimeOfDay",
        update_frequency: timedelta = timedelta(minutes=1),
        normalisation_on: bool = False,
        max_norm_len: int = 100_000,
        n_buckets: int = 10,
    ):
        super().__init__(name, 0, n_buckets - 1, update_frequency, 0, normalisation_on, max_norm_len)
        self.n_buckets = n_buckets
        self.min_time = datetime.combine(date(1, 1, 1), MIN_TRADING_TIME)
        self.max_time = datetime.combine(date(1, 1, 1), MAX_TRADING_TIME)
        self.bucket_size = (self.max_time - self.min_time) / self.n_buckets

    def reset(self, state: State, first_usage_time: Optional[datetime] = None):
        self.min_time = datetime.combine(state.now_is.date(), MIN_TRADING_TIME)
        self.max_time = datetime.combine(state.now_is.date(), MAX_TRADING_TIME)
        super()._reset(state, first_usage_time)

    def _update(self, state: State) -> None:
        self.current_value = max((state.now_is - self.min_time) // self.bucket_size, 0)  # to avoid issues with warm-up
