import abc

import pandas as pd


class CannotUpdateError(Exception):
    pass


class BookFeature(metaclass=abc.ABCMeta):
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

    def calculate(self, book_snapshots: pd.DataFrame) -> float:
        return self.clamp(self._calculate(book_snapshots), min_value=self.min_value, max_value=self.max_value)

    def update(self, book_snapshot: pd.Series) -> float:
        return self.clamp(self._update(book_snapshot), min_value=self.min_value, max_value=self.max_value)

    @abc.abstractmethod
    def _calculate(self, book_snapshots: pd.DataFrame) -> float:
        pass

    @abc.abstractmethod
    def _update(self, book_snapshot: pd.Series) -> float:
        pass

    @staticmethod
    def clamp(number: float, min_value: float, max_value: float):
        return max(min(number, max_value), min_value)


class Spread(BookFeature):
    max_value = int(100 * 100)  # 100 ticks
    min_value = 0.0

    def __init__(self, min_value=0.0, max_value=int(100 * 100)):
        self.min_value = min_value
        self.max_value = max_value

    def _calculate(self, book_snapshots: pd.DataFrame):
        current_book = book_snapshots.iloc[-1]
        return self._update(current_book)

    def _update(self, book_snapshot: pd.Series):
        return book_snapshot.sell_price_0 - book_snapshot.buy_price_0


class MidpriceMove(BookFeature):
    max_value = 100 * 100  # 100 tick upward move
    min_value = -100 * 100  # 100 tick downward move

    def __init__(self, lookback_period: int = 1):
        self.lookback_period = lookback_period
        self.midprice_move = 0
        self.current_midprice = 0
        self.midprice_series = pd.Series([], dtype=int)

    def _calculate(self, book_snapshots: pd.DataFrame):
        self.midprice_series = (book_snapshots.sell_price_0 + book_snapshots.buy_price_0) / 2
        self.midprice_series = self.midprice_series.iloc[range(-(1 + self.lookback_period), 0)]
        return self.midprice_series.iloc[-1] - self.midprice_series.iloc[-(1 + self.lookback_period)]

    def _update(self, book_snapshot: pd.Series):
        if len(self.midprice_series) == 0:
            raise CannotUpdateError("Must call calculate() at least one to instantiate the midprice series.")
        self.midprice_series.drop(index=self.midprice_series.index[0], inplace=True)
        new_midprice = (book_snapshot.sell_price_0 + book_snapshot.buy_price_0) / 2
        self.midprice_series = pd.concat([self.midprice_series, pd.Series(new_midprice, index=[book_snapshot.name])])
        return self.midprice_series.iloc[-1] - self.midprice_series.iloc[-(1 + self.lookback_period)]


class PriceRange(BookFeature):
    max_value = int(100 * 100)  # 100 ticks
    min_value = 0

    def __init__(self):
        self.price_range = 0
        self.max_price = 0
        self.min_price = 0

    def _calculate(self, book_snapshots: pd.DataFrame):
        self.max_price = book_snapshots.sell_price_0.max()
        self.min_price = book_snapshots.buy_price_0.min()
        self.price_range = self.max_price - self.min_price
        return self.price_range

    def _update(self, book_snapshot: pd.Series):
        raise CannotUpdateError("Price range must be calculated using a dataframe of prices.")


class Volatility(BookFeature):
    min_value = 0
    max_value = (100 * 100) ** 2

    def __init__(self):
        self.returns_df = pd.DataFrame([], dtype=int)

    def _calculate(self, book_snapshots: pd.DataFrame):
        midprice_df = (book_snapshots.sell_price_0 + book_snapshots.buy_price_0) / 2
        self.returns_df = midprice_df.diff()
        return self.returns_df.var()

    def _update(self, book_snapshot: pd.Series):
        raise CannotUpdateError("Price range must be calculated using a dataframe of prices.")


class Microprice(BookFeature):
    min_value = 0
    max_value = 1000 * 10000  # Here, we assume that stock prices are less than $1000

    def _calculate(self, book_snapshots: pd.DataFrame):
        current_book = book_snapshots.iloc[-1]
        return self._update(current_book)

    def _update(self, book_snapshot: pd.Series):
        sell_weighting = book_snapshot.buy_volume_0 / (book_snapshot.buy_volume_0 + book_snapshot.sell_volume_0)
        return book_snapshot.sell_price_0 * sell_weighting + book_snapshot.buy_price_0 * (1 - sell_weighting)
