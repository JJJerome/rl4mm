from datetime import datetime
from unittest import TestCase

import numpy as np
import pandas as pd
from RL4MM.features.Features import (
    Spread,
    MidpriceMove,
    PriceRange,
    Volatility,
    MicroPrice,
    InternalState,
)
from RL4MM.orderbook.helpers import get_book_columns

columns = get_book_columns(n_levels=1)
timestamps = pd.date_range(start=datetime(2012, 6, 21, 10), end=datetime(2012, 6, 21, 10, 0, 9), freq="S")
sell_prices = pd.Series([10.2, 10.3, 10.0, 10.1, 9.8, 10.2, 10.0, 10.1, 10.1, 9.9], index=timestamps, name=columns[0])
sell_volumes = pd.Series([200, 300, 120, 150, 100, 100, 250, 200, 10, 300], index=timestamps, name=columns[1])
buy_prices = pd.Series([10.0, 10.1, 9.9, 9.9, 9.7, 10.0, 9.9, 9.9, 10.0, 9.8], index=timestamps, name=columns[2])
buy_volumes = pd.Series([100, 300, 200, 150, 200, 110, 65, 200, 50, 200], index=timestamps, name=columns[3])
buy_prices *= 10000
sell_prices *= 10000
MOCK_BOOK_SNAPSHOTS = pd.concat([sell_prices, sell_volumes, buy_prices, buy_volumes], axis=1)
MOCK_BOOK_SNAPSHOTS = MOCK_BOOK_SNAPSHOTS.astype(int)
MIDPRICE_SERIES = pd.Series([10.1, 10.2, 9.95, 10.0, 9.75, 10.1, 9.95, 10.0, 10.05, 9.85], index=timestamps) * 10000
RETURNS_SERIES = pd.Series([np.nan, 0.1, -0.25, 0.05, -0.25, 0.35, -0.15, 0.05, 0.05, -0.2], index=timestamps) * 10000
MOCK_INTERNAL_STATE = InternalState(inventory=0, cash=0, asset_price=0, book_snapshots=MOCK_BOOK_SNAPSHOTS)


class TestBookFeatures(TestCase):
    def test_spread_clamp(self):
        spread = Spread(max_value=200)
        big_spread_df = MOCK_BOOK_SNAPSHOTS.copy()
        big_spread_df.loc[big_spread_df.iloc[0].name, "sell_price_0"] = 1000000
        expected = 200
        big_spread_state = MOCK_INTERNAL_STATE.copy()
        big_spread_state["book_snapshots"] = big_spread_df
        actual = spread.calculate(big_spread_state)
        self.assertEqual(expected, actual)

    def test_spread(self):
        spread = Spread()
        expected = int(9.9 * 10000) - int(9.8 * 10000)
        self.assertEqual(expected, spread.calculate(internal_state=MOCK_INTERNAL_STATE))

    def test_midprice_move_calculate(self):
        midprice_move = MidpriceMove(lookback_period=1)
        midprice_move_5 = MidpriceMove(lookback_period=5)
        current_midprice = (9.9 * 10000 + 9.8 * 10000) / 2
        previous_midprice = (10.1 * 10000 + 10.0 * 10000) / 2
        midprice_t_minus_5 = (9.8 * 10000 + 9.7 * 10000) / 2
        expected = current_midprice - previous_midprice
        actual = midprice_move.calculate(MOCK_INTERNAL_STATE)
        self.assertEqual(expected, actual)
        expected = current_midprice - midprice_t_minus_5
        actual = midprice_move_5.calculate(MOCK_INTERNAL_STATE)
        self.assertEqual(expected, actual)

    def test_price_range(self):
        price_range = PriceRange()
        expected = 10.3 * 10000 - 9.7 * 10000
        actual = price_range.calculate(MOCK_INTERNAL_STATE)
        self.assertEqual(expected, actual)

    def test_volatility(self):
        volatility = Volatility()
        expected = np.nanvar(RETURNS_SERIES, ddof=1)
        actual = volatility.calculate(MOCK_INTERNAL_STATE)
        self.assertEqual(expected, actual)

    def test_microprice(self):
        microprice = MicroPrice()
        expected = (9.9 * 200 / (200 + 300) + 9.8 * 300 / (200 + 300)) * 10000
        actual = microprice.calculate(MOCK_INTERNAL_STATE)
        self.assertEqual(expected, actual)