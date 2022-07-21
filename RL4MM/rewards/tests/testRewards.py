from datetime import datetime
from unittest import TestCase

import copy
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
from RL4MM.rewards.RewardFunctions import RollingSharpe

# columns = get_book_columns(n_levels=1)
# timestamps = pd.date_range(start=datetime(2012, 6, 21, 10), end=datetime(2012, 6, 21, 10, 0, 9), freq="S")
# sell_prices = pd.Series([10.2, 10.3, 10.0, 10.1, 9.8, 10.2, 10.0, 10.1, 10.1, 9.9], index=timestamps, name=columns[0])
# sell_volumes = pd.Series([200, 300, 120, 150, 100, 100, 250, 200, 10, 300], index=timestamps, name=columns[1])
# buy_prices = pd.Series([10.0, 10.1, 9.9, 9.9, 9.7, 10.0, 9.9, 9.9, 10.0, 9.8], index=timestamps, name=columns[2])
# buy_volumes = pd.Series([100, 300, 200, 150, 200, 110, 65, 200, 50, 200], index=timestamps, name=columns[3])
# buy_prices *= 10000
# sell_prices *= 10000
# MOCK_BOOK_SNAPSHOTS = pd.concat([sell_prices, sell_volumes, buy_prices, buy_volumes], axis=1)
# MOCK_BOOK_SNAPSHOTS = MOCK_BOOK_SNAPSHOTS.astype(int)
# MIDPRICE_SERIES = pd.Series([10.1, 10.2, 9.95, 10.0, 9.75, 10.1, 9.95, 10.0, 10.05, 9.85], index=timestamps) * 10000
# RETURNS_SERIES = pd.Series([np.nan, 0.1, -0.25, 0.05, -0.25, 0.35, -0.15, 0.05, 0.05, -0.2], index=timestamps) * 10000


MOCK_INTERNAL_STATE_1 = InternalState(inventory=1, cash=0, asset_price=100, book_snapshots=None)
MOCK_INTERNAL_STATE_2 = InternalState(inventory=1, cash=0, asset_price=110, book_snapshots=None)
MOCK_INTERNAL_STATE_3 = InternalState(inventory=1, cash=0, asset_price=100, book_snapshots=None)


class TestRollingSharpe(TestCase):
    def test_calculate(self):

        rs = RollingSharpe(max_window_size=3, min_window_size=3)

        sharpe1 = rs.calculate(MOCK_INTERNAL_STATE_1, MOCK_INTERNAL_STATE_1)
        sharpe2 = rs.calculate(MOCK_INTERNAL_STATE_1, MOCK_INTERNAL_STATE_2)
        sharpe3 = rs.calculate(MOCK_INTERNAL_STATE_2, MOCK_INTERNAL_STATE_3)

        print(sharpe1)
        print(sharpe2)
        print(sharpe3)

        # big_spread_state["book_snapshots"] = big_spread_df
        # actual = spread.calculate(big_spread_state)
        # self.assertEqual(expected, actual)

