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

MOCK_INTERNAL_STATE_1 = InternalState(inventory=1, cash=0, asset_price=100, book_snapshots=None)
MOCK_INTERNAL_STATE_2 = InternalState(inventory=1, cash=0, asset_price=110, book_snapshots=None)
MOCK_INTERNAL_STATE_3 = InternalState(inventory=1, cash=0, asset_price=100, book_snapshots=None)
MOCK_INTERNAL_STATE_4 = InternalState(inventory=1, cash=0, asset_price=0, book_snapshots=None)

class TestRollingSharpe(TestCase):

    def test_calculate(self):

        rs = RollingSharpe(max_window_size=3, min_window_size=3)

        sharpe1 = rs.calculate(MOCK_INTERNAL_STATE_1, MOCK_INTERNAL_STATE_1)
        # only 1 entry in aum array so expect 0 reward
        self.assertEqual(0, sharpe1)

        # only 2 entries in aum array so expect 0 reward
        sharpe2 = rs.calculate(MOCK_INTERNAL_STATE_1, MOCK_INTERNAL_STATE_2)
        self.assertEqual(0, sharpe2)

        # now 3 entries so we get a proper Sharpe calculation
        sharpe3 = rs.calculate(MOCK_INTERNAL_STATE_2, MOCK_INTERNAL_STATE_3)
        # produces aum array [100,110,100] by sharp3 giving via 
        # separate calculation in R:
        # > rets <- c(110/100,100/110) - 1
        # > mean(rets)/sd(rets)
        # [1] 0.03367175
        expected = 0.03367175
        self.assertAlmostEqual(expected, sharpe3)

        with self.assertRaises(Exception):
            # now we get a 0 aum value 
            rs.calculate(MOCK_INTERNAL_STATE_3, MOCK_INTERNAL_STATE_4)

        rs.reset()
        # Now test that we get 0 when aum doesn't change
        sharpe1 = rs.calculate(MOCK_INTERNAL_STATE_1, MOCK_INTERNAL_STATE_1)
        sharpe2 = rs.calculate(MOCK_INTERNAL_STATE_1, MOCK_INTERNAL_STATE_1)
        sharpe3 = rs.calculate(MOCK_INTERNAL_STATE_1, MOCK_INTERNAL_STATE_1)
        
        self.assertEqual(0, sharpe3)
