from collections import deque
from copy import deepcopy
from datetime import datetime
from unittest import TestCase

import numpy as np
from RL4MM.features.Features import (
    Spread,
    PriceMove,
    PriceRange,
    Volatility,
    Price,
    State,
    Portfolio,
    Inventory,
)
from RL4MM.orderbook.tests.mock_orders import (
    get_mock_orderbook,
    submission_4,
)

MIDPRICES = np.array([30.1, 30.2, 29.95, 30.0, 29.75, 30.1, 29.95, 30.0, 30.05, 29.85]) * 10000
PCT_RETURNS = (MIDPRICES[1:] - MIDPRICES[:-1]) / MIDPRICES[:1]

MOCK_PORTFOLIO = Portfolio(inventory=100, cash=-20)
MOCK_ORDERBOOK = get_mock_orderbook()

MOCK_STATE = State(
    filled_orders=([], []),
    orderbook=MOCK_ORDERBOOK,
    price=MIDPRICES[0],
    portfolio=MOCK_PORTFOLIO,
    now_is=datetime(2022, 1, 1, 10),
)


class TestBookFeatures(TestCase):
    def test_spread_clamp(self):
        spread = Spread(max_value=1000)
        big_spread_book = deepcopy(MOCK_ORDERBOOK)
        big_spread_book.sell.pop(int(30.3 * 10000))
        low_price_submission = deepcopy(submission_4)
        low_price_submission.price = int(40.3 * 10000)
        big_spread_book.sell[int(40.3 * 10000)] = deque([low_price_submission])
        big_spread_state = deepcopy(MOCK_STATE)
        big_spread_state.orderbook = big_spread_book
        expected = 1000
        spread.update(big_spread_state)
        actual = spread.current_value
        self.assertEqual(expected, actual)

    def test_spread(self):
        spread = Spread()
        expected = int(30.3 * 10000) - int(30.2 * 10000)
        spread.update(state=MOCK_STATE)
        self.assertEqual(expected, spread.current_value)

    def test_price_move_calculate(self):
        price_move = PriceMove(lookback_periods=1)
        price_move_5 = PriceMove(lookback_periods=5)
        price_move.reset(state=MOCK_STATE)
        price_move_5.reset(state=MOCK_STATE)
        calculated_price_moves = []
        calculated_price_moves_5 = []
        for price in MIDPRICES[1:]:
            state = deepcopy(MOCK_STATE)
            state.price = price
            price_move.update(state)
            price_move_5.update(state)
            calculated_price_moves.append(price_move.current_value)
            calculated_price_moves_5.append(price_move_5.current_value)
        expected_price_moves = [1000.0, -2500.0, 500.0, -2500.0, 3500.0, -1500.0, 500.0, 500.0, -2000.0]
        expected_price_moves_5 = [1000.0, -1500.0, -1000.0, -3500.0, 0.0, -2500.0, 500.0, 500.0, 1000.0]
        self.assertEqual(expected_price_moves, calculated_price_moves)
        self.assertEqual(expected_price_moves_5, calculated_price_moves_5)

    def test_price_range(self):
        price_range = PriceRange(lookback_periods=3)
        price_range.reset(state=MOCK_STATE)
        calculated_price_ranges = []
        for price in MIDPRICES[1:]:
            state = deepcopy(MOCK_STATE)
            state.price = price
            price_range.update(state)
            calculated_price_ranges.append(price_range.current_value)
        expected_price_ranges = [1000.0, 2500.0, 2500.0, 4500.0, 3500.0, 3500.0, 3500.0, 1500.0, 2000.0]
        self.assertEqual(expected_price_ranges, calculated_price_ranges)

    def test_volatility(self):
        volatility = Volatility(lookback_periods=5)
        volatility.reset(MOCK_STATE)
        calculated_volatilities = []
        for price in MIDPRICES[1:]:
            state = deepcopy(MOCK_STATE)
            state.price = price
            volatility.update(state)
            calculated_volatilities.append(volatility.current_value)
        expected_pre_min_updates = [0 for _ in range(4)]
        expected_post_min_updates = [sum(PCT_RETURNS[[i - 4, i - 3, i - 2, i - 1, i]] ** 2) / 5 for i in range(4, 9)]
        expected_volatilities = expected_pre_min_updates + expected_post_min_updates
        for index in range(len(expected_volatilities)):
            self.assertAlmostEqual(expected_volatilities[index], calculated_volatilities[index], places=2)

    def test_price(self):
        price = Price()
        price.reset(state=MOCK_STATE)
        price.update(state=MOCK_STATE)
        actual = price.current_value
        self.assertEqual(MOCK_STATE.price, actual)

    def test_inventory(self):
        inventory = Inventory()
        inventory.reset(state=MOCK_STATE)
        inventory.update(state=MOCK_STATE)
        actual = inventory.current_value
        self.assertEqual(MOCK_STATE.portfolio.inventory, actual)
