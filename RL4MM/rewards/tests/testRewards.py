from datetime import datetime
from unittest import TestCase

from RL4MM.features.Features import (
    State,
    Portfolio,
)
from RL4MM.orderbook.tests.mock_orders import get_mock_orderbook
from RL4MM.rewards.RewardFunctions import RollingSharpe

MOCK_PORTFOLIO = Portfolio(inventory=1, cash=0)
MOCK_ORDERBOOK = get_mock_orderbook()


MOCK_INTERNAL_STATE_1 = State(
    filled_orders=([], []),
    orderbook=MOCK_ORDERBOOK,
    price=100,
    portfolio=MOCK_PORTFOLIO,
    now_is=datetime(2022, 1, 1, 10, 1),
)
MOCK_INTERNAL_STATE_2 = State(
    filled_orders=([], []),
    orderbook=MOCK_ORDERBOOK,
    price=110,
    portfolio=MOCK_PORTFOLIO,
    now_is=datetime(2022, 1, 1, 10, 2),
)
MOCK_INTERNAL_STATE_3 = State(
    filled_orders=([], []),
    orderbook=MOCK_ORDERBOOK,
    price=100,
    portfolio=MOCK_PORTFOLIO,
    now_is=datetime(2022, 1, 1, 10, 3),
)
MOCK_INTERNAL_STATE_4 = State(
    filled_orders=([], []),
    orderbook=MOCK_ORDERBOOK,
    price=0,
    portfolio=MOCK_PORTFOLIO,
    now_is=datetime(2022, 1, 1, 10, 4),
)


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
