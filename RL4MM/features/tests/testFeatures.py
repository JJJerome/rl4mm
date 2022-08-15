from collections import deque
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from unittest import TestCase

import numpy as np
from sqlalchemy import create_engine

import RL4MM
from RL4MM.database.HistoricalDatabase import HistoricalDatabase
from RL4MM.database.populate_database import populate_database
from RL4MM.features.Features import (
    Spread,
    PriceMove,
    PriceRange,
    Volatility,
    Price,
    State,
    Portfolio,
    Inventory,
    TradeDirectionImbalance,
    TradeVolumeImbalance,
)
from RL4MM.orderbook.Exchange import Exchange
from RL4MM.orderbook.models import FilledOrders
from RL4MM.orderbook.tests.mock_orders import (
    get_mock_orderbook,
    submission_4,
)
from RL4MM.simulation.HistoricalOrderGenerator import HistoricalOrderGenerator
from RL4MM.simulation.OrderbookSimulator import OrderbookSimulator

MIDPRICES = np.array([30.1, 30.2, 29.95, 30.0, 29.75, 30.1, 29.95, 30.0, 30.05, 29.85]) * 10000
PCT_RETURNS = (MIDPRICES[1:] - MIDPRICES[:-1]) / MIDPRICES[:1]

MOCK_PORTFOLIO = Portfolio(inventory=100, cash=-20)
MOCK_ORDERBOOK = get_mock_orderbook()

MOCK_STATE = State(
    filled_orders=FilledOrders(internal=list(), external=list()),
    orderbook=MOCK_ORDERBOOK,
    price=MIDPRICES[0],
    portfolio=MOCK_PORTFOLIO,
    now_is=datetime(2022, 1, 1, 10),
)


class TestBookFeatures(TestCase):
    path_to_test_data = str(Path(RL4MM.__file__).parent.parent) + "/test_data/"
    ticker = "MSFT"
    trading_date = datetime(2012, 6, 21)
    n_levels = 50
    exchange_name = "NASDAQ"
    test_engine = create_engine("sqlite:///:memory:")  # spin up a temporary sql db in RAM
    test_db = HistoricalDatabase(engine=test_engine)
    generator = HistoricalOrderGenerator(ticker, test_db, save_messages_locally=False)
    simulator = OrderbookSimulator(ticker, Exchange(ticker), [generator], 50, test_db, preload_messages=False)

    @classmethod
    def setUpClass(cls) -> None:
        populate_database(
            (cls.ticker,),
            (cls.trading_date,),
            database=cls.test_db,
            path_to_lobster_data=cls.path_to_test_data,
            book_snapshot_freq=None,
            max_rows=1000,
            batch_size=1000,
        )

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

    def test_order_direction_and_volume_imbalance(self):
        # To test order flow, we use actual historical data - testOrderbookSimulator needs to pass for this to work.
        start_of_trading_day = datetime(2012, 6, 21, 9, 30)
        start_of_episode = self.test_db.get_next_snapshot(start_of_trading_day, self.ticker).name
        start_of_episode += timedelta(microseconds=10**6 - start_of_episode.microsecond)
        end_of_episode = start_of_episode + timedelta(seconds=2)
        self.simulator.reset_episode(start_of_episode)
        trade_dir_imbalance = TradeDirectionImbalance(lookback_periods=10, update_frequency=timedelta(seconds=0.1))
        trade_vol_imbalance = TradeVolumeImbalance(
            lookback_periods=10, update_frequency=timedelta(seconds=0.1), track_internal=True
        )
        trade_dir_imbalance.reset(MOCK_STATE)
        trade_vol_imbalance.reset(MOCK_STATE)
        now_is = start_of_episode

        def trade_imbalance_update_step(now_is: datetime):
            filled = self.simulator.forward_step(now_is)
            state = deepcopy(MOCK_STATE)
            state.filled_orders = filled
            trade_dir_imbalance.update(state)
            trade_vol_imbalance.update(state)

        for _ in range(10 - 1):
            now_is += timedelta(seconds=0.1)
            trade_imbalance_update_step(now_is)
            self.assertEqual(trade_dir_imbalance.current_value, 0)
            self.assertEqual(trade_vol_imbalance.current_value, 0)
        expected_dir_imbalances = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 6 / 36, 13 / 43, 9 / 39, 6 / 36]
        expected_volume_imbalances = [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            -988 / 14642,
            -188 / 15442,
            -638 / 14992,
            -778 / 14852,
        ]
        for i in range(0, 9):
            now_is += timedelta(seconds=0.1)
            trade_imbalance_update_step(now_is)
            self.assertAlmostEqual(trade_dir_imbalance.current_value, expected_dir_imbalances[i])
            self.assertAlmostEqual(trade_vol_imbalance.current_value, expected_volume_imbalances[i])
