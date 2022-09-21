from datetime import timedelta, datetime
from pathlib import Path
from typing import List, Union
from unittest import TestCase

import numpy as np
from sqlalchemy import create_engine

import rl4mm
from rl4mm.database.HistoricalDatabase import HistoricalDatabase
from rl4mm.database.populate_database import populate_database
from rl4mm.features.Features import Inventory, PriceMove, Spread, PriceRange
from rl4mm.gym.HistoricalOrderbookEnvironment import HistoricalOrderbookEnvironment
from rl4mm.orderbook.models import Cancellation, LimitOrder
from rl4mm.simulation.HistoricalOrderGenerator import HistoricalOrderGenerator
from rl4mm.simulation.OrderbookSimulator import OrderbookSimulator

ACTION_1 = np.array([1, 1, 1, 1])
ACTION_2 = np.array([1, 2, 1, 2])


class testHistoricalOrderbookEnvironment(TestCase):
    path_to_test_data = str(Path(rl4mm.__file__).parent.parent) + "/test_data/"
    ticker = "MSFT"
    trading_date = datetime(2012, 6, 21)
    n_levels = 50
    test_engine = create_engine("sqlite:///:memory:")  # spin up a temporary sql db in RAM
    test_db = HistoricalDatabase(engine=test_engine)
    generator = HistoricalOrderGenerator(ticker, test_db, preload_orders=False)
    simulator = OrderbookSimulator(
        ticker=ticker, order_generators=[generator], n_levels=50, database=test_db, preload_orders=False
    )
    env = HistoricalOrderbookEnvironment(
        step_size=timedelta(milliseconds=100),
        episode_length=timedelta(seconds=1),
        min_date=datetime(2012, 6, 21),
        max_date=datetime(2012, 6, 21),
        min_start_timedelta=timedelta(hours=10, seconds=1),
        max_end_timedelta=timedelta(hours=10, seconds=2),
        max_quote_level=10,
        simulator=simulator,
        preload_orders=False,
        features=[Inventory(), Spread(), PriceMove(lookback_periods=1), PriceRange(lookback_periods=1)],
    )

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
        cls.env.reset()

    def test_reset(self):
        self.env.reset()
        expected = np.array([0, 100, -343.1, 343.1])
        actual = self.env.reset()
        for i in range(len(expected)):
            self.assertAlmostEqual(actual[i], expected[i], places=1)

    def test_get_prices(self):
        self.env.reset()
        best_prices = self.env._get_best_prices()
        # Check that arrays are ordered (ascending for sell and descending for buy)
        self.assertTrue(np.array_equal(best_prices["buy"], np.flip(np.sort(best_prices["buy"]))))
        self.assertTrue(np.array_equal(best_prices["sell"], np.sort(best_prices["sell"])))

    def test_convert_action_to_orders(self):
        self.env.reset()
        # to avoid typing errors
        internal_orders_1: List[Union[Cancellation, LimitOrder]]
        internal_orders_2: List[Union[Cancellation, LimitOrder]]
        internal_orders_3: List[Union[Cancellation, LimitOrder]]
        # Testing order placement in empty book
        internal_orders_1 = self.env.convert_action_to_orders(action=ACTION_1)  # type: ignore
        total_volume = sum(order.volume for order in internal_orders_1)
        self.assertEqual(200, total_volume)
        for order in internal_orders_1:
            self.assertEqual(order.volume, 10)  # BetaBinom(1,1) corresponds to Uniform
        internal_orders_2 = self.env.convert_action_to_orders(action=ACTION_2)  # type: ignore
        best_prices = self.env._get_best_prices()
        order_prices = np.concatenate((best_prices["buy"],best_prices["sell"]))
        expected_order_sizes = [19, 17, 15, 13, 11, 9, 7, 5, 3, 1] * 2  # Placing more orders towards the best price
        for i, order in enumerate(internal_orders_2):
            self.assertEqual(expected_order_sizes[i], order.volume)
            self.assertEqual(order_prices[i], order.price)
        # Testing update
        for order in internal_orders_1:
            self.env.simulator.exchange.process_order(order)  # Add orders to the orderbooks
        internal_orders_3 = self.env.convert_action_to_orders(action=ACTION_2)  # type: ignore
        expected_order_sizes = [19 - 10, 17 - 10, 15 - 10, 13 - 10, 11 - 10, 9 - 10, 7 - 10, 5 - 10, 3 - 10, 1 - 10] * 2
        for i, order in enumerate(internal_orders_3):
            if expected_order_sizes[i] > 0:
                self.assertIsInstance(order, LimitOrder)
                self.assertEqual(order.volume, expected_order_sizes[i])
            elif expected_order_sizes[i] < 0:
                self.assertIsInstance(order, Cancellation)
                self.assertEqual(order.volume, abs(expected_order_sizes[i]))

    def test_volume_diff_to_orders(self):
        self.env.reset()
        # to avoid typing errors
        internal_orders_1: List[Union[Cancellation, LimitOrder]]
        internal_orders_1 = self.env.convert_action_to_orders(action=ACTION_1)  # type: ignore
        for order in internal_orders_1:
            self.env.simulator.exchange.process_order(order)  # Add orders to the orderbooks
        for order in internal_orders_1:
            self.assertEqual(order.volume, 10)  # BetaBinom(1,1) corresponds to Uniform
        internal_orders_2 = self.env.convert_action_to_orders(action=ACTION_2)  # type: ignore
        expected_order_sizes = [19, 17, 15, 13, 11, 9, 7, 5, 3, 1] * 2  # Placing more orders towards the best price
        for i, order in enumerate(internal_orders_2):
            self.assertEqual(expected_order_sizes[i], order.volume)
        # Testing update


