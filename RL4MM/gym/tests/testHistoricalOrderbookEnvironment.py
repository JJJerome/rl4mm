from datetime import timedelta, datetime
from pathlib import Path
from typing import List, Union
from unittest import TestCase

import numpy as np
from sqlalchemy import create_engine

import RL4MM
from RL4MM.database.HistoricalDatabase import HistoricalDatabase
from RL4MM.database.populate_database import populate_database
from RL4MM.features.Features import Inventory, PriceMove, Spread, PriceRange
from RL4MM.gym.HistoricalOrderbookEnvironment import HistoricalOrderbookEnvironment
from RL4MM.orderbook.models import Cancellation, LimitOrder
from RL4MM.simulation.HistoricalOrderGenerator import HistoricalOrderGenerator
from RL4MM.simulation.OrderbookSimulator import OrderbookSimulator

ACTION_1 = np.array([1, 1, 1, 1])
ACTION_2 = np.array([1, 2, 1, 2])


class testHistoricalOrderbookEnvironment(TestCase):
    path_to_test_data = str(Path(RL4MM.__file__).parent.parent) + "/test_data/"
    ticker = "MSFT"
    trading_date = datetime(2012, 6, 21)
    n_levels = 50
    test_engine = create_engine("sqlite:///:memory:")  # spin up a temporary sql db in RAM
    test_db = HistoricalDatabase(engine=test_engine)
    generator = HistoricalOrderGenerator(ticker, test_db, save_messages_locally=False)
    simulator = OrderbookSimulator(
        ticker=ticker, order_generators=[generator], n_levels=200, database=test_db, preload_messages=False
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
        preload_messages=False,
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
        expected_order_sizes = [19, 17, 15, 13, 11, 9, 7, 5, 3, 1] * 2  # Placing more orders towards the best price
        for i, order in enumerate(internal_orders_2):
            self.assertEqual(expected_order_sizes[i], order.volume)
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
