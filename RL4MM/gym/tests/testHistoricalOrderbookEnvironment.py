from datetime import timedelta, datetime
from pathlib import Path
from unittest import TestCase

import numpy as np
from sqlalchemy import create_engine

import RL4MM
from RL4MM.database.HistoricalDatabase import HistoricalDatabase
from RL4MM.database.populate_database import populate_database
from RL4MM.gym.HistoricalOrderbookEnvironment import HistoricalOrderbookEnvironment
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
        ticker=ticker, order_generators=[generator], n_levels=200, database=test_db, save_messages_locally=False
    )
    env = HistoricalOrderbookEnvironment(
        step_size=timedelta(milliseconds=100),
        episode_length=timedelta(seconds=1),
        min_date=datetime(2012, 6, 21),
        max_date=datetime(2012, 6, 21),
        min_start_timedelta=timedelta(hours=10, seconds=1),
        max_end_timedelta=timedelta(hours=10, seconds=2),
        simulator=simulator,
        max_feature_window_size=3,
        save_messages_locally=False,
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
        expected = np.array([100, 0, 833.33, 0, 1, 308531.11])
        actual = self.env.reset()
        for i in range(len(expected)):
            self.assertAlmostEqual(actual[i], expected[i], places=2)

    def test_convert_action_to_orders(self):
        self.env.reset()
        internal_orders = self.env.convert_action_to_orders(action=ACTION_1)
        total_volume = sum(order.volume for order in internal_orders)
        self.assertEqual(200, total_volume)
        for order in internal_orders:  #
            self.assertEqual(order.volume, 10)  # BetaBinom(1,1) corresponds to Uniform
        internal_orders = self.env.convert_action_to_orders(action=ACTION_2)
        total_volume = sum(order.volume for order in internal_orders)
        expected_order_sizes = [18, 16, 15, 13, 11, 9, 7, 5, 4, 2] * 2  # Placing more orders towards the best price
        for i, order in enumerate(internal_orders):
            self.assertEqual(expected_order_sizes[i], order.volume)
