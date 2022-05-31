from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from unittest import TestCase

from sqlalchemy import create_engine

import RL4MM
from RL4MM.database.HistoricalDatabase import HistoricalDatabase
from RL4MM.database.populate_database import populate_database
from RL4MM.extras.orderbook_comparison import convert_to_lobster_format
from RL4MM.orderbook.Exchange import Exchange
from RL4MM.orderbook.tests.mock_orders import CANCELLATION_1, CANCELLATION_2, LIMIT_1, LIMIT_2
from RL4MM.simulation.HistoricalOrderGenerator import HistoricalOrderGenerator
from RL4MM.simulation.OrderbookSimulator import OrderbookSimulator


class TestHistoricalOrderGenerator(TestCase):
    path_to_test_data = str(Path(RL4MM.__file__).parent.parent) + "/test_data/"
    ticker = "MSFT"
    trading_date = "2012-06-21"
    n_levels = 50
    exchange_name = "NASDAQ"
    test_engine = create_engine("sqlite:///:memory:")  # spin up a temporary sql db in RAM
    test_db = HistoricalDatabase(engine=test_engine)
    generator = HistoricalOrderGenerator(ticker, test_db)
    simulator = OrderbookSimulator(ticker, Exchange(ticker), [generator], 50, test_db)

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

    def test_agreement_with_lobster(self):
        start_of_trading_day = datetime(2012, 6, 21, 9, 30)
        end_of_trading_day = datetime(2012, 6, 21, 16)
        start_of_episode = self.test_db.get_next_snapshot(start_of_trading_day, self.ticker).name
        end_of_episode = self.test_db.get_last_snapshot(end_of_trading_day, self.ticker).name
        # Test agreement at the start of the episode
        self.simulator.reset_episode(start_of_episode)
        expected_book = self.test_db.get_last_snapshot(start_of_episode, self.ticker).to_dict()
        actual_book = convert_to_lobster_format(self.simulator.exchange.orderbook, self.n_levels)
        self.assertDictEqual(expected_book, actual_book)
        # Test agreement after 1 second
        time_1 = start_of_episode + timedelta(seconds=1)
        self.simulator.forward_step(time_1)
        expected_book = self.test_db.get_last_snapshot(time_1, self.ticker).to_dict()
        actual_book = convert_to_lobster_format(self.simulator.exchange.orderbook, self.n_levels)
        keys_to_drop = expected_book.keys() - actual_book.keys()
        for key in keys_to_drop:
            expected_book.pop(key)
        self.assertDictEqual(expected_book, actual_book)
        # Test agreement at end of episode
        self.simulator.forward_step(end_of_episode)
        expected_book = self.test_db.get_last_snapshot(end_of_episode, self.ticker).to_dict()
        actual_book = convert_to_lobster_format(self.simulator.exchange.orderbook, self.n_levels)
        keys_to_drop = expected_book.keys() - actual_book.keys()
        keys_to_drop = keys_to_drop.union({"buy_price_45", "buy_volume_45"})  # Here, orders come in at price levels not
        for key in keys_to_drop:  # in the book at start_of_episode, due to
            expected_book.pop(key)  # only having 50 price levels
            if key in actual_book.keys():
                actual_book.pop(key)
        self.assertDictEqual(expected_book, actual_book)

    def test_compare_order_dict(self):
        order_dict = {"gen_1": deque([LIMIT_1, CANCELLATION_1]), "gen_2": deque([LIMIT_2, CANCELLATION_2])}
        orders = self.simulator._compress_order_dict(order_dict)
        expected = [LIMIT_1, CANCELLATION_1, LIMIT_2, CANCELLATION_2]
        self.assertEqual(expected, orders)
