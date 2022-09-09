from collections import deque
from typing import cast
from datetime import datetime, timedelta
from pathlib import Path
from unittest import TestCase

from sqlalchemy import create_engine

import rl4mm
from rl4mm.database.HistoricalDatabase import HistoricalDatabase
from rl4mm.database.populate_database import populate_database
from rl4mm.extras.orderbook_comparison import convert_to_lobster_format
from rl4mm.orderbook.Exchange import Exchange
from rl4mm.orderbook.create_order import create_order
from rl4mm.orderbook.models import OrderDict
from rl4mm.orderbook.tests.mock_orders import CANCELLATION_1, CANCELLATION_2, LIMIT_1, LIMIT_2
from rl4mm.simulation.HistoricalOrderGenerator import HistoricalOrderGenerator
from rl4mm.simulation.OrderbookSimulator import OrderbookSimulator


class TestOrderbookSimulator(TestCase):
    path_to_test_data = str(Path(rl4mm.__file__).parent.parent) + "/test_data/"
    ticker = "MSFT"
    trading_date = datetime(2012, 6, 21)
    n_levels = 50
    exchange_name = "NASDAQ"
    test_engine = create_engine("sqlite:///:memory:")  # spin up a temporary sql db in RAM
    test_db = HistoricalDatabase(engine=test_engine)
    generator = HistoricalOrderGenerator(ticker, test_db, save_messages_locally=False)
    simulator = OrderbookSimulator(
        ticker, Exchange(ticker), [generator], 50, test_db, preload_messages=False, outer_levels=48
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

    def test_agreement_with_lobster(self):
        start_of_trading_day = datetime(2012, 6, 21, 9, 30)
        start_of_episode = self.test_db.get_next_snapshot(start_of_trading_day, self.ticker).name
        start_of_episode += timedelta(microseconds=10**6 - start_of_episode.microsecond)
        end_of_episode = start_of_episode + timedelta(seconds=2)
        # Test agreement at the start of the episode
        self.simulator.reset_episode(start_of_episode)
        expected_book = self.test_db.get_last_snapshot(start_of_episode, self.ticker).to_dict()
        actual_book = convert_to_lobster_format(self.simulator.exchange.central_orderbook, self.n_levels)
        self.assertDictEqual(expected_book, actual_book)
        # Test agreement after 1 second
        time_1 = start_of_episode + timedelta(seconds=1)
        self.simulator.forward_step(time_1)
        expected_book = self.test_db.get_last_snapshot(time_1, self.ticker).to_dict()
        actual_book = convert_to_lobster_format(self.simulator.exchange.central_orderbook, self.n_levels)
        keys_to_drop = expected_book.keys() - actual_book.keys()
        for key in keys_to_drop:
            expected_book.pop(key)
        self.assertDictEqual(expected_book, actual_book)
        # Test agreement at end of episode
        self.simulator.forward_step(end_of_episode)
        expected_book = self.test_db.get_last_snapshot(end_of_episode, self.ticker).to_dict()
        actual_book = convert_to_lobster_format(self.simulator.exchange.central_orderbook, self.n_levels)
        keys_to_drop = expected_book.keys() - actual_book.keys()
        keys_to_drop = keys_to_drop.union({"buy_price_45", "buy_volume_45"})  # Here, orders come in at price levels not
        for key in keys_to_drop:  # in the book at start_of_episode, due to
            expected_book.pop(key)  # only having 50 price levels
            if key in actual_book.keys():
                actual_book.pop(key)
        self.assertDictEqual(expected_book, actual_book)

    def test_update_outer_levels_with_internal_orders(self):
        start_of_trading_day = datetime(2012, 6, 21, 9, 30)
        start_of_episode = self.test_db.get_next_snapshot(start_of_trading_day, self.ticker).name
        start_of_episode += timedelta(microseconds=10**6 - start_of_episode.microsecond)
        end_of_episode = start_of_episode + timedelta(seconds=1)
        self.simulator.reset_episode(start_of_episode)
        min_buy_price_0 = self.simulator.min_buy_price
        orderbook = self.simulator.exchange.central_orderbook
        worst_buy_price = min(orderbook.buy.keys())
        worst_sell_price = max(orderbook.sell.keys())
        worst_buy_order_dict = cast(OrderDict, orderbook.buy[worst_buy_price][0].__dict__)
        worst_sell_order_dict = cast(OrderDict, orderbook.sell[worst_sell_price][0].__dict__)
        worst_buy_order_dict["volume"] = 100
        worst_sell_order_dict["volume"] = 200
        worst_buy_order_dict["is_external"] = False
        worst_sell_order_dict["is_external"] = False
        worst_buy_order_dict["price"] -= 100
        internal_buy = create_order("limit", worst_buy_order_dict)
        internal_sell = create_order("limit", worst_sell_order_dict)
        # Test agreement after 1 second
        self.simulator.forward_step(end_of_episode, internal_orders=[internal_buy, internal_sell])
        # Check that min_buy_price_is updated
        self.assertLess(self.simulator.min_buy_price, min_buy_price_0)
        internal_level = self.simulator.exchange.internal_orderbook.buy[worst_buy_order_dict["price"]]
        external_level = self.simulator.exchange.central_orderbook.buy[worst_buy_order_dict["price"]]
        # Check that internal order is replaced in internal book
        self.assertGreater(internal_level[0].internal_id, 10)
        # Assert that it joins the back of the queue in the central book
        self.assertTrue(external_level[0].is_external)
        self.assertFalse(external_level[1].is_external)
        self.assertEqual(external_level[0].volume, 200)
        self.assertEqual(external_level[1].volume, 100)

    def test_compare_order_dict(self):
        order_dict = {"gen_1": deque([LIMIT_1, CANCELLATION_1]), "gen_2": deque([LIMIT_2, CANCELLATION_2])}
        orders = self.simulator._compress_order_dict(order_dict)
        expected = [LIMIT_1, CANCELLATION_1, LIMIT_2, CANCELLATION_2]
        self.assertEqual(expected, orders)
