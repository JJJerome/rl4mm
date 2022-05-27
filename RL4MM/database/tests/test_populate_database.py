from pathlib import Path
from unittest import TestCase

import pandas as pd
from sqlalchemy import create_engine

from RL4MM.database.HistoricalDatabase import HistoricalDatabase
from populate_database import (
    populate_database,
    _get_book_and_message_paths,
    _get_book_and_message_columns,
    _get_timestamps,
    _get_interval_series,
    reformat_message_data,
)
from datetime import datetime


PATH_TO_TEST_DATA = "../../../test_data/"
TRADING_DATE = "2012-06-21"
TICKER = "MSFT"
START_OF_TRADING = datetime(2012, 6, 21, 9, 30)
END_OF_TRADING = datetime(2012, 6, 21, 16)
N_LEVELS = 50


class Test_populate_database(TestCase):
    def test_get_book_and_messages_paths(self):
        book_path, message_path = _get_book_and_message_paths(PATH_TO_TEST_DATA, TICKER, TRADING_DATE, N_LEVELS)
        expected_message_path = Path(PATH_TO_TEST_DATA + "MSFT_2012-06-21_34200000_37800000_message_50.csv")
        expected_book_path = Path(PATH_TO_TEST_DATA + "MSFT_2012-06-21_34200000_37800000_orderbook_50.csv")
        self.assertEqual(expected_message_path, message_path)
        self.assertEqual(expected_book_path, book_path)

    def test_get_interval_series(self):
        _, message_cols = _get_book_and_message_columns(50)
        _, message_path = _get_book_and_message_paths(PATH_TO_TEST_DATA, TICKER, TRADING_DATE, N_LEVELS)
        messages = pd.read_csv(
            message_path,
            header=None,
            names=message_cols,
            usecols=[0, 1, 2, 3, 4, 5],
        )
        messages = reformat_message_data(messages, TRADING_DATE)
        interval_series = _get_interval_series(messages)
        expected = messages.timestamp.iloc[[0, 468, 821]]
        self.assertDictEqual(dict(expected), dict(interval_series))

    def test_insert_all_books(self):
        test_engine = create_engine(f"sqlite:///:memory:")  # spin up a temporary sql db in RAM
        test_db = HistoricalDatabase(engine=test_engine)
        populate_database(
            (TICKER,),
            (TRADING_DATE,),
            database=test_db,
            path_to_lobster_data=PATH_TO_TEST_DATA,
            book_snapshot_freq=None,
            max_rows=1000,
            batch_size=1000,
        )
        messages = test_db.get_messages(
            start_date=START_OF_TRADING, end_date=END_OF_TRADING, exchange="NASDAQ", ticker=TICKER
        )
        books = test_db.get_book_snapshots(
            start_date=START_OF_TRADING, end_date=END_OF_TRADING, exchange="NASDAQ", ticker=TICKER
        )
        self.assertEqual(1000, len(messages))
        self.assertEqual(1000, len(books))

