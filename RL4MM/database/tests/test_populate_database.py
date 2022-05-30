from pathlib import Path
from unittest import TestCase

import pandas as pd
from sqlalchemy import create_engine

from RL4MM.database.HistoricalDatabase import HistoricalDatabase
from populate_database import (
    populate_database,
    _get_book_and_message_paths,
    _get_book_and_message_columns,
    _get_interval_series,
    reformat_message_data,
    get_book_snapshots,
)
from datetime import datetime


PATH_TO_TEST_DATA = "../../../test_data/"
TRADING_DATE = "2012-06-21"
TICKER = "MSFT"
START_OF_TRADING = datetime(2012, 6, 21, 9, 30)
END_OF_TRADING = datetime(2012, 6, 21, 16)
N_LEVELS = 50
TOTAL_DAILY_MESSAGES = 1000
BOOK_PATH, MESSAGE_PATH = _get_book_and_message_paths(PATH_TO_TEST_DATA, TICKER, TRADING_DATE, N_LEVELS)
BOOK_COLS, MESSAGE_COLS = _get_book_and_message_columns(N_LEVELS)


class Test_populate_database(TestCase):
    def test_get_book_and_messages_paths(self):
        book_path, message_path = _get_book_and_message_paths(PATH_TO_TEST_DATA, TICKER, TRADING_DATE, N_LEVELS)
        expected_message_path = Path(PATH_TO_TEST_DATA + "MSFT_2012-06-21_34200000_37800000_message_50.csv")
        expected_book_path = Path(PATH_TO_TEST_DATA + "MSFT_2012-06-21_34200000_37800000_orderbook_50.csv")
        self.assertEqual(expected_message_path, message_path)
        self.assertEqual(expected_book_path, book_path)

    def test_get_interval_series(self):
        messages, _ = self.get_all_messages_and_books()
        interval_series_seconds = _get_interval_series(messages)
        interval_series_100_ms = _get_interval_series(messages, "100ms")
        expected_seconds = messages.timestamp.iloc[[0, 468, 821, 999]]
        expected_100ms = messages.timestamp.iloc[[0, 1, 5, 229, 368, 386, 442, 454, 468, 625, 642, 644, 651, 665, 669]]
        self.assertDictEqual(dict(expected_seconds), dict(interval_series_seconds))
        self.assertDictEqual(dict(interval_series_100_ms.head(15)), dict(expected_100ms))

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

    def test_get_book_snapshots(self):
        messages, all_books = self.get_all_messages_and_books()
        for snapshot_freq in ["S", "100ms"]:
            books = get_book_snapshots(BOOK_PATH, BOOK_COLS, messages, snapshot_freq, N_LEVELS, TOTAL_DAILY_MESSAGES)
            self.assert_book_snapshots_are_correct(books, all_books)


    def test_get_book_snapshots_for_smaller_batches(self):
        freq = "S"
        all_messages, all_books = self.get_all_messages_and_books()
        books = pd.DataFrame(columns=all_books.columns)
        for messages in pd.read_csv(
            MESSAGE_PATH,
            header=None,
            names=MESSAGE_COLS,
            usecols=[0, 1, 2, 3, 4, 5],
            chunksize=100,
        ):
            messages = reformat_message_data(messages, TRADING_DATE)
            snapshots = get_book_snapshots(BOOK_PATH, BOOK_COLS, messages, freq, N_LEVELS, TOTAL_DAILY_MESSAGES)
            books = books.append(snapshots, ignore_index=True)
        # self.assert_book_snapshots_are_correct(books,all_books)


    @staticmethod
    def get_all_messages_and_books():
        _, message_cols = _get_book_and_message_columns(N_LEVELS)
        _, message_path = _get_book_and_message_paths(PATH_TO_TEST_DATA, TICKER, TRADING_DATE, N_LEVELS)
        messages = pd.read_csv(
            message_path,
            header=None,
            names=message_cols,
            usecols=[0, 1, 2, 3, 4, 5],
        )
        messages = reformat_message_data(messages, TRADING_DATE)
        book_path, _ = _get_book_and_message_paths(PATH_TO_TEST_DATA, TICKER, TRADING_DATE, N_LEVELS)
        book_cols, _ = _get_book_and_message_columns(N_LEVELS)
        books = get_book_snapshots(book_path, book_cols, messages, None, N_LEVELS, TOTAL_DAILY_MESSAGES)
        return messages, books

    def assert_book_snapshots_are_correct(self, books:pd.DataFrame, all_books:pd.DataFrame):
        for _, book in books.iterrows():
            expected_book = all_books.loc[all_books.timestamp == book.timestamp]
            if len(expected_book) > 1:
                expected_book = expected_book.iloc[-1]
            else:
                expected_book = expected_book.iloc[0]
            expected_book = expected_book.drop("timestamp").astype("float64")
            actual_book = book.drop("timestamp").astype("float64")
            self.assertDictEqual(dict(expected_book), dict(actual_book))

