from pathlib import Path
from typing import Optional
from unittest import TestCase
import unittest

import pandas as pd
from sqlalchemy import create_engine

import rl4mm
from rl4mm.database.HistoricalDatabase import HistoricalDatabase
from rl4mm.database.populate_database import (
    populate_database,
)
from rl4mm.database.database_population_helpers import (
    get_book_snapshots,
    get_book_and_message_columns,
    get_book_and_message_paths,
    reformat_message_data,
    get_interval_series,
)
from datetime import datetime

print(Path(rl4mm.__file__).parent)


class Test_populate_database(TestCase):
    path_to_test_data = str(Path(rl4mm.__file__).parent.parent) + "/test_data/"
    ticker = "MSFT"
    trading_date = datetime(2012, 6, 21)
    n_levels = 50
    start_of_trading = datetime(2012, 6, 21, 9, 30)
    end_of_trading = datetime(2012, 6, 21, 16)
    total_daily_messages = 1000
    td = datetime.strftime(trading_date, "%Y-%m-%d")
    book_path, message_path = get_book_and_message_paths(path_to_test_data, ticker, td, n_levels)
    book_cols, message_cols = get_book_and_message_columns(n_levels)

    def test_get_book_and_messages_paths(self):
        book_path, message_path = get_book_and_message_paths(
            self.path_to_test_data, self.ticker, self.td, self.n_levels
        )
        expected_message_path = Path(self.path_to_test_data + "MSFT_2012-06-21_34200000_37800000_message_50.csv")
        expected_book_path = Path(self.path_to_test_data + "MSFT_2012-06-21_34200000_37800000_orderbook_50.csv")
        self.assertEqual(expected_message_path, message_path)
        self.assertEqual(expected_book_path, book_path)

    def test_get_interval_series(self):
        messages, _ = self.get_all_messages_and_books()
        interval_series_seconds = get_interval_series(messages)
        interval_series_100_ms = get_interval_series(messages, "100ms")
        expected_seconds = messages.timestamp.iloc[[0, 468, 821, 999]]
        expected_100ms = messages.timestamp.iloc[[0, 1, 5, 229, 368, 386, 442, 454, 468, 625, 642, 644, 651, 665, 669]]
        self.assertDictEqual(dict(expected_seconds), dict(interval_series_seconds))
        self.assertDictEqual(dict(interval_series_100_ms.head(15)), dict(expected_100ms))

    def test_insert_all_books(self):
        test_engine = create_engine("sqlite:///:memory:")  # spin up a temporary sql db in RAM
        test_db = HistoricalDatabase(engine=test_engine)
        populate_database(
            (self.ticker,),
            (self.trading_date,),
            database=test_db,
            path_to_lobster_data=self.path_to_test_data,
            book_snapshot_freq=None,
            max_rows=1000,
            batch_size=1000,
        )
        messages = test_db.get_messages(
            start_date=self.start_of_trading, end_date=self.end_of_trading, ticker=self.ticker
        )
        books = test_db.get_book_snapshots(
            start_date=self.start_of_trading, end_date=self.end_of_trading, ticker=self.ticker
        )
        self.assertEqual(1000, len(messages))
        self.assertEqual(1000, len(books))

    def test_get_book_snapshots(self):
        messages, all_books = self.get_all_messages_and_books()
        for snapshot_freq in ["S", "100ms"]:
            books = get_book_snapshots(
                self.book_path, self.book_cols, messages, snapshot_freq, self.n_levels, self.total_daily_messages
            )
            self.assert_book_snapshots_are_correct(books, all_books, batch_size=None)

    def test_get_book_snapshots_for_smaller_batches(self):
        freq = "S"
        batch_size = 100
        all_messages, all_books = self.get_all_messages_and_books()
        books = pd.DataFrame(columns=all_books.columns)
        for messages in pd.read_csv(
            self.message_path,
            header=None,
            names=self.message_cols,
            usecols=[0, 1, 2, 3, 4, 5],
            chunksize=batch_size,
        ):
            messages = reformat_message_data(messages, self.td)
            snapshots = get_book_snapshots(
                self.book_path, self.book_cols, messages, freq, self.n_levels, self.total_daily_messages
            )
            books = books.append(snapshots, ignore_index=True)
        self.assert_book_snapshots_are_correct(books, all_books, batch_size)

    def get_all_messages_and_books(self):
        _, message_cols = get_book_and_message_columns(self.n_levels)
        _, message_path = get_book_and_message_paths(self.path_to_test_data, self.ticker, self.td, self.n_levels)
        messages = pd.read_csv(
            message_path,
            header=None,
            names=message_cols,
            usecols=[0, 1, 2, 3, 4, 5],
        )
        messages = reformat_message_data(messages, self.td)
        book_path, _ = get_book_and_message_paths(self.path_to_test_data, self.ticker, self.td, self.n_levels)
        book_cols, _ = get_book_and_message_columns(self.n_levels)
        books = get_book_snapshots(book_path, book_cols, messages, None, self.n_levels, self.total_daily_messages)
        return messages, books

    def assert_book_snapshots_are_correct(
        self, books: pd.DataFrame, all_books: pd.DataFrame, batch_size: Optional[int]
    ):
        for _, book in books.iterrows():
            expected_books = all_books.loc[all_books.timestamp == book.timestamp]
            if batch_size is not None and (batch_size - 1 in expected_books.index % batch_size):
                expected_book = expected_books[expected_books.index % batch_size == batch_size - 1].iloc[0]
            else:
                expected_book = expected_books.iloc[-1]
            expected_book = expected_book.drop("timestamp").astype("float64")
            actual_book = book.drop("timestamp").astype("float64")
            self.assertDictEqual(dict(expected_book), dict(actual_book))


if __name__ == "__main__":
    unittest.main()
