from datetime import datetime
from typing import Tuple, Optional

import argparse
import logging
import numpy as np
import pandas as pd
import shutil

from contextlib import suppress

from pandas import DatetimeIndex
from sqlalchemy.exc import IntegrityError

from tqdm import tqdm

from rl4mm.database.HistoricalDatabase import HistoricalDatabase
from rl4mm.database.database_population_helpers import (
    make_temporary_data_path,
    download_lobster_sample_data,
    create_tables,
    get_book_snapshots,
    get_file_len,
    convert_messages_and_books_to_dicts,
    get_book_and_message_columns,
    get_book_and_message_paths,
    reformat_message_data,
)
from rl4mm.utils.utils import get_next_trading_dt, get_last_trading_dt, daterange_in_db, get_date_time

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def populate_database(
    tickers: Tuple[str] = ("MSFT",),
    trading_datetimes: DatetimeIndex = DatetimeIndex(
        [
            "2012-06-21",
        ]
    ),
    n_levels: int = 50,
    database: HistoricalDatabase = HistoricalDatabase(),
    path_to_lobster_data: str = "",
    book_snapshot_freq: Optional[str] = None,
    max_rows: int = 99999999999,
    batch_size: int = 1000000,
):
    create_tables()
    is_sample_data = False
    if path_to_lobster_data == "":
        path_to_lobster_data = make_temporary_data_path()
        is_sample_data = True
    book_cols, message_cols = get_book_and_message_columns(n_levels)
    for ticker in tickers:
        for trading_datetime in trading_datetimes:
            trading_date = datetime.strftime(trading_datetime, "%Y-%m-%d")
            next_trading_dt = get_next_trading_dt(trading_datetime)
            last_trading_dt = get_last_trading_dt(trading_datetime)
            if daterange_in_db(next_trading_dt, last_trading_dt, ticker):
                logging.info(f"Data for {ticker} on {trading_date} already in database and so not re-added.")
                continue
            if is_sample_data:
                download_lobster_sample_data(ticker, trading_date, n_levels, path_to_lobster_data)
            book_path, message_path = get_book_and_message_paths(path_to_lobster_data, ticker, trading_date, n_levels)
            n_messages = get_file_len(message_path)
            for messages in tqdm(
                pd.read_csv(
                    message_path,
                    header=None,
                    names=message_cols,
                    usecols=[0, 1, 2, 3, 4, 5],
                    chunksize=batch_size,
                    nrows=min(n_messages, max_rows),
                ),
                total=np.ceil(min(n_messages, max_rows) / batch_size),
                desc=f"Adding data for {ticker} on {trading_date} into database",
            ):
                messages = reformat_message_data(messages, trading_date)
                books = get_book_snapshots(book_path, book_cols, messages, book_snapshot_freq, n_levels, n_messages)
                message_dict, book_dict = convert_messages_and_books_to_dicts(
                    messages, books, ticker, trading_date, n_levels, book_snapshot_freq
                )
                try:
                    database.insert_books_from_dicts(book_dict)
                    database.insert_messages_from_dicts(message_dict)
                except IntegrityError:
                    logging.warning(f"Data for {ticker} on {trading_date} already in database and so not re-added.")
            logging.info(f"Data for {ticker} on {trading_date} successfully added to database.")
            if is_sample_data:
                with suppress(FileNotFoundError):
                    shutil.rmtree(path_to_lobster_data)


parser = argparse.ArgumentParser(description="Populate a postgres database with LOBSTER data")
parser.add_argument("--ticker", action="store", type=str, default="MSFT", help="the ticker to add")
parser.add_argument(
    "-mintd",
    "--min_trading_date",
    action="store",
    type=str,
    default="2012-06-21",
    help="The mininum trading date to add.",
)
parser.add_argument(
    "-maxtd",
    "--max_trading_date",
    action="store",
    type=str,
    default="2012-06-21",
    help="The maxinum trading date to add.",
)
parser.add_argument("-nl", "--n_levels", action="store", type=int, default=50, help="the number of orderbook levels")
parser.add_argument(
    "--path_to_lobster_data",
    action="store",
    type=str,
    default="",
    help="the path to the folder containing the LOBSTER message and book data",
)
parser.add_argument(
    "-bsf",
    "--book_snapshot_freq",
    action="store",
    type=str,
    default="S",
    help="the frequency of book snapshots added to database",
)
parser.add_argument(
    "-mr",
    "--max_rows",
    action="store",
    type=int,
    default=99999999999,
    help="the max number of rows to add for a given date",
)
parser.add_argument(
    "-bs",
    "--batch_size",
    action="store",
    type=int,
    default=1000000,
    help="the batch size used to populate the db (for reducing memory requirements)",
)

if __name__ == "__main__":
    args = parser.parse_args()
    populate_database(
        tickers=(args.ticker,),
        trading_datetimes=pd.bdate_range(get_date_time(args.min_trading_date), get_date_time(args.max_trading_date)),
        n_levels=args.n_levels,
        path_to_lobster_data=args.path_to_lobster_data,
        book_snapshot_freq=args.book_snapshot_freq,
        max_rows=args.max_rows,
        batch_size=args.batch_size,
    )
