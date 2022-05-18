import argparse
import os
from functools import partial
from typing import Tuple, Optional

import glob
import logging
import pandas as pd
import shutil
import ssl

from contextlib import suppress
from datetime import datetime, timedelta
from io import BytesIO
from itertools import chain
from pathlib import Path
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from urllib.request import urlopen
from zipfile import ZipFile

from tqdm import tqdm

from RL4MM.database.HistoricalDatabase import HistoricalDatabase
from RL4MM.database.models import Book, Message
from RL4MM.database.PostgresEngine import PostgresEngine

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

EXCHANGE = "NASDAQ"
BATCH_SIZE = 10000


def populate_database(
    tickers: Tuple[str] = ("AMZN",),
    trading_dates: Tuple[str] = ("2012-06-21",),
    n_levels: int = 10,
    database: HistoricalDatabase = HistoricalDatabase(),
    path_to_lobster_data: Optional[str] = None,
    book_snapshot_freq: str = "S",
):
    create_tables()
    if path_to_lobster_data is None:
        path_to_lobster_data = _make_temporary_data_path()
    is_sample_data = path_to_lobster_data is None
    book_cols, message_cols = get_book_and_message_columns(n_levels)
    for ticker in tickers:
        for trading_date in trading_dates:
            if is_sample_data:
                download_lobster_sample_data(ticker, trading_date, n_levels, path_to_lobster_data)
            book_path, message_path = _get_book_and_message_paths(path_to_lobster_data, ticker, trading_date, n_levels)
            n_messages = count_lines(message_path)
            for messages in tqdm(
                pd.read_csv(
                    message_path, header=None, names=message_cols, usecols=[0, 1, 2, 3, 4, 5], chunksize=BATCH_SIZE
                ),
                total=n_messages // BATCH_SIZE + 1,
                desc=f"Adding data for {ticker} on {trading_date} into database!",
            ):
                messages = reformat_message_data(messages, trading_date)
                books = get_book_snapshots(book_path, book_cols, messages, book_snapshot_freq, n_levels, n_messages)
                # logging.info(f"Converting message and book data for {ticker} on {trading_date} into internal format.")
                messages, books = _convert_messages_and_books_to_internal(
                    messages, books, ticker, trading_date, n_levels
                )
                try:
                    # logging.info("Inserting books into database.")
                    database.insert_books(books)
                    # logging.info("Inserting messages into database.")
                    database.insert_messages(messages)
                except IntegrityError:
                    logging.warning(f"Data for {ticker} on {trading_date} already in database and so not re-added.")
            logging.info(f"Data for {ticker} on {trading_date} successfully added to database.")
            if is_sample_data:
                with suppress(FileNotFoundError):
                    shutil.rmtree(path_to_lobster_data)


def _make_temporary_data_path() -> str:
    with suppress(FileExistsError):
        os.mkdir("temporary_data")
    return "temporary_data"


def download_lobster_sample_data(ticker: str, trading_date: str, n_levels: int = 10, data_path: str = "temporary_data"):
    logging.info(f"Downloading book and message data for {ticker} on {trading_date} from LOBSTER.")
    zip_url = f"https://lobsterdata.com/info/sample/LOBSTER_SampleFile_{ticker}_{trading_date}_{n_levels}.zip"
    ssl._create_default_https_context = ssl._create_unverified_context  # This is a hack, and not ideal.
    with urlopen(zip_url) as zip_resp:
        with ZipFile(BytesIO(zip_resp.read())) as zip_file:
            zip_file.extractall(data_path)
    logging.info(f"Book and message data for {ticker} on {trading_date} successfully downloaded.")


def create_tables():
    engine = PostgresEngine().engine
    Session = sessionmaker(engine)
    session = Session()
    Book.metadata.create_all(engine)
    session.close()


def get_book_snapshots(
    book_path: Path,
    book_cols: list,
    messages: pd.DataFrame,
    snapshot_freq: str,
    n_levels: int,
    total_daily_messages: int,
):
    interval_series = get_interval_series(messages, snapshot_freq)
    first_index, last_index = messages.iloc[[0, -1]].index
    rows_to_skip = (
        list(range(0, first_index))
        + list(messages[~messages.index.isin(interval_series.index)].index)
        + list(range(last_index + 1, total_daily_messages))
    )
    books = pd.read_csv(book_path, nrows=len(interval_series), skiprows=rows_to_skip, header=None, names=book_cols)
    books = rescale_book_data(books, n_levels)
    return pd.concat([pd.Series(interval_series.values, name="timestamp"), books], axis=1)


def count_lines(file_name):
    for line_count, line in enumerate(open(file_name, "r")):
        pass
    return line_count


def _convert_messages_and_books_to_internal(
    messages: pd.DataFrame, books: pd.DataFrame, ticker: str, trading_date: str, n_levels: int
):
    start_index = messages.iloc[0].name
    message_convertor = partial(
        get_message_from_series,
        **{"ticker": ticker, "trading_date": trading_date, "n_levels": n_levels, "start_index": start_index},
    )
    books_to_internal = partial(
        get_book_from_series,
        **{"ticker": ticker, "trading_date": trading_date, "n_levels": n_levels, "start_index": start_index},
    )
    messages = messages.apply(message_convertor, axis=1).values
    books = books.apply(books_to_internal, axis=1).values
    return messages, books


def get_book_and_message_columns(n_levels: int = 10):
    price_cols = list(chain(*[("ask_price_{0},bid_price_{0}".format(i)).split(",") for i in range(n_levels)]))
    volume_cols = list(chain(*[("ask_volume_{0},bid_volume_{0}".format(i)).split(",") for i in range(n_levels)]))
    book_cols = list(chain(*zip(price_cols, volume_cols)))
    message_cols = ["time", "type", "external_id", "volume", "price", "direction"]
    return book_cols, message_cols


def _get_book_and_message_paths(data_path: str, ticker: str, trading_date: str, n_levels: int) -> Tuple[Path, Path]:
    try:
        book_path = glob.glob(data_path + "/" + f"{ticker}_{trading_date}_*_orderbook_{n_levels}.csv")[0]
        message_path = glob.glob(data_path + "/" + f"{ticker}_{trading_date}_*_message_{n_levels}.csv")[0]
    except IndexError:
        raise FileNotFoundError(f"Level {n_levels} data for ticker {ticker} on {trading_date} not found in {data_path}")
    return Path(book_path), Path(message_path)


def reformat_message_data(messages: pd.DataFrame, trading_date: str) -> pd.DataFrame:
    messages.time = pd.to_timedelta(messages.time, unit="s")
    messages["trading_date"] = pd.to_datetime(trading_date)
    messages["timestamp"] = messages.trading_date.add(messages.time)
    messages.drop(["trading_date", "time"], axis=1, inplace=True)
    type_dict = get_external_internal_type_dict()
    messages.type.replace(type_dict.keys(), type_dict.values(), inplace=True)
    messages.direction.replace([-1, 1], ["ask", "bid"], inplace=True)
    messages["price"] /= 10000
    messages.astype({"external_id": int})
    return messages


def get_interval_series(messages: pd.DataFrame, freq: str = "S"):
    trading_date = datetime.combine(messages.timestamp.iloc[0].date(), datetime.min.time())
    start_date = trading_date + timedelta(hours=9, minutes=30)
    end_date = trading_date + timedelta(hours=16, minutes=0)
    target_times = pd.date_range(start_date, end_date, freq=freq)
    unique_timestamps = messages.timestamp.drop_duplicates(keep="last")
    mask = pd.DatetimeIndex(unique_timestamps).get_indexer(target_times[1:], method="ffill")
    mask = unique_timestamps.iloc[mask[mask > 0]].index.unique() - messages.iloc[0].name
    return messages.timestamp.iloc[mask].drop_duplicates()


def rescale_book_data(books: pd.DataFrame, n_levels: int = 10) -> pd.DataFrame:
    # TODO: speed up by applying all at once
    for message_type in ["ask", "bid"]:
        for i in range(n_levels):
            books[message_type + "_price_" + str(i)] /= 10000
    return books


def get_external_internal_type_dict():
    return {
        1: "limit",
        2: "cancellation",
        3: "deletion",
        4: "market",
        5: "market_hidden",
        6: "cross_trade",
        7: "trading_halt",
    }


def generate_internal_index(index: int, ticker: str, trading_date: str, n_levels: int = 10):
    return f"L{str(n_levels).zfill(3)}_{EXCHANGE}_{ticker}_{trading_date}_" + str(index)


def get_message_from_series(message: pd.Series, ticker: str, trading_date: str, n_levels: int, start_index: int):
    # TODO: Can we accelerate this and the below?
    return Message(
        id=generate_internal_index(message.name + start_index, ticker, trading_date, n_levels),
        timestamp=message.timestamp,
        exchange=EXCHANGE,
        ticker=ticker,
        direction=message.direction,
        volume=message.volume,
        price=message.price,
        external_id=message.external_id,
        message_type=message.type,
    )


def get_book_from_series(book: pd.Series, ticker: str, trading_date: str, n_levels: int, start_index: int):
    return Book(
        id=generate_internal_index(book.name + start_index, ticker, trading_date, n_levels),
        timestamp=book.timestamp,
        exchange=EXCHANGE,
        ticker=ticker,
        data=book.drop("timestamp").to_json(),
    )


parser = argparse.ArgumentParser(description="Populate a postgres database with LOBSTER data")
parser.add_argument("--ticker", action="store", type=str, default="MSFT", help="the ticker to add")
parser.add_argument("--trading_date", action="store", type=str, default="2012-06-21", help="the date to add")
parser.add_argument("--n_levels", action="store", type=int, default=200, help="the number of orderbook levels")
parser.add_argument(
    "--path_to_lobster_data",
    action="store",
    type=Optional[str],
    default=None,
    help="the path to the folder containing the LOBSTER message and book data",
)
parser.add_argument(
    "--book_snapshot_freq",
    action="store",
    type=str,
    default="S",
    help="the frequency of book snapshots added to database",
)


if __name__ == "__main__":
    args = parser.parse_args()

    populate_database(
        tickers=(args.ticker,),
        trading_dates=(args.trading_date,),
        n_levels=args.n_levels,
        path_to_lobster_data=args.path_to_lobster_data,
        book_snapshot_freq=args.book_snapshot_freq,
    )
