import os
from functools import partial
from typing import List, Tuple, Optional

import glob
import logging
import shutil
import ssl

from contextlib import suppress
from io import BytesIO
from itertools import chain
from contextlib import suppress
from pandas import concat, DataFrame, read_csv, Series, to_datetime, to_timedelta
from pathlib import Path
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from urllib.request import urlopen
from zipfile import ZipFile

from RL4MM.database.HistoricalDatabase import HistoricalDatabase
from RL4MM.database.models import Book, Message
from RL4MM.database.PostgresEngine import PostgresEngine

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def populate_database(
    tickers: List[str] = None,
    trading_dates: List[str] = None,
    n_levels: int = 10,
    database: HistoricalDatabase = None,
    exchange: str = "NASDAQ",
    path_to_lobster_data: Optional[str] = None,
):
    if tickers is None:
        tickers = ["AMZN"]
    if trading_dates is None:
        trading_dates = ["2012-06-21"]
    if not database:
        database = HistoricalDatabase()
    create_tables()
    if exchange != "NASDAQ":
        raise NotImplementedError
    is_sample_data = path_to_lobster_data is None
    if is_sample_data:
        path_to_lobster_data = "temporary_data"
        with suppress(FileExistsError):
            os.mkdir(path_to_lobster_data)
    for ticker in tickers:
        for trading_date in trading_dates:
            if is_sample_data:
                download_lobster_sample_data(ticker, trading_date, n_levels, path_to_lobster_data)
            book_path, message_path = get_book_and_message_paths(path_to_lobster_data, ticker, trading_date, n_levels)
            book_cols, message_cols = get_book_and_message_columns(n_levels)
            books = read_csv(book_path, header=None, names=book_cols)
            messages = read_csv(message_path, header=None, names=message_cols, usecols=[0, 1, 2, 3, 4, 5])
            assert len(books) == len(messages), "Length of the order book csv and message csv differ"
            messages = reformat_message_data(messages, trading_date)
            books = rescale_book_data(books, n_levels)
            books = concat([books, messages.timestamp], axis=1)
            messages_to_internal = partial(
                get_message_from_series,
                **{"ticker": ticker, "trading_date": trading_date, "exchange": exchange, "n_levels": n_levels},
            )
            books_to_internal = partial(
                get_book_from_series,
                **{"ticker": ticker, "trading_date": trading_date, "exchange": exchange, "n_levels": n_levels},
            )
            logging.info("Formatting message data into internal format.")
            messages_to_insert = messages.apply(messages_to_internal, axis=1).values
            logging.info("Formatting book data into internal format.")
            books_to_insert = books.apply(books_to_internal, axis=1).values
            logging.info(f"Inserting book and message data for {ticker} on {trading_date} into database.")
            try:
                database.insert_books(messages_to_insert)
                database.insert_messages(books_to_insert)
                logging.info(f"Data for {ticker} on {trading_date} successfully added to database.")
            except IntegrityError:
                logging.warning(f"Data for {ticker} on {trading_date} already exists in database and so was not added.")
            if is_sample_data:
                with suppress(FileNotFoundError):
                    shutil.rmtree(path_to_lobster_data)


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


def get_book_and_message_columns(n_levels: int = 10):
    price_cols = list(chain(*[("ask_price_{0},bid_price_{0}".format(i)).split(",") for i in range(n_levels)]))
    volume_cols = list(chain(*[("ask_volume_{0},bid_volume_{0}".format(i)).split(",") for i in range(n_levels)]))
    book_cols = list(chain(*zip(price_cols, volume_cols)))
    message_cols = ["time", "type", "external_id", "volume", "price", "direction"]
    return book_cols, message_cols


def get_book_and_message_paths(data_path: str, ticker: str, trading_date: str, n_levels: int) -> Tuple[Path, Path]:
    try:
        book_path = glob.glob(data_path + "/" + f"{ticker}_{trading_date}_*_orderbook_{n_levels}.csv")[0]
        message_path = glob.glob(data_path + "/" + f"{ticker}_{trading_date}_*_message_{n_levels}.csv")[0]
    except IndexError:
        raise FileNotFoundError(f"Level {n_levels} data for ticker {ticker} on {trading_date} not found in {data_path}")
    return book_path, message_path


def reformat_message_data(messages: DataFrame, trading_date: str) -> DataFrame:
    messages.time = to_timedelta(messages.time, unit="s")
    messages["trading_date"] = to_datetime(trading_date)
    messages["timestamp"] = messages.trading_date.add(messages.time)
    messages.drop(["trading_date", "time"], axis=1, inplace=True)
    type_dict = get_external_internal_type_dict()
    messages.type.replace(type_dict.keys(), type_dict.values(), inplace=True)
    messages.direction.replace([-1, 1], ["ask", "bid"], inplace=True)
    messages["price"] /= 10000
    messages.astype({"external_id": int})
    return messages


def rescale_book_data(books: DataFrame, n_levels: int = 10) -> DataFrame:
    for message_type in ["ask", "bid"]:
        for i in range(n_levels):
            books[message_type + "_price_" + str(i)] /= 10000
    return books


def get_external_internal_type_dict():
    return {
        1: "submission",
        2: "cancellation",
        3: "deletion",
        4: "execution_visible",
        5: "execution_hidden",
        6: "cross_trade",
        7: "trading_halt",
    }


def generate_internal_index(index: int, ticker: str, trading_date: str, exchange: str, n_levels: int = 10):
    return f"L{str(n_levels).zfill(3)}_{exchange}_{ticker}_{trading_date}_" + str(index)


def get_message_from_series(message: Series, ticker: str, trading_date: str, exchange: str, n_levels: int):
    return Message(
        id=generate_internal_index(message.name, ticker, trading_date, exchange, n_levels),
        timestamp=message.timestamp,
        exchange=exchange,
        ticker=ticker,
        direction=message.direction,
        volume=message.volume,
        price=message.price,
        external_id=message.external_id,
        message_type=message.type,
    )


def get_book_from_series(book: Series, ticker: str, trading_date: str, exchange: str, n_levels: int):
    return Book(
        id=generate_internal_index(book.name, ticker, trading_date, exchange, n_levels),
        timestamp=book.timestamp,
        exchange=exchange,
        ticker=ticker,
        data=book.drop("timestamp").to_json(),
    )


def get_message_and_book_lists(
    messages: DataFrame, books: DataFrame, exchange: str, ticker: str
) -> Tuple[List[Message], List[Book]]:
    message_list, book_list = list(), list()
    for message in messages.itertuples():
        message_list.append(
            Message(
                id=message.internal_index,
                timestamp=message.timestamp,
                exchange=exchange,
                ticker=ticker,
                direction=message.direction,
                volume=message.volume,
                price=message.price,
                external_id=message.external_id,
                message_type=message.type,
            )
        )
        book_list.append(
            Book(
                id=message.internal_index,
                timestamp=message.timestamp,
                exchange=exchange,
                ticker=ticker,
                data=books.iloc[message.Index].to_json(),
            )
        )
    return message_list, book_list


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    populate_database()
