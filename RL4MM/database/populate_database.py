from os import popen
from typing import List

import logging
import shutil

from io import BytesIO
from itertools import chain
from contextlib import suppress
from pandas import DataFrame, read_csv, to_datetime, to_timedelta
from pathlib import Path
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from urllib.request import urlopen
from zipfile import ZipFile

from RL4MM.database.HistoricalDatabase import HistoricalDatabase
from RL4MM.database.models import Book, Trade
from RL4MM.database.PostgresEngine import PostgresEngine

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")


def populate_database(
    tickers: List = None,
    trading_dates: List = None,
    levels: int = 10,
    database: HistoricalDatabase = None,
    exchange: str = "NASDAQ",
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
    for ticker in tickers:
        for trading_date in trading_dates:
            logging.info(f"Downloading book and trade data for {ticker} on {trading_date} from LOBSTER.")
            books, messages = get_lobster_data(ticker, trading_date, levels)
            messages = reformat_message_data(messages, trading_date)
            messages = add_internal_index(messages, ticker, trading_date, exchange)
            books = rescale_book_data(books)
            trade_list, book_list = get_trade_and_book_lists(messages, books, exchange, ticker)
            logging.info(f"Inserting book and trade data for {ticker} on {trading_date} into database.")
            try:
                database.insert_books(book_list)
                database.insert_trades(trade_list)
            except IntegrityError:
                logging.warning(f"Data for {ticker} on {trading_date} already exists in database and so was not added.")


def create_tables():
    engine = PostgresEngine().engine
    Session = sessionmaker(engine)
    session = Session()
    Book.metadata.create_all(engine)
    session.close()


def get_book_and_message_columns():
    price_cols = list(chain(*[("ask_price_{0},bid_price_{0}".format(i)).split(",") for i in range(10)]))
    size_cols = list(chain(*[("ask_size_{0},bid_size_{0}".format(i)).split(",") for i in range(10)]))
    book_cols = list(chain(*zip(price_cols, size_cols)))
    message_cols = ["time", "type", "external_id", "size", "price", "direction"]
    return book_cols, message_cols


def make_temporary_data_folder(ticker: str, trading_date: str, levels: int) -> Path:
    temp_data_path = Path("data") / ticker / trading_date / f"level_{levels}"
    with suppress(FileExistsError):
        temp_data_path.mkdir(parents=True)
    return temp_data_path


def get_book_and_message_paths(data_path: Path, ticker: str, trading_date: str, levels: int) -> List[Path]:
    book_path = data_path / f"{ticker}_{trading_date}_34200000_57600000_orderbook_{levels}.csv"
    message_path = data_path / f"{ticker}_{trading_date}_34200000_57600000_message_{levels}.csv"
    return book_path, message_path


def get_lobster_data(ticker: str, trading_date: str, levels: int) -> List[DataFrame]:
    temp_data_path = make_temporary_data_folder(ticker, trading_date, levels)
    zip_url = f"https://lobsterdata.com/info/sample/LOBSTER_SampleFile_{ticker}_{trading_date}_{levels}.zip"
    with urlopen(zip_url) as zip_resp:
        with ZipFile(BytesIO(zip_resp.read())) as zip_file:
            zip_file.extractall(temp_data_path)
    book_path, message_path = get_book_and_message_paths(temp_data_path, ticker, trading_date, levels)
    book_cols, message_cols = get_book_and_message_columns()
    books = read_csv(book_path, header=None, names=book_cols)
    messages = read_csv(message_path, header=None, names=message_cols)
    assert len(books) == len(messages), "Length of the order book csv and message csv differ"
    with suppress(FileNotFoundError):
        shutil.rmtree(temp_data_path)
    return [books, messages]


def reformat_message_data(messages: DataFrame, trading_date: str) -> DataFrame:
    messages.time = to_timedelta(messages.time, unit="s")
    messages["trading_date"] = to_datetime(trading_date)
    messages["timestamp"] = messages.trading_date.add(messages.time)
    messages.drop(["trading_date", "time"], axis=1, inplace=True)
    type_dict = get_external_internal_type_dict()
    messages.type.replace(type_dict.keys(), type_dict.values(), inplace=True)
    messages.direction.replace([-1, 1], ["sell", "buy"], inplace=True)
    messages["price"] /= 10000
    return messages


def rescale_book_data(books: DataFrame) -> DataFrame:
    for order_type in ["ask", "bid"]:
        for i in range(10):
            books[order_type + "_price_" + str(i)] /= 10000
    return books


def get_external_internal_type_dict():
    return {
        1: "submission",
        2: "cancellation",
        3: "deletion",
        4: "execution_visible",
        5: "execution_hidden",
        7: "trading_halt",
    }


def add_internal_index(dataframe: DataFrame, ticker: str, trading_date: str, exchange: str):
    dataframe["internal_index"] = f"{exchange}_{ticker}_{trading_date}_" + dataframe.index.astype("str")
    return dataframe


def get_trade_and_book_lists(messages: DataFrame, books: DataFrame, exchange: str, ticker: str) -> [Trade, Book]:
    trade_list, book_list = list(), list()
    for message in messages.itertuples():
        trade_list.append(
            Trade(
                id=message.internal_index,
                timestamp=message.timestamp,
                exchange=exchange,
                ticker=ticker,
                direction=message.direction,
                size=message.size,
                price=message.price,
                external_id=message.external_id,
                order_type=message.type,
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
    return trade_list, book_list


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )
    populate_database()
