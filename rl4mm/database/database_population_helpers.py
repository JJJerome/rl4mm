import glob
import logging
import os
import ssl
from contextlib import suppress
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd
from sqlalchemy.orm import sessionmaker

from rl4mm.database.PostgresEngine import PostgresEngine
from rl4mm.database.models import Book
from rl4mm.orderbook.helpers import get_book_columns


def make_temporary_data_path() -> str:
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
    snapshot_freq: Optional[str],
    n_levels: int,
    total_daily_messages: int,
):
    first_index, last_index = messages.iloc[[0, -1]].index
    interval_series = get_interval_series(messages, snapshot_freq)
    rows_to_skip = (
        list(range(0, first_index))
        + list(messages[~messages.index.isin(interval_series.index)].index)
        + list(range(last_index + 1, total_daily_messages))
    )
    books = pd.read_csv(book_path, nrows=len(interval_series), skiprows=rows_to_skip, header=None, names=book_cols)
    books = pd.concat([pd.Series(interval_series.values, name="timestamp"), books], axis=1)
    return books


def get_file_len(filename):
    with open(filename) as f:
        for i, _ in enumerate(f):
            pass
    return i + 1


def convert_messages_and_books_to_dicts(
    messages: pd.DataFrame, books: pd.DataFrame, ticker: str, trading_date: str, n_levels: int, freq: Optional[str]
):
    start_index = messages.iloc[0].name
    message_convertor = partial(
        get_message_dict_from_series,
        **{
            "ticker": ticker,
            "trading_date": trading_date,
            "n_levels": n_levels,
            "start_index": start_index,
            "freq": freq,
        },
    )
    books_to_internal = partial(
        get_book_dict_from_series,
        **{
            "ticker": ticker,
            "trading_date": trading_date,
            "n_levels": n_levels,
            "start_index": start_index,
            "freq": freq,
        },
    )
    messages = messages.apply(message_convertor, axis=1).values
    books = books.apply(books_to_internal, axis=1).values
    return messages, books


def get_book_and_message_columns(n_levels: int = 50):
    book_cols = get_book_columns(n_levels)
    message_cols = ["time", "message_type", "external_id", "volume", "price", "direction"]
    return book_cols, message_cols


def get_book_and_message_paths(data_path: str, ticker: str, trading_date: str, n_levels: int) -> Tuple[Path, Path]:
    try:
        book_path = glob.glob(data_path + "/" + f"{ticker}_{trading_date}_*_orderbook_{n_levels}.csv")[0]
        message_path = glob.glob(data_path + "/" + f"{ticker}_{trading_date}_*_message_{n_levels}.csv")[0]
    except IndexError:
        raise FileNotFoundError(f"Level {n_levels} data for ticker {ticker} on {trading_date} not found in {data_path}")
    return Path(book_path), Path(message_path)


def reformat_message_data(messages: pd.DataFrame, trading_date: str) -> pd.DataFrame:
    messages["timestamp"] = get_timestamps(messages, trading_date)
    messages.drop(["trading_date", "time"], axis=1, inplace=True)
    type_dict = get_external_internal_type_dict()
    messages.message_type.replace(type_dict.keys(), type_dict.values(), inplace=True)
    update_direction(messages)
    messages.astype({"external_id": int})
    return messages


def get_timestamps(messages: pd.DataFrame, trading_date: str) -> pd.Series:
    messages.time = pd.to_timedelta(messages.time, unit="s")
    messages["trading_date"] = pd.to_datetime(trading_date)
    return messages.trading_date.add(messages.time)


def update_direction(messages: pd.DataFrame) -> None:
    messages.loc[(messages.direction == -1) & (messages.message_type != "market"), "direction"] = "sell"
    messages.loc[(messages.direction == 1) & (messages.message_type == "market"), "direction"] = "sell"
    messages.loc[(messages.direction == 1) & (messages.message_type != "market"), "direction"] = "buy"
    messages.loc[(messages.direction == -1) & (messages.message_type == "market"), "direction"] = "buy"


def get_interval_series(messages: pd.DataFrame, freq: Optional[str] = "S"):
    if freq is None:
        return messages.timestamp
    start_date = messages.iloc[0].timestamp
    end_date = messages.iloc[-1].timestamp
    target_times = pd.date_range(start_date.ceil(freq), end_date.ceil(freq), freq=freq)
    unique_timestamps = messages.timestamp.drop_duplicates(keep="last")
    mask = pd.DatetimeIndex(unique_timestamps).get_indexer(target_times, method="ffill")
    mask = unique_timestamps.iloc[mask[mask >= 0]].index.unique()
    return messages.timestamp.loc[mask].drop_duplicates()


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


def generate_internal_index(index: int, ticker: str, trading_date: str, n_levels: int, freq: str):
    return f"{freq}_L{str(n_levels).zfill(3)}_{EXCHANGE}_{ticker}_{trading_date}_" + str(index)


def get_message_dict_from_series(
    message: pd.Series, ticker: str, trading_date: str, n_levels: int, start_index: int, freq: str
):
    return dict(
        id=generate_internal_index(message.name + start_index, ticker, trading_date, n_levels, freq),
        timestamp=message.timestamp,
        exchange=EXCHANGE,
        ticker=ticker,
        direction=message.direction,
        volume=message.volume,
        price=message.price,
        external_id=message.external_id,
        message_type=message.message_type,
    )


def get_book_dict_from_series(
    book: pd.Series, ticker: str, trading_date: str, n_levels: int, start_index: int, freq: str
):
    return dict(
        id=generate_internal_index(book.name + start_index, ticker, trading_date, n_levels, freq),
        timestamp=book.timestamp,
        exchange=EXCHANGE,
        ticker=ticker,
        data=book.drop("timestamp").to_json(),
    )


EXCHANGE = "NASDAQ"
