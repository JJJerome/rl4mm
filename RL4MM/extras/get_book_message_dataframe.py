from typing import Optional

import pandas as pd

from RL4MM.database.populate_database import (
    download_lobster_sample_data,
    reformat_message_data,
    rescale_book_data,
    _make_temporary_data_path,
    get_book_and_message_columns,
    _get_book_and_message_paths,
)


def get_book_message_dataframe(
    ticker: str = "AMZN",
    trading_date: str = "2012-06-21",
    n_levels: int = 10,
    path_to_lobster_data: Optional[str] = None,
    nrows: int = 100000,
):
    if path_to_lobster_data is None:
        path_to_lobster_data = _make_temporary_data_path()
    is_sample_data = path_to_lobster_data is None
    book_cols, message_cols = get_book_and_message_columns(n_levels)
    if is_sample_data:
        download_lobster_sample_data(ticker, trading_date, n_levels, path_to_lobster_data)
    book_path, message_path = _get_book_and_message_paths(path_to_lobster_data, ticker, trading_date, n_levels)
    messages = pd.read_csv(message_path, header=None, names=message_cols, usecols=[0, 1, 2, 3, 4, 5], nrows=nrows)
    messages = reformat_message_data(messages, trading_date)
    books = pd.read_csv(book_path, nrows=nrows, header=None, names=book_cols)
    books = rescale_book_data(books, n_levels)
    book_message_df = pd.concat([messages, books], axis=1)
    return book_message_df.set_index("timestamp")
