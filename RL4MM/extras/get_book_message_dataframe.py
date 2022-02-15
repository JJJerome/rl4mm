import pandas as pd

from RL4MM.database.populate_database import (
    get_lobster_data,
    reformat_message_data,
    rescale_book_data,
)


def get_book_message_dataframe(
    ticker: str = "AMZN",
    trading_date: str = "2012-06-21",
    levels: int = 10,
):
    print(f"Getting data for ticker {ticker} on {trading_date}.")
    books, messages = get_lobster_data(ticker, trading_date, levels)
    messages = reformat_message_data(messages, trading_date)
    books = rescale_book_data(books)
    book_message_df = pd.concat([messages, books], axis=1)
    book_message_df.columns = book_message_df.columns.str.replace("_", " ")
    return book_message_df.set_index("timestamp")
