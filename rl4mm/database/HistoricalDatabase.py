from datetime import datetime
from typing import List

import ast
import pandas as pd

from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import sessionmaker

from rl4mm.database.PostgresEngine import PostgresEngine
from rl4mm.database.models import Base, Book, Message
from rl4mm.orderbook.helpers import get_book_columns


class HistoricalDatabase:
    def __init__(self, engine: Engine = None) -> None:
        self.engine = engine or PostgresEngine().engine
        self.session_maker = sessionmaker(bind=self.engine)
        Base.metadata.create_all(bind=self.engine)
        self.exchange = "NASDAQ"  # This database currently only retrieves NASDAQ (LOBSTER) data

    def insert_messages(self, messages: List[Message]) -> None:
        session = self.session_maker()
        session.add_all(messages)
        session.commit()
        session.close()

    def insert_books(self, books: List[Book]) -> None:
        with self.session_maker() as session:
            session.add_all(books)
            session.commit()
            session.close()

    def insert_books_from_dicts(self, book_dicts: List[dict]) -> None:
        with self.session_maker() as session:
            session.bulk_insert_mappings(Book, book_dicts)
            session.commit()
            session.close()

    def insert_messages_from_dicts(self, message_dicts: List[dict]) -> None:
        with self.session_maker() as session:
            session.bulk_insert_mappings(Message, message_dicts)
            session.commit()
            session.close()

    def get_last_snapshot(self, timestamp: datetime, ticker: str) -> pd.DataFrame:
        session = self.session_maker()
        snapshot = (
            session.query(Book)
            .filter(Book.exchange == self.exchange)
            .filter(Book.ticker == ticker)
            .filter(Book.timestamp <= timestamp)
            .order_by(Book.timestamp.desc(), Book.id.desc())
            .first()
        )
        session.close()
        if snapshot is None:
            return pd.DataFrame()
        else:
            book_data = pd.DataFrame([snapshot.__dict__]).data[0]
            ts = pd.DataFrame([snapshot.__dict__]).timestamp[0]
            return pd.Series(ast.literal_eval(book_data), name=ts)

    def get_next_snapshot(self, timestamp: datetime, ticker: str) -> pd.DataFrame:
        session = self.session_maker()
        snapshot = (
            session.query(Book)
            .filter(Book.exchange == self.exchange)
            .filter(Book.ticker == ticker)
            .filter(Book.timestamp >= timestamp)
            .order_by(Book.timestamp.asc(), Book.id.asc())
            .first()
        )
        session.close()
        if snapshot is None:
            return pd.DataFrame()
        else:
            book_data = pd.DataFrame([snapshot.__dict__]).data[0]
            ts = pd.DataFrame([snapshot.__dict__]).timestamp[0]
            return pd.Series(ast.literal_eval(book_data), name=ts)

    def get_book_snapshots(self, start_date: datetime, end_date: datetime, ticker: str) -> pd.DataFrame:
        session = self.session_maker()
        snapshots = (
            session.query(Book)
            .filter(Book.exchange == self.exchange)
            .filter(Book.ticker == ticker)
            .filter(Book.timestamp.between(start_date, end_date))
            .order_by(Book.timestamp.asc())
            .all()
        )
        session.close()
        snapshots_dict = [s.__dict__ for s in snapshots]
        if len(snapshots_dict) > 0:
            book_levels = pd.DataFrame(snapshots_dict).data
            book_info = pd.DataFrame(snapshots_dict).drop(columns=["_sa_instance_state", "data"])
            book_info.reset_index()
            book_levels = book_levels.apply(self.convert_book_to_series)
            return pd.concat([book_info, book_levels], axis=1)
        else:
            return pd.DataFrame()

    def get_messages(self, start_date: datetime, end_date: datetime, ticker: str) -> pd.DataFrame:
        session = self.session_maker()
        messages = (
            session.query(Message)
            .filter(Message.exchange == self.exchange)
            .filter(Message.ticker == ticker)
            .filter(Message.timestamp > start_date)
            .filter(Message.timestamp <= end_date)
            .order_by(Message.timestamp.asc(), Message.id.asc())
            .all()
        )
        session.close()
        messages_dict = [t.__dict__ for t in messages]
        if len(messages_dict) > 0:
            return pd.DataFrame(messages_dict).drop(columns=["_sa_instance_state"])
        else:
            return pd.DataFrame()

    def get_book_snapshot_series(
        self, start_date: datetime, end_date: datetime, ticker: str, freq: str = "S", n_levels: int = 10
    ) -> pd.DataFrame:
        # TODO: speed this up by using postgres' "generate_series"
        timestamp_series = pd.date_range(start_date, end_date, freq=freq)
        book_columns = get_book_columns(n_levels)
        book_df = pd.DataFrame(columns=book_columns)
        for timestamp in timestamp_series:
            book = pd.DataFrame(self.get_last_snapshot(timestamp=timestamp, ticker=ticker)).T[book_columns]
            book_df = pd.concat([book_df, book])
        return book_df

    @staticmethod
    def convert_book_to_series(book_data: str) -> pd.Series:
        return pd.Series(ast.literal_eval(book_data))
