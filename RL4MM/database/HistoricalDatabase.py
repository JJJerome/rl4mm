from datetime import datetime
from typing import List

import ast
import pandas as pd

from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import sessionmaker

from RL4MM.database.PostgresEngine import PostgresEngine
from RL4MM.database.models import Base, Book, Message


class HistoricalDatabase:
    def __init__(self, engine: Engine = None, preprocess: bool = True) -> None:
        self.engine = engine or PostgresEngine().engine
        self.session_maker = sessionmaker(bind=self.engine)
        self.preprocess = preprocess
        Base.metadata.create_all(bind=self.engine)

    def insert_messages(self, messages: List[Message]) -> None:
        session = self.session_maker()
        session.add_all(messages)
        session.commit()
        session.close()

    def insert_books(self, books: List[Book]) -> None:
        session = self.session_maker()
        session.add_all(books)
        session.commit()
        session.close()

    def get_last_snapshot(self, timestamp: datetime, exchange: str, ticker: str) -> pd.DataFrame:
        session = self.session_maker()
        snapshot = (
            session.query(Book)
            .filter(Book.exchange == exchange)
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

    def get_next_snapshot(self, timestamp: datetime, exchange: str, ticker: str) -> pd.DataFrame:
        session = self.session_maker()
        snapshot = (
            session.query(Book)
            .filter(Book.exchange == exchange)
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

    def get_book_snapshots(self, start_date: datetime, end_date: datetime, exchange: str, ticker: str) -> pd.DataFrame:
        session = self.session_maker()
        snapshots = (
            session.query(Book)
            .filter(Book.exchange == exchange)
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

    def get_messages(self, start_date: datetime, end_date: datetime, exchange: str, ticker: str) -> pd.DataFrame:
        session = self.session_maker()
        messages = (
            session.query(Message)
            .filter(Message.exchange == exchange)
            .filter(Message.ticker == ticker)
            .filter(Message.timestamp.between(start_date, end_date))
            .order_by(Message.timestamp.asc(), Message.id.asc())
            .all()
        )
        session.close()
        messages_dict = [t.__dict__ for t in messages]
        if len(messages_dict) > 0:
            return pd.DataFrame(messages_dict).drop(columns=["_sa_instance_state"])
        else:
            return pd.DataFrame()

    @staticmethod
    def convert_book_to_series(book_data: str) -> pd.Series:
        return pd.Series(ast.literal_eval(book_data))
