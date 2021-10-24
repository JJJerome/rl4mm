from datetime import datetime
from typing import List

import ast
import pandas as pd

from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import sessionmaker

from RL4MM.database.PostgresEngine import PostgresEngine
from RL4MM.database.models import Base, Book, Event


class HistoricalDatabase:
    def __init__(self, engine: Engine = None) -> None:
        self.engine = engine or PostgresEngine().engine
        self.session_maker = sessionmaker(bind=self.engine)
        Base.metadata.create_all(bind=self.engine)

    def insert_events(self, events: List[Event]) -> None:
        session = self.session_maker()
        session.add_all(events)
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
            .order_by(Book.timestamp.desc())
            .order_by(Book.id.desc())
            .first()
        )
        session.close()
        if snapshot is None:
            return pd.DataFrame()
        else:
            book_data = pd.DataFrame([snapshot.__dict__]).data[0]
            ts = pd.DataFrame([snapshot.__dict__]).timestamp[0]
            return pd.Series(ast.literal_eval(book_data),name=ts)

    def get_next_snapshot(self, timestamp: datetime, exchange: str, ticker: str) -> pd.DataFrame:
        session = self.session_maker()
        snapshot = (
            session.query(Book)
            .filter(Book.exchange == exchange)
            .filter(Book.ticker == ticker)
            .filter(Book.timestamp >= timestamp)
            .order_by(Book.timestamp.asc())
            .first()
        )
        session.close()
        if snapshot is None:
            return pd.DataFrame()
        else:
            book_data = pd.DataFrame([snapshot.__dict__]).drop(columns=["_sa_instance_state"]).data[0]
            ts = pd.DataFrame([snapshot.__dict__]).timestamp[0]
            return pd.Series(ast.literal_eval(book_data),name=ts)

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
            return pd.DataFrame(snapshots_dict).drop(columns=["_sa_instance_state"])
        else:
            return pd.DataFrame()

    def get_events(self, start_date: datetime, end_date: datetime, exchange: str, ticker: str) -> pd.DataFrame:
        session = self.session_maker()
        events = (
            session.query(Event)
            .filter(Event.exchange == exchange)
            .filter(Event.ticker == ticker)
            .filter(Event.timestamp.between(start_date, end_date))
            .order_by(Event.timestamp.asc())
            .all()
        )
        session.close()
        events_dict = [t.__dict__ for t in events]
        if len(events_dict) > 0:
            return pd.DataFrame(events_dict).drop(columns=["_sa_instance_state"])
        else:
            return pd.DataFrame()
