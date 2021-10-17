from datetime import datetime
from typing import List

import pandas as pd

from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import sessionmaker

from RL4MM.database.PostgresEngine import PostgresEngine
from RL4MM.database.models import Base, Book, Trade


class HistoricalDatabase:
    def __init__(self, engine: Engine = None) -> None:
        self.engine = engine or PostgresEngine().engine
        self.session_maker = sessionmaker(bind=self.engine)
        Base.metadata.create_all(bind=self.engine)

    def insert_trades(self, trades: List[Trade]) -> None:
        session = self.session_maker()
        session.add_all(trades)
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
            .first()
        )
        session.close()
        if snapshot is None:
            return pd.DataFrame()
        else:
            return pd.DataFrame([snapshot.__dict__]).drop(columns=["_sa_instance_state"])

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
            return pd.DataFrame([snapshot.__dict__]).drop(columns=["_sa_instance_state"])

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

    def get_trades(self, start_date: datetime, end_date: datetime, exchange: str, ticker: str) -> pd.DataFrame:
        session = self.session_maker()
        trades = (
            session.query(Trade)
            .filter(Trade.exchange == exchange)
            .filter(Trade.ticker == ticker)
            .filter(Trade.timestamp.between(start_date, end_date))
            .order_by(Trade.timestamp.asc())
            .all()
        )
        session.close()
        trades_dict = [t.__dict__ for t in trades]
        if len(trades_dict) > 0:
            return pd.DataFrame(trades_dict).drop(columns=["_sa_instance_state"])
        else:
            return pd.DataFrame()
