from typing import Any

from sqlalchemy import Column, DateTime, Float, Integer, JSON, String
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()  # type: Any


class Message(Base):
    __tablename__ = "messages"

    id = Column(String, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    exchange = Column(String, nullable=False, index=True)
    ticker = Column(String, nullable=False, index=True)
    direction = Column(String, nullable=False)
    volume = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    external_id = Column(Integer, nullable=True)
    message_type = Column(String, nullable=True)

    def __repr__(self):
        return f"{self.__class__.__name__}_{self.id}"


class Book(Base):
    __tablename__ = "book"

    id = Column(String, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    exchange = Column(String, nullable=False, index=True)
    ticker = Column(String, nullable=False, index=True)
    data = Column(JSON, nullable=False)

    def __repr__(self):
        return f"{self.__class__.__name__}_{self.id}"
