import abc
from datetime import datetime


class OrderGenerator(abc.ABCMeta):
    @abc.abstractmethod
    def generate_orders(cls, start: datetime, end: datetime):
        pass
