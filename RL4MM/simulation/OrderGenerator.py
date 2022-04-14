import abc
from collections import deque
from datetime import datetime

from RL4MM.orderbook.models import Order


class OrderGenerator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def generate_orders(self, start_date: datetime, end_date: datetime) -> deque[Order]:
        pass  # return a deque of orders, _ordered by arrival time_.
