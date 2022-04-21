from typing import Deque

import abc
from datetime import datetime

from RL4MM.orderbook.models import Order


class OrderGenerator(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def name(self):
        pass

    @abc.abstractmethod
    def generate_orders(self, start_date: datetime, end_date: datetime) -> Deque[Order]:
        pass  # return a deque of orders, _ordered by arrival time_.
