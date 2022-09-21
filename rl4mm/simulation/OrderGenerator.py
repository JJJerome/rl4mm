from typing import Deque

import abc
from datetime import datetime

from rl4mm.orderbook.models import Order


class OrderGenerator(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def name(self):
        pass

    @abc.abstractmethod
    def preload_episode_orders(self, min_date: datetime, max_date: datetime):
        pass

    @abc.abstractmethod
    def generate_orders(self, start_date: datetime, end_date: datetime) -> Deque[Order]:
        pass  # return a list of orders, _ordered by arrival time_.
