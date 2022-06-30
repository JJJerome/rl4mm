from typing import List

import abc
from datetime import datetime

from RL4MM.orderbook.models import Order


class OrderGenerator(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def name(self):
        pass

    @abc.abstractmethod
    def generate_orders(self, start_date: datetime, end_date: datetime) -> List[Order]:
        pass  # return a list of orders, _ordered by arrival time_.
