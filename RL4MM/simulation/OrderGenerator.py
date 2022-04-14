import abc
from datetime import datetime


class MessageGenerator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def generate_messages(self, start_date: datetime, end_date: datetime):
        pass
