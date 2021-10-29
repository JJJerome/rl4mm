import abc
from datetime import datetime

import pandas as pd


class Agent(metaclass=abc.ABCMeta):
    def generate_messages(self, timestamp: datetime, book: pd.Series):
        pass
