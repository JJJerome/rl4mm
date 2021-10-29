from datetime import datetime

import pandas as pd

from RL4MM.agents.Agent import Agent
from RL4MM.simulator.OrderbookSimulator import OrderbookMessage


class AddTouchAgent(Agent):
    def __init__(self, ticker: str, order_size: int):
        self.ticker = ticker
        self.order_size = order_size

    def generate_messages(self, timestamp: datetime, book: pd.Series):
        touch_prices = self._get_touch_prices(book)
        return [
            OrderbookMessage(
                _id=-1,
                datetime=datetime,
                message_type="submission",
                ticker=self.ticker,
                size=self.order_size,
                price=price,
                side=side,
            )
            for side, price in touch_prices.items()
        ]

    @staticmethod
    def _get_touch_prices(book: pd.Series):
        touch_prices = dict()
        touch_prices["bid"] = book.loc["bid_price_0"]
        touch_prices["ask"] = book.loc["ask_price_0"]
        return touch_prices
