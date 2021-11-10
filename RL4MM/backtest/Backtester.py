from datetime import datetime, timedelta
from tqdm import tqdm

import pandas as pd

from RL4MM.agents.Agent import Agent
from RL4MM.simulator.OrderbookSimulator import OrderbookSimulator, HistoricalOrderbookSimulator


class Backtester:
    def __init__(
        self, agent: Agent, simulator: OrderbookSimulator = None, initial_portfolio: dict = {"cash": 0, "stock": 0}
    ) -> None:
        self.agent = agent
        self.simulator = simulator or self._get_default_simulator()
        self.initial_portfolio = initial_portfolio
        self.results = dict()

    def run(self, start_date: datetime, end_date: datetime, step_size: timedelta, start_book: pd.Series) -> None:
        messages_to_fill = list()
        filled_messages = list()
        book = start_book
        for timestamp in tqdm(self._get_daterange(start_date, end_date, step_size)):
            new_messages = self.agent.generate_messages(timestamp, book)
            messages_to_fill += new_messages
            results = self.simulator.simulate_step(timestamp, timestamp + step_size, messages_to_fill, book)
            filled_messages += results["filled_messages"]
            messages_to_fill = results["messages_to_fill"]
            book = results["orderbook"]
        results = {
            "start date": start_date,
            "end date": end_date,
            "step size": step_size,
            "start book": start_book,
            "remaining messages": messages_to_fill,
            "filled messages": filled_messages,
            "terminal book": book,
        }
        self.results = results

    @staticmethod
    def _get_default_simulator():
        return HistoricalOrderbookSimulator(exchange="NASDAQ", ticker="MSFT")

    @staticmethod
    def _get_daterange(start_date: datetime, end_date: datetime, step_size: timedelta):
        for n in range(int((end_date - start_date) / step_size)):
            yield start_date + n * step_size
