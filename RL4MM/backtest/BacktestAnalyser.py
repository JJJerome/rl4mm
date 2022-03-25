import abc

import pandas as pd

from RL4MM.backtest.Backtester import Backtester
from RL4MM.database.HistoricalDatabase import HistoricalDatabase


class BacktestAnalyser(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def summary(cls, results):
        pass


class HistoricalBacktestAnalyser(BacktestAnalyser):
    def __init__(self, backtester: Backtester, historical_database: HistoricalDatabase = None):
        self.backtester = backtester
        self.database = historical_database or HistoricalDatabase()

    def summary(self):
        summary = pd.DataFrame(
            columns=["Results"],
            index=["Start", "End", "", "Total Return", "Fill Percentage"],
        )
        summary.loc["Start"] = self.backtester.results["start date"]
        summary.loc["End"] = self.backtester.results["start date"]
        summary.loc[""] = "***"
        summary.loc["Total Return"] = self.get_pnl(self.backtester.results, self.backtester.initial_portfolio)
        summary.loc["Fill Percentage"] = self.get_fill_percentage(self.backtester.results)
        return summary

    @staticmethod
    def get_fill_percentage(results: dict):
        return len(results["filled messages"]) / len(results["remaining messages"] + results["filled messages"])

    @staticmethod
    def get_pnl(results: dict, initial_portfolio: dict) -> float:
        portfolio = initial_portfolio.copy()
        for message in results["filled messages"]:
            if message.side == "ask":
                portfolio["stock"] -= message.volume
                portfolio["cash"] += message.volume * message.price
            if message.side == "bid":
                portfolio["stock"] += message.volume
                portfolio["cash"] -= message.volume * message.price

        ticker = results["filled messages"][0].ticker
        for message in results["filled messages"]:
            assert message.ticker == ticker, f"Some of the filled messages correspond to a ticker other than {ticker}."

        initial_midprice = (results["start book"].loc["bid_price_0"] + results["start book"].loc["ask_price_0"]) / 2
        terminal_midprice = (
            results["terminal book"].loc["bid_price_0"] + results["terminal book"].loc["ask_price_0"]
        ) / 2
        initial_value = initial_portfolio["cash"] + initial_portfolio["stock"] * initial_midprice
        terminal_value = portfolio["cash"] + portfolio["stock"] * terminal_midprice
        return terminal_value - initial_value
