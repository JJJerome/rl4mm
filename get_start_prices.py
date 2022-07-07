from datetime import datetime
from RL4MM.database.HistoricalDatabase import HistoricalDatabase

tickers_file = open("scripts/tickers_all.txt", "r")
tickers = tickers_file.readlines()
tickers = [t.strip("\n") for t in tickers]

print(tickers)


# tickers = ["KO"]  # Choose the tickers we want

start_price_dict = {}
start_date = datetime(2022, 3, 1)

database = HistoricalDatabase()

for ticker in tickers:
    start_book = database.get_next_snapshot(start_date, ticker)
    start_price_dict[ticker] = (start_book.sell_price_0 + start_book.buy_price_0) / 2

print(start_price_dict)
