from itertools import chain

import pandas as pd
import plotly.express as px

from RL4MM.orderbook.models import Orderbook


def visualise_orderbook(orderbook: Orderbook, n_levels: int = 10, tick_size: float = 0.01):
    df = convert_orderbook_to_dataframe(orderbook, n_levels)
    fig = px.bar(
        df,
        x="price",
        y="volume",
        color="direction",
        title=orderbook.ticker,
        color_discrete_sequence=["green", "red"],
    )
    fig.update_traces(width=tick_size)
    fig.show()


def convert_orderbook_to_dataframe(orderbook: Orderbook, n_levels: int = 10):
    order_dict = {}
    for direction in ["buy", "sell"]:
        prices = reversed(getattr(orderbook, direction)) if direction == "buy" else getattr(orderbook, direction)
        for level, price in enumerate(prices):
            if level >= n_levels:
                break
            total_volume = sum(order.volume for order in getattr(orderbook, direction)[price])  # type: ignore
            order_dict[direction + "_" + str(level)] = (direction, price, total_volume)
    df = pd.DataFrame(order_dict).T
    return df.rename(columns={0: "direction", 1: "price", 2: "volume"})


def convert_orderbook_to_series(orderbook: Orderbook, n_levels: int = 10):
    order_dict = {}
    for direction in ["buy", "sell"]:
        prices = reversed(getattr(orderbook, direction)) if direction == "buy" else getattr(orderbook, direction)
        for level, price in enumerate(prices):
            if level >= n_levels:
                break
            total_volume = sum(order.volume for order in getattr(orderbook, direction)[price])  # type: ignore
            order_dict[direction + "_price_" + str(level)] = price
            order_dict[direction + "_volume_" + str(level)] = total_volume
    return pd.DataFrame(order_dict, index=[0])


def get_book_columns(n_levels: int = 50):
    price_cols = list(chain(*[("sell_price_{0},buy_price_{0}".format(i)).split(",") for i in range(n_levels)]))
    volume_cols = list(chain(*[("sell_volume_{0},buy_volume_{0}".format(i)).split(",") for i in range(n_levels)]))
    return list(chain(*zip(price_cols, volume_cols)))
