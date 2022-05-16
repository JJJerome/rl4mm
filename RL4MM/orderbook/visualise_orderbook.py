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
        title=orderbook["ticker"],
        color_discrete_sequence=["green", "red"],
    )
    fig.update_traces(width=tick_size)
    fig.show()


def convert_orderbook_to_dataframe(orderbook: Orderbook, n_levels: int = 10):
    order_dict = {}
    for side in ["bid", "ask"]:
        prices = reversed(orderbook[side]) if side == "bid" else orderbook[side]  # type: ignore
        for level, price in enumerate(prices):
            if level >= n_levels:
                break
            total_volume = sum(order.volume for order in orderbook[side][price])  # type: ignore
            order_dict[side + str(level)] = (side, price, total_volume)
    df = pd.DataFrame(order_dict).T
    return df.rename(columns={0: "direction", 1: "price", 2: "volume"})
