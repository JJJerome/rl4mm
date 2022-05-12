import pandas as pd
import plotly.express as px

from RL4MM.orderbook.models import Orderbook


def visualise_orderbook(orderbook: Orderbook, n_levels: int = 10):
    df = convert_orderbook_to_dataframe(orderbook, n_levels)
    fig = px.bar(df, x="price", y="volume", color="direction")  # , title=f"Orderbook at {x.name}")
    fig.show()


def convert_orderbook_to_dataframe(orderbook: Orderbook, n_levels: int = 10):
    order_dict = {}
    for side in ["bid", "ask"]:
        prices = reversed(orderbook[side]) if side == "bid" else orderbook[side]
        for level, price in enumerate(prices):
            if level >= n_levels:
                break
            order_dict[side + str(level)] = (side, price, sum(order.volume for order in orderbook[side][price]))
    df = pd.DataFrame(order_dict).T
    return df.rename(columns={0: "direction", 1: "price", 2: "volume"})
