import pandas as pd
import plotly.express as px

from RL4MM.orderbook.Exchange import Exchange


def visualise_orderbook(orderbook: Exchange):
    price_volume_dict = {
        price: (sum([order.volume for order in orders]), "bid") for price, orders in orderbook.orderbook["bid"].items()
    }
    price_volume_dict.update(
        {
            price: (sum([order.volume for order in orders]), "ask")
            for price, orders in orderbook.orderbook["ask"].items()
        }
    )
    price_volume_df = pd.DataFrame(price_volume_dict).T
    price_volume_df.reset_index(inplace=True)
    price_volume_df.columns = ["price", "size", "direction"]
    price_volume_df = price_volume_df.convert_dtypes(float, int, str)
    fig = px.bar(price_volume_df, x="price", y="size", color="direction")
    fig.show()
