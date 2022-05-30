import pandas as pd

from RL4MM.orderbook.models import Orderbook


def convert_to_lobster_format(orderbook: Orderbook, n_levels: int = 10):
    lobster_book = dict()
    for direction in ["buy", "sell"]:
        half_book = orderbook[direction]  # type: ignore
        if direction == "buy":
            half_book = reversed(half_book)
        for level, price in enumerate(half_book):
            if level < n_levels:
                lobster_book[direction + "_price_" + str(level)] = price
                volume = 0
                for order in orderbook[direction][price]:  # type: ignore
                    volume += order.volume
                lobster_book[direction + "_volume_" + str(level)] = volume
    return lobster_book


def compare_elements_of_books(book_1: pd.DataFrame, book_2: pd.DataFrame, verbose: bool = False):
    if verbose:
        for key in book_1.keys():
            print(f"{key}: {book_1[key]==book_2[key]}")
    else:
        print(all([book_1[key] == book_2[key] for key in book_1.keys()]))
