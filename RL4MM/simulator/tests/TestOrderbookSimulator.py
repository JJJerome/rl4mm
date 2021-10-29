from pandas.testing import assert_frame_equal, assert_series_equal
from unittest import TestCase

import numpy as np
import pandas as pd

from RL4MM.simulator.OrderbookSimulator import HistoricalOrderbookSimulator

book_dict = dict()
book_dict.update({f"ask_price_{i}": 100 + i for i in range(10)})
book_dict.update({f"bid_price_{i}": 99 - i for i in range(10)})
book_dict.update({f"ask_size_{i}": 10 for i in range(10)})
book_dict.update({f"bid_size_{i}": 10 for i in range(10)})

BOOK_1 = pd.Series(book_dict)
BOOK_2 = BOOK_1.copy()
BOOK_2.loc["bid_size_0"] = 20
BOOK_2.loc["ask_size_1"] = 5


class TestHistoricalOrderBookSimulator(TestCase):
    def test_get_book_delta(self) -> None:
        expected = pd.Series(np.zeros(40, dtype="int64"), index=BOOK_1.index)
        expected["bid_size_0"] = 20 - 10
        expected["ask_size_1"] = 5 - 10
        actual = HistoricalOrderbookSimulator.get_book_delta(BOOK_1, BOOK_2)
        assert_series_equal(actual, expected)
