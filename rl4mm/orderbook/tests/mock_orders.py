from collections import deque
from copy import deepcopy
from datetime import datetime

from sortedcontainers import SortedDict

from rl4mm.orderbook.models import LimitOrder, Cancellation, Deletion, MarketOrder, Orderbook

LIMIT_1 = LimitOrder(
    timestamp=datetime(2012, 6, 21, 12, 0),
    price=int(30.1 * 10000),
    volume=1000,
    direction="buy",
    ticker="MSFT",
    external_id=50,
    internal_id=None,
    is_external=True,
)
LIMIT_2 = LimitOrder(
    timestamp=datetime(2012, 6, 21, 12, 1),
    price=int(30.1 * 10000),
    volume=200,
    direction="buy",
    ticker="MSFT",
    external_id=100,
    internal_id=None,
    is_external=True,
)
LIMIT_3 = LimitOrder(
    timestamp=datetime(2012, 6, 21, 12, 1),
    price=int(30.2 * 10000),
    volume=200,
    direction="buy",
    ticker="MSFT",
    external_id=110,
    internal_id=None,
    is_external=True,
)
LIMIT_4 = LimitOrder(
    timestamp=datetime(2012, 6, 21, 12, 1),
    price=int(30.3 * 10000),
    volume=200,
    direction="sell",
    ticker="MSFT",
    external_id=None,
    internal_id=None,
    is_external=False,
)
LIMIT_5 = LimitOrder(
    timestamp=datetime(2012, 6, 21, 12, 1),
    price=int(30.2 * 10000),
    volume=300,
    direction="sell",
    ticker="MSFT",
    external_id=None,
    internal_id=None,
    is_external=False,
)
CANCELLATION_1 = Cancellation(
    timestamp=datetime(2012, 6, 21, 12, 1),
    price=int(30.1 * 10000),
    volume=200,
    direction="buy",
    ticker="MSFT",
    external_id=50,
    internal_id=None,
    is_external=True,
)
CANCELLATION_2 = Cancellation(
    timestamp=datetime(2012, 6, 21, 12, 2),
    price=int(30.1 * 10000),
    volume=1100,
    direction="buy",
    ticker="MSFT",
    external_id=50,
    internal_id=None,
    is_external=False,
)
CANCELLATION_3 = Cancellation(
    timestamp=datetime(2012, 6, 21, 12, 3),
    price=int(30.1 * 10000),
    volume=200,
    direction="buy",
    ticker="MSFT",
    external_id=110,
    internal_id=None,
    is_external=False,
)
DELETION_1 = Deletion(
    timestamp=datetime(2012, 6, 21, 12, 1),
    price=int(30.1 * 10000),
    volume=1000,
    direction="buy",
    ticker="MSFT",
    external_id=50,
    internal_id=None,
    is_external=True,
)
DELETION_2 = Deletion(
    timestamp=datetime(2012, 6, 21, 12, 2),
    price=int(30.1 * 10000),
    volume=None,
    direction="buy",
    ticker="MSFT",
    external_id=100,
    internal_id=None,
    is_external=True,
)
DELETION_3 = Deletion(
    timestamp=datetime(2012, 6, 21, 12, 3),
    price=int(30.3 * 10000),
    volume=None,
    direction="sell",
    ticker="MSFT",
    external_id=None,
    internal_id=2,
    is_external=False,
)
MARKET_1 = MarketOrder(
    timestamp=datetime(2012, 6, 21, 12, 1),
    volume=1000,
    direction="sell",
    ticker="MSFT",
    external_id=50,
    internal_id=None,
    is_external=False,
)
MARKET_2 = MarketOrder(
    timestamp=datetime(2012, 6, 21, 12, 2),
    volume=200,
    direction="sell",
    ticker="MSFT",
    external_id=100,
    internal_id=None,
    is_external=True,
)
MARKET_3 = MarketOrder(
    timestamp=datetime(2012, 6, 21, 12, 3),
    volume=200,
    direction="buy",
    ticker="MSFT",
    external_id=110,
    internal_id=None,
    is_external=True,
)
submission_1 = deepcopy(LIMIT_1)
submission_2 = deepcopy(LIMIT_2)
submission_3 = deepcopy(LIMIT_3)
submission_4 = deepcopy(LIMIT_4)
submission_1.internal_id = 1
submission_2.internal_id = 2
submission_3.internal_id = 3
submission_4.internal_id = 4


TICKER = "MSFT"
TICK_SIZE = 100


def get_mock_orderbook():
    orderbook = Orderbook(
        buy=SortedDict(
            {int(30.1 * 10000): deque([submission_1, submission_2]), int(30.2 * 10000): deque([submission_3])}
        ),
        sell=SortedDict({int(30.3 * 10000): deque([submission_4])}),
        ticker=TICKER,
        tick_size=TICK_SIZE,
    )
    return deepcopy(orderbook)

    # TODO: write test for internal order executing when another internal order is present on opposing side
