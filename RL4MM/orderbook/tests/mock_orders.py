from datetime import datetime

from RL4MM.orderbook.models import LimitOrder, Cancellation, Deletion, MarketOrder

LIMIT_1 = LimitOrder(
    timestamp=datetime(2012, 6, 21, 12, 0),
    price=30.1,
    volume=1000,
    direction="buy",
    ticker="MSFT",
    external_id=50,
    internal_id=None,
    is_external=True,
)
LIMIT_2 = LimitOrder(
    timestamp=datetime(2012, 6, 21, 12, 1),
    price=30.1,
    volume=200,
    direction="buy",
    ticker="MSFT",
    external_id=100,
    internal_id=None,
    is_external=True,
)
LIMIT_3 = LimitOrder(
    timestamp=datetime(2012, 6, 21, 12, 1),
    price=30.2,
    volume=200,
    direction="buy",
    ticker="MSFT",
    external_id=110,
    internal_id=None,
    is_external=True,
)
LIMIT_4 = LimitOrder(
    timestamp=datetime(2012, 6, 21, 12, 1),
    price=30.3,
    volume=200,
    direction="sell",
    ticker="MSFT",
    external_id=None,
    internal_id=None,
    is_external=False,
)
LIMIT_5 = LimitOrder(
    timestamp=datetime(2012, 6, 21, 12, 1),
    price=30.2,
    volume=300,
    direction="sell",
    ticker="MSFT",
    external_id=None,
    internal_id=None,
    is_external=False,
)
CANCELLATION_1 = Cancellation(
    timestamp=datetime(2012, 6, 21, 12, 1),
    price=30.1,
    volume=200,
    direction="buy",
    ticker="MSFT",
    external_id=50,
    internal_id=None,
    is_external=True,
)
CANCELLATION_2 = Cancellation(
    timestamp=datetime(2012, 6, 21, 12, 2),
    price=30.1,
    volume=1100,
    direction="buy",
    ticker="MSFT",
    external_id=50,
    internal_id=None,
    is_external=False,
)
CANCELLATION_3 = Cancellation(
    timestamp=datetime(2012, 6, 21, 12, 3),
    price=30.1,
    volume=200,
    direction="buy",
    ticker="MSFT",
    external_id=110,
    internal_id=None,
    is_external=False,
)
DELETION_1 = Deletion(
    timestamp=datetime(2012, 6, 21, 12, 1),
    price=30.1,
    volume=1000,
    direction="buy",
    ticker="MSFT",
    external_id=50,
    internal_id=None,
    is_external=True,
)
DELETION_2 = Deletion(
    timestamp=datetime(2012, 6, 21, 12, 2),
    price=30.1,
    volume=None,
    direction="buy",
    ticker="MSFT",
    external_id=100,
    internal_id=None,
    is_external=True,
)
DELETION_3 = Deletion(
    timestamp=datetime(2012, 6, 21, 12, 3),
    price=30.1,
    volume=200,
    direction="buy",
    ticker="MSFT",
    external_id=110,
    internal_id=None,
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
    is_external=False,
)
