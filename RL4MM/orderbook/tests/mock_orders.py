from datetime import datetime

from RL4MM.orderbook.models import Order, OrderType

LIMIT_1 = Order(
    timestamp=datetime(2012, 6, 21, 12, 0),
    price=30.1,
    volume=1000,
    direction="bid",
    ticker="MSFT",
    external_id=50,
    internal_id=None,
    type=OrderType.LIMIT,
    is_external=True,
)
LIMIT_2 = Order(
    timestamp=datetime(2012, 6, 21, 12, 1),
    price=30.1,
    volume=200,
    direction="bid",
    ticker="MSFT",
    external_id=100,
    internal_id=None,
    type=OrderType.LIMIT,
    is_external=True,
)
LIMIT_3 = Order(
    timestamp=datetime(2012, 6, 21, 12, 1),
    price=30.2,
    volume=200,
    direction="bid",
    ticker="MSFT",
    external_id=110,
    internal_id=None,
    type=OrderType.LIMIT,
    is_external=True,
)
LIMIT_4 = Order(
    timestamp=datetime(2012, 6, 21, 12, 1),
    price=30.3,
    volume=200,
    direction="ask",
    ticker="MSFT",
    external_id=None,
    internal_id=None,
    type=OrderType.LIMIT,
    is_external=False,
)
LIMIT_5 = Order(
    timestamp=datetime(2012, 6, 21, 12, 1),
    price=30.2,
    volume=300,
    direction="ask",
    ticker="MSFT",
    external_id=None,
    internal_id=None,
    type=OrderType.LIMIT,
    is_external=False,
)
CANCELLATION_1 = Order(
    timestamp=datetime(2012, 6, 21, 12, 1),
    price=30.1,
    volume=200,
    direction="bid",
    ticker="MSFT",
    external_id=50,
    internal_id=None,
    type=OrderType.CANCELLATION,
    is_external=True,
)
CANCELLATION_2 = Order(
    timestamp=datetime(2012, 6, 21, 12, 2),
    price=30.1,
    volume=1100,
    direction="bid",
    ticker="MSFT",
    external_id=50,
    internal_id=None,
    type=OrderType.CANCELLATION,
    is_external=False,
)
CANCELLATION_3 = Order(
    timestamp=datetime(2012, 6, 21, 12, 3),
    price=30.1,
    volume=200,
    direction="bid",
    ticker="MSFT",
    external_id=110,
    internal_id=None,
    type=OrderType.CANCELLATION,
    is_external=False,
)
DELETION_1 = Order(
    timestamp=datetime(2012, 6, 21, 12, 1),
    price=30.1,
    volume=1000,
    direction="bid",
    ticker="MSFT",
    external_id=50,
    internal_id=None,
    type=OrderType.DELETION,
    is_external=True,
)
DELETION_2 = Order(
    timestamp=datetime(2012, 6, 21, 12, 2),
    price=30.1,
    volume=None,
    direction="bid",
    ticker="MSFT",
    external_id=100,
    internal_id=None,
    type=OrderType.DELETION,
    is_external=True,
)
DELETION_3 = Order(
    timestamp=datetime(2012, 6, 21, 12, 3),
    price=30.1,
    volume=200,
    direction="bid",
    ticker="MSFT",
    external_id=110,
    internal_id=None,
    type=OrderType.DELETION,
    is_external=False,
)
MARKET_1 = Order(
    timestamp=datetime(2012, 6, 21, 12, 1),
    price=None,
    volume=1000,
    direction="ask",
    ticker="MSFT",
    external_id=50,
    internal_id=None,
    type=OrderType.MARKET,
    is_external=False,
)
MARKET_2 = Order(
    timestamp=datetime(2012, 6, 21, 12, 2),
    price=None,
    volume=200,
    direction="ask",
    ticker="MSFT",
    external_id=100,
    internal_id=None,
    type=OrderType.MARKET,
    is_external=True,
)
MARKET_3 = Order(
    timestamp=datetime(2012, 6, 21, 12, 3),
    price=None,
    volume=200,
    direction="bid",
    ticker="MSFT",
    external_id=110,
    internal_id=None,
    type=OrderType.MARKET,
    is_external=False,
)
