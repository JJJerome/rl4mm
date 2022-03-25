from datetime import datetime

from RL4MM.orderbook.models import Order, OrderType

SUBMISSION_1 = Order(
    timestamp=datetime(2012, 6, 21, 12, 0),
    price=30.1,
    size=1000,
    direction="bid",
    ticker="MSFT",
    external_id=-1,
    internal_id=None,
    type=OrderType.SUBMISSION,
    is_external=False,
)
SUBMISSION_2 = Order(
    timestamp=datetime(2012, 6, 21, 12, 1),
    price=30.1,
    size=200,
    direction="bid",
    ticker="MSFT",
    external_id=100,
    internal_id=None,
    type=OrderType.SUBMISSION,
    is_external=False,
)
SUBMISSION_3 = Order(
    timestamp=datetime(2012, 6, 21, 12, 1),
    price=30.2,
    size=200,
    direction="bid",
    ticker="MSFT",
    external_id=110,
    internal_id=None,
    type=OrderType.SUBMISSION,
    is_external=False,
)
SUBMISSION_4 = Order(
    timestamp=datetime(2012, 6, 21, 12, 1),
    price=30.2,
    size=200,
    direction="ask",
    ticker="MSFT",
    external_id=110,
    internal_id=None,
    type=OrderType.SUBMISSION,
    is_external=False,
)
CANCELLATION_1 = Order(
    timestamp=datetime(2012, 6, 21, 12, 1),
    price=30.1,
    size=1000,
    direction="bid",
    ticker="MSFT",
    external_id=50,
    internal_id=None,
    type=OrderType.CANCELLATION,
    is_external=False,
)
CANCELLATION_2 = Order(
    timestamp=datetime(2012, 6, 21, 12, 2),
    price=30.1,
    size=200,
    direction="bid",
    ticker="MSFT",
    external_id=100,
    internal_id=None,
    type=OrderType.CANCELLATION,
    is_external=False,
)
CANCELLATION_3 = Order(
    timestamp=datetime(2012, 6, 21, 12, 3),
    price=30.1,
    size=200,
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
    size=1000,
    direction="bid",
    ticker="MSFT",
    external_id=50,
    internal_id=None,
    type=OrderType.DELETION,
    is_external=False,
)
DELETION_2 = Order(
    timestamp=datetime(2012, 6, 21, 12, 2),
    price=30.1,
    size=200,
    direction="bid",
    ticker="MSFT",
    external_id=100,
    internal_id=None,
    type=OrderType.DELETION,
    is_external=False,
)
DELETION_3 = Order(
    timestamp=datetime(2012, 6, 21, 12, 3),
    price=30.1,
    size=200,
    direction="bid",
    ticker="MSFT",
    external_id=110,
    internal_id=None,
    type=OrderType.DELETION,
    is_external=False,
)
EXECUTION_1 = Order(
    timestamp=datetime(2012, 6, 21, 12, 1),
    price=30.1,
    size=1000,
    direction="bid",
    ticker="MSFT",
    external_id=50,
    internal_id=None,
    type=OrderType.EXECUTION,
    is_external=False,
)
EXECUTION_2 = Order(
    timestamp=datetime(2012, 6, 21, 12, 2),
    price=30.1,
    size=200,
    direction="bid",
    ticker="MSFT",
    external_id=100,
    internal_id=None,
    type=OrderType.EXECUTION,
    is_external=False,
)
EXECUTION_3 = Order(
    timestamp=datetime(2012, 6, 21, 12, 3),
    price=30.1,
    size=200,
    direction="bid",
    ticker="MSFT",
    external_id=110,
    internal_id=None,
    type=OrderType.EXECUTION,
    is_external=False,
)
