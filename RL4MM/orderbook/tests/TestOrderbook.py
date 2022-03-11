from datetime import datetime
from unittest import TestCase

from RL4MM.orderbook.models import Order, OrderType

SUBMISSION_1 = Order(
    timestamp=datetime(2012, 6, 21, 12, 0),
    price=30.1,
    size=1000,
    direction="bid",
    id_=-1,
    type=OrderType.SUBMISSION,
    is_internal=False,
)

SUBMISSION_1 = Order(
    timestamp=datetime(2012, 6, 21, 12, 1),
    price=30.1,
    size=200,
    direction="bid",
    id_=100,
    type=OrderType.SUBMISSION,
    is_internal=False,
)

SUBMISSION_3 = Order(
    timestamp=datetime(2012, 6, 21, 12, 1),
    price=30.1,
    size=200,
    direction="bid",
    id_=100,
    type=OrderType.SUBMISSION,
    is_internal=False,
)


class TestOrder(TestCase):
    def test_lt(self):
        pass
