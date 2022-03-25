from copy import copy
from datetime import datetime, timedelta
from unittest import TestCase

from RL4MM.orderbook.OrderIDConvertor import OrderIdConvertor
from RL4MM.orderbook.models import Order, OrderType

INTERNAL_TEST_SUBMISSION = Order(
    timestamp=datetime(2020, 1, 1, 12, 0),
    price=100,
    size=10,
    direction="bid",
    type=OrderType.SUBMISSION,
    ticker="MSFT",
    is_external=False,
)

EXTERNAL_TEST_SUBMISSION = copy(INTERNAL_TEST_SUBMISSION)
EXTERNAL_TEST_SUBMISSION.external_id = 123
EXTERNAL_TEST_SUBMISSION.is_external = True

INTERNAL_TEST_CANCELLATION = copy(INTERNAL_TEST_SUBMISSION)
INTERNAL_TEST_CANCELLATION.timestamp += timedelta(minutes=1)
INTERNAL_TEST_CANCELLATION.internal_id = 1
INTERNAL_TEST_CANCELLATION.volume = 5

EXTERNAL_TEST_CANCELLATION = copy(INTERNAL_TEST_CANCELLATION)
EXTERNAL_TEST_CANCELLATION.external_id = 123
EXTERNAL_TEST_CANCELLATION.is_external = True


class TestOrderIdConvertor(TestCase):
    def setUp(self) -> None:
        self.convertor = OrderIdConvertor()

    def test_add_internal_id_to_order_and_track(self):
        self.convertor.reset()
        self.UPDATED_INTERNAL_ORDER = self.convertor.add_internal_id_to_order_and_track(INTERNAL_TEST_SUBMISSION)
        self.UPDATED_EXTERNAL_ORDER = self.convertor.add_internal_id_to_order_and_track(EXTERNAL_TEST_SUBMISSION)
        self.assertEqual(self.UPDATED_INTERNAL_ORDER.internal_id, 1)
        self.assertEqual(self.UPDATED_EXTERNAL_ORDER.internal_id, 2)

    def test_get_internal_order_id(self):
        self.test_add_internal_id_to_order_and_track()  # We need to add the orders before looking them up.
        actual_internal_id_1 = self.convertor.get_internal_order_id(INTERNAL_TEST_CANCELLATION)
        actual_internal_id_2 = self.convertor.get_internal_order_id(EXTERNAL_TEST_CANCELLATION)
        self.assertEqual(actual_internal_id_1, 1)
        self.assertEqual(actual_internal_id_2, 2)

    def test_remove_external_order_id(self):
        self.test_add_internal_id_to_order_and_track()  # We need to add the orders before removing their keys
        self.convertor.remove_external_order_id(external_id=EXTERNAL_TEST_SUBMISSION.external_id)
        with self.assertRaises(KeyError):
            self.convertor.external_to_internal_lookup[EXTERNAL_TEST_SUBMISSION.external_id]
