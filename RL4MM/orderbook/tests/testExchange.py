from collections import deque
from copy import copy, deepcopy
from unittest import TestCase

from sortedcontainers import SortedDict

from RL4MM.orderbook.Exchange import Exchange, CancellationVolumeExceededError
from RL4MM.orderbook.models import Orderbook
from RL4MM.orderbook.tests.mock_orders import (
    LIMIT_1,
    LIMIT_2,
    LIMIT_3,
    LIMIT_4,
    LIMIT_5,
    CANCELLATION_1,
    CANCELLATION_2,
    CANCELLATION_3,
    DELETION_1,
    DELETION_2,
    DELETION_3,
    MARKET_1,
    MARKET_2,
    MARKET_3,
)

cancellations = [CANCELLATION_1, CANCELLATION_2, CANCELLATION_3]
deletions = [DELETION_1, DELETION_2, DELETION_3]
TICKER = "MSFT"

submission_1 = deepcopy(LIMIT_1)
submission_2 = deepcopy(LIMIT_2)
submission_3 = deepcopy(LIMIT_3)
submission_4 = deepcopy(LIMIT_4)
submission_1.internal_id = 1
submission_2.internal_id = 2
submission_3.internal_id = 3
submission_4.internal_id = 4


class TestExchange(TestCase):
    def test_post_init(self):
        exchange = Exchange(TICKER)
        self.assertEqual(exchange.name, "NASDAQ")
        empty_orderbook = {"buy": SortedDict(), "sell": SortedDict(), "ticker": TICKER}
        self.assertEqual(exchange.orderbook, empty_orderbook)

    def test_get_initial_orderbook_from_orders(self):
        exchange = Exchange()
        initial_orders = [deepcopy(LIMIT_1), deepcopy(LIMIT_3)]
        with self.assertRaises(AssertionError):  # internal Order ID for initial orders must be -1
            exchange.get_initial_orderbook_from_orders(initial_orders)
        for order in initial_orders:
            order.internal_id = -1
        initial_orderbook = exchange.get_initial_orderbook_from_orders(initial_orders)
        expected = Orderbook(
            buy=SortedDict({30.1: deque([initial_orders[0]]), 30.2: deque([initial_orders[1]])}),
            sell=SortedDict(),
            ticker=TICKER,
        )
        self.assertEqual(initial_orderbook, expected)

    def test_order_tracking(self):
        exchange = Exchange(TICKER)
        for order in LIMIT_1, LIMIT_2, LIMIT_3, LIMIT_4:
            exchange.submit_order(order)
        count = 1
        for direction in ["buy", "sell"]:
            for level in exchange.orderbook[direction].keys():  # type: ignore
                for order in exchange.orderbook[direction][level]:  # type: ignore
                    self.assertEqual(count, order.internal_id)
                    count += 1
        for count, order in enumerate([LIMIT_1, LIMIT_2, LIMIT_3, submission_4]):
            self.assertEqual(count + 1, exchange.order_id_convertor.get_internal_order_id(order))

    def test_submit_order_basic(self):
        exchange = Exchange(TICKER)
        for order in [LIMIT_1, LIMIT_2, LIMIT_3, LIMIT_4]:
            exchange.submit_order(order)
        expected = self.get_demo_orderbook()
        actual = exchange.orderbook
        self.assertEqual(expected, actual)

    def test_execute_market_order(self):
        orderbook = self.get_demo_orderbook()
        exchange = Exchange(TICKER, deepcopy(orderbook))
        # Execute first order
        exchange.execute_order(MARKET_1)
        orderbook["buy"].pop(30.2)
        orderbook["buy"][30.1][0].volume -= MARKET_1.volume - LIMIT_3.volume
        self.assertEqual(orderbook, exchange.orderbook)
        # Execute second order
        exchange.execute_order(MARKET_2)
        orderbook["buy"][30.1].popleft()
        self.assertEqual(orderbook, exchange.orderbook)
        # Execute third order
        exchange.execute_order(MARKET_3)
        orderbook["sell"].pop(30.3)
        self.assertEqual(orderbook, exchange.orderbook)

    def test_cancel_order_basic(self):
        exchange = Exchange(TICKER)
        modified_submission = deepcopy(submission_1)
        modified_submission.volume = 1000 - 200
        expected = Orderbook(
            buy=SortedDict({30.1: deque([modified_submission])}),
            sell=SortedDict({}),
            ticker=TICKER,
        )
        exchange.submit_order(LIMIT_1)
        exchange.remove_order(CANCELLATION_1)
        self.assertEqual(expected, exchange.orderbook)

    def test_large_cancellation_fails(self):
        exchange = Exchange(TICKER)
        exchange.submit_order(LIMIT_1)
        exchange.remove_order(CANCELLATION_1)
        self.assertRaises(CancellationVolumeExceededError)

    def test_delete_order_basic(self):
        exchange = Exchange(TICKER)
        exchange.submit_order(LIMIT_1)
        exchange.remove_order(DELETION_1)
        expected = exchange.get_empty_orderbook()
        self.assertEqual(expected, exchange.orderbook)

    def test_delete_order_with_no_volume_given(self):
        exchange = Exchange(TICKER)
        exchange.submit_order(LIMIT_2)
        exchange.remove_order(DELETION_2)
        expected = exchange.get_empty_orderbook()
        self.assertEqual(expected, exchange.orderbook)

    def test_submit_limit_order_crossing_spread(self):
        exchange = Exchange(TICKER, self.get_demo_orderbook())
        submission_5 = copy(LIMIT_5)
        submission_5.internal_id = 1  # all other orders are already being tracked
        exchange.submit_order(submission_5)
        partially_filled_submission = copy(submission_5)
        partially_filled_submission.volume = LIMIT_5.volume - LIMIT_3.volume
        expected = Orderbook(
            buy=SortedDict({30.1: deque([submission_1, submission_2])}),
            sell=SortedDict({30.2: deque([partially_filled_submission]), 30.3: deque([submission_4])}),
            ticker=TICKER,
        )
        self.assertEqual(expected, exchange.orderbook)

    @staticmethod
    def get_demo_orderbook():
        orderbook = Orderbook(
            buy=SortedDict({30.1: deque([submission_1, submission_2]), 30.2: deque([submission_3])}),
            sell=SortedDict({30.3: deque([submission_4])}),
            ticker=TICKER,
        )
        return deepcopy(orderbook)


if __name__ == "__main__":
    import nose2

    nose2.main()
