from collections import deque
from copy import copy
from unittest import TestCase

from sortedcontainers import SortedDict

from RL4MM.orderbook.Exchange import Exchange
from RL4MM.orderbook.models import Orderbook
from RL4MM.orderbook.tests.mock_orders import (
    SUBMISSION_1,
    SUBMISSION_2,
    SUBMISSION_3,
    CANCELLATION_1,
    CANCELLATION_2,
    CANCELLATION_3,
    DELETION_1,
    DELETION_2,
    DELETION_3,
    EXECUTION_1,
    EXECUTION_2,
    EXECUTION_3,
)

submissions = [SUBMISSION_1, SUBMISSION_2, SUBMISSION_3]
cancellations = [CANCELLATION_1, CANCELLATION_2, CANCELLATION_3]
deletions = [DELETION_1, DELETION_2, DELETION_3]
executions = [EXECUTION_1, EXECUTION_2, EXECUTION_3]
TICKER = "MSFT"


class TestExchange(TestCase):
    def test_post_init(self):
        exchange = Exchange(TICKER)
        self.assertEqual(exchange.name, "NASDAQ")
        empty_orderbook = {"bid": SortedDict(), "ask": SortedDict(), "ticker": TICKER}
        self.assertEqual(exchange.orderbook, empty_orderbook)

    def test_submit_order(self):
        exchange = Exchange(TICKER)
        for submission in submissions:
            exchange.submit_order(submission)
        submission_1 = copy(SUBMISSION_1)
        submission_2 = copy(SUBMISSION_2)
        submission_3 = copy(SUBMISSION_3)
        submission_1.internal_id = 1
        submission_2.internal_id = 2
        submission_3.internal_id = 3
        expected = Orderbook(
            bid=SortedDict({30.1: deque([submission_1, submission_2]), 30.2: deque([submission_3])}),
            ask=SortedDict(),
            ticker=TICKER,
        )
        self.assertEqual(exchange.orderbook, expected)

    def test_initialise_orderbook_from_orders(self):
        exchange = Exchange()
        initial_orders = [copy(SUBMISSION_1), copy(SUBMISSION_3)]
        with self.assertRaises(AssertionError):
            exchange.get_initial_orderbook_from_orders(initial_orders)
        for order in initial_orders:
            order.internal_id = -1
        # expected_orderbook = Orderbook()
