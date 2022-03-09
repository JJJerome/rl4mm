from typing import Deque, Literal, OrderedDict, Tuple, TypedDict

import warnings

from collections import deque
from enum import Enum

from dataclasses import dataclass
from datetime import datetime


class OrderType(Enum):
    SUBMISSION = "submission"
    CANCELLATION = "cancellation"
    DELETION = "deletion"
    EXECUTION = "execution_visible"


@dataclass
class Order:
    timestamp: datetime
    price: float
    size: int
    direction: Literal["bid", "ask"]
    id_: int
    type: OrderType
    is_internal: bool = False


class BookData(TypedDict):
    bid: OrderedDict[float, Deque[Order]]
    ask: OrderedDict[float, Deque[Order]]


@dataclass
class Orderbook:
    data: BookData
    ticker: str

    def submit_order(self, order: Order):
        self._assert_order_type(order, OrderType.SUBMISSION)
        try:
            self.data[order.direction][order.price].append(order)
        except KeyError:
            self.data[order.direction][order.price] = deque([order])
        return order

    def execute_order(self, order: Order):
        self._assert_order_type(order, OrderType.EXECUTION)
        if order.size == self.data[order.direction][order.price][0]:
            return self.data[order.direction][order.price].popleft()
        else:
            return self.partially_remove_order_with_queue_position(order, 0)

    def cancel_order(self, order: Order):
        self._assert_order_type(order, OrderType.CANCELLATION)
        queue_position, _ = self._find_queue_position(order)
        return self.partially_remove_order_with_queue_position(order, queue_position)

    def delete_order(self, order: Order):
        queue_position, order_id_exists = self._find_queue_position(order)
        if order_id_exists:
            del self.data[order.direction][order.price][queue_position]
        else:
            self.partially_remove_order_with_queue_position(order, 0)
            if self.data[order.direction][order.price][0].size == 0:
                self.data[order.direction][order.price].popleft()
        return order

    def _find_queue_position(self, order: Order) -> Tuple[int, bool]:
        book_level = self.data[order.direction][order.price]
        left, right = 0, len(book_level) - 1
        while left <= right:
            middle = (left + right) // 2
            if book_level[middle].id_ == order.id_:
                return middle, True
            if book_level[middle].id_ < order.id_:
                left = middle + 1
            elif book_level[middle].id_ > order.id_:
                right = middle - 1
        warnings.warn(f"No order found with order ID {order.id_}")
        self._linearly_assert_order_id_absent(order)  # This is just for safety, due to nonincreasing order id!
        assert self.data[order.direction][order.price][0].id_ == -1
        return 0, False

    def partially_remove_order_with_queue_position(self, order: Order, queue_position: int):
        self.data[order.direction][order.price][queue_position].size -= order.size
        return order

    def _linearly_assert_order_id_absent(self, order: Order):
        for o in self.data[order.direction][order.price]:
            assert o.id_ != order.id_, "Binary search has failed due to non increasing order id!!!"

    @staticmethod
    def _assert_order_type(order: Order, order_type: OrderType):
        assert order.type is order_type, f"Attempting {order_type} of a {order.type} order."
