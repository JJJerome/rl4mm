import warnings
from collections import deque
from dataclasses import dataclass
from typing import Optional, List

from RL4MM.orderbook.OrderIDConvertor import OrderIdConvertor
from RL4MM.orderbook.models import Orderbook, Order, OrderType


@dataclass
class Exchange:
    orderbook: Orderbook
    order_id_convertor: OrderIdConvertor
    ticker: str
    name: str = "NASDAQ"

    def __post_init__(self):
        assert self.orderbook["ticker"] == self.ticker, "The Exchange orderbook's ticker must agree with the Exchange."

    def submit_order(self, order: Order):
        order = self.order_id_convertor.add_internal_id_to_order_and_track(order)
        try:
            self.orderbook[order.direction][order.price].append(order)
        except KeyError:
            self.orderbook[order.direction][order.price] = deque([order])
        return order

    def execute_order(self, order: Order) -> List[Order]:
        self._assert_order_type(order, OrderType.EXECUTION)
        executed_orders = list()
        remaining_size = order.size
        while remaining_size > 0:
            best_limit_order = self.orderbook[order.direction][order.price][0]
            if remaining_size < best_limit_order.size:
                executed_orders.append(self._partially_remove_order_with_queue_position(order, 0))
                remaining_size = 0
            elif remaining_size >= best_limit_order.size:
                self.orderbook[order.direction][order.price].popleft()
                executed_orders.append(best_limit_order)
                remaining_size -= best_limit_order.size
                if order.external_id is not None:
                    self.order_id_convertor.remove_external_order_id(order.external_id)  # Stop tracking order_id
        return executed_orders

    def cancel_order(self, order: Order) -> Optional[Order]:
        self._assert_order_type(order, OrderType.CANCELLATION)
        queue_position = self._find_queue_position(order)
        if not queue_position:
            if self.orderbook[order.direction][order.price][0].internal_id == -1:  # Initial orders remaining in level
                queue_position = 0
            else:  # trying to cancel order that has already been filled
                return None
        return self._partially_remove_order_with_queue_position(order, queue_position)

    def delete_order(self, order: Order) -> Optional[Order]:
        self._assert_order_type(order, OrderType.DELETION)
        queue_position = self._find_queue_position(order)
        if queue_position is not None:
            del self.orderbook[order.direction][order.price][queue_position]
        else:
            if self.orderbook[order.direction][order.price][0].internal_id == -1:  # Initial orders remaining in level
                self._partially_remove_order_with_queue_position(order, 0)
                if self.orderbook[order.direction][order.price][0].size == 0:
                    self.orderbook[order.direction][order.price].popleft()
            else:
                return None
        return order

    def _find_queue_position(self, order: Order) -> Optional[int]:
        internal_id = self.order_id_convertor.get_internal_order_id(order)
        book_level = self.orderbook[order.direction][order.price]
        left, right = 0, len(book_level) - 1
        while left <= right:
            middle = (left + right) // 2
            middle_id: int = book_level[middle].internal_id  # type: ignore
            if middle_id == internal_id:
                return middle
            if middle_id < internal_id:
                left = middle + 1
            elif middle_id > internal_id:
                right = middle - 1
        warnings.warn(f"No order found with internal_id = {internal_id}")
        return None

    def _partially_remove_order_with_queue_position(self, order: Order, queue_position: int):
        self.orderbook[order.direction][order.price][queue_position].size -= order.size
        return order

    @staticmethod
    def _assert_order_type(order: Order, order_type: OrderType):
        assert order.type is order_type, f"Attempting {order_type} of a {order.type} order."
