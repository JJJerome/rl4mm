import warnings
from collections import deque
from dataclasses import dataclass
from sortedcontainers import SortedDict
from typing import Optional, List

from RL4MM.orderbook.OrderIDConvertor import OrderIdConvertor
from RL4MM.orderbook.models import Orderbook, Order, OrderType

from datetime import datetime


@dataclass
class Exchange:
    ticker: str
    orderbook: Orderbook = None  # type: ignore
    name: str = "NASDAQ"
    order_id_convertor: OrderIdConvertor = None  # type: ignore

    def __post_init__(self):
        self.orderbook = self.orderbook or self.get_empty_orderbook()
        assert self.orderbook["ticker"] == self.ticker, "The Exchange orderbook's ticker must agree with the Exchange."
        self.order_id_convertor = OrderIdConvertor()

    def process_order(self, order: Order):
        if order.type == OrderType.SUBMISSION:
            return self.submit_order(order)
        elif order.type == OrderType.EXECUTION:
            return self.execute_order(order)
        elif order.type == OrderType.CANCELLATION:
            return self.cancel_order(order)
        elif order.type == OrderType.DELETION:
            return self.delete_order(order)
        else:
            raise NotImplementedError

    def submit_order(self, order: Order) -> List[Order]:
        order = self.order_id_convertor.add_internal_id_to_order_and_track(order)
        try:
            self.orderbook[order.direction][order.price].append(order)
        except KeyError:
            self.orderbook[order.direction][order.price] = deque([order])
        return [order]

    def execute_order(self, order: Order) -> List[Order]:
        self._assert_order_type(order, OrderType.EXECUTION)
        executed_orders = list()
        remaining_volume = order.volume
        while remaining_volume > 0:
            if order.direction == "bid":
                best_price = next(reversed(self.orderbook[order.direction]))  # best price is highest
            elif order.direction == "ask":
                best_price = next(iter(self.orderbook[order.direction].keys()))  # best price is lowest
            best_limit_order = self.orderbook[order.direction][best_price][0]
            if remaining_volume < best_limit_order.volume:
                executed_order = self._partially_remove_order_with_queue_position(order, 0)
                executed_orders.append(executed_order)
                remaining_volume = 0
            elif remaining_volume >= best_limit_order.volume:
                self.orderbook[order.direction][order.price].popleft()
                if not self.orderbook[order.direction][order.price]:
                    del self.orderbook[order.direction][order.price]  # If price level is empty, delete from orderbook
                executed_orders.append(best_limit_order)
                remaining_volume -= best_limit_order.volume
                if best_limit_order.is_external and best_limit_order.internal_id != -1:
                    self.order_id_convertor.remove_external_order_id(order.external_id)  # Stop tracking order_id
        return executed_orders

    def cancel_order(self, order: Order) -> List[Order]:
        self._assert_order_type(order, OrderType.CANCELLATION)
        queue_position = self._find_queue_position(order)
        if not queue_position:
            if self.orderbook[order.direction][order.price][0].internal_id == -1:  # Initial orders remaining in level
                queue_position = 0
            else:  # trying to cancel order that has already been filled
                return list()
        return [self._partially_remove_order_with_queue_position(order, queue_position)]

    def delete_order(self, order: Order) -> List[Order]:
        self._assert_order_type(order, OrderType.DELETION)
        queue_position = self._find_queue_position(order)
        if queue_position is not None:
            del self.orderbook[order.direction][order.price][queue_position]
        else:
            if self.orderbook[order.direction][order.price][0].internal_id == -1:  # Initial orders remaining in level
                self._partially_remove_order_with_queue_position(order, 0)
                if self.orderbook[order.direction][order.price][0].volume == 0:
                    self.orderbook[order.direction][order.price].popleft()
            else:
                return list()
        return [order]

    def initialise_orderbook_from_orders(self, orders: List[Order]) -> List[Order]:
        assert all(order.internal_id == -1 for order in orders), "internal_ids of orders in the initial book must be -1"
        assert len(orders) == len(set([order.price for order in orders])), "each order must have a unique price"
        for order in orders:
            self.orderbook[order.direction][order.price] = deque([order])
        return orders

    def get_empty_orderbook(self):
        return Orderbook(bid=SortedDict(), ask=SortedDict(), ticker=self.ticker)

    def _find_queue_position(self, order: Order) -> Optional[int]:
        internal_id = order.internal_id or self.order_id_convertor.get_internal_order_id(order)
        if internal_id is None and order.is_external:  # This is due to the external order being submitted before start
            return None
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
        self.orderbook[order.direction][order.price][queue_position].volume -= order.volume
        return order

    @staticmethod
    def _assert_order_type(order: Order, order_type: OrderType):
        assert order.type is order_type, f"Attempting {order_type} of a {order.type} order."
