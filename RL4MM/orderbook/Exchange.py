import warnings
from collections import deque
from copy import copy
from dataclasses import dataclass
from sortedcontainers import SortedDict
from typing import Optional, List

from RL4MM.orderbook.OrderIDConvertor import OrderIdConvertor
from RL4MM.orderbook.models import Orderbook, Order, OrderType


@dataclass
class Exchange:
    ticker: str = "MSFT"
    orderbook: Orderbook = None  # type: ignore

    def __post_init__(self):
        self.orderbook = self.orderbook or self.get_empty_orderbook()
        assert self.orderbook["ticker"] == self.ticker, "The Exchange orderbook's ticker must agree with the Exchange."
        self.order_id_convertor = OrderIdConvertor()
        self.name = "NASDAQ"

    def process_order(self, order: Order):
        if order.external_id == 10213347:
            print("boop")
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
                executed_order = self._partially_remove_order(best_price, order.direction, 0, remaining_volume)
                executed_orders.append(executed_order)
                remaining_volume = 0
            elif remaining_volume >= best_limit_order.volume:
                self.orderbook[order.direction][best_price].popleft()
                if not self.orderbook[order.direction][best_price]:
                    del self.orderbook[order.direction][best_price]  # If price level is empty, delete from orderbook
                executed_orders.append(best_limit_order)
                remaining_volume -= best_limit_order.volume
                if best_limit_order.is_external:
                    self.order_id_convertor.remove_external_order_id(order.external_id)  # Stop tracking order_id
        if 10213347 in {order.external_id for order in executed_orders}:
            print("boop")
        return executed_orders

    def cancel_order(self, order: Order) -> List[Order]:
        self._assert_order_type(order, OrderType.CANCELLATION)
        queue_position = self._find_queue_position(order)
        if not queue_position:
            if self.orderbook[order.direction][order.price][0].internal_id == -1:  # Initial orders remaining in level
                queue_position = 0
            else:  # trying to cancel order that has already been filled
                return list()
        return [self._partially_remove_order(order.price, order.direction, queue_position, order.volume)]

    def delete_order(self, order: Order) -> List[Order]:
        self._assert_order_type(order, OrderType.DELETION)
        queue_position = self._find_queue_position(order)
        if queue_position is None:
            try:
                oldest_order = self.orderbook[order.direction][order.price][0].internal_id
            except KeyError:
                return list()
            if oldest_order == -1:  # Initial orders remaining in level
                self._partially_remove_order(order.price, order.direction, 0, order.volume)
                if self.orderbook[order.direction][order.price][0].volume == 0:
                    self.orderbook[order.direction][order.price].popleft()
            else:
                return list()  # If non-existent order is "deleted", nothing happens
        elif queue_position == 0:
            self.orderbook[order.direction][order.price].popleft()
        elif queue_position > 0:
            del self.orderbook[order.direction][order.price][queue_position]
            if order.is_external:
                self.order_id_convertor.remove_external_order_id(order.external_id)  # Stop tracking order_id
        else:
            raise NotImplementedError
        if len(self.orderbook[order.direction][order.price]) == 0:
            self.orderbook[order.direction].pop(order.price)
        return [order]

    def get_initial_orderbook_from_orders(self, orders: List[Order]) -> List[Order]:
        assert all(order.internal_id == -1 for order in orders), "internal_ids of orders in the initial book must be -1"
        orderbook = self.get_empty_orderbook()
        for order in orders:
            orderbook[order.direction][order.price] = deque([order])
        return orderbook

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

    def _partially_remove_order(
        self, order_price: float, order_direction: str, queue_position: int, volume_to_remove: int
    ):
        self.orderbook[order_direction][order_price][queue_position].volume -= volume_to_remove
        removed_order = copy(self.orderbook[order_direction][order_price][queue_position])
        removed_order.volume = volume_to_remove
        return removed_order

    @staticmethod
    def _assert_order_type(order: Order, order_type: OrderType):
        assert order.type is order_type, f"Attempting {order_type} of a {order.type} order."
