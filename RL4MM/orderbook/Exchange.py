import warnings
from collections import deque
from copy import copy, deepcopy
from dataclasses import dataclass

import numpy as np
from sortedcontainers import SortedDict
from typing import Optional, List, Literal

from RL4MM.orderbook.OrderIDConvertor import OrderIdConvertor
from RL4MM.orderbook.models import Orderbook, Order, OrderType


class EmptyOrderbookError(Exception):
    pass


class CancellationVolumeExceededError(Exception):
    pass


@dataclass
class Exchange:
    ticker: str = "MSFT"
    orderbook: Orderbook = None  # type: ignore

    def __post_init__(self):
        self.orderbook = self.orderbook or self.get_empty_orderbook()
        assert self.orderbook["ticker"] == self.ticker, "The orderbook ticker must agree with the exchange ticker."
        self.order_id_convertor = OrderIdConvertor()
        self.name = "NASDAQ"

    def process_order(self, order: Order) -> Optional[List[Order]]:
        if order.type == OrderType.LIMIT:
            return self.submit_order(order)
        elif order.type == OrderType.MARKET:
            return self.execute_order(order)
        elif order.type == OrderType.CANCELLATION:
            return self.remove_order(order)
        elif order.type == OrderType.DELETION:
            return self.remove_order(order)
        else:
            raise NotImplementedError

    def submit_order(self, order: Order) -> Optional[List[Order]]:
        self._check_order_type(order, [OrderType.LIMIT])
        if self._does_order_cross_spread(order):
            return self.execute_order(order)  # Execute against orders already in the book
        order = self.order_id_convertor.add_internal_id_to_order_and_track(order)
        try:
            self.orderbook[order.direction][order.price].append(order)
        except KeyError:
            self.orderbook[order.direction][order.price] = deque([order])
        return None

    def execute_order(self, order: Order) -> List[Order]:
        self._check_order_type(order, [OrderType.MARKET, OrderType.LIMIT])
        executed_orders = list()
        remaining_volume = order.volume
        while remaining_volume > 0 and self._does_order_cross_spread(order):
            best_limit_order = self._get_highest_priority_matching_order(order)
            volume_to_execute = min(remaining_volume, best_limit_order.volume)
            executed_order = self._reduce_order_with_queue_position(
                order_price=best_limit_order.price,
                order_direction=best_limit_order.direction,
                queue_position=0,
                volume_to_remove=volume_to_execute,
            )
            executed_orders.append(executed_order)
            remaining_volume -= volume_to_execute
        if remaining_volume > 0:
            remaining_order = copy(order)
            remaining_order.volume = remaining_volume
            self.submit_order(remaining_order)  # submit a limit order with the remaining volume
        return executed_orders

    def remove_order(self, order: Order) -> List[Order]:
        self._check_order_type(order, [OrderType.CANCELLATION, OrderType.DELETION])
        queue_position = self._find_queue_position(order)
        if queue_position is None:
            if self.orderbook[order.direction][order.price][0].internal_id == -1:  # Initial orders remain in book
                assert order.volume is not None, "If attempting to delete an initial order, a volume must be provided."
                # NOTE: here, we are assuming that none of the order trying to be cancelled/deleted has been filled!
                order.internal_id = -1
                queue_position = 0
            else:  # trying to remove order that has already been filled
                return None
        elif order.type is OrderType.DELETION:
            order.volume = self.orderbook[order.direction][order.price][queue_position].volume
        self._reduce_order_with_queue_position(order.price, order.direction, queue_position, order.volume)
        return None

    def get_empty_orderbook(self):
        return Orderbook(bid=SortedDict(), ask=SortedDict(), ticker=self.ticker)

    @property
    def best_ask_price(self):
        return next(iter(self.orderbook["ask"].keys()), np.infty)

    @property
    def best_bid_price(self):
        return next(reversed(self.orderbook["bid"]), 0)

    def get_initial_orderbook_from_orders(self, orders: List[Order]) -> List[Order]:
        assert all(order.internal_id == -1 for order in orders), "internal_ids of orders in the initial book must be -1"
        orderbook = self.get_empty_orderbook()
        for order in orders:
            orderbook[order.direction][order.price] = deque([order])
        return orderbook

    def _get_highest_priority_matching_order(self, order: Order):
        opposite_direction = "ask" if order.direction == "bid" else "bid"
        best_price = self.best_ask_price if opposite_direction == "ask" else self.best_bid_price
        try:
            return self.orderbook[opposite_direction][best_price][0]
        except KeyError:
            raise EmptyOrderbookError(f"Trying take liquidity from empty {opposite_direction} side of the book.")

    def _execute_entire_order(self, best_limit_order: Order, remaining_volume: int):
        return

    def _does_order_cross_spread(self, order: Order):
        if order.type == OrderType.MARKET:
            return True
        if order.direction == "bid":
            return order.price >= self.best_ask_price
        if order.direction == "ask":
            return order.price <= self.best_bid_price

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

    def _reduce_order_with_queue_position(
        self, order_price: float, order_direction: Literal["bid", "ask"], queue_position: int, volume_to_remove: int
    ) -> Order:
        order_to_partially_remove = self.orderbook[order_direction][order_price][queue_position]
        if volume_to_remove > order_to_partially_remove.volume:
            raise CancellationVolumeExceededError(
                f"Attempting to remove volume {volume_to_remove} from order of size {order_to_partially_remove.volume}."
            )
        removed_order = deepcopy(self.orderbook[order_direction][order_price][queue_position])
        removed_order.volume = volume_to_remove
        order_to_partially_remove.volume -= volume_to_remove
        self._clear_empty_orders_and_prices(order_price, order_direction, queue_position)
        return removed_order

    def _clear_empty_orders_and_prices(self, price: float, direction: Literal["bid", "ask"], queue_position: int):
        if self.orderbook[direction][price][queue_position].volume == 0:
            order_to_remove = self.orderbook[direction][price][queue_position]
            if order_to_remove.is_external:
                self.order_id_convertor.remove_external_order_id(order_to_remove.external_id)  # Stop tracking order_id
            del self.orderbook[direction][price][queue_position]
        if len(self.orderbook[direction][price]) == 0:
            self.orderbook[direction].pop(price)

    @staticmethod
    def _check_order_type(order: Order, order_types: List[OrderType]):
        assert order.type in order_types, f"Expected order types {order_types}"
