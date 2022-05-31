import warnings
from collections import deque
from copy import copy, deepcopy
from dataclasses import dataclass

import numpy as np
from sortedcontainers import SortedDict
from typing import Optional, List, Literal, Union

from RL4MM.orderbook.OrderIDConvertor import OrderIdConvertor
from RL4MM.orderbook.models import Orderbook, Order, LimitOrder, MarketOrder, Cancellation, Deletion


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

    def process_order(self, order: Order) -> Optional[List[LimitOrder]]:
        if isinstance(order, LimitOrder):
            return self.submit_order(order)
        elif isinstance(order, MarketOrder):
            return self.execute_order(order)
        elif isinstance(order, (Cancellation, Deletion)):
            self.remove_order(order)
            return None
        else:
            raise NotImplementedError

    def submit_order(self, order: LimitOrder) -> Optional[List[LimitOrder]]:
        if self._does_order_cross_spread(order):
            return self.execute_order(order)  # Execute against orders already in the book
        order = self.order_id_convertor.add_internal_id_to_order_and_track(order)
        try:
            self.orderbook[order.direction][order.price].append(order)
        except KeyError:
            self.orderbook[order.direction][order.price] = deque([order])
        return None

    def execute_order(self, order: Union[MarketOrder, LimitOrder]) -> List[LimitOrder]:
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
        if remaining_volume > 0 and isinstance(order, LimitOrder):
            remaining_order = copy(order)
            remaining_order.volume = remaining_volume
            self.submit_order(remaining_order)  # submit a limit order with the remaining volume
        return executed_orders

    def remove_order(self, order: Union[Cancellation, Deletion]) -> None:
        queue_position = self._find_queue_position(order)
        if queue_position is None:
            if self.orderbook[order.direction][order.price][0].internal_id == -1:  # Initial orders remain in book
                assert order.volume is not None, "If attempting to delete an initial order, a volume must be provided."
                # NOTE: here, we are assuming that none of the order trying to be cancelled/deleted has been filled!
                order.internal_id = -1
                queue_position = 0
            else:  # trying to remove order that has already been filled
                return None
        elif isinstance(order, Deletion) and order.volume is None:
            order.volume = self.orderbook[order.direction][order.price][queue_position].volume
        self._reduce_order_with_queue_position(order.price, order.direction, queue_position, order.volume)
        return None

    def get_empty_orderbook(self):
        return Orderbook(buy=SortedDict(), sell=SortedDict(), ticker=self.ticker)

    @property
    def best_ask_price(self):
        return next(iter(self.orderbook["sell"].keys()), np.infty)

    @property
    def best_bid_price(self):
        return next(reversed(self.orderbook["buy"]), 0)

    def get_initial_orderbook_from_orders(self, orders: List[LimitOrder]) -> Orderbook:
        assert all(order.internal_id == -1 for order in orders), "internal_ids of orders in the initial book must be -1"
        orderbook = self.get_empty_orderbook()
        for order in orders:
            orderbook[order.direction][order.price] = deque([order])
        return orderbook

    def _get_highest_priority_matching_order(self, order: Order):
        opposite_direction = "sell" if order.direction == "buy" else "buy"
        best_price = self.best_ask_price if opposite_direction == "sell" else self.best_bid_price
        try:
            return self.orderbook[opposite_direction][best_price][0]  # type: ignore
        except KeyError:
            raise EmptyOrderbookError(f"Trying take liquidity from empty {opposite_direction} side of the book.")

    def _does_order_cross_spread(self, order: Union[LimitOrder, MarketOrder]):
        if isinstance(order, MarketOrder):
            return True
        if order.direction == "buy":
            return order.price >= self.best_ask_price
        if order.direction == "sell":
            return order.price <= self.best_bid_price

    def _find_queue_position(self, order: Union[Cancellation, Deletion, LimitOrder]) -> Optional[int]:
        internal_id = order.internal_id or self.order_id_convertor.get_internal_order_id(order)
        if internal_id is None and order.is_external:  # This is due to the external order being submitted before start
            return None
        book_level = self.orderbook[order.direction][order.price]
        left, right = 0, len(book_level) - 1
        while left <= right:
            middle = (left + right) // 2
            middle_id: int = book_level[middle].internal_id
            if middle_id == internal_id:
                return middle
            if middle_id < internal_id:  # type: ignore
                left = middle + 1
            elif middle_id > internal_id:  # type: ignore
                right = middle - 1
        warnings.warn(f"No order found with internal_id = {internal_id}")
        return None

    def _reduce_order_with_queue_position(
        self, order_price: int, order_direction: Literal["buy", "sell"], queue_position: int, volume_to_remove: int
    ) -> LimitOrder:
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

    def _clear_empty_orders_and_prices(self, price: int, direction: Literal["buy", "sell"], queue_position: int):
        if self.orderbook[direction][price][queue_position].volume == 0:
            order_to_remove = self.orderbook[direction][price][queue_position]
            if order_to_remove.is_external:
                self.order_id_convertor.remove_external_order_id(order_to_remove.external_id)  # Stop tracking order_id
            del self.orderbook[direction][price][queue_position]
        if len(self.orderbook[direction][price]) == 0:
            self.orderbook[direction].pop(price)
