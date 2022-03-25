from copy import copy
from typing import Optional

from RL4MM.orderbook.models import Order, OrderType


class OrderIdConvertor:
    def __init__(self):
        self.external_to_internal_lookup = dict()
        self.counter = 0

    def add_internal_id_to_order_and_track(self, order: Order) -> Order:
        if order.type != OrderType.SUBMISSION:
            raise TypeError("Only submissions can be tracked. Order IDs for other order types are already present.")
        self.counter += 1
        new_order = copy(order)
        new_order.internal_id = self.counter
        if order.is_external:
            self.external_to_internal_lookup[order.external_id] = self.counter
        return new_order

    def get_internal_order_id(self, order: Order) -> Optional[int]:
        if not order.is_external:
            return order.internal_id
        elif order.is_external:
            try:
                return self.external_to_internal_lookup[order.external_id]
            except KeyError:
                return None

    def remove_external_order_id(self, external_id: int) -> None:
        del self.external_to_internal_lookup[external_id]

    def reset(self):
        self.external_to_internal_lookup = dict()
        self.counter = 0
