from RL4MM.orderbook.models import MarketOrder, LimitOrder, Cancellation, Deletion, OrderDict


def create_order(order_type: str, order_dict: OrderDict):
    order_creator = _get_order_creator(order_type)
    return order_creator(order_dict)


def _get_order_creator(order_type: str):
    if order_type == "market":
        return _create_market_order
    elif order_type == "limit":
        return _create_limit_order
    elif order_type == "cancellation":
        return _create_cancellation
    elif order_type == "deletion":
        return _create_deletion


def _create_market_order(order_dict: OrderDict):
    market_dict = dict(order_dict)
    market_dict.pop("price")
    return MarketOrder(**market_dict)


def _create_limit_order(order_dict: OrderDict):
    return LimitOrder(**order_dict)


def _create_cancellation(order_dict: OrderDict):
    return Cancellation(**order_dict)


def _create_deletion(order_dict: OrderDict):
    return Deletion(**order_dict)
