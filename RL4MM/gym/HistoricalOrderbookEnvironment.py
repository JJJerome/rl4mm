from __future__ import annotations
from copy import deepcopy, copy
from datetime import datetime, timedelta
import sys
if sys.version_info[0] == 3 and sys.version_info[1] >= 8:
    from typing import List, TypedDict, Literal
else:
    from typing import List
    from typing_extensions import TypedDict, Literal

import gym
import numpy as np
import pandas as pd

from gym.spaces import Box
from gym.utils import seeding

from RL4MM.extras.orderbook_comparison import convert_to_lobster_format
from RL4MM.features.Features import (
    Feature,
    Spread,
    MicroPrice,
    InternalState,
    MidpriceMove,
    Volatility,
    Inventory,
    TimeRemaining,
)
from RL4MM.gym.action_interpretation.OrderDistributors import OrderDistributor, BetaOrderDistributor
from RL4MM.orderbook.create_order import create_order
from RL4MM.orderbook.models import Orderbook, Order, FillableOrder, OrderDict, Cancellation
from RL4MM.rewards.RewardFunctions import RewardFunction, PnL, InventoryAdjustedPnL
from RL4MM.simulation.HistoricalOrderGenerator import HistoricalOrderGenerator
from RL4MM.simulation.OrderbookSimulator import OrderbookSimulator

MINIMUM_START_DELTA = timedelta(hours=10)  # We ignore the first half an hour of trading, as it is too chaotic
MAXIMUM_END_DELTA = timedelta(hours=11, minutes=20)  # Same for the last half an hour
ORDERBOOK_FREQ = "1S"
ORDERBOOK_MIN_STEP = pd.to_timedelta(ORDERBOOK_FREQ)
LEVELS_FOR_FEATURE_CALCULATION = 1


class Portfolio(TypedDict):
    inventory: int
    cash: float


class HistoricalOrderbookEnvironment(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        features: List[Feature] = None,
        max_beta_param: float = 1000.0,
        ticker: str = "MSFT",
        step_size: timedelta = ORDERBOOK_MIN_STEP,
        episode_length: timedelta = timedelta(minutes=30),
        initial_portfolio: Portfolio = None,
        quote_levels: int = 10,
        num_book_snapshots: int = 10,
        min_date: datetime = datetime(2019, 1, 2),
        max_date: datetime = datetime(2019, 1, 2),
        simulator: OrderbookSimulator = None,
        order_distributor: OrderDistributor = None,
        per_step_reward_function: RewardFunction = PnL(),
        terminal_reward_function: RewardFunction = InventoryAdjustedPnL(inventory_aversion=0.1),
    ):
        super(HistoricalOrderbookEnvironment, self).__init__()
        # Actions are the parameters governing the distribution over levels in the orderbook
        self.action_space = Box(low=0.0, high=max_beta_param, shape=(4,), dtype=np.float64)
        # Observation space is determined by the features used
        self.features = features or [Spread(), MidpriceMove(), Volatility(), Inventory(), TimeRemaining(), MicroPrice()]
        self.observation_space = Box(
            low=np.array([feature.min_value for feature in self.features]),
            high=np.array([feature.max_value for feature in self.features]),
            dtype=np.float64,
        )
        self.max_beta_param = max_beta_param
        self.ticker = ticker
        self.step_size = step_size
        self.quote_levels = quote_levels
        assert episode_length % self.step_size == timedelta(0), "Episode length must be a multiple of step size!"
        self.n_steps = int(episode_length / self.step_size)
        self.episode_length = episode_length
        self.initial_portfolio = initial_portfolio or Portfolio(inventory=0, cash=0)
        book_snapshots = pd.DataFrame([], dtype=int)
        self.internal_state = InternalState(
            inventory=0, cash=0, asset_price=0, book_snapshots=book_snapshots, proportion_of_episode_remaining=1.0
        )
        self.min_date = min_date
        self.max_date = max_date
        self.simulator = simulator or OrderbookSimulator(
            ticker=ticker, order_generators=[HistoricalOrderGenerator(ticker)], n_levels=200
        )
        self.order_distributor = order_distributor or BetaOrderDistributor(self.quote_levels)
        self.per_step_reward_function = per_step_reward_function
        self.terminal_reward_function = terminal_reward_function
        self.now_is = min_date
        self.num_book_snapshots = num_book_snapshots
        self.price = MicroPrice()
        self.reset()

    def reset(self):
        self.now_is = self._get_random_start_time()
        self.simulator.reset_episode(start_date=self.now_is)
        self._reset_internal_state()
        observation = self.get_observation()
        return observation

    def step(self, action: tuple):
        done = False # rllib requires a bool
        info = {}
        internal_orders = self.convert_action_to_orders(action=action)
        filled_orders = self.simulator.forward_step(until=self.now_is + self.step_size, internal_orders=internal_orders)
        previous_internal_state = deepcopy(self.internal_state)
        self.now_is += self.step_size
        self.update_internal_state(filled_orders)
        reward = self.per_step_reward_function.calculate(self.internal_state, previous_internal_state)
        observation = self.get_observation()
        if np.isclose(self.internal_state["proportion_of_episode_remaining"], 0):
            reward = self.terminal_reward_function.calculate(self.internal_state, previous_internal_state)
            done = True # rllib requires a bool
        return observation, reward, done, info 

    def get_observation(self) -> np.ndarray:
        return np.array([feature.calculate(self.internal_state) for feature in self.features])

    def _get_random_start_time(self):
        return self.min_date + timedelta(days=self._random_offset_days()) + self._random_offset_timestamp()

    def update_internal_state(self, filled_orders: List[FillableOrder]):
        self._update_portfolio(filled_orders)
        self._update_book_snapshots(self.central_orderbook)
        self._update_asset_price()
        self._update_time_remaining()

    def convert_action_to_orders(self, action: tuple) -> List[Order]:
        desired_volumes = self.order_distributor.convert_action(action)
        current_volumes = self._get_current_internal_order_volumes()
        difference_in_volumes = {d: desired_volumes[d] - current_volumes[d] for d in ["buy", "sell"]}  # type: ignore
        orders = self._convert_volume_diff_to_orders(difference_in_volumes)
        return orders

    def _convert_volume_diff_to_orders(self, volume_difference: dict[str, np.ndarray]) -> List[Order]:
        best_prices = self._get_best_prices()
        orders = list()
        for side in ["buy", "sell"]:
            for level, price in enumerate(best_prices[side]):
                order_dict = self._get_default_order_dict(side)  # type:ignore
                order_volume = volume_difference[side][level]
                order_dict["price"] = price
                if order_volume > 0:
                    order_dict["volume"] = order_volume
                    order = create_order("limit", order_dict)
                    orders.append(order)
                if order_volume < 0:
                    current_orders = copy(self.internal_orderbook[side][price])
                    while order_volume < 0:
                        worst_order = current_orders[-1]
                        volume_to_remove = min(worst_order.volume, order_volume)
                        order_dict["volume"] = volume_to_remove
                        order_dict["internal_id"] = worst_order.internal_id
                        cancellation = create_order("cancellation", order_dict)
                        orders.append(cancellation)
                        order_volume -= volume_to_remove
            for price in set(self.internal_orderbook[side].keys()) - set(best_prices):
                try:
                    wide_orders = list(self.internal_orderbook[side][price])
                except KeyError:
                    wide_orders = list()
                for order in wide_orders:
                    cancellation = Cancellation(**copy(order.__dict__))
                    orders.append(cancellation)

        return orders

    def _get_default_order_dict(self, direction: Literal["buy", "sell"]) -> OrderDict:
        return OrderDict(
            timestamp=self.now_is,
            price=None,
            volume=None,
            direction=direction,
            ticker=self.ticker,
            internal_id=None,
            external_id=None,
            is_external=False,
        )

    def _update_portfolio(self, filled_orders):
        for order in filled_orders:
            if order.direction == "sell":
                self.internal_state["inventory"] -= order.volume
                self.internal_state["cash"] += order.volume * order.price
            if order.direction == "buy":
                self.internal_state["inventory"] += order.volume
                self.internal_state["cash"] -= order.volume * order.price

    def _update_book_snapshots(self, orderbook: Orderbook) -> None:
        current_book_snapshots = self.internal_state["book_snapshots"][1:]
        new_book_dict = convert_to_lobster_format(orderbook, LEVELS_FOR_FEATURE_CALCULATION)
        new_book_snapshot = pd.DataFrame.from_dict({self.now_is: new_book_dict}).T
        self.internal_state["book_snapshots"] = pd.concat([current_book_snapshots, new_book_snapshot])

    def _update_asset_price(self):
        self.internal_state["asset_price"] = self.price.calculate(self.internal_state)

    def _update_time_remaining(self):
        self.internal_state["proportion_of_episode_remaining"] -= 1 / self.n_steps

    def _get_current_internal_order_volumes(self) -> dict[Literal["buy", "sell"], tuple[np.ndarray]]:
        best_prices = self._get_best_prices()
        internal_volumes = dict()
        for side in ["buy", "sell"]:
            internal_volumes[side] = self._get_volumes_at_prices("buy", best_prices[side], self.internal_orderbook)
        return internal_volumes  # type: ignore

    def _get_best_prices(self):
        best_buy = self.simulator.exchange.best_bid_price
        best_sell = self.simulator.exchange.best_ask_price
        tick_size = self.central_orderbook["tick_size"]
        buy_prices = np.arange(best_buy - self.quote_levels * tick_size, best_buy, tick_size)
        sell_prices = np.arange(best_sell, best_sell + self.quote_levels * tick_size, tick_size)
        return {"buy": buy_prices, "sell": sell_prices}

    def _get_volumes_at_prices(self, direction: Literal["buy", "sell"], price_levels: np.ndarray, orderbook: Orderbook):
        volumes = list()
        for price in price_levels:
            try:
                volumes.append(sum([order.volume for order in orderbook[direction][price]]))
            except KeyError:
                volumes.append(0)
        return np.array(volumes)

    def _get_features_from_underlying(self, book_snapshots: pd.DataFrame):
        return (feature.calculate(book_snapshots) for feature in self.features)

    def _random_offset_timestamp(self):
        max_offset_steps = int((MAXIMUM_END_DELTA - self.episode_length - MINIMUM_START_DELTA) / self.step_size)
        random_offset_steps = np.random.randint(low=0, high=max_offset_steps)
        return MINIMUM_START_DELTA + random_offset_steps * self.step_size

    def _random_offset_days(self):
        return np.random.randint(int((self.max_date.date() - self.min_date.date()) / timedelta(days=1)) + 1)

    def _reset_internal_state(self):
        snapshot_start = self.now_is - self.step_size * self.num_book_snapshots
        book_snapshots = self.simulator.database.get_book_snapshot_series(
            start_date=snapshot_start,
            end_date=self.now_is,
            ticker=self.ticker,
            freq=ORDERBOOK_FREQ,
            n_levels=LEVELS_FOR_FEATURE_CALCULATION,
        )
        self.internal_state = InternalState(
            inventory=self.initial_portfolio["inventory"],
            cash=self.initial_portfolio["cash"],
            asset_price=self.price.calculate_from_current_book(book_snapshots.iloc[-1]),
            book_snapshots=book_snapshots,
            proportion_of_episode_remaining=1.0,
        )

    def render(self, mode="human"):
        pass

    def seed(self, seed=42):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @property
    def central_orderbook(self):
        return self.simulator.exchange.central_orderbook

    @property
    def internal_orderbook(self):
        return self.simulator.exchange.internal_orderbook
