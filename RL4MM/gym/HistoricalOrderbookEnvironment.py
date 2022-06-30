from __future__ import annotations

import warnings
from copy import deepcopy, copy
from datetime import datetime, timedelta
import sys

from RL4MM.database.HistoricalDatabase import HistoricalDatabase
from RL4MM.gym.order_tracking.InfoCalculators import InfoCalculator, SimpleInfoCalculator
from RL4MM.utils.utils import convert_timedelta_to_freq

if sys.version_info[0] == 3 and sys.version_info[1] >= 8:
    from typing import List, TypedDict, Literal, Union, Optional
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
    MidPrice,
)
from RL4MM.gym.action_interpretation.OrderDistributors import OrderDistributor, BetaOrderDistributor
from RL4MM.orderbook.create_order import create_order
from RL4MM.orderbook.models import Orderbook, FillableOrder, OrderDict, Cancellation, LimitOrder
from RL4MM.rewards.RewardFunctions import RewardFunction, InventoryAdjustedPnL
from RL4MM.simulation.HistoricalOrderGenerator import HistoricalOrderGenerator
from RL4MM.simulation.OrderbookSimulator import OrderbookSimulator

ORDERBOOK_FREQ = "1S"
ORDERBOOK_MIN_STEP = pd.to_timedelta(ORDERBOOK_FREQ)
LEVELS_FOR_FEATURE_CALCULATION = 1
MARKET_ORDER_FRACTION_OF_INVENTORY = 0.1


class Portfolio(TypedDict):
    inventory: int
    cash: float


class HistoricalOrderbookEnvironment(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        features: List[Feature] = None,
        max_distribution_param: float = 10.0,  # 1000.0,
        ticker: str = "MSFT",
        step_size: timedelta = ORDERBOOK_MIN_STEP,
        episode_length: timedelta = timedelta(minutes=30),
        initial_portfolio: Portfolio = None,
        min_quote_level: int = 0,
        max_quote_level: int = 10,
        feature_window_size: int = 10,
        min_date: datetime = datetime(2019, 1, 2),
        max_date: datetime = datetime(2019, 1, 2),
        min_start_timedelta: timedelta = timedelta(hours=10),  # Ignore the first half an hour of trading
        max_end_timedelta: timedelta = timedelta(hours=15, minutes=30),  # Same for the last half an hour
        simulator: OrderbookSimulator = None,
        market_order_clearing: bool = False,
        inc_prev_action_in_obs: bool = False,
        max_inventory: int = 100000,
        per_step_reward_function: RewardFunction = InventoryAdjustedPnL(inventory_aversion=10 ** (-4)),
        terminal_reward_function: RewardFunction = InventoryAdjustedPnL(inventory_aversion=0.1),
        info_calculator: InfoCalculator = None,
        order_distributor: OrderDistributor = None,
        concentration: float = 10.0,
        market_order_fraction_of_inventory: Optional[float] = None,
        enter_spread: bool = False,
        save_messages_locally: bool = True,
    ):
        super(HistoricalOrderbookEnvironment, self).__init__()

        # Actions are the parameters governing the distribution over levels in the orderbook
        if concentration is not None or (order_distributor is not None and order_distributor.c is not None):
            assert concentration >= max_distribution_param, "concentration is less than max_distribution_param"
            self.action_space = Box(low=0.0, high=concentration, shape=(2,), dtype=np.float64)  # alpha < kappa for beta
        else:
            self.action_space = Box(low=0.0, high=max_distribution_param, shape=(4,), dtype=np.float64)
        if market_order_clearing:
            low = np.append(self.action_space.low, [0.0])
            high = np.append(self.action_space.high, [max_inventory])
            self.action_space = Box(low=low, high=high, dtype=np.float64)

        # Observation space is determined by the features used
        self.features = features or [Spread(), MidpriceMove(), Volatility(), Inventory(), TimeRemaining(), MicroPrice()]
        self.inc_prev_action_in_obs = inc_prev_action_in_obs
        low_obs = np.array([feature.min_value for feature in self.features])
        high_obs = np.array([feature.max_value for feature in self.features])
        # If the previous action is included in the observation:
        if self.inc_prev_action_in_obs:
            low_obs = np.concatenate((low_obs, self.action_space.low))
            high_obs = np.concatenate((high_obs, self.action_space.high))
        self.observation_space = Box(
            low=low_obs,
            high=high_obs,
            dtype=np.float64,
        )
        self.max_distribution_param = max_distribution_param
        self.ticker = ticker
        self.step_size = step_size
        self.min_quote_level = min_quote_level
        self.max_quote_level = max_quote_level
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
        self.min_start_timedelta = min_start_timedelta
        self.max_end_timedelta = max_end_timedelta
        self.simulator = simulator or OrderbookSimulator(
            ticker=ticker,
            order_generators=[HistoricalOrderGenerator(ticker, HistoricalDatabase(), save_messages_locally)],
            n_levels=200,
            save_messages_locally=save_messages_locally,
            episode_length=episode_length,
        )
        self.order_distributor = order_distributor or BetaOrderDistributor(
            self.max_quote_level - self.min_quote_level, concentration=concentration
        )
        self.market_order_clearing = market_order_clearing
        self.market_order_fraction_of_inventory = market_order_fraction_of_inventory
        self._check_market_order_clearing_well_defined()
        self.per_step_reward_function = per_step_reward_function
        self.terminal_reward_function = terminal_reward_function
        self.now_is = min_date
        self.feature_window_size = feature_window_size
        self.enter_spread = enter_spread
        self.info_calculator = info_calculator or SimpleInfoCalculator(market_order_fraction_of_inventory, enter_spread)
        self.price = MidPrice()
        self._check_params()

    def reset(self):
        self.now_is = self._get_random_start_time()
        self.simulator.reset_episode(start_date=self.now_is)
        self._reset_internal_state()
        if self.inc_prev_action_in_obs:
            return self.get_observation(np.zeros(shape=self.action_space.shape))
        else:
            return self.get_observation()

    def step(self, action: np.ndarray):
        done = False  # rllib requires a bool
        internal_orders = self.convert_action_to_orders(action=action)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            filled = self.simulator.forward_step(
                until=self.now_is + self.step_size, internal_orders=internal_orders  # type: ignore
            )
        current_state = deepcopy(self.internal_state)
        self.now_is += self.step_size
        self.update_internal_state(filled)
        if self.info_calculator is not None:
            info = self.info_calculator.calculate(
                filled_orders=filled, internal_state=self.internal_state, action=action
            )
        next_state = self.internal_state
        reward = self.per_step_reward_function.calculate(current_state, next_state)
        observation = self.get_observation(action) if self.inc_prev_action_in_obs else self.get_observation()
        if np.isclose(self.internal_state["proportion_of_episode_remaining"], 0):
            reward = self.terminal_reward_function.calculate(current_state, next_state)
            done = True
        return observation, reward, done, info

    def get_observation(self, previous_action: np.ndarray = None) -> np.ndarray:
        obs = np.array([feature.calculate(self.internal_state) for feature in self.features])
        if previous_action is not None:
            return np.concatenate((obs, previous_action))
        else:
            return obs

    def _get_random_start_time(self):
        return self._get_random_trading_day() + self._random_offset_timestamp()

    def update_internal_state(self, filled_orders: List[FillableOrder]):
        self._update_portfolio(filled_orders)
        self._update_book_snapshots(self.central_orderbook)
        self._update_asset_price()
        self._update_time_remaining()

    def convert_action_to_orders(self, action: np.ndarray) -> List[Union[Cancellation, LimitOrder]]:
        desired_volumes = self.order_distributor.convert_action(action)
        if self.market_order_clearing and np.abs(self.internal_state["inventory"]) > action[-1]:  # cancel all orders
            desired_volumes = {d: np.zeros(self.max_quote_level - self.min_quote_level) for d in ["buy", "sell"]}  # type: ignore
        current_volumes = self._get_current_internal_order_volumes()
        difference_in_volumes = {d: desired_volumes[d] - current_volumes[d] for d in ["buy", "sell"]}  # type: ignore
        orders = self._volume_diff_to_orders(difference_in_volumes)
        if self.market_order_clearing and np.abs(self.internal_state["inventory"]) > action[-1]:
            # place a market order to reduce inventory to zero
            orders += self._get_inventory_clearing_market_order()
        return orders

    def _volume_diff_to_orders(self, volume_diff: dict[str, np.ndarray]) -> List[Union[Cancellation, LimitOrder]]:
        best_prices = self._get_best_prices()
        orders = list()
        for side in ["buy", "sell"]:
            for level, price in enumerate(best_prices[side]):
                order_dict = self._get_default_order_dict(side)  # type:ignore
                order_volume = volume_diff[side][level]
                order_dict["price"] = price
                if order_volume > 0:
                    order_dict["volume"] = order_volume
                    order = create_order("limit", order_dict)
                    orders.append(order)
                if order_volume < 0:
                    try:
                        current_orders = deepcopy(self.internal_orderbook[side][price])
                    except:
                        raise Exception("DEBUG")
                    while order_volume < 0:
                        worst_order = current_orders[-1]
                        volume_to_remove = min(worst_order.volume, order_volume)
                        order_dict["volume"] = volume_to_remove
                        order_dict["internal_id"] = worst_order.internal_id
                        cancellation = create_order("cancellation", order_dict)
                        orders.append(cancellation)
                        order_volume -= volume_to_remove
                        current_orders.pop()
            for price in set(self.internal_orderbook[side].keys()) - set(best_prices):
                try:
                    wide_orders = list(self.internal_orderbook[side][price])
                except KeyError:
                    wide_orders = list()
                for order in wide_orders:
                    cancellation = Cancellation(**copy(order.__dict__))
                    orders.append(cancellation)
        return orders

    def _get_inventory_clearing_market_order(self):
        inventory = self.internal_state["inventory"]
        order_direction = "buy" if inventory < 0 else "sell"
        order_dict = self._get_default_order_dict(order_direction)  # type:ignore
        order_dict["volume"] = np.abs(inventory) * self.market_order_fraction_of_inventory
        market_order = create_order("market", order_dict)
        return [market_order]

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

    def _update_portfolio(self, filled_orders: List[FillableOrder]):
        for order in filled_orders:
            if order.price is None:
                raise Exception("Cannot update portfolio from a market order with no fill price.")
            elif order.direction == "sell":
                self.internal_state["inventory"] -= order.volume
                self.internal_state["cash"] += order.volume * order.price
            elif order.direction == "buy":
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
            internal_volumes[side] = self._get_volumes_at_prices(side, best_prices[side], self.internal_orderbook)
        return internal_volumes  # type: ignore

    def _get_best_prices(self):
        tick_size = self.central_orderbook["tick_size"]
        if self.enter_spread:
            midprice = (self.simulator.exchange.best_buy_price + self.simulator.exchange.best_sell_price) / 2
            best_buy = int(np.floor(midprice / tick_size) * tick_size)
            best_sell = int(np.ceil(midprice / tick_size) * tick_size)  # TODO: check me when quoting withing spread.
        if not self.enter_spread:
            best_buy = self.simulator.exchange.best_buy_price
            best_sell = self.simulator.exchange.best_sell_price
        buy_prices = np.arange(
            best_buy - (self.max_quote_level - 1) * tick_size,
            best_buy - (self.min_quote_level - 1) * tick_size,
            tick_size,
            dtype=int,
        )
        sell_prices = np.arange(
            best_sell + self.min_quote_level * tick_size,
            best_sell + self.max_quote_level * tick_size,
            tick_size,
            dtype=int,
        )
        return {"buy": buy_prices, "sell": sell_prices}

    def Up_get_volumes_at_prices(self, direction: Literal["buy", "sell"], price_levels: np.ndarray, orderbook: Orderbook):
        volumes = list()
        for price in price_levels:
            try:
                volumes.append(sum(order.volume for order in orderbook[direction][price]))
            except KeyError:
                volumes.append(0)
        return np.array(volumes)

    def _random_offset_timestamp(self):
        max_offset_steps = int(
            (self.max_end_timedelta - self.episode_length - self.min_start_timedelta) / self.step_size
        )
        try:
            random_offset_steps = np.random.randint(low=0, high=max_offset_steps)
        except ValueError:
            random_offset_steps = 0
        return self.min_start_timedelta + random_offset_steps * self.step_size

    def _random_offset_days(self):
        return np.random.randint(int((self.max_date.date() - self.min_date.date()) / timedelta(days=1)) + 1)

    def _get_random_trading_day(self):
        trading_dates = pd.bdate_range(self.min_date, self.max_date)
        return pd.to_datetime(np.random.choice(trading_dates))

    def _reset_internal_state(self):
        snapshot_start = self.now_is - self.step_size * self.feature_window_size
        book_snapshots = self.simulator.database.get_book_snapshot_series(
            start_date=snapshot_start,
            end_date=self.now_is,
            ticker=self.ticker,
            freq=convert_timedelta_to_freq(self.step_size),
            n_levels=LEVELS_FOR_FEATURE_CALCULATION,
        )
        self.internal_state = InternalState(
            inventory=self.initial_portfolio["inventory"],
            cash=self.initial_portfolio["cash"],
            asset_price=self.price.calculate_from_current_book(book_snapshots.iloc[-1]),
            book_snapshots=book_snapshots,
            proportion_of_episode_remaining=1.0,
        )

    def _check_params(self):
        assert self.min_start_timedelta + self.episode_length <= self.max_end_timedelta, "Episode is too long"

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

    def mark_to_market_value(self):
        return self.internal_state["inventory"] * self.internal_state["asset_price"] + self.internal_state["cash"]

    def _check_market_order_clearing_well_defined(self):
        if (self.market_order_clearing and self.market_order_fraction_of_inventory is None) or (
            not self.market_order_clearing and self.market_order_fraction_of_inventory is not None
        ):
            raise Exception(
                f"market_order_fraction_of_inventory {self.market_order_fraction_of_inventory} "
                "must be defined if and only if market order clearing (self.market_order_clearing} is on"
            )
