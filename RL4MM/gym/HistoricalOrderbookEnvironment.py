from __future__ import annotations

import warnings
from copy import deepcopy, copy
from datetime import datetime, timedelta
import sys

from RL4MM.database.HistoricalDatabase import HistoricalDatabase
from RL4MM.gym.order_tracking.InfoCalculators import InfoCalculator
from RL4MM.utils.utils import get_next_trading_dt

if sys.version_info[0] == 3 and sys.version_info[1] >= 8:
    from typing import List, Literal, Optional
else:
    from typing import List
    from typing_extensions import Literal

import gym
import numpy as np
import pandas as pd

from gym.spaces import Box
from gym.utils import seeding

from RL4MM.features.Features import (
    Feature,
    Spread,
    State,
    PriceMove,
    Volatility,
    Inventory,
    EpisodeProportion,
    TimeOfDay,
    Portfolio,
    TradeDirectionImbalance,
    TradeVolumeImbalance,
)
from RL4MM.gym.action_interpretation.OrderDistributors import OrderDistributor, BetaOrderDistributor
from RL4MM.orderbook.create_order import create_order
from RL4MM.orderbook.models import Order, Orderbook, OrderDict, Cancellation, FilledOrders, MarketOrder
from RL4MM.rewards.RewardFunctions import RewardFunction, InventoryAdjustedPnL
from RL4MM.simulation.HistoricalOrderGenerator import HistoricalOrderGenerator
from RL4MM.simulation.TimeDrivenOrderbookSimulator import TimeDrivenOrderbookSimulator

ORDERBOOK_FREQ = "1S"
ORDERBOOK_MIN_STEP = pd.to_timedelta(ORDERBOOK_FREQ)
LEVELS_FOR_FEATURE_CALCULATION = 1
MARKET_ORDER_FRACTION_OF_INVENTORY = 0.1


class HistoricalOrderbookEnvironment(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        features: List[Feature] = None,
        max_distribution_param: float = 10.0,  # 1000.0,
        ticker: str = "MSFT",
        step_size: timedelta = timedelta(seconds=0.1),
        episode_length: timedelta = timedelta(minutes=30),
        initial_portfolio: Portfolio = None,
        min_quote_level: int = 0,
        max_quote_level: int = 10,
        min_date: datetime = datetime(2019, 1, 2),
        max_date: datetime = datetime(2019, 1, 2),
        min_start_timedelta: timedelta = timedelta(hours=10),  # Ignore the first half an hour of trading
        max_end_timedelta: timedelta = timedelta(hours=15, minutes=30),  # Same for the last half an hour
        simulator: TimeDrivenOrderbookSimulator = None,
        market_order_clearing: bool = False,
        inc_prev_action_in_obs: bool = False,
        max_inventory: int = 100000,
        per_step_reward_function: RewardFunction = InventoryAdjustedPnL(inventory_aversion=10 ** (-4)),
        terminal_reward_function: RewardFunction = InventoryAdjustedPnL(inventory_aversion=0.1),
        info_calculator: InfoCalculator = None,
        order_distributor: OrderDistributor = None,
        concentration: Optional[float] = None,
        market_order_fraction_of_inventory: float = 0.0,
        enter_spread: bool = False,
        n_levels: int = 50,
        preload_messages: bool = True,
    ):
        super(HistoricalOrderbookEnvironment, self).__init__()

        # Actions are the parameters governing the distribution over levels in the orderbook
        if concentration is not None:
            assert order_distributor is None, "When specifying concentration, no order distributor should be passed."
            assert concentration >= max_distribution_param, "Concentration is less than max_distribution_param."
            self.action_space = Box(low=0.0, high=concentration, shape=(2,), dtype=np.float64)  # alpha < kappa for beta
        else:
            self.action_space = Box(low=0.0, high=max_distribution_param, shape=(4,), dtype=np.float64)
        if market_order_clearing:
            low_action = np.append(self.action_space.low, [0.0])
            high_action = np.append(self.action_space.high, [max_inventory])
            self.action_space = Box(low=low_action, high=high_action, dtype=np.float64)
        self.max_distribution_param = max_distribution_param
        self.ticker = ticker
        self.step_size = step_size
        self.min_quote_level = min_quote_level
        self.max_quote_level = max_quote_level
        assert episode_length % self.step_size == timedelta(0), "Episode length must be a multiple of step size!"
        self.n_steps = int(episode_length / self.step_size)
        self.episode_length = episode_length
        self.terminal_time = datetime.max
        self.initial_portfolio = initial_portfolio or Portfolio(inventory=0, cash=1000)
        self.min_date = min_date
        self.max_date = max_date
        self.min_start_timedelta = min_start_timedelta
        self.max_end_timedelta = max_end_timedelta
        self.order_distributor = order_distributor or BetaOrderDistributor(
            self.max_quote_level - self.min_quote_level, concentration=concentration
        )
        self.market_order_clearing = market_order_clearing
        self.market_order_fraction_of_inventory = market_order_fraction_of_inventory
        self.per_step_reward_function = per_step_reward_function
        self.terminal_reward_function = terminal_reward_function
        self.enter_spread = enter_spread
        self.n_levels = n_levels
        self.info_calculator = info_calculator
        self.pricer = lambda orderbook: orderbook.microprice  # Can change this to midprice or any other notion of price
        self._check_params()
        # Observation space is determined by the features used
        self.max_inventory = max_inventory
        self.features = features or self.get_default_features(step_size, episode_length)
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
        self.max_feature_window_size = max([feature.window_size for feature in self.features])
        self.simulator = simulator or TimeDrivenOrderbookSimulator(
            ticker=ticker,
            order_generators=[HistoricalOrderGenerator(ticker, HistoricalDatabase(), preload_messages)],
            n_levels=self.n_levels,
            preload_messages=preload_messages,
            episode_length=episode_length,
            warm_up=self.max_feature_window_size,
        )
        self.state: State = self._get_default_state()

    def reset(self):
        episode_start = self._get_random_start_time()
        self.terminal_time = episode_start + self.episode_length
        now_is = episode_start - self.max_feature_window_size
        self.simulator.reset_episode(start_date=now_is)
        price = self.pricer(self.central_orderbook)
        self.state = State(FilledOrders(), self.central_orderbook, price, self.initial_portfolio, now_is)
        self._reset_features(episode_start)
        for step in range(int(self.max_feature_window_size / self.step_size)):
            self._forward(list())
            self._update_features()
        if self.inc_prev_action_in_obs:
            return self.get_observation(np.zeros(shape=self.action_space.shape))
        else:
            return self.get_observation()

    def step(self, action: np.ndarray):
        done = False
        internal_orders = self.convert_action_to_orders(action=action)
        current_state = deepcopy(self.state)
        self._forward(internal_orders)
        self._update_features()
        next_state = self.state
        reward = self.per_step_reward_function.calculate(current_state, next_state)
        observation = self.get_observation(action) if self.inc_prev_action_in_obs else self.get_observation()
        if self.terminal_time - next_state.now_is < self.step_size / 2:
            reward = self.terminal_reward_function.calculate(current_state, next_state)
            done = True
        info = {}
        if self.info_calculator is not None:
            info = self.info_calculator.calculate(internal_state=self.state, action=action)
        return observation, reward, done, info

    def _forward(self, internal_orders: List[Order]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            filled = self.simulator.forward_step(
                until=self.state.now_is + self.step_size, internal_orders=internal_orders
            )
        self.update_internal_state(filled)
        return filled

    def get_observation(self, previous_action: np.ndarray = None) -> np.ndarray:
        obs = np.array([feature.current_value for feature in self.features])
        if previous_action is not None:
            return np.concatenate((obs, previous_action))
        else:
            return obs

    def _get_random_start_time(self):
        return self._get_random_trading_day() + self._random_offset_timestamp()

    def update_internal_state(self, filled_orders: FilledOrders):
        self._update_portfolio(filled_orders)
        self.state.filled_orders = filled_orders
        self.state.orderbook = self.central_orderbook
        self.state.price = self.pricer(self.central_orderbook)
        self.state.now_is += self.step_size

    def convert_action_to_orders(self, action: np.ndarray) -> List[Order]:
        desired = self.order_distributor.convert_action(action)
        if self.market_order_clearing and np.abs(self.state.portfolio.inventory) > action[-1]:  # cancel all orders
            desired = {d: np.zeros(self.max_quote_level - self.min_quote_level) for d in ["buy", "sell"]}  # type:ignore
        current_volumes = self._get_current_internal_order_volumes()
        difference_in_volumes = {d: desired[d] - current_volumes[d] for d in ["buy", "sell"]}  # type: ignore
        orders = self._volume_diff_to_orders(difference_in_volumes)
        if self.market_order_clearing and np.abs(self.state.portfolio.inventory) > action[-1]:
            # place a market order to reduce inventory to zero
            orders += self._get_inventory_clearing_market_order()
        return orders

    def _reset_features(self, episode_start: datetime):
        for feature in self.features:
            first_usage_time = episode_start - feature.window_size
            feature.reset(self.state, first_usage_time)

    def _update_features(self):
        for feature in self.features:
            feature.update(self.state)

    def _volume_diff_to_orders(self, volume_diff: dict[str, np.ndarray]) -> List[Order]:
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
                    current_orders = deepcopy(getattr(self.internal_orderbook, side)[price])
                    while order_volume < 0:
                        worst_order = current_orders[-1]
                        volume_to_remove = min(worst_order.volume, abs(order_volume))
                        order_dict["volume"] = volume_to_remove
                        order_dict["internal_id"] = worst_order.internal_id
                        cancellation = create_order("cancellation", order_dict)
                        orders.append(cancellation)
                        order_volume += volume_to_remove
                        current_orders.pop()
            for price in set(getattr(self.internal_orderbook, side).keys()) - set(best_prices[side]):
                try:
                    wide_orders = list(getattr(self.internal_orderbook, side)[price])
                except KeyError:
                    wide_orders = list()
                for order in wide_orders:
                    cancellation = Cancellation(**copy(order.__dict__))
                    orders.append(cancellation)
        return orders

    def _get_inventory_clearing_market_order(self) -> List[MarketOrder]:
        inventory = self.state.portfolio.inventory
        order_direction = "buy" if inventory < 0 else "sell"
        order_dict = self._get_default_order_dict(order_direction)  # type:ignore
        order_dict["volume"] = np.round(np.abs(inventory) * self.market_order_fraction_of_inventory)
        market_order = create_order("market", order_dict)
        return [market_order]

    def _get_default_order_dict(self, direction: Literal["buy", "sell"]) -> OrderDict:
        return OrderDict(
            timestamp=self.state.now_is,
            price=None,
            volume=None,
            direction=direction,
            ticker=self.ticker,
            internal_id=None,
            external_id=None,
            is_external=False,
        )

    def _update_portfolio(self, filled_orders: FilledOrders):
        for order in filled_orders.internal:
            if order.price is None:
                raise Exception("Cannot update portfolio from a market order with no fill price.")
            elif order.direction == "sell":
                self.state.portfolio.inventory -= order.volume
                self.state.portfolio.cash += order.volume * order.price
            elif order.direction == "buy":
                self.state.portfolio.inventory += order.volume
                self.state.portfolio.cash -= order.volume * order.price

    # def _update_book_snapshots_STALE(self, orderbook: Orderbook) -> None:
    #     current_book_snapshots = self.state["book_snapshots"][1:]
    #     new_book_dict = convert_to_lobster_format(orderbook, LEVELS_FOR_FEATURE_CALCULATION)
    #     new_book_snapshot = pd.DataFrame.from_dict({self.state.now_is: new_book_dict}).T
    #     self.state["book_snapshots"] = pd.concat([current_book_snapshots, new_book_snapshot])

    def _get_current_internal_order_volumes(self) -> dict[Literal["buy", "sell"], tuple[np.ndarray]]:
        best_prices = self._get_best_prices()
        internal_volumes = dict()
        for side in ["buy", "sell"]:
            internal_volumes[side] = self._get_volumes_at_prices(side, best_prices[side], self.internal_orderbook)
        return internal_volumes  # type: ignore

    def _get_best_prices(self):
        tick_size = self.central_orderbook.tick_size
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

    def _get_volumes_at_prices(self, direction: str, price_levels: np.ndarray, orderbook: Orderbook):
        volumes = list()
        for price in price_levels:
            try:
                volumes.append(sum(order.volume for order in getattr(orderbook, direction)[price]))
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
        random_offset_timestamp = self.min_start_timedelta + random_offset_steps * self.step_size
        random_offset_timestamp -= timedelta(microseconds=random_offset_timestamp.microseconds)  # Start episode on sec
        return random_offset_timestamp

    def _random_offset_days(self):
        return np.random.randint(int((self.max_date.date() - self.min_date.date()) / timedelta(days=1)) + 1)

    def _get_random_trading_day(self):
        trading_dates = pd.bdate_range(self.min_date, self.max_date)
        trading_date = get_next_trading_dt(pd.to_datetime(np.random.choice(trading_dates)))
        return datetime.combine(trading_date.date(), datetime.min.time())

    # def _reset_internal_state_slow_STALE(self):
    #     snapshot_start = self.state.now_is - self.step_size * self.max_feature_window_size
    #     book_snapshots = self.simulator.database.get_book_snapshot_series(
    #         start_date=snapshot_start,
    #         end_date=self.state.now_is,
    #         ticker=self.ticker,
    #         freq=convert_timedelta_to_freq(self.step_size),
    #         n_levels=LEVELS_FOR_FEATURE_CALCULATION,
    #     )
    #     self.state = State(
    #         inventory=self.initial_portfolio["inventory"],
    #         cash=self.initial_portfolio["cash"],
    #         asset_price=self.price.calculate_from_current_book(book_snapshots.iloc[-1]),
    #         book_snapshots=book_snapshots,
    #         proportion_of_episode_remaining=1.0,
    #     )

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
        return self.state.portfolio.inventory * self.state.price + self.state.portfolio.cash

    def _check_params(self):
        assert self.min_start_timedelta + self.episode_length <= self.max_end_timedelta, "Episode is too long"
        assert self.max_quote_level - self.min_quote_level == self.order_distributor.quote_levels
        self._check_market_order_clearing_well_defined()

    def _check_market_order_clearing_well_defined(self):
        if (self.market_order_clearing and self.market_order_fraction_of_inventory <= 0.0) or (
            not self.market_order_clearing
            and (self.market_order_fraction_of_inventory is not None and self.market_order_fraction_of_inventory > 0.0)
        ):
            raise Exception(
                f"market_order_fraction_of_inventory {self.market_order_fraction_of_inventory} "
                "must be positive if and only if market order clearing (self.market_order_clearing} is on"
            )

    def _get_default_state(self):
        return State(
            filled_orders=FilledOrders(),
            orderbook=self.simulator.exchange.get_empty_orderbook(),
            price=0.0,
            portfolio=self.initial_portfolio,
            now_is=datetime.min,
        )

    @staticmethod
    def get_default_features(step_size: timedelta, episode_length: timedelta, normalisation_on: bool = False):
        time_of_day_buckets = 10
        assert step_size <= timedelta(seconds=0.1), "Default features require a minimum step size of 0.1 seconds."
        return [
            Spread(update_frequency=step_size, normalisation_on=normalisation_on),
            PriceMove(
                name="price_move_0.1_s",
                update_frequency=timedelta(seconds=0.1),
                lookback_periods=1,
                normalisation_on=normalisation_on,
            ),
            PriceMove(
                name="price_move_10_s",
                update_frequency=timedelta(seconds=1),
                lookback_periods=10,
                normalisation_on=normalisation_on,
            ),
            Volatility(
                name="volatility_1_min",
                update_frequency=timedelta(seconds=0.1),
                lookback_periods=int(10 * 60),
                normalisation_on=normalisation_on,
            ),
            Volatility(
                name="volatility_5_min",
                update_frequency=timedelta(seconds=1),
                lookback_periods=int(5 * 60),
                normalisation_on=normalisation_on,
            ),
            Inventory(update_frequency=step_size, normalisation_on=normalisation_on),
            EpisodeProportion(
                update_frequency=step_size, episode_length=episode_length, normalisation_on=normalisation_on
            ),
            TimeOfDay(n_buckets=time_of_day_buckets, normalisation_on=normalisation_on),
            TradeDirectionImbalance(
                update_frequency=timedelta(seconds=0.1),
                lookback_periods=int(60 * 10),
                normalisation_on=normalisation_on,
            ),
            TradeVolumeImbalance(
                update_frequency=timedelta(seconds=0.1),
                lookback_periods=int(60 * 10),
                normalisation_on=normalisation_on,
            ),
        ]
