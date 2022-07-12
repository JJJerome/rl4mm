from __future__ import annotations
import abc
import sys
import warnings
from contextlib import suppress

if sys.version_info[0] == 3 and sys.version_info[1] >= 8:
    from typing import Literal
else:
    from typing_extensions import Literal

import numpy as np
from scipy.stats import beta

EPS = 0.000001


class OrderDistributor(metaclass=abc.ABCMeta):
    def convert_action(self, action: np.ndarray) -> dict[Literal["buy", "sell"], tuple[np.ndarray]]:
        action = np.array(action, dtype=float) + EPS  # for strict positivity
        return self._convert_action(action)

    @abc.abstractmethod
    def _convert_action(self, action: np.ndarray) -> dict[Literal["buy", "sell"], tuple[np.ndarray]]:
        pass


class BetaOrderDistributor(OrderDistributor):
    def __init__(self, quote_levels: int = 10, active_volume: int = 100, concentration: float = None):
        self.n_levels = quote_levels
        self.distribution = beta
        self.tick_range = range(0, self.n_levels)
        self.active_volume = active_volume
        self.c = concentration
        self.midpoints = 1 / self.n_levels * np.array([i + 0.5 for i in range(self.n_levels)])

    def _convert_action(self, action: np.ndarray) -> dict[Literal["buy", "sell"], tuple[np.ndarray]]:
        assert all(action) > 0, "Action must be positive"
        assert (self.c is None and len(action) in (4, 5)) or (
            self.c is not None and len(action) in (2, 3)
        ), f"Concentration is set to {self.c} and the action taken is of length {len(action)}"
        (a_buy, b_buy) = (action[0], self.c - action[0] + EPS) if self.c is not None else (action[0], action[1])
        (a_sell, b_sell) = (action[1], self.c - action[1] + EPS) if self.c is not None else (action[2], action[3])
        beta_buy = self.distribution(a=a_buy, b=b_buy)
        beta_sell = self.distribution(a=a_sell, b=b_sell)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            buy_dist = np.array([beta_buy.pdf(midpoint) for midpoint in self.midpoints])
            sell_dist = np.array([beta_sell.pdf(midpoint) for midpoint in self.midpoints])
        buy_dist /= buy_dist.sum()  # normalise to one
        sell_dist /= sell_dist.sum()  # normalise to one
        buy_volumes = np.round(buy_dist * self.active_volume).astype(int)
        sell_volumes = np.round(sell_dist * self.active_volume).astype(int)
        return {"buy": buy_volumes, "sell": sell_volumes}
