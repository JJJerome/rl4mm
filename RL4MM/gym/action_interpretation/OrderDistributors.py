from __future__ import annotations
import abc
import sys

if sys.version_info[0] == 3 and sys.version_info[1] >= 8:
    from typing import Literal
else:
    from typing_extensions import Literal

import numpy as np
from scipy.stats import betabinom


class OrderDistributor(metaclass=abc.ABCMeta):
    def convert_action(self, action: np.ndarray) -> dict[Literal["buy", "sell"], tuple[np.ndarray]]:
        action = np.array(action, dtype=float) + sys.float_info.min
        return self._convert_action(action)

    @abc.abstractmethod
    def _convert_action(self, action: np.ndarray) -> dict[Literal["buy", "sell"], tuple[np.ndarray]]:
        pass


class BetaOrderDistributor(OrderDistributor):
    def __init__(self, quote_levels: int = 10, active_volume: int = 100, concentration: float = None):
        self.n_levels = quote_levels
        self.distribution = betabinom
        self.tick_range = range(0, self.n_levels)
        self.active_volume = active_volume
        self.c = concentration

    def _convert_action(self, action: np.ndarray) -> dict[Literal["buy", "sell"], tuple[np.ndarray]]:
        assert all(action) > 0, "Action must be positive"
        (a_buy, b_buy) = (action[0], self.c - action[0]) if self.c is not None else (action[0], action[1])
        (a_sell, b_sell) = (action[1], self.c - action[1]) if self.c is not None else (action[2], action[3])
        beta_binom_buy = betabinom(n=self.n_levels - 1, a=a_buy, b=b_buy)
        beta_binom_sell = betabinom(n=self.n_levels - 1, a=a_sell, b=b_sell)
        buy_volumes = np.round(beta_binom_buy.pmf(self.tick_range) * self.active_volume).astype(int)
        sell_volumes = np.round(beta_binom_sell.pmf(self.tick_range) * self.active_volume).astype(int)
        return {"buy": buy_volumes, "sell": sell_volumes}
