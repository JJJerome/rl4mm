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
    def __init__(self, quote_levels: int = 10, active_volume: int = 100):
        self.n_levels = quote_levels
        self.distribution = betabinom
        self.tick_range = range(0, self.n_levels)
        self.active_volume = active_volume

    def _convert_action(self, action: np.ndarray) -> dict[Literal["buy", "sell"], tuple[np.ndarray]]:
        beta_binom_buy = betabinom(n=self.n_levels - 1, a=action[0], b=action[1])
        beta_binom_sell = betabinom(n=self.n_levels - 1, a=action[2], b=action[3])
        buy_volumes = np.round(beta_binom_buy.pmf(self.tick_range) * self.active_volume).astype(int)
        sell_volumes = np.round(beta_binom_sell.pmf(self.tick_range) * self.active_volume).astype(int)
        return {"buy": buy_volumes, "sell": sell_volumes}
