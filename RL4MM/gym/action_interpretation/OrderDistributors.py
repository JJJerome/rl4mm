import abc
from typing import Literal

import numpy as np
from scipy.stats import betabinom


class OrderDistributor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def convert_action(self, action: tuple) -> dict[Literal["buy", "sell"], tuple[np.ndarray]]:
        pass


class BetaOrderDistributor(OrderDistributor):
    def __init__(self, quote_levels: int = 10, active_volume: int = 100):
        self.n_levels = quote_levels
        self.distribution = betabinom
        self.tick_range = range(0, self.n_levels)
        self.active_volume = active_volume

    def convert_action(self, action: tuple) -> dict[Literal["buy", "sell"], tuple[np.ndarray]]:
        beta_binom_buy = betabinom(n=self.n_levels - 1, a=action[0], b=action[1])
        beta_binom_sell = betabinom(n=self.n_levels - 1, a=action[2], b=action[3])
        buy_volumes = np.round(beta_binom_buy.pmf(self.tick_range) * self.active_volume)
        sell_volumes = np.round(beta_binom_sell.pmf(self.tick_range) * self.active_volume)
        return {"buy": buy_volumes, "sell": sell_volumes}
