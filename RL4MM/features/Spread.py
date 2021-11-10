from typing import List

import numpy as np

from RL4MM.features.Feature import Feature
from RL4MM.simulator.OrderbookSimulator import ResultsDict


class Spread(Feature):
    def __init__(self, max_spread: float, values_taken: int, precision: int = 6) -> None:
        self.max_spread = max_spread
        self.values_taken = values_taken
        self._precision = precision

    @property
    def name(self) -> str:
        return "spread"

    @property
    def feature_space(self) -> List[float]:
        return [round(v, self.precision) for v in np.linspace(0, self.max_spread, self.values_taken)]

    @property
    def precision(self):
        return self._precision

    def _calculate(self, results: ResultsDict) -> float:
        orderbook = results["orderbook"]
        return orderbook["ask_price_0"] - orderbook["bid_price_0"]

    def _discretise(self, feature_value: float) -> float:
        rounding_factor = self.max_spread / (self.values_taken - 1)
        return np.minimum(np.round(feature_value / rounding_factor) * rounding_factor, self.max_spread)
