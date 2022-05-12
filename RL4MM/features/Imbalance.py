from typing import List

import numpy as np

from RL4MM.features.Feature import Feature
from RL4MM.simulation.StaleOrderbookSimulator import ResultsDict


class Imbalance(Feature):
    def __init__(self, values_taken: int, precision: int = 6) -> None:
        self.values_taken = values_taken
        self._precision = precision

    @property
    def name(self) -> str:
        return "imbalance"

    @property
    def feature_space(self) -> List[float]:
        return [round(v, self.precision) for v in np.linspace(0, 1, self.values_taken)]

    @property
    def precision(self):
        return self._precision

    def _calculate(self, results: ResultsDict) -> float:
        orderbook = results["orderbook"]
        return orderbook["bid_size_0"] / (orderbook["bid_size_0"] + orderbook["ask_size_0"])

    def _discretise(self, feature_value: float) -> float:
        rounding_factor = 1 / (self.values_taken - 1)
        return np.round(feature_value / rounding_factor) * rounding_factor
