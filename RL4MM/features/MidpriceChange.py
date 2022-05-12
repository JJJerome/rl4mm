from typing import List

import numpy as np

from RL4MM.features.Feature import Feature
from RL4MM.simulation.StaleOrderbookSimulator import ResultsDict


class MidpriceChange(Feature):
    def __init__(self, max_change: float, values_taken: int, precision: int = 6) -> None:
        self.max_change = max_change
        self.values_taken = values_taken
        self._precision = precision

    @property
    def name(self) -> str:
        return "midprice change"

    @property
    def feature_space(self) -> List[float]:
        return [round(v, self.precision) for v in np.linspace(0, self.max_change, self.values_taken)]

    @property
    def precision(self):
        return self._precision

    def _calculate(self, results: ResultsDict) -> float:
        return results["midprice_change"]

    def _discretise(self, feature_value: float) -> float:
        rounding_factor = 2 * self.max_change / (self.values_taken - 1)
        return np.maximum(
            np.minimum(
                np.round(feature_value / rounding_factor) * rounding_factor,
                self.max_change,
            ),
            -self.max_change,
        )
