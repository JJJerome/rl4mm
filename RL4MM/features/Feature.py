import abc

from typing import List, Union

import pandas as pd


class Feature(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def feature_space(self) -> List[float]:
        pass

    @property
    @abc.abstractmethod
    def precision(self):
        pass

    @abc.abstractmethod
    def _calculate(self, *args) -> Union[float, pd.Series]:
        pass

    @abc.abstractmethod
    def _discretise(self, feature_value: Union[float, pd.Series]) -> Union[float, pd.Series]:
        pass

    def calculate(self, *args) -> Union[float, pd.Series]:
        return self._discretise(self._calculate(*args))
