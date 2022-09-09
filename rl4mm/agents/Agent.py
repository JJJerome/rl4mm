import abc
import numpy as np


class Agent(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_action(self, state: np.ndarray) -> np.ndarray:
        pass
