import abc

from scipy.stats import betabinom


class OrderDistributor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def convert_action(self, action: tuple) -> tuple[int]:
        pass


class BetaOrderDistributor(OrderDistributor):
    def __init__(self, n_levels: int = 10):
        self.n_levels = n_levels
        self.distribution = betabinom
        self.tick_range = range(0, self.n_levels - 1)

    def convert_action(self, action: tuple) -> tuple[int]:
        beta_binom = betabinom(n=self.n_levels - 1, a=action[0], b=action[1])
        return beta_binom.pmf(self.tick_range)
