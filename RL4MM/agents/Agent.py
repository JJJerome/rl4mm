import abc

from RL4MM.base import State, Action


class Agent(metaclass=abc.ABCMeta):
    def get_action(self, state: State) -> Action:
        pass
