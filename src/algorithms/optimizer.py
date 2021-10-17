"""Super class for all swarm based optimizers
"""
import abc
from src.problems.problem import Problem
from src.utils.logger import Logger

class Optimizer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def optimize(self, problem: Problem) -> Logger:
        """The optimize function runs the specific swarm algorithm
        """
