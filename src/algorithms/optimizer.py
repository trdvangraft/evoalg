"""Super class for all swarm based optimizers
"""
from __future__ import annotations
import abc
from src.problems.problem import Problem
from src.utils.logger import Logger


class Optimizer(metaclass=abc.ABCMeta):
    def __init__(self, number_of_iterations: int, population_size: int) -> None:
        self.number_of_iterations = number_of_iterations
        self.population_size = population_size

    @abc.abstractmethod
    def optimize(self) -> Logger:
        """The optimize function runs the specific swarm algorithm
        """
    
    @abc.abstractmethod
    def reset(self):
        """Resets the algorithm to its original state without having to redefine hyperparameters
        """
    
    def register_problem(self, problem: Problem) -> Optimizer:
        if problem is None:
            return
        elif hasattr(self, 'problem') and self.problem is not None:
            self.reset()
            self.problem = problem
            return
        self.problem = problem
        return self
    
    def register_logger(self, logger: Logger=None) -> Optimizer:
        if logger is None and not hasattr(self, 'logger'):
            self.logger = Logger(self.number_of_iterations, self.population_size, self.problem.get_ndim())
        else: 
            self.logger = logger
        return self
    
