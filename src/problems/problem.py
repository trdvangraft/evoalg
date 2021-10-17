import numpy as np
import abc

"""The problem definition class contains the necessary function to define a problem that can be solved.
"""
class Problem(metaclass=abc.ABCMeta):
    def evaluate(self, solution: np.ndarray) -> np.ndarray:
        if solution.ndim == 1:
            return self.fitness(solution.reshape(1, -1))
        return self.fitness(solution)
    
    @abc.abstractmethod
    def fitness(self, solution: np.ndarray) -> np.ndarray:
        """The fitness function calculate the solutions fitness.

        Args:
            solution (np.array): a solution which fitness needs to be computed

        Returns:
            float: the fitness of the solution
        """

    @abc.abstractmethod
    def is_solved(self, best_solution) -> bool:
        """The is solved method allows the solver to check if the problem is solved.
        If the problem has a VTR then this function will return True if we got there

        Args:
            best_solution (np.array): the best possible solution in the current population

        Returns:
            bool: True if the problem is solved else False
        """
    
    @abc.abstractmethod
    def get_ndim(self) -> int:
        pass
    
    @abc.abstractmethod
    def get_lowerbound(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def get_upperbound(self) -> np.ndarray:
        pass