from src.algorithms.optimizer import Optimizer
from src.problems.problem import Problem


class Solver:
    def solve(self, problem: Problem, optimizer: Optimizer) -> bool:
        optimizer.solve(problem)