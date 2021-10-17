from src.problems.problem import Problem
from dataclasses import dataclass
import numpy as np

class Minimization2DProblem(Problem):
    x_min: int = -6
    y_min: int = -6
    x_max: int = 6
    y_max: int = 6
    vtr: float

    def is_solved(self, best_solution: np.ndarray):
        return self.evaluate(best_solution) - self.vtr <= 10e-6
    
    def get_ndim(self):
        return 2
    
    def get_lowerbound(self) -> np.ndarray:
        return np.array([self.x_min, self.y_min])
    
    def get_upperbound(self) -> np.ndarray:
        return np.array([self.x_max, self.y_max])

@dataclass
class ParabolicRidge(Minimization2DProblem):
    vtr: float = 0.0

    def fitness(self, solution: np.ndarray) -> np.ndarray:
        return -1 * solution[:, 0] + 100 * np.sum(solution[:, 0:] ** 2, axis=1)

@dataclass
class RosenBrock2D(Minimization2DProblem):
    vtr: float = 0.0

    def fitness(self, solution: np.ndarray) -> np.ndarray:
        return (1 - solution[:, 0]) ** 2 + ((solution[:, 1] - solution[:, 0] ** 2) ** 2) * 100.

@dataclass
class DeExample(Minimization2DProblem):
    vtr: float = 0.0

    def fitness(self, solution: np.ndarray) -> np.ndarray:
        return 3 * (1 - solution[:, 0]) ** 2 * np.exp(-1 * solution[:, 0]**2 - (solution[:, 1] + 1)**2) \
        - 10 * (1/5 * solution[:, 0] - solution[:, 0]**3 - solution[:, 1]**5) * np.exp(-1 * solution[:, 0]**2 -1 * solution[:, 1]**2) \
        - 1/3 * np.exp(-1 * (solution[:, 0] + 1) ** 2 - solution[:, 1] ** 2) 