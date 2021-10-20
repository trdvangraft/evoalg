from dataclasses import dataclass, field
from email.policy import default
from typing import Tuple
import numpy as np
import math

from numpy.lib.ufunclike import fix

from src.algorithms.optimizer import Optimizer
from src.problems.problem import Problem
from src.utils.logger import Logger
from src.utils.utils import GreedySelector, RouletteWheelSelector

@dataclass
class BeeClassMemory:
    solution: np.ndarray
    fitness: np.ndarray
    prev_fitness: np.ndarray = field(default=np.zeros(0), init=False)

    def update_solution(self, solution: np.ndarray, fitness: np.ndarray):
        # first we store the previous solution to use in our scout phase
        self.prev_fitness = self.fitness.copy()
        # override the previous solution
        self.solution = solution
        self.fitness = fitness

class ABC(Optimizer): 
    def __init__(self,  
        number_of_workers: int=10, \
        number_of_onlookers: int=10,  \
        alpha: int=10, \
        abandon_criteria: int = 5, \
        number_of_iterations: int=100, \
    ) -> None:
        super().__init__(number_of_iterations, number_of_onlookers + number_of_workers)

        self.number_of_workers = number_of_workers
        self.number_of_onlookers = number_of_onlookers
        self.number_of_bees = number_of_onlookers + number_of_workers
        self.alpha = alpha

        self.selector = GreedySelector()

        self.number_of_iterations = number_of_iterations
        self.abandon_criteria = abandon_criteria
    
    def fitness(self, x: np.ndarray) -> np.ndarray:
        f_x = self.problem.evaluate(x)
        f_x[f_x >= 0] = 1/(1 + f_x[f_x >= 0])
        f_x[f_x < 0] = 1 + np.abs(f_x[f_x < 0])
        return f_x

    def optimize(self) -> Logger:
        if self.problem is None:
            raise AttributeError()

        # initialize the memory of both the employee and onlooker bees
        employee_memory, onlooker_memory = self.__initialize_bees()

        limit = np.zeros(self.number_of_workers)

        for i in range(self.number_of_iterations):
            # employeed bees phase
            employee_solutions, employee_fitness = self.employee_phase(employee_memory)
            self.__selection_step(employee_solutions, employee_fitness, employee_memory)
            
            # onlooker bees phase
            onlooker_solutions, onlooker_fitness = self.onlooker_phase(onlooker_memory)
            self.__selection_step(onlooker_solutions, onlooker_fitness, employee_memory)

            # scout bees phase
            scout_solution, scout_fitness, limit = self.scout_phase(employee_memory, limit)
            employee_memory.update_solution(scout_solution, scout_fitness)

            # remember best solution
            break
    
    def employee_phase(self, memory: BeeClassMemory):
        employee_solutions = self.__neightbour_search(memory.solution)
        employee_fitness = self.fitness(employee_solutions)
        return employee_solutions, employee_fitness
    
    def onlooker_phase(self, memory: BeeClassMemory):
        wheel = RouletteWheelSelector()
        food_sources_of_interest = memory.solution[wheel.select(memory.fitness, n=self.number_of_onlookers)]
        onlooker_solutions = self.__neightbour_search(food_sources_of_interest)
        onlooker_fitness = self.fitness(onlooker_solutions)
        return onlooker_solutions, onlooker_fitness
    
    def scout_phase(self, memory: BeeClassMemory, limit: np.ndarray):
        limit += memory.prev_fitness >= memory.fitness
        random_solution = self.__initialize_solution(self.number_of_workers)

        scout_solution = np.where(limit >= self.abandon_criteria, random_solution, memory.solution)
        scout_fitness = self.fitness(scout_solution)

        limit[limit >= self.abandon_criteria] = 0
        return scout_solution, scout_fitness, limit
    
    def memorize(self, i: int, employee_memory: BeeClassMemory, onlooker_memory: BeeClassMemory):
        solutions = np.vstack((employee_memory.solution, onlooker_memory.solution))
        fitness = np.vstack((employee_memory.fitness, onlooker_memory.fitness))

        self.logger.log(i, solutions, fitness)

    
    def reset():
        pass
    
    def __selection_step(self, solution: np.ndarray, fitness: np.ndarray, memory: BeeClassMemory) -> BeeClassMemory:
        new_solution, new_fitness = self.selector.select(memory.solution, memory.fitness, solution, fitness)
        memory.update_solution(new_solution, new_fitness)

    def __neightbour_search(self, solution: np.ndarray) -> np.ndarray:
        # setup search - standard for every phase
        number_of_bees = len(solution)
        phi = np.random.uniform(-self.alpha, self.alpha, size=(number_of_bees, self.problem.get_ndim()))
        selected_variables = np.random.randint(low=0, high=self.problem.get_ndim(), size=number_of_bees)
        selected_food_sources = np.random.randint(low=0, high=number_of_bees, size=number_of_bees)
        bee_idx = np.arange(number_of_bees)

        pointer = (bee_idx, selected_variables)

        v = solution.copy()
        dist = solution[pointer] - solution[(selected_food_sources, selected_variables)]
        v[pointer] = solution[pointer] + (phi[pointer] * dist)

        return v
    
    def __initialize_bees(self) -> Tuple[BeeClassMemory, BeeClassMemory]:
        return self.__initialize_memory(self.number_of_workers), self.__initialize_memory(self.number_of_onlookers)

    def __initialize_memory(self, n: int) -> BeeClassMemory:
        solution = self.__initialize_solution(n)
        fitness = self.fitness(solution)

        return BeeClassMemory(solution, fitness)

    def __initialize_solution(self, n: int) -> np.ndarray:
        margin = self.problem.get_upperbound() - self.problem.get_lowerbound()
        solution = self.problem.get_lowerbound() + np.random.random((n, self.problem.get_ndim())) * margin

        return solution

