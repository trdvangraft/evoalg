from src.problems.problem import Problem
from src.algorithms.optimizer import Optimizer
import numpy as np
from src.utils.logger import Logger

class PSO(Optimizer):
    def __init__(self, c_1=0.8, c_2=0.4, number_of_particles=20, number_of_iterations=100):
        self.number_of_iterations = number_of_iterations
        self.number_of_particles = number_of_particles
        
        self.c_1 = c_1
        self.c_2 = c_2
        self.wt = np.linspace(0.9, 0.4, self.number_of_iterations)

    def optimize(self, problem: Problem) -> Logger:
        location, velocity, pbest, fitness, gbest = self.__initialize_particles(problem)
        logger = Logger(self.number_of_iterations, self.number_of_particles, problem.get_ndim())
        r = np.random.uniform(size=(self.number_of_iterations, self.number_of_particles, 2))

        for i in range(self.number_of_iterations):
            cognitive_comp = self.c_1 * np.diag(r[i, :, 0]) @ (pbest - location)
            social_comp = self.c_2 * np.diag(r[i, :, 1]) @ (gbest - location)

            velocity = self.wt[i] * velocity + cognitive_comp + social_comp
            location = np.clip(location + velocity, problem.get_lowerbound(), problem.get_upperbound())
            new_fitness = problem.evaluate(location)

            pbest[new_fitness < fitness] = location[new_fitness < fitness]
            gbest = location[np.argmin(new_fitness)] if min(new_fitness) <= problem.evaluate(gbest) else gbest

            fitness = new_fitness

            logger.log(i, location, fitness)

        return logger
    
    def __initialize_particles(self, problem: Problem):
        initial_location = initial_pbest = np.random.uniform(
            problem.get_lowerbound(),
            problem.get_upperbound(),
            size=(self.number_of_particles, problem.get_ndim())
        )
        initial_velocity = np.random.uniform(
            problem.get_lowerbound(),
            problem.get_upperbound(),
            size=(self.number_of_particles, problem.get_ndim())
        )
        initial_fitness = problem.evaluate(initial_location)
        gbest = initial_location[np.argmin(initial_fitness)]

        return initial_location, initial_velocity, initial_pbest, initial_fitness, gbest
    
    def __str__(self):
        return f"{self.number_of_iterations=} \n {self.number_of_particles=} \n {self.c_1=} \n {self.c_2=}"