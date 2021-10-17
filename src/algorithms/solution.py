import abc

import numpy as np

class Solution(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def unpack(self):
        pass

class PSOSolution:
    def __init__(self, location, velocity):
        self.location = location
        self.velocity = velocity
        self.fitness = np.inf

        self.pbest_location = None
        self.pbest_fitness = np.inf
    
    def set_fitness(self, fitness):
        self.fitness = fitness

        if fitness > self.pbest_fitness:
            self.pbest_fitness = fitness
            self.pbest_location = self.location
    
    def update_solution(self, new_location, new_velocity):
        self.location = new_location
        self.velocity = new_velocity