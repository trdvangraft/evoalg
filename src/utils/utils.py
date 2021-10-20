from typing import Tuple
import numpy as np
import torch

def MSELoss(y_true, y_pred):
    return torch.mean((y_pred - y_true) ** 2)

class RouletteWheelSelector:
    def select(self, fitness: np.array, n=None) -> np.array:
        n = n if n is not None else len(fitness)
        total_fitness = np.sum(fitness)
        selection_prob = fitness / total_fitness
        return np.random.choice(n, size=fitness.shape, p=selection_prob)

class GreedySelector:
    def select(self, \
        prev_population: np.ndarray, \
        prev_fitness: np.array, \
        population: np.ndarray, \
        fitness: np.array
    ) -> Tuple[np.ndarray, np.array]:
        """
        """
        new_fit_x = np.where(fitness > prev_fitness, fitness, prev_fitness)
        new_x = np.where(fitness > prev_fitness, population, prev_population)

        prev_population[fitness > prev_fitness] = population[fitness > prev_fitness]

        return new_x, new_fit_x

