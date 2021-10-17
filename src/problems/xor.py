import dataclasses
from typing import List

import numpy as np
import torch
from src.problems.problem import Problem
from src.utils.nn import Linear, Net, ReLU
from src.utils.utils import MSELoss

@dataclasses
class XOR(Problem):
    number_of_inputs: int = 2
    neurons_per_hidden_layer: List[int] = [2]
    number_of_outputs: int = 2

    samples = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    targets = torch.tensor([[1, 0], [0, 1], [0, 1], [1, 0]])

    def get_ndim(self) -> int:
        _all_neurons = [self.number_of_inputs, *self.neurons_per_hidden_layer, self.number_of_outputs]
        return sum([_all_neurons[i] * _all_neurons[i+1] + _all_neurons[i+1] for i in range(len(_all_neurons) - 1)])

    def fitness(self, solution: np.ndarray) -> np.ndarray:
        return [self.__network_fitness(s) for s in solution]
    
    def get_lowerbound(self) -> np.ndarray:
        return np.full(self.get_ndim(), -10)
    
    def get_upperbound(self) -> np.ndarray:
        return np.full(self.get_ndim(), 10)

    def is_solved(self, best_solution) -> bool:
        return np.mean(self.evaluate(best_solution) - self.targets) <= 10e-6

    def get_net(self):
        layers = [
            Linear(self.number_of_inputs, self.neurons_per_hidden_layer[0]),
            ReLU(),
            Linear(self.neurons_per_hidden_layer[0], self.number_of_outputs)
        ]

        return Net(layers)

    def __network_fitness(self, solution: np.ndarray) -> np.ndarray:
        net = self.get_net().set_params(solution)
        y_pred = net.forward(self.samples)

        return MSELoss(self.targets, y_pred)

   
