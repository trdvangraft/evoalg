import numpy as np
import math

from numpy.lib.ufunclike import fix

def de_example(X):
    return 3 * (1 - X[:, 0]) ** 2 * np.exp(-1 * X[:, 0]**2 - (X[:, 1] + 1)**2) \
        - 10 * (1/5 * X[:, 0] - X[:, 0]**3 - X[:, 1]**5) * np.exp(-1 * X[:, 0]**2 -1 * X[:, 1]**2) \
        - 1/3 * np.exp(-1 * (X[:, 0] + 1) ** 2 - X[:, 1] ** 2)

DIMENSION = 2

class ABC: 
    def __init__(self,  
        fit_func,
        number_of_workers: int=10, \
        number_of_scouts: int=10, \
        number_of_onlookers: int=10,  \
        number_of_iterations: int=100, \
    ) -> None:
        self.number_of_workers = number_of_workers
        self.number_of_scouts = number_of_scouts
        self.number_of_onlookers = number_of_onlookers
        self.number_of_bees = number_of_onlookers + number_of_scouts + number_of_workers

        self.fit_func = fit_func
        self.number_of_iterations = number_of_iterations
    
    def fitness(self, x):
        f_x = self.fit_func(x)
        f_x[f_x >= 0] = 1/(1 + f_x[f_x >= 0])
        f_x[f_x < 0] = 1 + np.abs(f_x[f_x < 0])
        return f_x

    def optimize(self):
        lb, ub = -10, 10
        alpha = 1

        x = lb + np.random.random((self.number_of_bees, DIMENSION)) * (ub - lb)
        fit_x = self.fitness(x)
        wo_idx, sc_idx, on_idx = np.split(np.random.permutation(self.number_of_bees), 3)

        print(wo_idx)
        print(sc_idx)
        print(on_idx)

        # init
        for i in range(self.number_of_iterations):
            phi = np.random.uniform(-alpha, alpha, size=(self.number_of_bees, DIMENSION))
            selected_variables = np.random.randint(low=0, high=DIMENSION, size=len(wo_idx))
            selected_food_sources = np.random.randint(low=0, high=self.number_of_bees, size=len(wo_idx))
            wo_selected_foot_sources = [wo_idx, selected_variables]


            # employeed bees phase
            v = x.copy()
            dist = x[wo_selected_foot_sources] - x[[selected_food_sources, selected_variables]]
            v[wo_selected_foot_sources] = v[wo_selected_foot_sources] + (phi[wo_selected_foot_sources] * dist)
            fit_v = self.fitness(v)
            fit_x = fit_v[fit_v > fit_x]
            
            # onlooker bees phase
            # scout bees phase
            # remember best solution
            break


def main():
    abc = ABC(de_example)
    abc.optimize()

if __name__ == '__main__':
    main()
