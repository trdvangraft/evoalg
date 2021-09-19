
from src.algorithms.plotter import Visualizer2D
from src.algorithms.pso import PSO

import numpy as np

X_MIN = Y_MIN = -6
X_MAX = Y_MAX = 6

def rosenbrock_fitness(X):
    if X.ndim == 1:
        return (1. - X[0]) ** 2 + ((X[1] - X[0] ** 2) ** 2) *100.
    return (1 - X[:, 0]) ** 2 + ((X[:, 1] - X[:, 0] ** 2) ** 2) * 100.

def main():
    print('rosenbrock example')

    plotter = Visualizer2D(X_MIN, X_MAX, Y_MIN, Y_MAX, rosenbrock_fitness)
    

    optimizer = PSO(2, np.array([[X_MIN, X_MAX], [Y_MIN, Y_MAX]]), rosenbrock_fitness)
    history = optimizer.optimize()

    plotter.animate_2d_landscape(history)

if __name__ == '__main__':
    main()