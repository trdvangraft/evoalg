from src.algorithms.pso import PSO
from src.algorithms.plotter import Visualizer2D

import numpy as np

X_MIN = Y_MIN = -6
X_MAX = Y_MAX = 6

def parabolic_ridge_fitness(X):
    if X.ndim == 1:
        return -1 * X[0] + 100 * np.sum(X[0:] ** 2)
    return -1 * X[:, 0] + 100 * np.sum(X[:, 0:] ** 2, axis=1)

def main():
    print('parabolic ridge example')
    plotter = Visualizer2D(X_MIN, X_MAX, Y_MIN, Y_MAX, parabolic_ridge_fitness)
    
    optimizer = PSO(2, np.array([[X_MIN, X_MAX], [Y_MIN, Y_MAX]]), parabolic_ridge_fitness)
    history = optimizer.optimize()

    plotter.animate_2d_landscape(history)

if __name__ == '__main__':
    main()