from src.algorithms.pso import PSO
from src.algorithms.plotter import Visualizer2D

import numpy as np

X_MIN = Y_MIN = -6
X_MAX = Y_MAX = 6

def de_example(X):
    if X.ndim == 1:
        return 3 * (1 - X[0]) ** 2 * np.exp(-1 * X[0]**2 - (X[1] + 1)**2) \
            - 10 * (1/5 * X[0] - X[0]**3 - X[1]**5) * np.exp(-1 * X[0]**2 -1 * X[1]**2) \
            - 1/3 * np.exp(-1 * (X[0] + 1) ** 2 - X[1] ** 2)
    return 3 * (1 - X[:, 0]) ** 2 * np.exp(-1 * X[:, 0]**2 - (X[:, 1] + 1)**2) \
        - 10 * (1/5 * X[:, 0] - X[:, 0]**3 - X[:, 1]**5) * np.exp(-1 * X[:, 0]**2 -1 * X[:, 1]**2) \
        - 1/3 * np.exp(-1 * (X[:, 0] + 1) ** 2 - X[:, 1] ** 2)


def main():
    print('differential algorithm example')
    plotter = Visualizer2D(X_MIN, X_MAX, Y_MIN, Y_MAX, de_example)
    
    optimizer = PSO(2, np.array([[X_MIN, X_MAX], [Y_MIN, Y_MAX]]), de_example)
    history = optimizer.optimize()

    plotter.animate_2d_landscape(history)

if __name__ == '__main__':
    main()