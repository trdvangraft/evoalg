
from src.utils.plotter import Visualizer2D
from src.problems.minimization2d import RosenBrock2D
from src.algorithms.pso import PSO

import numpy as np

def main():
    print('rosenbrock example')
    optimizer = PSO(number_of_particles=40, number_of_iterations=100)
    problem = RosenBrock2D()
    history = optimizer.optimize(problem)

    plotter = Visualizer2D(problem)
    plotter.animate_2d_landscape(history)

if __name__ == '__main__':
    main()