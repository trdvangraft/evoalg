from torch import optim
from src.algorithms.abc import ABC
from src.problems.minimization2d import DeExample
from src.algorithms.pso import PSO
from src.utils.plotter import Visualizer2D

def main():
    print('differential algorithm example')
    optimizer = PSO()
    abc_optimizer = ABC()
    problem = DeExample()
    # history = optimizer.optimize(problem)
    abc_history = abc_optimizer.register_problem(problem).register_logger().optimize() 

    # plotter = Visualizer2D(problem)
    # plotter.animate_2d_landscape(history)

if __name__ == '__main__':
    main()