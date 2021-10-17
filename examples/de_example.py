from src.problems.minimization2d import DeExample
from src.algorithms.pso import PSO
from src.utils.plotter import Visualizer2D

def main():
    print('differential algorithm example')
    optimizer = PSO()
    problem = DeExample()
    history = optimizer.optimize(problem)

    plotter = Visualizer2D(problem)
    plotter.animate_2d_landscape(history)

if __name__ == '__main__':
    main()