from src.utils.plotter import Visualizer2D
from src.problems.minimization2d import ParabolicRidge
from src.algorithms.pso import PSO

def main():
    print('parabolic ridge example')
    optimizer = PSO()
    problem = ParabolicRidge()
    history = optimizer.optimize(problem)

    plotter = Visualizer2D(problem)
    plotter.animate_2d_landscape(history)

if __name__ == '__main__':
    main()