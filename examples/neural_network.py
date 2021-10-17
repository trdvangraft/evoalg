from src.algorithms.plotter import NeuralPlotter
from src.algorithms.pso import PSO
from src.problems.xor import XOR

def main():
    print('neural network example')
    optimizer = PSO(number_of_iterations=250, number_of_particles=160, c_1=0.7, c_2=2.5)
    problem = XOR()
    history = optimizer.optimize(problem)

    location_hist, fitness_hist, gbest_hist, gbest_location = history.report()

    best_net = problem.get_net().set_params(gbest_location[-1])

    visualizer = NeuralPlotter()
    visualizer.plot_xor(problem.samples, problem.targets, best_net, save=True, file_name="XOR_decision")
    visualizer.plot_loss(fitness_hist, gbest_hist)
    # print(predict(gbest_location[-1]))

if __name__ == '__main__':
    main()
