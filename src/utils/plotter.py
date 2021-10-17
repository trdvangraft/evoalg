from src.problems.problem import Problem
from src.algorithms.pso import Logger
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import torch

from mpl_toolkits import mplot3d

STEP = 100

class Visualizer2D():
    def __init__(self, problem: Problem):
        self.problem = problem

    def plot_fitness_landscape(self, show=False):
        X, Y, points = self._precompute_grid(self.problem.get_lowerbound(), self.problem.get_upperbound())
        Z = self.problem.evaluate(points).reshape((STEP, STEP))    

        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

        if show:
            plt.show()

        return fig, ax
    
    def contour_plot(self, ax):
        X, Y, points = self._precompute_grid([-1, -1], [1, 1])
        Z = self.__precompute_fitness(points)

        print(np.min(Z))
        min_value = np.min(Z)

        epsillon = 0.0001
        levels = np.arange(min_value - epsillon, min_value + epsillon)

        CS = ax.contour(X, Y, Z, levels=40, origin='lower', cmap='RdGy')
        ax.clabel(CS, inline=True, fontsize=8)

    def animate_2d_landscape(self, history: Logger, save=False, file_name: str = ""):
        location_hist, fitness_hist, gbest_hist, gbest_location = history.report()
        mean_fitness = np.mean(fitness_hist, axis=1)

        fig, surface_ax = self.plot_fitness_landscape()
        fitness_ax = fig.add_subplot(2, 2, 2)
        contour_ax = fig.add_subplot(2, 2, 4)

        self.contour_plot(contour_ax)

        fitness_ax.set_yscale('log')

        fig.axes.append(fitness_ax)

        scatter, = surface_ax.plot(
            location_hist[0, :, 0],
            location_hist[0, :, 1],
            fitness_hist[0],
            linestyle="", marker="o"
        )

        gbest_line, = fitness_ax.plot(
            range(len(gbest_hist)),
            gbest_hist
        )

        mean_line, = fitness_ax.plot(
            range(len(gbest_hist)),
            mean_fitness
        )

        lines = [gbest_line, mean_line]

        ani = animation.FuncAnimation(
            fig,
            self.animate, fargs=[lines, scatter, location_hist, fitness_hist, mean_fitness, gbest_hist], 
            blit=True, frames=len(fitness_hist) - 1
        )

        if save:
            ani.save(file_name)

        plt.show()

    
    def animate(self, i, lines, scatter, location_hist, fit_hist, mean_fit, gbest_hist):
        lines = self._animate_line(i, gbest_hist, mean_fit, lines)
        scatter, = self._animate_scatter(i, location_hist, fit_hist, scatter)

        return scatter, lines[0], lines[1], 


    def _animate_line(self, i, gbest_hist, mean_fit, lines):
        
        lines[0].set_data(range(i), gbest_hist[:i])
        lines[1].set_data(range(i), mean_fit[:i])
        # lines.set_data(range(i), mean_fit[:i])
        return lines

    def _animate_scatter(self, i, location_hist, fit_hist, scatter):
        scatter.set_data(location_hist[i, :, 0], location_hist[i, :, 1])
        scatter.set_3d_properties(fit_hist[i])

        return scatter,

    def _precompute_grid(self, lower_limit, upper_limit):
        x = np.linspace(lower_limit[0], upper_limit[0], STEP)
        y = np.linspace(lower_limit[1], upper_limit[1], STEP)

        X, Y = np.meshgrid(x, y)

        points = np.asarray([(p1, p2) for p1 in x for p2 in y])
        
        return X, Y, points
    
    def __precompute_fitness(self, points):
        return self.problem.evaluate(points).reshape((STEP, STEP))

class NeuralPlotter():
    def plot_xor(self, x, y, net=None, save=False, file_name:str=""):
        """
        Plotter function for XOR dataset and classifier boundaries (optional).

        Args:
            x: Nx2 dimensional data
            y: N dimensional labels
            net: Model which has a forward function
        """
        # Convert one-hot to class id
        y = torch.argmax(y, dim=1)

        # Plot decision boundary if net is given
        if net:
            h = 0.005
            x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
            y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1

            xx, yy = torch.meshgrid(torch.arange(x_min, x_max, h),
                                    torch.arange(y_min, y_max, h))
            
            in_tensor = torch.cat((xx.reshape((-1,1)), yy.reshape((-1,1))), dim=1)

            z = net.forward(in_tensor)
            z = torch.argmax(z, dim=1)
            z = z.reshape(xx.shape)
            plt.contourf(xx, yy, z, cmap=plt.cm.coolwarm)

        # Plot data points
        plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
        plt.title('XOR problem')
        plt.xlabel('x0')
        plt.ylabel('x1')

        if save:
            plt.savefig(file_name)
        
        plt.show()
    
    def plot_loss(self, fitness_hist, gbest_hist):
        plt.plot(gbest_hist, label="Global fitness")
        print(np.mean(fitness_hist, axis=1).shape)
        plt.plot(np.mean(fitness_hist, axis=1), label="Mean fitness")
        plt.legend()
        plt.show()