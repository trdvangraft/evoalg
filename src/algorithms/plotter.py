from src.algorithms.pso import Logger
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import torch

from mpl_toolkits import mplot3d

STEP = 100

class Visualizer2D():
    def __init__(self, x_min, x_max, y_min, y_max, fit_func):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.fit_func = fit_func

    def plot_fitness_landscape(self, show=False):
        X, Y, points = self._precompute_grid()
        Z = self.fit_func(points).reshape((STEP, STEP))    

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

        if show:
            plt.show()

        return fig, ax

    def animate_2d_landscape(self, history: Logger, save=False, file_name: str = ""):
        location_hist, fitness_hist, gbest_hist, gbest_location = history.report()
        fig, ax = self.plot_fitness_landscape()

        scatter, = ax.plot(
            location_hist[0, :, 0],
            location_hist[0, :, 1],
            fitness_hist[0],
            frames=len(fitness_hist) - 1, linestyle="", marker="o"
        )

        ani = animation.FuncAnimation(
            fig, self._animate_scatter, fargs=[location_hist, fitness_hist, scatter], blit=True
        )

        if save:
            ani.save(file_name)

        plt.show()


    def _animate_line(self, i, gbest_hist, mean_fit, lines):
        # print(f"{i=}; {gbest_hist[:i]}")
        lines[0].set_data(range(i), gbest_hist[:i])
        lines[1].set_data(range(i), mean_fit[:i])
        return lines

    def _animate_scatter(self, i, location_hist, fit_hist, scatter):
        scatter.set_data(location_hist[i, :, 0], location_hist[i, :, 1])
        scatter.set_3d_properties(fit_hist[i])

        return scatter,

    def _precompute_grid(self):
        x = np.linspace(self.x_min, self.x_max, STEP)
        y = np.linspace(self.y_min, self.y_max, STEP)

        X, Y = np.meshgrid(x, y)

        points = np.asarray([(p1, p2) for p1 in x for p2 in y])
        
        return X, Y, points

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