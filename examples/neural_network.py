from src.algorithms.plotter import NeuralPlotter
from src.algorithms.pso import PSO
import numpy as np

import torch

class Linear(object):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.weight = torch.Tensor(in_features, out_features)
        self.bias = torch.Tensor(out_features)

    def forward(self, x):
        y = torch.mm(x, self.weight) + self.bias
        return y

class ReLU(object):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        y = torch.clamp(x, min=0)
        return y
    
class Net(object):
    def __init__(self, layers):
        self.layers = layers
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

def MSELoss(y_true, y_pred):
    return torch.mean((y_pred - y_true) ** 2)

NUMBER_OF_INPUTS = 2
NEURONS_PER_HIDDEN_LAYER = [2]
NUMBER_OF_OUTPUTS = 2

SAMPLES = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
TARGETS = torch.tensor([[1, 0], [0, 1], [0, 1], [1, 0]])

def calculate_dimension():
    n_dim = 0

    _all_neurons = [NUMBER_OF_INPUTS, *NEURONS_PER_HIDDEN_LAYER, NUMBER_OF_OUTPUTS]

    for i in range(len(_all_neurons) - 1):
        neur_curr, neur_next = _all_neurons[i], _all_neurons[i+1]
        n_dim += neur_curr * neur_next + neur_next
    
    return n_dim

def _build_network(network_parameters):
    layers = [
        Linear(NUMBER_OF_INPUTS, NEURONS_PER_HIDDEN_LAYER[0]),
        ReLU(),
        Linear(NEURONS_PER_HIDDEN_LAYER[0], NUMBER_OF_OUTPUTS)
    ]

    net = Net(layers)
    
    i = NUMBER_OF_INPUTS * NEURONS_PER_HIDDEN_LAYER[0]
    
    net.layers[0].weight = torch.tensor(network_parameters[:i].reshape((NUMBER_OF_INPUTS, NEURONS_PER_HIDDEN_LAYER[0])), dtype=torch.float32)
    net.layers[0].bias = torch.tensor(network_parameters[i:(i + NEURONS_PER_HIDDEN_LAYER[0])].reshape((1, NEURONS_PER_HIDDEN_LAYER[0])), dtype=torch.float32)
    
    i += NEURONS_PER_HIDDEN_LAYER[0]
    ii = i + NEURONS_PER_HIDDEN_LAYER[0] * NUMBER_OF_OUTPUTS

    net.layers[2].weight = torch.tensor(network_parameters[i:ii].reshape((NEURONS_PER_HIDDEN_LAYER[0], NUMBER_OF_OUTPUTS)), dtype=torch.float32)
    net.layers[2].bias = torch.tensor(network_parameters[ii:(ii + NUMBER_OF_OUTPUTS)].reshape((1, NUMBER_OF_OUTPUTS)), dtype=torch.float32)

    return net

def calculate_loss(network_parameters):
    if network_parameters.ndim == 1:
        return network_fitness(network_parameters)
    return [network_fitness(network_parameter) for network_parameter in network_parameters]


def network_fitness(network_parameter):    
    net = _build_network(network_parameter)
    y_pred = net.forward(SAMPLES)

    return MSELoss(TARGETS, y_pred)

def predict(network_parameter):
    return _build_network(network_parameter)

def main():
    print('neural network example')
    n_dim = calculate_dimension()

    optimizer = PSO(n_dim, np.array([[-1, 1]]), calculate_loss, number_of_iterations=250, number_of_particles=160, c_1=0.7, c_2=2.5)
    history = optimizer.optimize()

    location_hist, fitness_hist, gbest_hist, gbest_location = history.report()

    best_net = predict(gbest_location[-1])

    visualizer = NeuralPlotter()
    visualizer.plot_xor(SAMPLES, TARGETS, best_net, save=True, file_name="XOR_decision")
    visualizer.plot_loss(fitness_hist, gbest_hist)
    # print(predict(gbest_location[-1]))

if __name__ == '__main__':
    main()