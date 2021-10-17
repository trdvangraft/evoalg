import torch
import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        self.weight = torch.Tensor(in_features, out_features)
        self.bias = torch.Tensor(out_features)

        self.weight_params = in_features * out_features
        self.bias_params = out_features
    
    def get_num_params(self) -> int:
        return self.weight_params + self.bias_params
    
    def set_params(self, params: np.ndarray):
        self.weight = torch.tensor(params[:self.weight_params].reshape(self.weight.shape), dtype=torch.float32)
        self.bias = torch.tensor(params[self.weight_params:self.get_num_params()].reshape(self.bias.shape), dtype=torch.float32)

    def forward(self, x):
        return torch.mm(x, self.weight) + self.bias

class ReLU:
    def forward(self, x):
        return torch.clamp(x, min=0)
    
class Net:
    def __init__(self, layers):
        self.layers = layers
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def set_params(self, network_params: np.ndarray):
        for layer in [layer for layer in self.layers if hasattr(layer, 'set_params')]:
            params, network_params = network_params[:layer.get_num_params()], network_params[layer.get_num_params():]
            layer.set_params(params)

        return self
