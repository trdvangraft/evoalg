import numpy as np
import torch

def dist(a, b):
    return np.linalg.norm(a - b, axis=1)

def MSELoss(y_true, y_pred):
    return torch.mean((y_pred - y_true) ** 2)