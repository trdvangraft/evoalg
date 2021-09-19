import numpy as np

def dist(a, b):
    return np.linalg.norm(a - b, axis=1)
