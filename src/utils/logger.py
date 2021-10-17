import numpy as np

class Logger():
    def __init__(self, number_of_iterations, number_of_particles, n_dim):
        self.location_hist = np.full((number_of_iterations, number_of_particles, n_dim), np.Inf)
        self.fitness_hist = np.full((number_of_iterations, number_of_particles), np.Inf)
        self.gbest_hist = np.full((number_of_iterations), np.Inf)
        self.gbest_location = np.full((number_of_iterations, n_dim), np.Inf)
    
    def log(self, iteration, location, fitness):
        self.location_hist[iteration] = location
        self.fitness_hist[iteration] = fitness

        self.gbest_hist[iteration] = np.min(self.fitness_hist)
        self.gbest_location[iteration] = self.location_hist[np.unravel_index(self.fitness_hist.argmin(), self.fitness_hist.shape)] 
    
    def report(self):
        return self.location_hist, self.fitness_hist, self.gbest_hist, self.gbest_location