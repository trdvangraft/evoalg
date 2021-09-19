import numpy as np

class Logger():
    def __init__(self, number_of_iterations, number_of_particles, n_dim, fit_func):
        self.location_hist = np.zeros((number_of_iterations, number_of_particles, n_dim))
        self.fitness_hist = np.zeros((number_of_iterations, number_of_particles))
        self.gbest_hist = np.zeros((number_of_iterations))
        self.gbest_location = np.zeros((number_of_iterations, n_dim))
        self._fit_func = fit_func
    
    def log(self, iteration, location, fitness, gbest):
        self.location_hist[iteration] = location
        self.fitness_hist[iteration] = fitness
        self.gbest_hist[iteration] = self._fit_func(gbest)
        self.gbest_location[iteration] = gbest
    
    def report(self):
        return self.location_hist, self.fitness_hist, self.gbest_hist, self.gbest_location

class PSO():
    def __init__(self, n_dim, problem_ranges, fit_func, c_1=0.8, c_2=0.4, number_of_particles=20, number_of_iterations=100):
        self.n_dim = n_dim
        self.problem_ranges = problem_ranges
        self.fit_func = fit_func
        
        self.c_1 = c_1
        self.c_2 = c_2
        self.number_of_iterations = number_of_iterations
        self.number_of_particles = number_of_particles

        self.print_info()

    def optimize(self) -> Logger:
        wt = np.linspace(0.9, 0.4, self.number_of_iterations)
        location, velocity, pbest, fitness, gbest = self._initialize_particles()

        logger = Logger(self.number_of_iterations, self.number_of_particles, self.n_dim, self.fit_func)

        for i in range(self.number_of_iterations):
            r = np.random.uniform(size=(self.number_of_particles, 2))

            cognitive_comp = self.c_1 * np.diag(r[:, 0]) @ (pbest - location)
            social_comp = self.c_2 * np.diag(r[:, 1]) @ (gbest - location)

            new_velocity = wt[i] * velocity + cognitive_comp + social_comp
            new_location = np.clip(location + new_velocity, self.problem_ranges[:, 0], self.problem_ranges[:, 1])
            new_fitness = self.fit_func(new_location)

            pbest[new_fitness < fitness] = new_location[new_fitness < fitness]
            gbest = new_location[np.argmin(new_fitness)] if min(new_fitness) <= self.fit_func(gbest) else gbest

            location = new_location
            velocity = new_velocity
            fitness = new_fitness

            logger.log(i, location, fitness, gbest)

        return logger
    
    def _initialize_particles(self):
        initial_location = initial_pbest = np.random.uniform(
            self.problem_ranges[:, 0],
            self.problem_ranges[:, 1],
            size=(self.number_of_particles, self.n_dim)
        )
        initial_velocity = np.random.uniform(
            self.problem_ranges[:, 0],
            self.problem_ranges[:, 1],
            size=(self.number_of_particles, self.n_dim)
        )
        initial_fitness = self.fit_func(initial_location)
        gbest = initial_location[np.argmin(initial_fitness)]

        return initial_location, initial_velocity, initial_pbest, initial_fitness, gbest
    
    def print_info(self):
        print(
            f"Dimensions: {self.n_dim}, \n \
            iterations: {self.number_of_iterations}, \n \
            particles: {self.number_of_particles}, \n \
            c_1: {self.c_1} \n \
            c_2: {self.c_2}"
        )