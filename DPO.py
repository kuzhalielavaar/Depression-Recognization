import numpy as np
import time


def normalize_population(population):
    """Normalize the population to a unit hypercube [0, 1]."""
    return (population - np.min(population, axis=0)) / (np.max(population, axis=0) - np.min(population, axis=0))


# dolphin_population_optimization
def DPO(population, evaluate_function, lb, ub, num_iterations):
    num_dolphins, dimensions = population.shape
    # Step 1: Initialize the population of dolphins (positions)
    population = np.random.rand(num_dolphins, dimensions)  # Random initial positions
    population = normalize_population(population)  # Normalize the population
    population = np.clip(population, lb, ub)

    # Step 2: Evaluate initial function values
    fitness_values = np.array([evaluate_function(population[i]) for i in range(num_dolphins)])

    best_dolphin_index = np.argmin(fitness_values)
    best_position = population[best_dolphin_index]
    best_fitness_value = fitness_values[best_dolphin_index]

    convergence = np.zeros(num_iterations)
    ct = time.time()
    # Main loop (Step 4)
    for iter in range(num_iterations):
        # Step 6: Evaluate each dolphin's position
        for j in range(num_dolphins):
            fitness_values[j] = evaluate_function(population[j])

        # Step 10: Find the best dolphin's position again
        best_dolphin_index = np.argmin(fitness_values)
        best_position = population[best_dolphin_index]
        best_fitness_value = fitness_values[best_dolphin_index]

        # Step 11: Update velocities and positions
        attraction_coef = 0.5  # Example coefficient, adjust as necessary
        phi = np.random.rand(num_dolphins, dimensions)  # Random phi values

        for j in range(num_dolphins):
            velocity = attraction_coef * (best_position - population[j]) + phi * np.random.uniform(-1, 1,
                                                                                                   population[j].shape)
            population[j] = population[j] + velocity[j]

            # Optional: Normalize the new position again if necessary
            population[j] = np.clip(population[j], 0, 1)  # Ensure positions stay within [0, 1]
        convergence[iter] = best_fitness_value

    ct = time.time() - ct
    return best_fitness_value, convergence, best_position, ct
