import numpy as np
import time


# Dove Swarm Optimization (DSO)
def DSO(W, objective_function, lower_bound, upper_bound, num_epochs):
    num_doves, dimensions = W.shape
    # Initialize other required variables
    S = np.zeros(num_doves)  # Satiety degree of doves
    best_fit = np.inf  # Best fitness found
    best_position = np.zeros(dimensions)  # Best position
    ct = time.time()
    convergence = np.zeros(num_epochs)
    # Main loop
    for epoch in range(num_epochs):
        # Step 2: Evaluate fitness of each dove
        fitness = np.array([objective_function(w) for w in W])

        # Step 3: Locate the dove nearest to the largest amount of crumbs (best dove)
        # Update the best fitness and best position
        for i in range(num_doves):
            if fitness[i] < best_fit:
                best_fit = fitness[i]
                best_position = W[i]

        # Step 5: Update each dove's satiety degree
        for i in range(num_doves):
            r = np.random.rand()
            # r = fitness[i] / ( np.max(fitness) + np.mean(fitness) - np.min(fitness) + fitness[i])
            S[i] = np.max(fitness) - fitness[i] * r  # Maximize satiety

        # Step 6: Select the most satisfied dove
        best_dove_index = np.argmax(S)
        d_f = W[best_dove_index]

        W = np.clip(W, lower_bound, upper_bound)
        # Step 7: Update each dove's position vector
        for i in range(num_doves):

            max_distance = np.max(np.linalg.norm(W - d_f, axis=1))
            distance = np.linalg.norm(W[i] - d_f)
            learning_rate = 0.1  # Initial learning rate

            W[i] = W[i] + learning_rate * (d_f - W[i]) * (1 - distance / max_distance)

        convergence[epoch] = best_fit
    ct = time.time() - ct
    return best_fit, convergence, best_position, ct
