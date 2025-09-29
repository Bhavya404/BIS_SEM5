import numpy as np

# Knapsack Problem
def knapsack_fitness(solution, values, weights, capacity):
    total_value = np.sum(values * solution)
    total_weight = np.sum(weights * solution)
    if total_weight > capacity:
        return 0  # penalty for invalid solution
    return total_value

# Grey Wolf Optimizer for Knapsack
def GWO_knapsack(values, weights, capacity, n_wolves=15, max_iter=50):
    n_items = len(values)

    # Initialize wolves randomly (binary positions)
    positions = np.random.randint(0, 2, (n_wolves, n_items))
    fitness = np.array([knapsack_fitness(pos, values, weights, capacity) for pos in positions])

    # Identify Alpha, Beta, Delta
    alpha, beta, delta = None, None, None
    alpha_score, beta_score, delta_score = -1, -1, -1  # maximizing

    for i in range(n_wolves):
        if fitness[i] > alpha_score:
            alpha_score, alpha = fitness[i], positions[i].copy()
        elif fitness[i] > beta_score:
            beta_score, beta = fitness[i], positions[i].copy()
        elif fitness[i] > delta_score:
            delta_score, delta = fitness[i], positions[i].copy()

    # Main loop
    for iteration in range(max_iter):
        a = 2 - (2 * iteration / max_iter)  # decrease linearly

        for i in range(n_wolves):
            for d in range(n_items):
                r1, r2 = np.random.rand(), np.random.rand()
                A1, C1 = 2*a*r1 - a, 2*r2
                r1, r2 = np.random.rand(), np.random.rand()
                A2, C2 = 2*a*r1 - a, 2*r2
                r1, r2 = np.random.rand(), np.random.rand()
                A3, C3 = 2*a*r1 - a, 2*r2

                # Influence by alpha, beta, delta
                X1 = alpha[d] - A1 * abs(C1 * alpha[d] - positions[i][d])
                X2 = beta[d]  - A2 * abs(C2 * beta[d]  - positions[i][d])
                X3 = delta[d] - A3 * abs(C3 * delta[d] - positions[i][d])

                # Update continuous position
                new_pos = (X1 + X2 + X3) / 3

                # Binarization using sigmoid
                sigmoid = 1 / (1 + np.exp(-new_pos))
                positions[i][d] = 1 if np.random.rand() < sigmoid else 0

            # Recalculate fitness
            fitness[i] = knapsack_fitness(positions[i], values, weights, capacity)

        # Update Alpha, Beta, Delta
        for i in range(n_wolves):
            if fitness[i] > alpha_score:
                delta_score, delta = beta_score, beta.copy()
                beta_score, beta = alpha_score, alpha.copy()
                alpha_score, alpha = fitness[i], positions[i].copy()
            elif fitness[i] > beta_score:
                delta_score, delta = beta_score, beta.copy()
                beta_score, beta = fitness[i], positions[i].copy()
            elif fitness[i] > delta_score:
                delta_score, delta = fitness[i], positions[i].copy()

        # Progress
        print(f"Iteration {iteration+1}, Best Value: {alpha_score}")

    return alpha, alpha_score


# Example usage
if __name__ == "__main__":
    # Knapsack example
    values  = np.array([60, 500, 120])  # profits
    weights = np.array([10, 20, 30])    # weights
    capacity = 50

    best_solution, best_value = GWO_knapsack(values, weights, capacity, n_wolves=20, max_iter=50)

    print("\nBest Solution (Items Selected):", best_solution)
    print("Best Value Achieved:", best_value)
