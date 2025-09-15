import random
import numpy as np

# Helper function to evaluate the fitness of a tour (sum of distances)
def evaluate_tour(tour, dist_matrix):
    distance = 0
    for i in range(len(tour) - 1):
        distance += dist_matrix[tour[i]][tour[i + 1]]
    distance += dist_matrix[tour[-1]][tour[0]]  # Closing the loop
    return distance

# Helper function to generate a random tour
def generate_random_tour(n):
    return random.sample(range(n), n)

# Helper function to apply Levy flight mutation (swap or reverse a segment)
def levy_flight(tour):
    new_tour = tour[:]
    a, b = random.sample(range(len(tour)), 2)
    if a > b:
        a, b = b, a
    new_tour[a:b+1] = reversed(new_tour[a:b+1])
    return new_tour

# Main CSA_TSP procedure
def csa_tsp(dist_matrix, N, Pa, MaxT):
    n = len(dist_matrix)  # Number of cities (nodes)
    
    # Step 1: Initialization
    Xi = [generate_random_tour(n) for _ in range(N)]  # Initial nests
    Fitness = [evaluate_tour(Xi[i], dist_matrix) for i in range(N)]  # Fitness of each tour
    BestNest = min(range(N), key=lambda i: Fitness[i])  # Index of the best nest
    
    # Initialize best fitness
    best_fitness = Fitness[BestNest]
    
    # Step 2: Main loop
    for t in range(MaxT):
        print(f"Iteration {t+1}:")
        
        # Generate new solutions using Levy flight
        for i in range(N):
            NewXi = levy_flight(Xi[i])  # Apply Levy flight mutation
            NewFitness = evaluate_tour(NewXi, dist_matrix)
            
            # Fitness evaluation and replacement
            j = random.randint(0, N - 1)  # Random nest index to replace
            if NewFitness < Fitness[j]:
                Xi[j] = NewXi
                Fitness[j] = NewFitness
        
        # Abandon worst nests
        worst_nests = sorted(range(N), key=lambda i: Fitness[i], reverse=True)[:int(Pa * N)]
        for k in worst_nests:
            Xi[k] = generate_random_tour(n)  # Replace with a random tour
            Fitness[k] = evaluate_tour(Xi[k], dist_matrix)
        
        # Keep the best solution
        BestNest = min(range(N), key=lambda i: Fitness[i])
        best_fitness = Fitness[BestNest]
        
        # Output the best solution at each iteration
        print(f"Best Fitness at Iteration {t+1}: {best_fitness}")
        print(f"Best Tour at Iteration {t+1}: {Xi[BestNest]}")
        print("-" * 50)
    
    # Step 3: Return BestNest
    return Xi[BestNest], best_fitness

# Example usage
if __name__ == "__main__":
    # Example distance matrix (symmetric, for simplicity)
    D = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
    
    N = 10  # Number of nests
    Pa = 0.3  # Fraction of worst nests to abandon
    MaxT = 10  # Maximum iterations
    
    best_tour, best_fitness = csa_tsp(D, N, Pa, MaxT)
    print("Final Best Tour:", best_tour)
    print("Final Best Fitness:", best_fitness)
