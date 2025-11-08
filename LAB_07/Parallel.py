import random

# Fix random seed for reproducibility
random.seed(42)

# Job processing times for 6 jobs
jobs = [4, 7, 2, 6, 5, 3]  # total workload = 27
N = len(jobs)
M = 3  # number of machines

# Grid size (5x5 = 25 cells)
grid_size = (5, 5)
iterations = 5

# Moore neighborhood (including self)
neighbors_offsets = [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1), (0, 0), (0, 1),
                     (1, -1), (1, 0), (1, 1)]

def initialize_population():
    population = []
    for _ in range(grid_size[0] * grid_size[1]):
        # Random assignment of jobs to machines
        assignment = [random.randint(0, M - 1) for _ in range(N)]
        population.append(assignment)
    return population

def calculate_makespan(assignment):
    loads = [0] * M
    for job_idx, machine in enumerate(assignment):
        loads[machine] += jobs[job_idx]
    return max(loads)

def get_neighbors_indices(idx):
    row = idx // grid_size[1]
    col = idx % grid_size[1]
    neighbors = []
    for dr, dc in neighbors_offsets:
        nr, nc = row + dr, col + dc
        if 0 <= nr < grid_size[0] and 0 <= nc < grid_size[1]:
            neighbors.append(nr * grid_size[1] + nc)
    return neighbors

def update_assignment(current, best_neighbor):
    # Copy one job assignment from best neighbor
    new_assignment = current[:]
    job_to_change = random.randint(0, N - 1)
    new_assignment[job_to_change] = best_neighbor[job_to_change]
    return new_assignment

# === PCA Execution ===

population = initialize_population()
best_solution = None
best_fitness = float('inf')

for iter in range(1, iterations + 1):
    fitness_values = [calculate_makespan(ind) for ind in population]

    # Track best
    min_fitness = min(fitness_values)
    if min_fitness < best_fitness:
        best_fitness = min_fitness
        best_solution = population[fitness_values.index(min_fitness)]

    # Display current best
    print(f"\nIteration {iter}")
    print("Best Makespan:", best_fitness)
    print("Best State: S =", best_solution)

    # Update population
    new_population = []
    for idx, individual in enumerate(population):
        neighbors = get_neighbors_indices(idx)
        neighbor_fitness = [(calculate_makespan(population[n]), n) for n in neighbors]
        best_neighbor_idx = min(neighbor_fitness)[1]
        best_neighbor = population[best_neighbor_idx]
        updated_assignment = update_assignment(individual, best_neighbor)
        new_population.append(updated_assignment)

    population = new_population
