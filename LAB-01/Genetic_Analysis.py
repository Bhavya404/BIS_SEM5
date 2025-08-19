import random

# Define the Knapsack problem: item weights, values, and maximum capacity
WEIGHTS = [2, 3, 4, 5, 9]  # Weights of items
VALUES = [3, 4, 5, 6, 10]  # Values of items
MAX_WEIGHT = 10           # Maximum weight capacity of the knapsack
NUM_ITEMS = len(WEIGHTS)

def initialize_population(size, num_items):
    """Initialize population with binary representation of inclusion/exclusion of items."""
    return [[random.randint(0, 1) for _ in range(num_items)] for _ in range(size)]

def fitness(individual):
    """
    Fitness function: sum of values of selected items, penalized if weight exceeds MAX_WEIGHT.
    An individual is a list of 0s and 1s representing item selection.
    """
    total_value = 0
    total_weight = 0
    for i in range(len(individual)):
        if individual[i] == 1:
            total_value += VALUES[i]
            total_weight += WEIGHTS[i]
    
    # Penalize if total weight exceeds the capacity by returning 0 fitness
    if total_weight > MAX_WEIGHT:
        return 0
    return total_value

def calculate_fitness(population):
    """Calculate the fitness for each individual in the population."""
    return [fitness(individual) for individual in population]

def select_mating_pool(population, fitness_scores, num_parents):
    """Select mating pool based on fitness using roulette wheel selection."""
    total_fitness = sum(fitness_scores)
    
    # If all individuals have a fitness of 0, select parents randomly.
    if total_fitness == 0:
        return random.sample(population, k=num_parents)
    
    # Use random.choices for weighted selection, which is more efficient.
    parents = random.choices(
        population=population,
        weights=fitness_scores,
        k=num_parents
    )
    return parents

def crossover(parents, offspring_size):
    """Crossover operation (single-point crossover)."""
    offspring = []
    for _ in range(offspring_size):
        parent1 = random.choice(parents)
        parent2 = random.choice(parents)
        
        # Choose a random crossover point (ensuring it's not at the very end)
        crossover_point = random.randint(1, NUM_ITEMS - 1)
        
        # Create child by combining parts from both parents
        child = parent1[:crossover_point] + parent2[crossover_point:]
        offspring.append(child)
    return offspring

def mutate(population, mutation_rate):
    """Mutation operation (randomly flip bits in the binary string)."""
    mutated_pop = []
    for individual in population:
        mutated_individual = list(individual) # Create a mutable copy
        for i in range(len(mutated_individual)):
            if random.random() < mutation_rate:
                # Flip the bit (0 becomes 1, 1 becomes 0)
                mutated_individual[i] = 1 - mutated_individual[i]
        mutated_pop.append(mutated_individual)
    return mutated_pop

def genetic_algorithm(pop_size, max_generations, num_parents, mutation_rate, patience):
    """
    Main function to run the genetic algorithm.
    'patience' is the number of generations to wait without improvement before stopping.
    """
    # 1. Initialization
    population = initialize_population(pop_size, NUM_ITEMS)
    
    best_fitness = 0
    best_individual = []
    unchanged_generations = 0

    for gen in range(max_generations):
        # 2. Fitness Calculation
        fitness_scores = calculate_fitness(population)
        
        # Find the best individual in the current generation
        current_max_fitness = max(fitness_scores)
        max_fitness_idx = fitness_scores.index(current_max_fitness)
        current_best_individual = population[max_fitness_idx]
        
        # Update the overall best solution found so far
        if current_max_fitness > best_fitness:
            best_fitness = current_max_fitness
            best_individual = current_best_individual
            unchanged_generations = 0
        else:
            unchanged_generations += 1

        print(f"Generation {gen + 1}: Best fitness={best_fitness}, Items={best_individual}")

        # Check for convergence (early stopping)
        if unchanged_generations >= patience:
            print(f"Converged after {gen + 1} generations.")
            break

        # 3. Selection
        parents = select_mating_pool(population, fitness_scores, num_parents)
        
        # 4. Crossover
        offspring_size = pop_size - len(parents)
        offspring = crossover(parents, offspring_size)
        
        # 5. Mutation
        mutated_offspring = mutate(offspring, mutation_rate)
        
        # 6. Create New Population
        # The new population consists of the best parents and the mutated offspring
        population = parents + mutated_offspring

    return best_individual

if __name__ == "__main__":
    result = genetic_algorithm(
        pop_size=10, 
        max_generations=100, 
        num_parents=4, 
        mutation_rate=0.1,
        patience=10
    )
    final_fitness = fitness(result)
    print("\n" + "="*40)
    print(f"Optimal solution found: Items selected: {result}")
    print(f"Fitness (Total Value): {final_fitness}")
    print("="*40)
