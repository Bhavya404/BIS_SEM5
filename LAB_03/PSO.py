import numpy as np

# --- 1. Define the Problem: Mathematical Function to Optimize ---
# We are trying to minimize this function. The global minimum is at (0, 0, ..., 0) with a value of 0.
def fitness_function(position):
    """
    Calculates the fitness of a particle.
    The fitness function used here is the sum of squares, a common benchmark function.
    """
    return np.sum(position**2)

class Particle:
    """
    Represents a single particle in the swarm.
    
    Attributes:
        position (np.ndarray): The current position of the particle in the search space.
        velocity (np.ndarray): The current velocity of the particle.
        pbest_position (np.ndarray): The personal best position found by this particle so far.
        pbest_fitness (float): The fitness value of the personal best position.
    """
    def __init__(self, dimensions):
        # Initialize position and velocity with random values within constraints
        self.position = np.random.uniform(low=-10.0, high=10.0, size=dimensions)
        self.velocity = np.random.uniform(low=-1.0, high=1.0, size=dimensions)
        
        # Initialize personal best position and fitness
        self.pbest_position = self.position.copy()
        self.pbest_fitness = fitness_function(self.position)

def particle_swarm_optimization(dimensions, num_particles, max_iterations):
    """
    Executes the Particle Swarm Optimization algorithm.
    
    Args:
        dimensions (int): The number of dimensions of the search space.
        num_particles (int): The number of particles in the swarm.
        max_iterations (int): The maximum number of iterations to run the algorithm.
    """
    # --- 2. Initialize Parameters ---
    w = 0.5   # Inertia weight
    c1 = 0.8  # Cognitive coefficient
    c2 = 0.9  # Social coefficient

    # --- 3. Initialize Particles (the Swarm) ---
    swarm = [Particle(dimensions) for _ in range(num_particles)]

    # --- Initialize Global Best ---
    # Start with infinite fitness and no position
    gbest_fitness = float('inf')
    gbest_position = np.zeros(dimensions)

    # --- 4. Evaluate Fitness of Initial Swarm ---
    # Find the best particle in the initial swarm to set the initial global best
    for particle in swarm:
        if particle.pbest_fitness < gbest_fitness:
            gbest_fitness = particle.pbest_fitness
            gbest_position = particle.pbest_position.copy()
            
    # --- 6. Iterate ---
    for iteration in range(max_iterations):
        # --- 5. Update Velocities and Positions ---
        for particle in swarm:
            # Generate random numbers for cognitive and social terms
            rand1 = np.random.rand(dimensions)
            rand2 = np.random.rand(dimensions)

            # Calculate velocity components
            inertia_term = w * particle.velocity
            personal_term = c1 * rand1 * (particle.pbest_position - particle.position)
            social_term = c2 * rand2 * (gbest_position - particle.position)
            
            # Update particle's velocity and position
            particle.velocity = inertia_term + personal_term + social_term
            particle.position += particle.velocity
            
            # --- 4. Evaluate Fitness ---
            current_fitness = fitness_function(particle.position)

            # Update personal best (pbest)
            if current_fitness < particle.pbest_fitness:
                particle.pbest_fitness = current_fitness
                particle.pbest_position = particle.position.copy()

            # Update global best (gbest)
            if current_fitness < gbest_fitness:
                gbest_fitness = current_fitness
                gbest_position = particle.position.copy()
        
        # Optional: Print progress
        if iteration % 10000 == 0:
            print(f"Iteration {iteration}, Best fitness: {gbest_fitness:.6f}")
    
    # --- 7. Output the Best Solution ---
    print("\nSOLUTION FOUND:")
    print(f"  Position: {gbest_position}")
    print(f"  Fitness: {gbest_fitness}")


if __name__ == "__main__":
    # Set the parameters for the optimization
    DIMENSIONS = 2
    NUM_PARTICLES = 300
    MAX_ITERATIONS = 100000
    
    # Run the PSO algorithm
    particle_swarm_optimization(DIMENSIONS, NUM_PARTICLES, MAX_ITERATIONS)
