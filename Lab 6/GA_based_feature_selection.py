import random

def simple_fitness(chromosome, X, y):
    """Simple fitness: count of selected features (demo only)"""
    # In real lab this would be SVM accuracy
    selected = [i for i, bit in enumerate(chromosome) if bit == 1]
    if not selected:
        return 0
    return len(selected)   # placeholder — replace with real accuracy if needed

def genetic_algorithm(n_features=4, pop_size=10, max_gen=20, mutation_rate=0.1):
    population = [[random.randint(0, 1) for _ in range(n_features)] for _ in range(pop_size)]
    avg_fitness_history = []
    
    for gen in range(max_gen):
        fitnesses = [simple_fitness(ind, X, y) for ind in population]
        avg_fit = sum(fitnesses) / len(fitnesses)
        avg_fitness_history.append(avg_fit)
        
        # Early stop if no improvement
        if gen > 5 and abs(avg_fitness_history[-1] - avg_fitness_history[-6]) < 0.01:
            print(f"Stopped at generation {gen}")
            break
        
        # Selection + Crossover + Mutation
        new_pop = []
        for _ in range(pop_size):
            p1 = max(random.sample(population, 3), key=lambda x: simple_fitness(x, X, y))
            p2 = max(random.sample(population, 3), key=lambda x: simple_fitness(x, X, y))
            point = random.randint(1, n_features - 1)
            child = p1[:point] + p2[point:]
            # Mutation
            for i in range(n_features):
                if random.random() < mutation_rate:
                    child[i] = 1 - child[i]
            new_pop.append(child)
        population = new_pop
    
    print("Average fitness per generation:", avg_fitness_history)
    return avg_fitness_history

print("\n=== GA Feature Selection ===")
history = genetic_algorithm()