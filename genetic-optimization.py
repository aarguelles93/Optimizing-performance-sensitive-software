import random
import time
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import psutil
import os

# Define the target function

def target_function(x):
    """
    A numerical computation function to calculate the sum of squares.
    
    Args:
    x (list): A list of numbers.

    Returns:
    int: The sum of squares of the list elements.
    """
    return sum([i**2 for i in x])

# Define the original function

def original_function(x):
    """
    The original function to calculate the squares of a list of numbers.

    Args:
    x (list): A list of numbers.

    Returns:
    list: The squares of the list elements.
    """
    return [i**2 for i in x]

# Create types

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Create toolbox

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -10, 10)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 10)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Performance metrics evaluation

def performance_metrics(func, *args):
    """
    Measures the execution time and memory usage of a function.

    Args:
    func (callable): The target function to be measured.
    *args: Arguments to be passed to the target function.

    Returns:
    tuple: A tuple containing the execution time (ms) and memory usage (MB).
    """
    process = psutil.Process(os.getpid())
    start_time = time.perf_counter()
    result = func(*args)
    end_time = time.perf_counter()
    execution_time = (end_time - start_time) * 1000  
    memory_usage = process.memory_info().rss / (1024 * 1024) 
    return execution_time, memory_usage, result

# Fitness evaluation function

def evaluate(individual):
    """
    Evaluates the fitness of an individual by calculating its execution time and memory usage.

    Args:
    individual (list): The individual (modified code parameters) to be evaluated.

    Returns:
    tuple: The execution time (ms) and memory usage (MB) as the fitness values.
    """
    x = list(individual)
    execution_time, memory_usage, _ = performance_metrics(target_function, x)
    return execution_time, memory_usage

# Additional Genetic operators

def mate(ind1, ind2):
    """
    Applies blend crossover between two individuals.

    Args:
    ind1 (Individual): The first individual.
    ind2 (Individual): The second individual.

    Returns:
    tuple: The two individuals after crossover.
    """
    tools.cxBlend(ind1, ind2, alpha=0.5)
    return ind1, ind2

def mate_two_point(ind1, ind2):
    """
    Applies two-point crossover between two individuals.

    Args:
    ind1 (Individual): The first individual.
    ind2 (Individual): The second individual.

    Returns:
    tuple: The two individuals after crossover.
    """
    tools.cxTwoPoint(ind1, ind2)
    return ind1, ind2

def mate_uniform(ind1, ind2):
    """
    Applies uniform crossover between two individuals.

    Args:
    ind1 (Individual): The first individual.
    ind2 (Individual): The second individual.

    Returns:
    tuple: The two individuals after crossover.
    """
    tools.cxUniform(ind1, ind2, indpb=0.5)
    return ind1, ind2

def mutate(individual):
    """
    Applies Gaussian mutation to an individual.

    Args:
    individual (Individual): The individual to be mutated.

    Returns:
    tuple: The individual after mutation.
    """
    tools.mutGaussian(individual, mu=0, sigma=1, indpb=0.2)
    return individual,

def mutate_shuffle(individual):
    """
    Applies shuffle mutation to an individual.

    Args:
    individual (Individual): The individual to be mutated.

    Returns:
    tuple: The individual after mutation.
    """
    tools.mutShuffleIndexes(individual, indpb=0.2)
    return individual,

def mutate_uniform(individual):
    """
    Applies uniform mutation to an individual.

    Args:
    individual (Individual): The individual to be mutated.

    Returns:
    tuple: The individual after mutation.
    """
    tools.mutUniformInt(individual, low=-10, up=10, indpb=0.2)
    return individual,

def mutate_memory_optimized(individual):
    """
    Applies a custom mutation to an individual focusing on reducing memory usage.

    Args:
    individual (Individual): The individual to be mutated.

    Returns:
    tuple: The individual after mutation.
    """
    for i in range(len(individual)):
        if random.random() < 0.2:
            individual[i] = random.uniform(-5, 5)  # Reduce range to potentially lower memory footprint
    return individual,

def select(population):
    """
    Selects individuals from the population using a tournament selection method.

    Args:
    population (list): The population from which individuals are selected.

    Returns:
    list: The selected individuals.
    """
    return tools.selTournament(population, k=len(population), tournsize=3)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", mate)
toolbox.register("mate_two_point", mate_two_point)
toolbox.register("mate_uniform", mate_uniform)
toolbox.register("mutate", mutate)
toolbox.register("mutate_shuffle", mutate_shuffle)
toolbox.register("mutate_uniform", mutate_uniform)
toolbox.register("mutate_memory_optimized", mutate_memory_optimized)
toolbox.register("select", select)

# Optimization process

def optimize_code(pop_size=100, ngen=50, cxpb=0.5, mutpb=0.2):
    """
    Runs the genetic algorithm to optimize the code.

    Initializes the population, applies genetic operators, and evaluates the fitness of the individuals
    over a number of generations to find the best performing code modifications.
    """
    print(f"Starting optimization with population size={pop_size}, generations={ngen}, crossover probability={cxpb}, mutation probability={mutpb}")

    population = toolbox.population(n=pop_size)
    logbook = tools.Logbook()
    logbook.header = ["gen", "min_time_ms", "avg_time_ms", "min_mem_mb", "avg_mem_mb", "improvement_time", "improvement_mem"]

    min_time_list = []
    avg_time_list = []
    min_mem_list = []
    avg_mem_list = []

    initial_time, initial_mem, _ = performance_metrics(target_function, [random.uniform(-10, 10) for _ in range(10)])
    print(f"Initial execution time: {initial_time:.6f} ms")
    print(f"Initial memory usage: {initial_mem:.6f} MB")
    
    for gen in range(ngen):
        offspring = toolbox.select(population)
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                operator_choice = random.choice([toolbox.mate, toolbox.mate_two_point, toolbox.mate_uniform])
                operator_choice(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutpb:
                operator_choice = random.choice([toolbox.mutate, toolbox.mutate_shuffle, toolbox.mutate_uniform, toolbox.mutate_memory_optimized])
                operator_choice(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring

        fits = [ind.fitness.values for ind in population]
        min_time = min(fit[0] for fit in fits)
        avg_time = sum(fit[0] for fit in fits) / len(population)
        min_mem = min(fit[1] for fit in fits)
        avg_mem = sum(fit[1] for fit in fits) / len(population)

        improvement_time = ((initial_time - min_time) / initial_time) * 100 if initial_time != 0 else 0
        improvement_mem = ((initial_mem - min_mem) / initial_mem) * 100 if initial_mem != 0 else 0

        min_time_list.append(min_time)
        avg_time_list.append(avg_time)
        min_mem_list.append(min_mem)
        avg_mem_list.append(avg_mem)

        logbook.record(gen=gen, min_time_ms=min_time, avg_time_ms=avg_time, min_mem_mb=min_mem, avg_mem_mb=avg_mem,
                       improvement_time=improvement_time, improvement_mem=improvement_mem)
        
        print(f"\nGeneration {gen}:")
        print(f"  Min Time = {min_time:.6f} ms")
        print(f"  Avg Time = {avg_time:.6f} ms")
        print(f"  Min Mem = {min_mem:.6f} MB")
        print(f"  Avg Mem = {avg_mem:.6f} MB")
        print(f"  Improvement Time = {improvement_time:.2f}%")
        print(f"  Improvement Mem = {improvement_mem:.2f}%")

    best_ind = tools.selBest(population, 1)[0]
    print("\nBest individual (parameters):", best_ind)
    print("Best fitness (execution time in ms, memory usage in MB):", best_ind.fitness.values)

    return min_time_list, avg_time_list, min_mem_list, avg_mem_list, initial_time, initial_mem

# Plotting the results

def plot_results(min_time_list, avg_time_list, min_mem_list, avg_mem_list):
    """
    Plots the performance metrics over generations.

    Args:
    min_time_list (list): Minimum execution times over generations.
    avg_time_list (list): Average execution times over generations.
    min_mem_list (list): Minimum memory usages over generations.
    avg_mem_list (list): Average memory usages over generations.
    """
    generations = list(range(len(min_time_list)))

    plt.figure(figsize=(14, 7))

    # Execution Time Plot

    plt.subplot(1, 2, 1)
    plt.plot(generations, min_time_list, label='Min Execution Time (ms)')
    plt.plot(generations, avg_time_list, label='Avg Execution Time (ms)')
    plt.xlabel('Generation')
    plt.ylabel('Execution Time (ms)')
    plt.title('Execution Time Over Generations')
    plt.legend()

    # Memory Usage Plot

    plt.subplot(1, 2, 2)
    plt.plot(generations, min_mem_list, label='Min Memory Usage (MB)')
    plt.plot(generations, avg_mem_list, label='Avg Memory Usage (MB)')
    plt.xlabel('Generation')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage Over Generations')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Run optimization and plot results

min_time_list, avg_time_list, min_mem_list, avg_mem_list, initial_time, initial_mem = optimize_code(pop_size=100, ngen=50, cxpb=0.5, mutpb=0.2)
plot_results(min_time_list, avg_time_list, min_mem_list, avg_mem_list)

# Final Summary of Improvements

if min_time_list:
    final_time = min_time_list[-1]
    final_mem = min_mem_list[-1]
    improvement_time = ((initial_time - final_time) / initial_time) * 100 if initial_time != 0 else 0
    improvement_mem = ((initial_mem - final_mem) / initial_mem) * 100 if initial_mem != 0 else 0

    print(f"\nInitial Execution Time: {initial_time:.2f} ms")
    print(f"Final Execution Time: {final_time:.2f} ms")
    print(f"Improvement in Execution Time: {improvement_time:.2f} %")

    print(f"\nInitial Memory Usage: {initial_mem:.2f} MB")
    print(f"Final Memory Usage: {final_mem:.2f} MB")
    print(f"Improvement in Memory Usage: {improvement_mem:.2f} %")
else:
    print("No valid generations to report.")

# Functionality verification

def test_optimized_function(original_func, optimized_func):
    """
    Tests the optimized function to ensure it produces the same results as the original function.

    Args:
    original_func (callable): The original target function.
    optimized_func (callable): The optimized function.

    Raises:
    AssertionError: If the optimized function does not produce the same results as the original function.
    """
    test_cases = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [-1, -2, -3, -4]
    ]
    for case in test_cases:
        original_result = original_func(case)
        optimized_result = optimized_func(case)
        assert original_result == optimized_result, f"Test failed for input {case}: {original_result} != {optimized_result}"


def optimized_function(x):
    """
    A placeholder for the optimized code generated by the genetic algorithm.

    Args:
    x (list): A list of numbers.

    Returns:
    list: The squares of the list elements.
    """
    return [i**2 for i in x]

test_optimized_function(original_function, optimized_function)
print("All tests passed!")
