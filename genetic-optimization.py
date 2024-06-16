import random
import time
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import pandas as pd

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
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Create toolbox
toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -10, 10)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 10)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Performance metrics evaluation
def performance_metrics(func, *args):
    """
    Measures the execution time of a function.

    Args:
    func (callable): The target function to be measured.
    *args: Arguments to be passed to the target function.

    Returns:
    tuple: A tuple containing the execution time (ms).
    """
    start_time = time.perf_counter()
    result = func(*args)
    end_time = time.perf_counter()
    execution_time = (end_time - start_time) * 1000  
    return execution_time, result

# Fitness evaluation function
def evaluate(individual):
    """
    Evaluates the fitness of an individual by calculating its execution time.

    Args:
    individual (list): The individual (modified code parameters) to be evaluated.

    Returns:
    float: The execution time (ms) as the fitness value.
    """
    x = list(individual)
    execution_time, _ = performance_metrics(target_function, x)
    return (execution_time,)

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

def select(population):
    """
    Selects individuals from the population using tournament selection.

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
    logbook.header = ["gen", "min_time_ms", "avg_time_ms", "median_time_ms", "variance_time_ms", "range_time_ms", "improvement_time"]

    min_time_list = []
    avg_time_list = []
    median_time_list = []
    variance_time_list = []
    range_time_list = []

    initial_time, _ = performance_metrics(target_function, [random.uniform(-10, 10) for _ in range(10)])
    print(f"Initial execution time: {initial_time:.10f} ms")
    
    results = []

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
                operator_choice = random.choice([toolbox.mutate, toolbox.mutate_shuffle, toolbox.mutate_uniform])
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
        median_time = np.median([fit[0] for fit in fits])
        variance_time = np.var([fit[0] for fit in fits])
        range_time = max([fit[0] for fit in fits]) - min([fit[0] for fit in fits])

        improvement_time = ((initial_time - min_time) / initial_time) * 100 if initial_time != 0 else 0

        min_time_list.append(min_time)
        avg_time_list.append(avg_time)
        median_time_list.append(median_time)
        variance_time_list.append(variance_time)
        range_time_list.append(range_time)

        logbook.record(gen=gen, min_time_ms=min_time, avg_time_ms=avg_time, median_time_ms=median_time, variance_time_ms=variance_time, range_time_ms=range_time, improvement_time=improvement_time)

        results.append({
            'Generation': gen,
            'Min Execution Time (ms)': min_time,
            'Avg Execution Time (ms)': avg_time,
            'Median Execution Time (ms)': median_time,
            'Variance Execution Time (ms)': variance_time,
            'Range Execution Time (ms)': range_time,
            'Improvement Time (%)': improvement_time
        })
        
        print(f"\nGeneration {gen}:")
        print(f"  Min Time = {min_time:.10f} ms")
        print(f"  Avg Time = {avg_time:.10f} ms")
        print(f"  Median Time = {median_time:.10f} ms")
        print(f"  Variance Time = {variance_time:.10f} ms")
        print(f"  Range Time = {range_time:.10f} ms")
        print(f"  Improvement Time = {improvement_time:.2f}%")

    
    results_df = pd.DataFrame(results)
    results_df.to_csv('optimization_results.csv', index=False)
    print("\nResults saved to optimization_results.csv")

    best_ind = tools.selBest(population, 1)[0]
    print("\nBest individual (parameters):", best_ind)
    print("Best fitness (execution time in ms):", best_ind.fitness.values)

    return min_time_list, avg_time_list, median_time_list, variance_time_list, range_time_list, initial_time, results

# Plotting the results
def plot_results(min_time_list, avg_time_list, median_time_list):
    """
    Plots the performance metrics over generations.

    Args:
    min_time_list (list): Minimum execution times over generations.
    avg_time_list (list): Average execution times over generations.
    median_time_list (list): Median execution times over generations.
    """
    generations = list(range(len(min_time_list)))

    plt.figure(figsize=(12, 8))

    # Execution Time Plot
    plt.subplot(311)
    plt.plot(generations, min_time_list, label='Min Execution Time (ms)')
    plt.plot(generations, avg_time_list, label='Avg Execution Time (ms)')
    plt.plot(generations, median_time_list, label='Median Execution Time (ms)')
    plt.xlabel('Generation')
    plt.ylabel('Execution Time (ms)')
    plt.title('Execution Time Over Generations')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Run optimization and plot results
min_time_list, avg_time_list, median_time_list, variance_time_list, range_time_list, initial_time, results = optimize_code(pop_size=100, ngen=50, cxpb=0.5, mutpb=0.2)
plot_results(min_time_list, avg_time_list, median_time_list)

# Final Summary of Improvements
if min_time_list:
    final_time = min_time_list[-1]
    improvement_time = ((initial_time - final_time) / initial_time) * 100 if initial_time != 0 else 0

    print(f"\nInitial Execution Time: {initial_time:.10f} ms")
    print(f"Final Execution Time: {final_time:.10f} ms")
    print(f"Improvement in Execution Time: {improvement_time:.2f} %")
else:
    print("No valid generations to report.")


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


def save_statistics_to_file(statistics, filename="statistics.txt"):
    """
    Save detailed statistics to a text file.

    Args:
    statistics (list): List of dictionaries containing statistics.
    filename (str): The name of the file to save the statistics to.
    """
    with open(filename, "w") as file:
        for stat in statistics:
            file.write(f"Generation {stat['Generation']}:\n")
            file.write(f"  Min Execution Time (ms) = {stat['Min Execution Time (ms)']:.10f}\n")
            file.write(f"  Avg Execution Time (ms) = {stat['Avg Execution Time (ms)']:.10f}\n")
            file.write(f"  Median Execution Time (ms) = {stat['Median Execution Time (ms)']:.10f}\n")
            file.write(f"  Variance Execution Time (ms) = {stat['Variance Execution Time (ms)']:.10f}\n")
            file.write(f"  Range Execution Time (ms) = {stat['Range Execution Time (ms)']:.10f}\n")
            file.write(f"  Improvement Time (%) = {stat['Improvement Time (%)']:.2f}\n\n")
    print(f"\nDetailed statistics saved to {filename}")


save_statistics_to_file(results)
