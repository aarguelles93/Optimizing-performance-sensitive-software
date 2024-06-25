import random
import time

class GeneticAlgorithm:
    def __init__(self, pop_size=100, num_generations=50, cxpb=0.5, mutpb=0.2):
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.population = self.create_population()

    def target_function(self, x):
        return sum([i**2 for i in x])

    def performance_metrics(self, func, *args):
        start_time = time.perf_counter()
        result = func(*args)
        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000  
        return execution_time, result

    def evaluate(self, individual):
        x = list(individual)
        execution_time, _ = self.performance_metrics(self.target_function, x)
        return execution_time

    def create_individual(self):
        return [random.uniform(-10, 10) for _ in range(10)]

    def create_population(self):
        return [self.create_individual() for _ in range(self.pop_size)]

    def select(self):
        selected = []
        for _ in range(len(self.population)):
            ind1, ind2 = random.sample(self.population, 2)
            selected.append(ind1 if self.evaluate(ind1) < self.evaluate(ind2) else ind2)
        return selected

    def crossover(self, ind1, ind2):
        if random.random() < self.cxpb:
            point = random.randint(1, len(ind1)-1)
            ind1[point:], ind2[point:] = ind2[point:], ind1[point:]
        return ind1, ind2

    def mutate(self, individual):
        if random.random() < self.mutpb:
            point = random.randint(0, len(individual)-1)
            individual[point] = random.uniform(-10, 10)
        return individual

    def optimize(self):
        best_ind = min(self.population, key=self.evaluate)
        best_eval = self.evaluate(best_ind)
        results = [{'Generation': 0, 'Best Eval': best_eval}]

        for gen in range(self.num_generations):
            selected = self.select()
            children = []
            for i in range(0, len(selected), 2):
                ind1, ind2 = selected[i], selected[i+1]
                ind1, ind2 = self.crossover(ind1, ind2)
                ind1 = self.mutate(ind1)
                ind2 = self.mutate(ind2)
                children.extend([ind1, ind2])
            self.population = children
            best_ind = min(self.population, key=self.evaluate)
            best_eval = self.evaluate(best_ind)
            results.append({'Generation': gen + 1, 'Best Eval': best_eval})
            print(f"Generation {gen + 1}: Best Eval = {best_eval:.10f} ms")

        return best_ind, best_eval, results
