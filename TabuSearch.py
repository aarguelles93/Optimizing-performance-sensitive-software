import random
import time

class TabuSearch:
    def __init__(self, num_iterations=100, tabu_list_size=10):
        self.num_iterations = num_iterations
        self.tabu_list_size = tabu_list_size
        self.tabu_list = []

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

    def create_solution(self):
        return [random.uniform(-10, 10) for _ in range(10)]

    def get_neighbors(self, solution):
        neighbors = []
        for i in range(len(solution)):
            neighbor = solution[:]
            neighbor[i] = random.uniform(-10, 10)
            neighbors.append(neighbor)
        return neighbors

    def optimize(self):
        best_solution = self.create_solution()
        best_eval = self.evaluate(best_solution)
        self.tabu_list.append(best_solution)
        results = [{'Iteration': 0, 'Best Eval': best_eval}]

        for i in range(self.num_iterations):
            neighbors = self.get_neighbors(best_solution)
            neighbors = [n for n in neighbors if n not in self.tabu_list]
            if not neighbors:
                continue

            best_neighbor = min(neighbors, key=self.evaluate)
            best_neighbor_eval = self.evaluate(best_neighbor)

            if best_neighbor_eval < best_eval:
                best_solution = best_neighbor
                best_eval = best_neighbor_eval
                self.tabu_list.append(best_solution)
                if len(self.tabu_list) > self.tabu_list_size:
                    self.tabu_list.pop(0)

            results.append({'Iteration': i + 1, 'Best Eval': best_eval})
            print(f"Iteration {i + 1}: Best Eval = {best_eval:.10f} ms")

        return best_solution, best_eval, results
