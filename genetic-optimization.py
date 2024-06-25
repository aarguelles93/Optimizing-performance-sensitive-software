import pandas as pd
import matplotlib.pyplot as plt
from GeneticAlgorithm import GeneticAlgorithm
from TabuSearch import TabuSearch

class GeneticOptimization:
    def __init__(self, ga_params, ts_params):
        self.ga = GeneticAlgorithm(**ga_params)
        self.ts = TabuSearch(**ts_params)

    def run_optimization(self):
        print("Running Genetic Algorithm (GA) optimization...")
        best_ind_ga, best_eval_ga, ga_results = self.ga.optimize()
        print(f"GA Best Evaluation: {best_eval_ga:.10f} ms")

        print("Running Tabu Search (TS) optimization...")
        best_ind_ts, best_eval_ts, ts_results = self.ts.optimize()
        print(f"TS Best Evaluation: {best_eval_ts:.10f} ms")

        self.save_results_to_file(ga_results, 'ga_optimization_results.csv')
        self.save_results_to_file(ts_results, 'ts_optimization_results.csv')

        self.plot_results(ga_results, ts_results)

        return best_eval_ga, best_eval_ts

    def save_results_to_file(self, results, filename):
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

    def plot_results(self, ga_results, ts_results):
        ga_gens = [result['Generation'] for result in ga_results]
        ga_evals = [result['Best Eval'] for result in ga_results]

        ts_gens = [result['Iteration'] for result in ts_results]
        ts_evals = [result['Best Eval'] for result in ts_results]

        plt.figure(figsize=(12, 8))

        plt.plot(ga_gens, ga_evals, label='Genetic Algorithm (GA)')
        plt.plot(ts_gens, ts_evals, label='Tabu Search (TS)')

        plt.xlabel('Generation/Iteration')
        plt.ylabel('Best Evaluation (ms)')
        plt.title('GA vs TS Optimization Performance')
        plt.legend()

        plt.tight_layout()
        plt.show()

def main():
    # Parameters for GA and TS
    ga_params = {
        'pop_size': 100,
        'num_generations': 50,
        'cxpb': 0.5,
        'mutpb': 0.2
    }

    ts_params = {
        'num_iterations': 100,
        'tabu_list_size': 10
    }

    # Create GeneticOptimization instance and run optimization
    optimizer = GeneticOptimization(ga_params, ts_params)
    best_eval_ga, best_eval_ts = optimizer.run_optimization()

if __name__ == "__main__":
    main()
