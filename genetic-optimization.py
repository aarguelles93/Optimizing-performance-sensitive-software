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
        self.analyze_and_save_results(ga_results, ts_results)

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
        plt.grid(True)
        plt.savefig('ga_vs_ts_optimization_performance.png')
        plt.show()

    def analyze_and_save_results(self, ga_results, ts_results):
        # Required transformations for GA results
        ga_df = pd.DataFrame(ga_results)
        ga_df['Min Exec Time (ms)'] = ga_df['Best Eval']
        ga_df['Avg Exec Time (ms)'] = ga_df['Best Eval'].expanding().mean()
        ga_df['Median Exec Time (ms)'] = ga_df['Best Eval'].expanding().median()
        ga_df['Variance Exec Time (ms)'] = ga_df['Best Eval'].expanding().var()
        ga_df['Range Exec Time (ms)'] = ga_df['Best Eval'].expanding().apply(lambda x: x.max() - x.min())
        ga_df['Improvement Time (%)'] = (1 - (ga_df['Best Eval'] / ga_df['Best Eval'].iloc[0])) * 100
        ga_df['Std Dev Exec Time (ms)'] = ga_df['Best Eval'].expanding().std()

        # Required transformations for TS results
        ts_df = pd.DataFrame(ts_results)
        ts_df['Min Exec Time (ms)'] = ts_df['Best Eval']
        ts_df['Avg Exec Time (ms)'] = ts_df['Best Eval'].expanding().mean()
        ts_df['Median Exec Time (ms)'] = ts_df['Best Eval'].expanding().median()
        ts_df['Variance Exec Time (ms)'] = ts_df['Best Eval'].expanding().var()
        ts_df['Range Exec Time (ms)'] = ts_df['Best Eval'].expanding().apply(lambda x: x.max() - x.min())
        ts_df['Improvement Time (%)'] = (1 - (ts_df['Best Eval'] / ts_df['Best Eval'].iloc[0])) * 100
        ts_df['Std Dev Exec Time (ms)'] = ts_df['Best Eval'].expanding().std()

        # Save GA and TS summary tables
        ga_df.to_csv('ga_optimization_summary.csv', index=False)
        ts_df.to_csv('ts_optimization_summary.csv', index=False)
        print("GA and TS results summary tables created and saved.")

        # Display the table
        print("GA Optimization Results:")
        print(ga_df.head())
        print("TS Optimization Results:")
        print(ts_df.head())

        # Plot summary statistics
        self.plot_summary_statistics(ga_df, 'GA')
        self.plot_summary_statistics(ts_df, 'TS')

    def plot_summary_statistics(self, df, method):
        plt.figure(figsize=(12, 8))

        plt.plot(df.index, df['Min Exec Time (ms)'], label='Min Exec Time (ms)')
        plt.plot(df.index, df['Avg Exec Time (ms)'], label='Avg Exec Time (ms)')
        plt.plot(df.index, df['Median Exec Time (ms)'], label='Median Exec Time (ms)')
        plt.plot(df.index, df['Variance Exec Time (ms)'], label='Variance Exec Time (ms)')
        plt.plot(df.index, df['Range Exec Time (ms)'], label='Range Exec Time (ms)')
        plt.plot(df.index, df['Improvement Time (%)'], label='Improvement Time (%)')
        plt.plot(df.index, df['Std Dev Exec Time (ms)'], label='Std Dev Exec Time (ms)')

        plt.xlabel('Generation/Iteration')
        plt.ylabel('Execution Time (ms)')
        plt.title(f'{method} Optimization Summary Statistics')
        plt.legend()

        plt.tight_layout()
        plt.grid(True)
        plt.savefig(f'{method.lower()}_optimization_summary_statistics.png')
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
