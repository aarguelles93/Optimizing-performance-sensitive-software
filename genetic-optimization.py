import pandas as pd
import matplotlib.pyplot as plt
import os
from GeneticAlgorithm import GeneticAlgorithm
from TabuSearch import TabuSearch

class GeneticOptimization:
    def __init__(self, ga_params, ts_params):
        self.ga_params = ga_params
        self.ts_params = ts_params
        
        if not os.path.exists('results'):
            os.makedirs('results')

    def run_single_optimization(self):
        ga = GeneticAlgorithm(**self.ga_params)
        ts = TabuSearch(**self.ts_params)
        
        print("Running Genetic Algorithm optimization...")
        best_ind_ga, best_eval_ga, ga_results = ga.optimize()
        print(f"GA Best: {best_eval_ga:.6f}")

        print("\nRunning Tabu Search optimization...")
        best_ind_ts, best_eval_ts, ts_results = ts.optimize()
        print(f"TS Best: {best_eval_ts:.6f}")

        return best_eval_ga, best_eval_ts, ga_results, ts_results

    def run_multiple_optimization(self, num_runs=5):
        ga_bests = []
        ts_bests = []
        all_ga_results = []
        all_ts_results = []
        
        print(f"Running {num_runs} independent trials...\n")
        
        for run in range(num_runs):
            print(f"=== Run {run + 1}/{num_runs} ===")
            ga_best, ts_best, ga_results, ts_results = self.run_single_optimization()
            
            ga_bests.append(ga_best)
            ts_bests.append(ts_best)
            all_ga_results.append(ga_results)
            all_ts_results.append(ts_results)
            print()

        # Save detailed results
        self.save_multiple_results(ga_bests, ts_bests, all_ga_results, all_ts_results)
        
        # Plot and analyze
        self.plot_multiple_results(all_ga_results, all_ts_results)
        self.create_statistical_summary(ga_bests, ts_bests)

        return ga_bests, ts_bests

    def save_multiple_results(self, ga_bests, ts_bests, all_ga_results, all_ts_results):
        # Save summary of all runs
        summary_df = pd.DataFrame({
            'Run': range(1, len(ga_bests) + 1),
            'GA_Best': ga_bests,
            'TS_Best': ts_bests
        })
        summary_df.to_csv('results/multiple_runs_summary.csv', index=False)
        
        # Save last run detailed results (like original code)
        pd.DataFrame(all_ga_results[-1]).to_csv('results/ga_results.csv', index=False)
        pd.DataFrame(all_ts_results[-1]).to_csv('results/ts_results.csv', index=False)

    def plot_multiple_results(self, all_ga_results, all_ts_results):
        plt.figure(figsize=(10, 6))
        
        # Find best performing runs
        ga_finals = [results[-1]['Best Eval'] for results in all_ga_results]
        ts_finals = [results[-1]['Best Eval'] for results in all_ts_results]
        
        best_ga_idx = ga_finals.index(min(ga_finals))
        best_ts_idx = ts_finals.index(min(ts_finals))
        
        # Plot all runs
        for i, (ga_results, ts_results) in enumerate(zip(all_ga_results, all_ts_results)):
            ga_gens = [r['Generation'] for r in ga_results]
            ga_evals = [r['Best Eval'] for r in ga_results]
            ts_gens = [r['Iteration'] for r in ts_results]
            ts_evals = [r['Best Eval'] for r in ts_results]
            
            # Highlight best performing runs
            ga_alpha = 1.0 if i == best_ga_idx else 0.3
            ts_alpha = 1.0 if i == best_ts_idx else 0.3
            ga_width = 2 if i == best_ga_idx else 1
            ts_width = 2 if i == best_ts_idx else 1
            
            plt.plot(ga_gens, ga_evals, 'b-', alpha=ga_alpha, linewidth=ga_width)
            plt.plot(ts_gens, ts_evals, 'r-', alpha=ts_alpha, linewidth=ts_width)

        # Add labels
        plt.plot([], [], 'b-', label='Genetic Algorithm (best run highlighted)', linewidth=2)
        plt.plot([], [], 'r-', label='Tabu Search (best run highlighted)', linewidth=2)
        
        plt.xlabel('Generation/Iteration')
        plt.ylabel('Best Function Value')
        plt.title('GA vs TS Optimization (Multiple Runs)')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')
        plt.savefig('results/optimization_comparison.png')
        plt.show()
        
        print(f"Best GA run: #{best_ga_idx+1} (final value: {min(ga_finals):.6f})")
        print(f"Best TS run: #{best_ts_idx+1} (final value: {min(ts_finals):.6f})")

    def create_statistical_summary(self, ga_bests, ts_bests):
        print("Statistical Summary:")
        print(f"GA - Mean: {sum(ga_bests)/len(ga_bests):.6f}, Best: {min(ga_bests):.6f}, Worst: {max(ga_bests):.6f}")
        print(f"TS - Mean: {sum(ts_bests)/len(ts_bests):.6f}, Best: {min(ts_bests):.6f}, Worst: {max(ts_bests):.6f}")
        print(f"Winner: {'GA' if sum(ga_bests)/len(ga_bests) < sum(ts_bests)/len(ts_bests) else 'TS'}")
        
        stats_df = pd.DataFrame({
            'Method': ['GA', 'TS'],
            'Mean': [sum(ga_bests)/len(ga_bests), sum(ts_bests)/len(ts_bests)],
            'Best': [min(ga_bests), min(ts_bests)],
            'Worst': [max(ga_bests), max(ts_bests)]
        })
        stats_df.to_csv('results/statistical_summary.csv', index=False)

def main():
    ga_params = {
        'pop_size': 100,
        'num_generations': 200,
        'cxpb': 0.5,
        'mutpb': 0.2
    }

    ts_params = {
        'num_iterations': 200,
        'tabu_list_size': 10
    }

    optimizer = GeneticOptimization(ga_params, ts_params)
    
    # Choose: single run or multiple runs
    print("Choose: (1) Single run or (2) Multiple runs")
    choice = input("Enter 1 or 2: ").strip()
    
    if choice == "2":
        num_runs = int(input("Number of runs (recommended: 5): ") or "5")
        optimizer.run_multiple_optimization(num_runs)
    else:
        ga_best, ts_best, ga_results, ts_results = optimizer.run_single_optimization()
        
        # Original plotting and saving logic
        plt.figure(figsize=(10, 6))
        plt.plot([r['Generation'] for r in ga_results], [r['Best Eval'] for r in ga_results], label='Genetic Algorithm')
        plt.plot([r['Iteration'] for r in ts_results], [r['Best Eval'] for r in ts_results], label='Tabu Search')
        plt.xlabel('Generation/Iteration')
        plt.ylabel('Best Function Value')
        plt.title('GA vs TS Optimization')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')
        plt.savefig('results/optimization_comparison.png')
        plt.show()
        
        # Save results
        pd.DataFrame(ga_results).to_csv('results/ga_results.csv', index=False)
        pd.DataFrame(ts_results).to_csv('results/ts_results.csv', index=False)
        
        print(f"\nResults: GA={ga_best:.6f}, TS={ts_best:.6f}")
        print(f"Winner: {'GA' if ga_best < ts_best else 'TS'}")

if __name__ == "__main__":
    main()
