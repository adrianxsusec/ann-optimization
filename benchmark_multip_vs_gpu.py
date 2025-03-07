import time
import numpy as np
import matplotlib.pyplot as plt
import logging
import pandas as pd
import os
import sys

# Import the implementations
import multiprocessing_cpu
import torch_improvements
import torch_improvements_gpu
import jit_opt

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_benchmark(iterations=[10, 20, 50], runs=3):
    """
    Run benchmark comparing different implementations

    Args:
        iterations: List of iteration counts to test
        runs: Number of times to run each test for averaging

    Returns:
        DataFrame with benchmark results
    """
    results = []

    implementations = [
        ("Multiprocessing", multiprocessing_cpu),
        ("Torch-CPU", torch_improvements),
        ("Torch-GPU", torch_improvements_gpu),
        ("Numba", jit_opt)
    ]

    for name, implementation in implementations:
        logger.info(f"Testing {name} implementation...")

        for max_iter in iterations:
            logger.info(f"  Running with {max_iter} iterations...")

            run_times = []
            accuracies = []

            for run in range(runs):
                logger.info(f"    Run {run + 1}/{runs}")

                start_time = time.time()
                accuracy, total_time = implementation.main(max_iter=max_iter)
                end_time = time.time()

                run_times.append(total_time)
                accuracies.append(accuracy)

            # Record average results
            results.append({
                'Implementation': name,
                'Iterations': max_iter,
                'Avg Time (s)': np.mean(run_times),
                'Std Dev (s)': np.std(run_times),
                'Avg Accuracy': np.mean(accuracies),
                'Std Dev Accuracy': np.std(accuracies)
            })

            logger.info(f"  Average time: {np.mean(run_times):.2f}s")
            logger.info(f"  Average accuracy: {np.mean(accuracies):.4f}")

    return pd.DataFrame(results)


def plot_results(results):
    """
    Plot benchmark results

    Args:
        results: DataFrame with benchmark results
    """
    plt.figure(figsize=(12, 8))

    # Plot training time vs iterations for each implementation
    plt.subplot(2, 1, 1)
    implementations = results['Implementation'].unique()
    iterations = results['Iterations'].unique()

    for impl in implementations:
        impl_data = results[results['Implementation'] == impl]
        plt.errorbar(impl_data['Iterations'], impl_data['Avg Time (s)'],
                     yerr=impl_data['Std Dev (s)'],
                     marker='o', label=impl)

    plt.xlabel('Iterations')
    plt.ylabel('Training Time (s)')
    plt.title('Performance Comparison of Neural Network Implementations')
    plt.grid(True)
    plt.legend()

    # Plot speedup ratio vs iterations
    plt.subplot(2, 1, 2)

    baseline = results[results['Implementation'] == implementations[0]]

    for impl in implementations[1:]:
        improved = results[results['Implementation'] == impl]
        speedup = pd.DataFrame({
            'Iterations': baseline['Iterations'],
            'Speedup': baseline['Avg Time (s)'].values / improved['Avg Time (s)'].values
        })
        plt.plot(speedup['Iterations'], speedup['Speedup'], marker='o', label=f'{impl} vs {implementations[0]}')

    plt.xlabel('Iterations')
    plt.ylabel('Speedup Ratio')
    plt.title('Speedup Relative to Baseline Implementation')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=300)
    plt.show()


def main():
    logger.info("Starting benchmark...")

    # Run benchmark with different iteration counts
    results = run_benchmark(iterations=[10, 20, 50], runs=3)

    # Save results to CSV
    results.to_csv('benchmark_results.csv', index=False)
    logger.info("Results saved to benchmark_results.csv")

    # Plot results
    plot_results(results)

    # Print summary
    print("\nBenchmark Summary:")
    print(results)

    # Calculate and print average speedup
    baseline = results[results['Implementation'] == 'Multiprocessing']
    improved = results[results['Implementation'] == 'GPU']
    avg_speedup = (baseline['Avg Time (s)'] / improved['Avg Time (s)']).mean()

    print(f"\nAverage speedup of GPU over Multiprocessing: {avg_speedup:.2f}x")

    logger.info("Benchmark completed")


if __name__ == "__main__":
    main()