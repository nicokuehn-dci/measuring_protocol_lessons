#!/usr/bin/env python3
"""
Algorithm Analyzer - Interactive tool for analyzing time complexity

This tool lets you run and analyze various algorithms with different
input sizes to observe how their performance scales.
"""

import time
import random
import matplotlib.pyplot as plt
import numpy as np
from functools import wraps
import sys


def measure_time(func):
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        return result, end_time - start_time
    return wrapper


class AlgorithmAnalyzer:
    def __init__(self):
        self.algorithms = {}
        self.results = {}
        
    def register_algorithm(self, name, func, complexity):
        """Register an algorithm for analysis"""
        self.algorithms[name] = {
            'function': func,
            'complexity': complexity
        }
        
    def run_analysis(self, sizes, runs=1):
        """Run analysis on all registered algorithms with given input sizes"""
        self.results = {'sizes': sizes}
        
        for name, algo_info in self.algorithms.items():
            print(f"Running {name}...")
            times = []
            
            for size in sizes:
                # Generate test data
                data = self.generate_test_data(size)
                
                # Run multiple times and take average
                total_time = 0
                for _ in range(runs):
                    _, execution_time = algo_info['function'](data)
                    total_time += execution_time
                
                avg_time = total_time / runs
                times.append(avg_time)
                
                # Print progress
                print(f"  Size {size}: {avg_time:.6f} seconds")
            
            self.results[name] = times
    
    def generate_test_data(self, size):
        """Generate test data for algorithms"""
        return [random.randint(1, 1000) for _ in range(size)]
    
    def plot_results(self, log_scale=False, save_path=None):
        """Plot the results of the analysis"""
        plt.figure(figsize=(12, 8))
        
        for name, times in self.results.items():
            if name != 'sizes':
                if log_scale:
                    plt.loglog(self.results['sizes'], times, 'o-', label=f"{name}")
                else:
                    plt.plot(self.results['sizes'], times, 'o-', label=f"{name}")
        
        plt.xlabel('Input Size')
        plt.ylabel('Time (seconds)')
        plt.title('Algorithm Performance Analysis' + (' (Log Scale)' if log_scale else ''))
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_comparison_dashboard(self, save_path=None):
        """Create a comprehensive performance dashboard"""
        if not self.results:
            print("No results to plot. Run analysis first.")
            return
        
        plt.figure(figsize=(15, 12))
        
        # 1. Linear scale plot
        plt.subplot(2, 2, 1)
        for name, times in self.results.items():
            if name != 'sizes':
                plt.plot(self.results['sizes'], times, 'o-', label=name)
        plt.xlabel('Input Size')
        plt.ylabel('Time (seconds)')
        plt.title('Linear Scale')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Log-log scale plot
        plt.subplot(2, 2, 2)
        for name, times in self.results.items():
            if name != 'sizes':
                plt.loglog(self.results['sizes'], times, 'o-', label=name)
        plt.xlabel('Input Size')
        plt.ylabel('Time (seconds)')
        plt.title('Log-Log Scale')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Relative performance plot
        plt.subplot(2, 2, 3)
        baseline_name = next(name for name in self.results.keys() if name != 'sizes')
        baseline = self.results[baseline_name]
        
        for name, times in self.results.items():
            if name != 'sizes' and name != baseline_name:
                relative = [t/b for t, b in zip(times, baseline)]
                plt.plot(self.results['sizes'], relative, 'o-', 
                         label=f"{name} / {baseline_name}")
        
        plt.xlabel('Input Size')
        plt.ylabel('Relative Time')
        plt.title(f'Performance Relative to {baseline_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Growth rate visualization
        plt.subplot(2, 2, 4)
        
        # Plot theoretical complexity curves
        sizes = self.results['sizes']
        max_time = max(max(times) for name, times in self.results.items() if name != 'sizes')
        
        # Scale factors to make curves visible in the same plot
        n_scale = max_time / sizes[-1]
        n2_scale = max_time / (sizes[-1]**2)
        nlogn_scale = max_time / (sizes[-1] * np.log(sizes[-1]))
        
        x = np.array(sizes)
        plt.plot(x, n_scale * x, '--', label='O(n)', alpha=0.5)
        plt.plot(x, n2_scale * x**2, '--', label='O(n²)', alpha=0.5)
        plt.plot(x, nlogn_scale * x * np.log(x), '--', label='O(n log n)', alpha=0.5)
        
        for name, times in self.results.items():
            if name != 'sizes':
                plt.plot(sizes, times, 'o-', label=name + ' (actual)')
        
        plt.xlabel('Input Size')
        plt.ylabel('Time (seconds)')
        plt.title('Comparison with Theoretical Complexity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


# ===== Example Algorithm Implementations =====

@measure_time
def constant_time_algorithm(data):
    """O(1) algorithm - only access first element"""
    if not data:
        return None
    return data[0]

@measure_time
def linear_time_algorithm(data):
    """O(n) algorithm - sum all elements"""
    total = 0
    for item in data:
        total += item
    return total

@measure_time
def quadratic_time_algorithm(data):
    """O(n²) algorithm - check all pairs"""
    n = len(data)
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            if data[i] == data[j]:
                count += 1
    return count

@measure_time
def nlogn_time_algorithm(data):
    """O(n log n) algorithm - sort then process"""
    # Sorting is O(n log n)
    sorted_data = sorted(data)
    
    # Then do a linear scan O(n)
    result = []
    for i in range(len(sorted_data) - 1):
        result.append(sorted_data[i+1] - sorted_data[i])
    
    return result

@measure_time
def linearithmic_complex_algorithm(data):
    """O(n log n) with more operations"""
    # Make a copy to avoid modifying input
    arr = data.copy()
    n = len(arr)
    
    # Perform merge-sort-like operations
    if n <= 1:
        return arr
    
    mid = n // 2
    left = arr[:mid]
    right = arr[mid:]
    
    # Recursive-like operations (but we don't actually recurse to keep it simpler)
    # This still has the operational complexity similar to merge sort
    for i in range(mid):
        # Do log n iterations of work
        j = i
        while j < n:
            arr[j] = (arr[j] * 2) % 1000  # Some operation
            j *= 2
    
    return arr

@measure_time
def log_time_algorithm(data):
    """O(log n) algorithm - binary search simulation"""
    if not data:
        return -1
    
    # Sort first (not counting this in the timing)
    data = sorted(data)
    
    # Find median value and do a binary search for it
    target = data[len(data) // 2]
    left, right = 0, len(data) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if data[mid] == target:
            return mid
        elif data[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1


def demo_mode():
    """Run a demonstration of algorithm analysis"""
    print("\n" + "=" * 60)
    print("🔬 ALGORITHM ANALYZER - DEMONSTRATION MODE")
    print("=" * 60)
    
    analyzer = AlgorithmAnalyzer()
    
    # Register algorithms
    analyzer.register_algorithm("O(1) Constant", constant_time_algorithm, "O(1)")
    analyzer.register_algorithm("O(n) Linear", linear_time_algorithm, "O(n)")
    analyzer.register_algorithm("O(n²) Quadratic", quadratic_time_algorithm, "O(n²)")
    analyzer.register_algorithm("O(n log n) Sorting", nlogn_time_algorithm, "O(n log n)")
    
    # Run analysis
    sizes = [100, 500, 1000, 5000, 10000]
    analyzer.run_analysis(sizes, runs=3)
    
    # Plot results
    analyzer.plot_results(log_scale=False)
    analyzer.plot_results(log_scale=True)
    
    # Create dashboard
    analyzer.create_comparison_dashboard("algorithm_comparison.png")
    
    print("\nAnalysis complete! Check the generated plots.")


def interactive_mode():
    """Run interactive algorithm analysis"""
    print("\n" + "=" * 60)
    print("🔬 ALGORITHM ANALYZER - INTERACTIVE MODE")
    print("=" * 60)
    
    analyzer = AlgorithmAnalyzer()
    
    # Show available algorithms
    algorithms = {
        1: ("O(1) Constant", constant_time_algorithm, "O(1)"),
        2: ("O(log n) Binary Search", log_time_algorithm, "O(log n)"),
        3: ("O(n) Linear", linear_time_algorithm, "O(n)"),
        4: ("O(n log n) Sorting", nlogn_time_algorithm, "O(n log n)"),
        5: ("O(n log n) Complex", linearithmic_complex_algorithm, "O(n log n)"),
        6: ("O(n²) Quadratic", quadratic_time_algorithm, "O(n²)")
    }
    
    print("\nAvailable algorithms:")
    for key, (name, _, complexity) in algorithms.items():
        print(f"{key}. {name} ({complexity})")
    
    # Get user selection
    try:
        print("\nSelect algorithms to analyze (comma-separated numbers, e.g., 1,3,6):")
        selection = input("> ")
        selected_ids = [int(x.strip()) for x in selection.split(",")]
        
        for algo_id in selected_ids:
            if algo_id in algorithms:
                name, func, complexity = algorithms[algo_id]
                analyzer.register_algorithm(name, func, complexity)
            else:
                print(f"Invalid selection: {algo_id}")
        
        if not analyzer.algorithms:
            print("No valid algorithms selected. Using defaults.")
            analyzer.register_algorithm("O(n) Linear", linear_time_algorithm, "O(n)")
            analyzer.register_algorithm("O(n²) Quadratic", quadratic_time_algorithm, "O(n²)")
        
        # Get input sizes
        print("\nEnter input sizes to test (comma-separated, e.g., 100,500,1000):")
        size_input = input("> ")
        if size_input.strip():
            sizes = [int(x.strip()) for x in size_input.split(",")]
        else:
            print("Using default sizes.")
            sizes = [100, 500, 1000, 5000]
        
        # Run analysis
        print("\nRunning analysis...")
        analyzer.run_analysis(sizes, runs=3)
        
        # Plot results
        analyzer.plot_results(log_scale=False)
        analyzer.plot_results(log_scale=True)
        
        # Create dashboard
        analyzer.create_comparison_dashboard("custom_analysis.png")
        
        print("\nAnalysis complete! Check the generated plots.")
        
    except ValueError as e:
        print(f"Error in input: {e}")
        print("Using demonstration mode instead.")
        demo_mode()


def main():
    """Main function"""
    print("\nWelcome to Algorithm Analyzer!")
    print("This tool helps you analyze and visualize algorithm performance.")
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_mode()
    else:
        print("\nChoose mode:")
        print("1. Demo (predefined algorithms and inputs)")
        print("2. Interactive (choose your own algorithms and inputs)")
        
        try:
            choice = int(input("> "))
            if choice == 1:
                demo_mode()
            else:
                interactive_mode()
        except ValueError:
            print("Invalid input. Running demo mode.")
            demo_mode()


if __name__ == "__main__":
    main()
