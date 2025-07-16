import time
import matplotlib.pyplot as plt
import numpy as np
from functools import wraps


def measure_time(func):
    """
    Decorator that measures the execution time of a function.
    
    Args:
        func: The function to be decorated
        
    Returns:
        A wrapper function that executes the original function and returns 
        a tuple of (result, execution_time_in_seconds)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time
    return wrapper


@measure_time
def constant_time_function(n):
    """
    O(1) - Constant time complexity
    Always performs the same number of operations regardless of input size
    """
    # Simple arithmetic operations - always takes the same time
    result = n * 2 + 5
    return result


@measure_time
def linear_time_function(n):
    """
    O(n) - Linear time complexity
    Number of operations grows linearly with input size
    """
    # Sum all numbers from 1 to n
    total = 0
    for i in range(n):
        total += i
    return total


@measure_time
def quadratic_time_function(n):
    """
    O(n²) - Quadratic time complexity
    Number of operations grows quadratically with input size
    """
    # Nested loops - for each i, we iterate through all j
    total = 0
    for i in range(n):
        for j in range(n):
            total += i * j
    return total


def run_benchmark():
    """
    Run benchmark tests for different input sizes and collect timing data
    """
    # Test different input sizes
    input_sizes = [10, 50, 100, 200, 500, 1000, 2000, 5000]
    
    # Store results for each function
    constant_times = []
    linear_times = []
    quadratic_times = []
    
    print("Running benchmark tests...")
    print("=" * 50)
    
    for n in input_sizes:
        print(f"Testing with n = {n}")
        
        # Test constant time function
        _, const_time = constant_time_function(n)
        constant_times.append(const_time)
        print(f"  Constant O(1): {const_time:.6f} seconds")
        
        # Test linear time function
        _, linear_time = linear_time_function(n)
        linear_times.append(linear_time)
        print(f"  Linear O(n): {linear_time:.6f} seconds")
        
        # Test quadratic time function (skip very large inputs to avoid long wait times)
        if n <= 2000:
            _, quad_time = quadratic_time_function(n)
            quadratic_times.append(quad_time)
            print(f"  Quadratic O(n²): {quad_time:.6f} seconds")
        else:
            quadratic_times.append(None)  # Skip large inputs for quadratic
            print(f"  Quadratic O(n²): Skipped (too slow)")
        
        print()
    
    return input_sizes, constant_times, linear_times, quadratic_times


def plot_results(input_sizes, constant_times, linear_times, quadratic_times):
    """
    Create plots to visualize the runtime vs input size for each function
    """
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Function Runtime Analysis: Time Complexity Comparison', fontsize=16)
    
    # Plot 1: All functions together (linear scale)
    ax1.plot(input_sizes, constant_times, 'b-o', label='O(1) - Constant', linewidth=2)
    ax1.plot(input_sizes, linear_times, 'g-s', label='O(n) - Linear', linewidth=2)
    
    # Only plot quadratic for smaller values
    quad_sizes = [size for size, time in zip(input_sizes, quadratic_times) if time is not None]
    quad_times_filtered = [time for time in quadratic_times if time is not None]
    ax1.plot(quad_sizes, quad_times_filtered, 'r-^', label='O(n²) - Quadratic', linewidth=2)
    
    ax1.set_xlabel('Input Size (n)')
    ax1.set_ylabel('Runtime (seconds)')
    ax1.set_title('All Functions - Linear Scale')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: All functions together (log scale)
    ax2.loglog(input_sizes, constant_times, 'b-o', label='O(1) - Constant', linewidth=2)
    ax2.loglog(input_sizes, linear_times, 'g-s', label='O(n) - Linear', linewidth=2)
    ax2.loglog(quad_sizes, quad_times_filtered, 'r-^', label='O(n²) - Quadratic', linewidth=2)
    
    ax2.set_xlabel('Input Size (n)')
    ax2.set_ylabel('Runtime (seconds)')
    ax2.set_title('All Functions - Log-Log Scale')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Individual function analysis
    ax3.plot(input_sizes, constant_times, 'b-o', label='O(1) - Constant', linewidth=2)
    ax3.plot(input_sizes, linear_times, 'g-s', label='O(n) - Linear', linewidth=2)
    
    ax3.set_xlabel('Input Size (n)')
    ax3.set_ylabel('Runtime (seconds)')
    ax3.set_title('Constant vs Linear Time Complexity')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Quadratic function separately
    ax4.plot(quad_sizes, quad_times_filtered, 'r-^', label='O(n²) - Quadratic', linewidth=2)
    ax4.set_xlabel('Input Size (n)')
    ax4.set_ylabel('Runtime (seconds)')
    ax4.set_title('Quadratic Time Complexity O(n²)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/nico-kuehn-dci/Desktop/lectures/measuring/time_complexity_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()


def analyze_growth_rates(input_sizes, constant_times, linear_times, quadratic_times):
    """
    Analyze and print the growth rates of each function
    """
    print("\n" + "=" * 60)
    print("GROWTH RATE ANALYSIS")
    print("=" * 60)
    
    # Calculate average growth ratios
    print("\nWhen input size doubles, runtime changes by:")
    
    for i in range(1, len(input_sizes)):
        if input_sizes[i] == 2 * input_sizes[i-1]:  # When size doubles
            const_ratio = constant_times[i] / constant_times[i-1] if constant_times[i-1] > 0 else 0
            linear_ratio = linear_times[i] / linear_times[i-1] if linear_times[i-1] > 0 else 0
            
            print(f"\nn = {input_sizes[i-1]} → n = {input_sizes[i]}:")
            print(f"  Constant O(1): {const_ratio:.2f}x")
            print(f"  Linear O(n): {linear_ratio:.2f}x")
            
            if quadratic_times[i] is not None and quadratic_times[i-1] is not None:
                quad_ratio = quadratic_times[i] / quadratic_times[i-1]
                print(f"  Quadratic O(n²): {quad_ratio:.2f}x")


def main():
    """
    Main function to run the complete analysis
    """
    print("Time Complexity Analysis with Decorator")
    print("=" * 50)
    print("This program demonstrates different time complexities:")
    print("- O(1): Constant time - always takes the same time")
    print("- O(n): Linear time - time grows linearly with input")
    print("- O(n²): Quadratic time - time grows quadratically with input")
    print("\n")
    
    # Run the benchmark
    input_sizes, constant_times, linear_times, quadratic_times = run_benchmark()
    
    # Analyze growth rates
    analyze_growth_rates(input_sizes, constant_times, linear_times, quadratic_times)
    
    # Create plots
    plot_results(input_sizes, constant_times, linear_times, quadratic_times)
    
    print("\n" + "=" * 60)
    print("Analysis complete! Check the generated plot: time_complexity_analysis.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
