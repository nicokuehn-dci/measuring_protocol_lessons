#!/usr/bin/env python3
"""
Complete Exercise Solution: Measure and Plot Function Runtimes

This script demonstrates:
1. A time measurement decorator
2. Functions with different time complexities (O(1), O(n), O(n²))
3. Runtime measurement and analysis
4. Plotting runtime vs input size
"""

import time
import matplotlib.pyplot as plt
from functools import wraps


def measure_time(func):
    """
    Decorator that measures the execution time of a function.
    
    Returns a tuple: (original_result, execution_time_in_seconds)
    
    Usage:
        @measure_time
        def my_function(arg):
            return some_result
            
        result, time_taken = my_function(input_value)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time
    return wrapper


# ============================================================================
# FUNCTIONS WITH DIFFERENT TIME COMPLEXITIES
# ============================================================================

@measure_time
def constant_time_o1(n):
    """
    O(1) - Constant Time Complexity
    
    No matter how large n is, this function always performs
    the same number of operations.
    """
    # Simple arithmetic - always takes the same time
    result = (n * 3 + 7) // 2
    return result


@measure_time
def linear_time_on(n):
    """
    O(n) - Linear Time Complexity
    
    The number of operations grows linearly with the input size.
    If n doubles, the runtime approximately doubles.
    """
    total = 0
    for i in range(n):
        total += i ** 2  # A bit more work than just addition
    return total


@measure_time
def quadratic_time_on2(n):
    """
    O(n²) - Quadratic Time Complexity
    
    The number of operations grows quadratically with input size.
    If n doubles, the runtime increases by approximately 4x.
    """
    result = 0
    for i in range(n):
        for j in range(n):
            result += (i + j) % 7  # Some operation in nested loops
    return result


# ============================================================================
# MEASUREMENT AND ANALYSIS
# ============================================================================

def run_time_analysis():
    """Run timing analysis for different input sizes"""
    
    # Test with various input sizes
    test_sizes = [10, 25, 50, 100, 200, 500, 1000]
    
    # Storage for results
    results = {
        'sizes': test_sizes,
        'constant_times': [],
        'linear_times': [],
        'quadratic_times': []
    }
    
    print("Running Time Complexity Analysis")
    print("=" * 50)
    print(f"{'Size':<8} {'O(1)':<12} {'O(n)':<12} {'O(n²)':<12}")
    print("-" * 50)
    
    for size in test_sizes:
        # Measure constant time function
        _, const_time = constant_time_o1(size)
        results['constant_times'].append(const_time)
        
        # Measure linear time function  
        _, linear_time = linear_time_on(size)
        results['linear_times'].append(linear_time)
        
        # Measure quadratic time function (limit size to avoid long waits)
        if size <= 1000:
            _, quad_time = quadratic_time_on2(size)
            results['quadratic_times'].append(quad_time)
        else:
            results['quadratic_times'].append(None)
        
        # Display results
        quad_display = f"{results['quadratic_times'][-1]:.6f}" if results['quadratic_times'][-1] else "skipped"
        print(f"{size:<8} {const_time:<12.6f} {linear_time:<12.6f} {quad_display:<12}")
    
    return results


def create_plots(results):
    """Create visualization plots of the timing results"""
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Function Runtime Analysis: Time Complexity Demonstration', fontsize=16)
    
    sizes = results['sizes']
    const_times = results['constant_times']
    linear_times = results['linear_times']
    quad_times = [t for t in results['quadratic_times'] if t is not None]
    quad_sizes = sizes[:len(quad_times)]
    
    # Plot 1: All functions on linear scale
    axes[0, 0].plot(sizes, const_times, 'bo-', label='O(1) Constant', linewidth=2, markersize=6)
    axes[0, 0].plot(sizes, linear_times, 'go-', label='O(n) Linear', linewidth=2, markersize=6)
    axes[0, 0].plot(quad_sizes, quad_times, 'ro-', label='O(n²) Quadratic', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Input Size (n)')
    axes[0, 0].set_ylabel('Runtime (seconds)')
    axes[0, 0].set_title('Linear Scale Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Log-log scale
    axes[0, 1].loglog(sizes, const_times, 'bo-', label='O(1) Constant', linewidth=2, markersize=6)
    axes[0, 1].loglog(sizes, linear_times, 'go-', label='O(n) Linear', linewidth=2, markersize=6)
    axes[0, 1].loglog(quad_sizes, quad_times, 'ro-', label='O(n²) Quadratic', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Input Size (n)')
    axes[0, 1].set_ylabel('Runtime (seconds)')
    axes[0, 1].set_title('Log-Log Scale')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Constant vs Linear only
    axes[1, 0].plot(sizes, const_times, 'bo-', label='O(1) Constant', linewidth=2, markersize=6)
    axes[1, 0].plot(sizes, linear_times, 'go-', label='O(n) Linear', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Input Size (n)')
    axes[1, 0].set_ylabel('Runtime (seconds)')
    axes[1, 0].set_title('O(1) vs O(n) Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Quadratic growth focus
    axes[1, 1].plot(quad_sizes, quad_times, 'ro-', label='O(n²) Quadratic', linewidth=2, markersize=6)
    axes[1, 1].set_xlabel('Input Size (n)')
    axes[1, 1].set_ylabel('Runtime (seconds)')
    axes[1, 1].set_title('Quadratic Growth Pattern')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = '/home/nico-kuehn-dci/Desktop/lectures/measuring/runtime_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: {output_file}")
    plt.show()


def demonstrate_decorator():
    """Simple demonstration of the decorator usage"""
    print("\nDecorator Usage Demonstration:")
    print("-" * 40)
    
    # Example: Using the decorator manually
    @measure_time
    def example_function(x):
        """Example function to demonstrate decorator"""
        time.sleep(0.01)  # Simulate some work
        return x ** 2
    
    # Call the decorated function
    result, execution_time = example_function(5)
    print(f"Function returned: {result}")
    print(f"Execution time: {execution_time:.6f} seconds")
    
    # Show that the decorator preserves function metadata
    print(f"Function name: {example_function.__name__}")
    print(f"Function doc: {example_function.__doc__}")


def main():
    """Main function orchestrating the complete analysis"""
    print("=" * 60)
    print("EXERCISE: MEASURE AND PLOT FUNCTION RUNTIMES")
    print("=" * 60)
    print("This program demonstrates:")
    print("• A time measurement decorator")
    print("• Functions with O(1), O(n), and O(n²) complexities")
    print("• Runtime analysis and visualization")
    print("=" * 60)
    
    # Demonstrate basic decorator usage
    demonstrate_decorator()
    
    # Run the complete timing analysis
    print("\n")
    results = run_time_analysis()
    
    # Create and display plots
    create_plots(results)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("Key Observations:")
    print("• O(1): Runtime stays constant regardless of input size")
    print("• O(n): Runtime grows linearly with input size")  
    print("• O(n²): Runtime grows quadratically with input size")
    print("=" * 60)


if __name__ == "__main__":
    main()
