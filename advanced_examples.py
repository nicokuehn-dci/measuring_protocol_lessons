"""
Additional examples demonstrating the measure_time decorator
with different types of functions and scenarios.
"""

import time
import random
from functools import wraps


def measure_time(func):
    """Time measurement decorator"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time
    return wrapper


@measure_time
def fibonacci_recursive(n):
    """
    O(2^n) - Exponential time complexity
    Recursive fibonacci implementation (very inefficient for large n)
    """
    if n <= 1:
        return n
    return fibonacci_recursive(n-1)[0] + fibonacci_recursive(n-2)[0]


@measure_time
def fibonacci_iterative(n):
    """
    O(n) - Linear time complexity
    Iterative fibonacci implementation
    """
    if n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


@measure_time
def bubble_sort(arr):
    """
    O(n²) - Quadratic time complexity
    Bubble sort algorithm
    """
    arr = arr.copy()  # Don't modify original
    n = len(arr)
    
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    
    return arr


@measure_time
def binary_search(arr, target):
    """
    O(log n) - Logarithmic time complexity
    Binary search algorithm
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1  # Not found


def demonstrate_decorator_usage():
    """Demonstrate various uses of the measure_time decorator"""
    print("Advanced Decorator Examples")
    print("=" * 50)
    
    # Test Fibonacci implementations
    print("\n1. Fibonacci Comparison (n=10):")
    print("-" * 30)
    
    # Recursive fibonacci (warning: gets very slow for n > 35)
    fib_result_recursive, time_recursive = fibonacci_recursive(10)
    print(f"Recursive: Result = {fib_result_recursive}, Time = {time_recursive:.6f}s")
    
    # Iterative fibonacci
    fib_result_iterative, time_iterative = fibonacci_iterative(10)
    print(f"Iterative: Result = {fib_result_iterative}, Time = {time_iterative:.6f}s")
    
    print(f"Speedup: {time_recursive/time_iterative:.1f}x faster with iterative approach")
    
    # Test sorting
    print("\n2. Bubble Sort (1000 random numbers):")
    print("-" * 30)
    
    # Generate random array
    original_array = [random.randint(1, 1000) for _ in range(1000)]
    
    sorted_array, sort_time = bubble_sort(original_array)
    print(f"Sorted {len(original_array)} numbers in {sort_time:.6f} seconds")
    print(f"First 10 sorted: {sorted_array[:10]}")
    
    # Test binary search
    print("\n3. Binary Search:")
    print("-" * 30)
    
    # Create sorted array for binary search
    sorted_test_array = list(range(0, 10000, 2))  # Even numbers 0 to 9998
    target = 5000
    
    index, search_time = binary_search(sorted_test_array, target)
    print(f"Searching for {target} in array of {len(sorted_test_array)} elements")
    print(f"Found at index: {index}, Time: {search_time:.6f} seconds")
    
    # Test with different complexity scenarios
    print("\n4. Time Complexity Comparison:")
    print("-" * 30)
    
    sizes = [100, 500, 1000]
    
    for size in sizes:
        print(f"\nArray size: {size}")
        
        # Linear search simulation
        test_array = list(range(size))
        target = size - 1  # Search for last element (worst case)
        
        _, search_time = binary_search(test_array, target)
        print(f"  Binary search O(log n): {search_time:.6f}s")
        
        # Generate random array for sorting
        random_array = [random.randint(1, 1000) for _ in range(size)]
        _, sort_time = bubble_sort(random_array)
        print(f"  Bubble sort O(n²): {sort_time:.6f}s")


if __name__ == "__main__":
    demonstrate_decorator_usage()
