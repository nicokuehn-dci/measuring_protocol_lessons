#!/usr/bin/env python3
"""
Recursion Analyzer - Educational tool for understanding recursive algorithms

This tool demonstrates various recursive algorithms, visualizes their
call patterns, and analyzes their time and space complexity.
"""

import time
import sys
import matplotlib.pyplot as plt
import numpy as np
from functools import wraps, lru_cache

# Increase recursion limit for deeper recursive calls
sys.setrecursionlimit(3000)


def measure_time(func):
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        return result, end_time - start_time
    return wrapper


class RecursionVisualizer:
    """Class for visualizing and analyzing recursive algorithms"""
    
    def __init__(self):
        self.call_count = 0
        self.call_history = []
        self.max_depth = 0
        self.current_depth = 0
    
    def reset_stats(self):
        """Reset tracking statistics"""
        self.call_count = 0
        self.call_history = []
        self.max_depth = 0
        self.current_depth = 0
    
    def track_recursive_calls(self, func):
        """Decorator to track recursive function calls"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.call_count += 1
            self.current_depth += 1
            self.max_depth = max(self.max_depth, self.current_depth)
            
            # Track this call
            call_info = {
                'args': args,
                'depth': self.current_depth,
                'index': self.call_count
            }
            self.call_history.append(call_info)
            
            # Call the original function
            result = func(*args, **kwargs)
            
            # Exiting this call
            self.current_depth -= 1
            
            return result
        return wrapper
    
    def print_call_tree(self, max_calls=100):
        """Print a visual representation of the recursive call tree"""
        if not self.call_history:
            print("No call history available. Run a tracked function first.")
            return
        
        print("\nRecursive Call Tree:")
        print("=" * 60)
        
        # Limit the number of calls shown to avoid overwhelming output
        shown_calls = min(len(self.call_history), max_calls)
        for i in range(shown_calls):
            call = self.call_history[i]
            indent = "  " * (call['depth'] - 1)
            args_str = ", ".join(str(arg) for arg in call['args'])
            print(f"{indent}Call {call['index']}: {func_name}({args_str})")
        
        if shown_calls < len(self.call_history):
            print(f"\n... and {len(self.call_history) - shown_calls} more calls (truncated)")
        
        print(f"\nTotal calls: {self.call_count}")
        print(f"Maximum recursion depth: {self.max_depth}")
    
    def plot_call_pattern(self, algorithm_name=None, save_path=None):
        """Plot the recursive call pattern"""
        if not self.call_history:
            print("No call history available. Run a tracked function first.")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Extract data
        depths = [call['depth'] for call in self.call_history]
        indices = [call['index'] for call in self.call_history]
        
        # Plot call depths over time
        plt.plot(indices, depths, 'b-', alpha=0.7)
        plt.scatter(indices, depths, c='r', s=30, alpha=0.5)
        
        plt.xlabel('Call Sequence')
        plt.ylabel('Recursion Depth')
        plt.title(f'Recursive Call Pattern{f" - {algorithm_name}" if algorithm_name else ""}')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_complexity_analysis(self, func, sizes, algorithm_name=None, save_path=None):
        """Plot time complexity analysis for a recursive function"""
        times = []
        
        print(f"Running complexity analysis for various input sizes...")
        for size in sizes:
            print(f"  Testing size {size}...", end=" ", flush=True)
            try:
                _, execution_time = func(size)
                times.append(execution_time)
                print(f"Done: {execution_time:.6f} seconds")
            except RecursionError:
                print(f"RecursionError! Input too large.")
                times.append(None)  # Mark as failed
        
        # Filter out None values
        valid_sizes = [size for size, time in zip(sizes, times) if time is not None]
        valid_times = [time for time in times if time is not None]
        
        if not valid_times:
            print("No valid data points to plot.")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Linear scale
        plt.subplot(2, 2, 1)
        plt.plot(valid_sizes, valid_times, 'bo-', linewidth=2)
        plt.xlabel('Input Size')
        plt.ylabel('Time (seconds)')
        plt.title('Linear Scale')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Log-Log scale
        plt.subplot(2, 2, 2)
        plt.loglog(valid_sizes, valid_times, 'ro-', linewidth=2)
        plt.xlabel('Input Size')
        plt.ylabel('Time (seconds)')
        plt.title('Log-Log Scale')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Try to determine complexity
        plt.subplot(2, 2, 3)
        
        # Generate potential complexity curves
        x = np.array(valid_sizes)
        max_time = max(valid_times)
        
        # Scale factors
        constant_scale = max_time
        log_scale = max_time / np.log(x[-1]) if x[-1] > 1 else 1
        linear_scale = max_time / x[-1] if x[-1] > 0 else 1
        nlogn_scale = max_time / (x[-1] * np.log(x[-1])) if x[-1] > 1 else 1
        quadratic_scale = max_time / (x[-1]**2) if x[-1] > 0 else 1
        exponential_scale = max_time / (2**x[-1]) if x[-1] < 30 else max_time / (2**30)
        
        # Plot reference complexity curves
        plt.plot(x, np.ones_like(x) * constant_scale, 'k--', label='O(1)', alpha=0.5)
        plt.plot(x, np.log(x) * log_scale, 'g--', label='O(log n)', alpha=0.5)
        plt.plot(x, x * linear_scale, 'b--', label='O(n)', alpha=0.5)
        plt.plot(x, x * np.log(x) * nlogn_scale, 'c--', label='O(n log n)', alpha=0.5)
        plt.plot(x, x**2 * quadratic_scale, 'm--', label='O(n²)', alpha=0.5)
        plt.plot(x, np.power(2, x) * exponential_scale, 'r--', label='O(2ⁿ)', alpha=0.5)
        
        # Plot actual data
        plt.plot(valid_sizes, valid_times, 'ko-', linewidth=2, label='Actual')
        
        plt.xlabel('Input Size')
        plt.ylabel('Time (seconds)')
        plt.title('Complexity Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Try to find best fit
        plt.subplot(2, 2, 4)
        
        # Determine best fit for complexity
        best_fit = self._determine_complexity(valid_sizes, valid_times)
        
        plt.plot(valid_sizes, valid_times, 'ko-', linewidth=2, label='Actual')
        plt.xlabel('Input Size')
        plt.ylabel('Time (seconds)')
        plt.title(f'Best Fit: {best_fit}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        
        if algorithm_name:
            plt.suptitle(f"{algorithm_name} - Time Complexity Analysis", fontsize=16, y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _determine_complexity(self, sizes, times):
        """Attempt to determine the time complexity by curve fitting"""
        # This is a simplified approximation based on growth patterns
        x = np.array(sizes)
        y = np.array(times)
        
        if len(x) < 3:
            return "Insufficient data points"
        
        # Try to fit different models
        models = {
            "O(1)": lambda x, a: a * np.ones_like(x),
            "O(log n)": lambda x, a: a * np.log(x),
            "O(n)": lambda x, a: a * x,
            "O(n log n)": lambda x, a: a * x * np.log(x),
            "O(n²)": lambda x, a: a * x**2,
            "O(2ⁿ)": lambda x, a: a * np.power(2, x)
        }
        
        best_model = None
        best_error = float('inf')
        
        for name, model_func in models.items():
            try:
                # Simple coefficient estimation
                if name == "O(1)":
                    a = np.mean(y)
                elif name == "O(log n)":
                    a = np.mean(y / np.log(x))
                elif name == "O(n)":
                    a = np.mean(y / x)
                elif name == "O(n log n)":
                    a = np.mean(y / (x * np.log(x)))
                elif name == "O(n²)":
                    a = np.mean(y / (x**2))
                elif name == "O(2ⁿ)":
                    # Avoid overflow for large x
                    valid_indices = x < 30
                    if np.any(valid_indices):
                        a = np.mean(y[valid_indices] / np.power(2, x[valid_indices]))
                    else:
                        continue
                
                # Compute fitted values
                y_fit = model_func(x, a)
                
                # Compute mean squared error
                error = np.mean((y - y_fit)**2)
                
                if error < best_error:
                    best_error = error
                    best_model = name
                    
                    # Plot the best fit line
                    plt.plot(x, y_fit, '--', label=f'{name} fit')
            except:
                # Skip models that cause numerical issues
                continue
        
        return best_model if best_model else "Undetermined"
    
    def compare_implementations(self, funcs, sizes, names=None, save_path=None):
        """Compare different implementations of the same algorithm"""
        if names is None:
            names = [f"Implementation {i+1}" for i in range(len(funcs))]
        
        results = {name: [] for name in names}
        
        for size in sizes:
            print(f"Testing size {size}...")
            for func, name in zip(funcs, names):
                try:
                    _, execution_time = func(size)
                    results[name].append(execution_time)
                    print(f"  {name}: {execution_time:.6f} seconds")
                except RecursionError:
                    results[name].append(None)
                    print(f"  {name}: RecursionError! Input too large.")
        
        # Plot comparison
        plt.figure(figsize=(12, 8))
        
        # Linear scale
        plt.subplot(2, 1, 1)
        for name in names:
            valid_times = [(size, time) for size, time in zip(sizes, results[name]) if time is not None]
            if valid_times:
                valid_sizes, valid_data = zip(*valid_times)
                plt.plot(valid_sizes, valid_data, 'o-', linewidth=2, label=name)
        
        plt.xlabel('Input Size')
        plt.ylabel('Time (seconds)')
        plt.title('Implementation Comparison (Linear Scale)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Log scale
        plt.subplot(2, 1, 2)
        for name in names:
            valid_times = [(size, time) for size, time in zip(sizes, results[name]) if time is not None]
            if valid_times:
                valid_sizes, valid_data = zip(*valid_times)
                plt.loglog(valid_sizes, valid_data, 'o-', linewidth=2, label=name)
        
        plt.xlabel('Input Size')
        plt.ylabel('Time (seconds)')
        plt.title('Implementation Comparison (Log-Log Scale)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


# ===== Recursive Algorithm Implementations =====

# Simple recursive factorial
@measure_time
def factorial(n):
    """Recursive factorial implementation: O(n) time, O(n) space"""
    if n <= 1:
        return 1
    return n * factorial(n - 1)[0]  # Extract result from the tuple

# Fibonacci implementations
@measure_time
def fibonacci_recursive(n):
    """Naive recursive Fibonacci: O(2ⁿ) time, O(n) space"""
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1)[0] + fibonacci_recursive(n - 2)[0]

@measure_time
def fibonacci_memo(n, memo=None):
    """Memoized recursive Fibonacci: O(n) time, O(n) space"""
    if memo is None:
        memo = {}
    
    if n in memo:
        return memo[n]
    
    if n <= 1:
        result = n
    else:
        result = fibonacci_memo(n - 1, memo)[0] + fibonacci_memo(n - 2, memo)[0]
    
    memo[n] = result
    return result

@measure_time
def fibonacci_iterative(n):
    """Iterative Fibonacci: O(n) time, O(1) space"""
    if n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    
    return b

# LRU cached fibonacci for higher values
@measure_time
@lru_cache(maxsize=None)
def fibonacci_lru(n):
    """LRU cached Fibonacci: O(n) time, O(n) space"""
    if n <= 1:
        return n
    return fibonacci_lru(n - 1) + fibonacci_lru(n - 2)

# Recursive binary search
@measure_time
def binary_search_recursive(arr, target, left=0, right=None):
    """Recursive binary search: O(log n) time, O(log n) space"""
    if right is None:
        right = len(arr) - 1
    
    if left > right:
        return -1
    
    mid = (left + right) // 2
    
    if arr[mid] == target:
        return mid
    elif arr[mid] > target:
        return binary_search_recursive(arr, target, left, mid - 1)[0]
    else:
        return binary_search_recursive(arr, target, mid + 1, right)[0]

# Recursive sum
@measure_time
def recursive_sum(arr):
    """Recursive sum of array: O(n) time, O(n) space"""
    if not arr:
        return 0
    return arr[0] + recursive_sum(arr[1:])[0]

# Recursive power function
@measure_time
def power_recursive(base, exponent):
    """Recursive power calculation: O(n) time, O(n) space"""
    if exponent == 0:
        return 1
    return base * power_recursive(base, exponent - 1)[0]

@measure_time
def power_optimized(base, exponent):
    """Optimized recursive power: O(log n) time, O(log n) space"""
    if exponent == 0:
        return 1
    
    half = power_optimized(base, exponent // 2)[0]
    
    if exponent % 2 == 0:
        return half * half
    else:
        return base * half * half

# Recursive GCD (Greatest Common Divisor)
@measure_time
def gcd_recursive(a, b):
    """Recursive GCD (Euclidean algorithm): O(log(min(a,b))) time and space"""
    if b == 0:
        return a
    return gcd_recursive(b, a % b)[0]

# Recursive Tower of Hanoi
@measure_time
def hanoi_moves(n):
    """
    Calculate number of moves for Tower of Hanoi: O(2ⁿ) time, O(n) space
    (Just calculates the move count, doesn't enumerate the moves)
    """
    if n <= 0:
        return 0
    return 2 * hanoi_moves(n - 1)[0] + 1

# Recursive merge sort
@measure_time
def merge_sort(arr):
    """Recursive merge sort: O(n log n) time, O(n) space"""
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])[0]  # Extract the result from the tuple
    right = merge_sort(arr[mid:])[0]
    
    return merge(left, right)

def merge(left, right):
    """Helper function for merge sort"""
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def explain_algorithm(name):
    """Print a detailed explanation of a recursive algorithm"""
    explanations = {
        "Factorial": """
        FACTORIAL RECURSION
        ==================
        
        Definition:
        The factorial of n (denoted as n!) is the product of all positive integers less than or equal to n.
        
        Mathematical Definition:
        n! = n × (n-1) × (n-2) × ... × 2 × 1
        
        Base Case:
        0! = 1
        1! = 1
        
        Recursive Relation:
        n! = n × (n-1)!
        
        Recursive Implementation:
        ```python
        def factorial(n):
            if n <= 1:
                return 1
            return n * factorial(n - 1)
        ```
        
        Time Complexity: O(n)
        Space Complexity: O(n) due to the recursion stack
        
        Call Tree for factorial(4):
        
        factorial(4)
        └── 4 × factorial(3)
            └── 3 × factorial(2)
                └── 2 × factorial(1)
                    └── return 1
                └── return 2 × 1 = 2
            └── return 3 × 2 = 6
        └── return 4 × 6 = 24
        
        Example Execution:
        factorial(4) = 4 × factorial(3)
                      = 4 × 3 × factorial(2)
                      = 4 × 3 × 2 × factorial(1)
                      = 4 × 3 × 2 × 1
                      = 24
        """,
        
        "Fibonacci": """
        FIBONACCI RECURSION
        ==================
        
        Definition:
        The Fibonacci sequence is a series where each number is the sum of the two preceding ones.
        
        Mathematical Definition:
        F(0) = 0
        F(1) = 1
        F(n) = F(n-1) + F(n-2) for n > 1
        
        The sequence starts: 0, 1, 1, 2, 3, 5, 8, 13, 21, ...
        
        Naive Recursive Implementation:
        ```python
        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n-1) + fibonacci(n-2)
        ```
        
        Time Complexity: O(2ⁿ) - Exponential!
        Space Complexity: O(n) due to the recursion stack
        
        Call Tree for fibonacci(5):
        
        fibonacci(5)
        ├── fibonacci(4)
        │   ├── fibonacci(3)
        │   │   ├── fibonacci(2)
        │   │   │   ├── fibonacci(1)
        │   │   │   └── fibonacci(0)
        │   │   └── fibonacci(1)
        │   └── fibonacci(2)
        │       ├── fibonacci(1)
        │       └── fibonacci(0)
        └── fibonacci(3)
            ├── fibonacci(2)
            │   ├── fibonacci(1)
            │   └── fibonacci(0)
            └── fibonacci(1)
        
        Problems with Naive Implementation:
        - Many redundant calculations (e.g., fibonacci(3) is computed twice)
        - Exponential time complexity makes it impractical for larger inputs
        
        Optimized Implementations:
        
        1. Memoization (Dynamic Programming):
        ```python
        def fibonacci_memo(n, memo={}):
            if n in memo:
                return memo[n]
            if n <= 1:
                return n
            memo[n] = fibonacci_memo(n-1) + fibonacci_memo(n-2)
            return memo[n]
        ```
        Time Complexity: O(n)
        Space Complexity: O(n)
        
        2. Iterative Implementation:
        ```python
        def fibonacci_iterative(n):
            if n <= 1:
                return n
            a, b = 0, 1
            for _ in range(2, n+1):
                a, b = b, a + b
            return b
        ```
        Time Complexity: O(n)
        Space Complexity: O(1) - Constant space!
        
        The dramatic difference in performance between the naive and optimized
        implementations demonstrates the importance of algorithm design.
        """,
        
        "Binary Search": """
        RECURSIVE BINARY SEARCH
        ======================
        
        Definition:
        Binary search is a search algorithm that finds the position of a target value
        within a sorted array by repeatedly dividing the search interval in half.
        
        Prerequisites:
        - The array must be sorted
        
        Recursive Implementation:
        ```python
        def binary_search(arr, target, left=0, right=None):
            if right is None:
                right = len(arr) - 1
                
            # Base case: element not found
            if left > right:
                return -1
                
            # Find middle element
            mid = (left + right) // 2
            
            # Check if middle element is the target
            if arr[mid] == target:
                return mid
            # If target is smaller, search left half
            elif arr[mid] > target:
                return binary_search(arr, target, left, mid - 1)
            # If target is larger, search right half
            else:
                return binary_search(arr, target, mid + 1, right)
        ```
        
        Time Complexity: O(log n) - Logarithmic
        Space Complexity: O(log n) due to the recursion stack
        
        Call Tree for binary_search([1, 2, 3, 4, 5, 6, 7], 5):
        
        binary_search([1,2,3,4,5,6,7], 5, 0, 6)
        └── mid = 3, arr[3] = 4 < 5
        └── binary_search([1,2,3,4,5,6,7], 5, 4, 6)
            └── mid = 5, arr[5] = 6 > 5
            └── binary_search([1,2,3,4,5,6,7], 5, 4, 4)
                └── mid = 4, arr[4] = 5 == 5
                └── return 4
                
        Explanation:
        1. Initially, we search the entire array (indices 0 to 6)
        2. We find that the middle element (index 3, value 4) is less than our target (5)
        3. We search the right half (indices 4 to 6)
        4. The middle element (index 5, value 6) is greater than our target
        5. We search the left half (indices 4 to 4)
        6. The middle element (index 4, value 5) is our target
        7. We return index 4
        
        The power of binary search is that it eliminates half of the remaining 
        elements at each step, leading to a logarithmic time complexity.
        """,
        
        "Merge Sort": """
        RECURSIVE MERGE SORT
        ===================
        
        Definition:
        Merge Sort is a divide-and-conquer algorithm that divides the input array
        into two halves, recursively sorts them, and then merges the sorted halves.
        
        Key Concept:
        - Divide the problem into smaller subproblems
        - Solve the subproblems recursively
        - Combine the solutions to solve the original problem
        
        Recursive Implementation:
        ```python
        def merge_sort(arr):
            # Base case: array with 0 or 1 element is already sorted
            if len(arr) <= 1:
                return arr
                
            # Divide array into two halves
            mid = len(arr) // 2
            left = merge_sort(arr[:mid])
            right = merge_sort(arr[mid:])
            
            # Conquer: merge sorted halves
            return merge(left, right)
            
        def merge(left, right):
            result = []
            i = j = 0
            
            # Compare elements from both halves and merge
            while i < len(left) and j < len(right):
                if left[i] <= right[j]:
                    result.append(left[i])
                    i += 1
                else:
                    result.append(right[j])
                    j += 1
                    
            # Add remaining elements
            result.extend(left[i:])
            result.extend(right[j:])
            return result
        ```
        
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        
        Call Tree for merge_sort([5, 3, 8, 4, 2]):
        
        merge_sort([5, 3, 8, 4, 2])
        ├── merge_sort([5, 3])
        │   ├── merge_sort([5])
        │   └── merge_sort([3])
        │   └── merge([5], [3]) → [3, 5]
        └── merge_sort([8, 4, 2])
            ├── merge_sort([8])
            └── merge_sort([4, 2])
                ├── merge_sort([4])
                └── merge_sort([2])
                └── merge([4], [2]) → [2, 4]
            └── merge([8], [2, 4]) → [2, 4, 8]
        └── merge([3, 5], [2, 4, 8]) → [2, 3, 4, 5, 8]
        
        Explanation:
        1. We divide the array [5, 3, 8, 4, 2] into [5, 3] and [8, 4, 2]
        2. We recursively sort [5, 3]:
           - Divide into [5] and [3]
           - Merge to get [3, 5]
        3. We recursively sort [8, 4, 2]:
           - Divide into [8] and [4, 2]
           - Sort [4, 2] to get [2, 4]
           - Merge to get [2, 4, 8]
        4. Finally, we merge [3, 5] and [2, 4, 8] to get [2, 3, 4, 5, 8]
        
        Merge sort is stable (maintains relative order of equal elements) and
        guarantees O(n log n) performance even in worst-case scenarios, but
        requires additional memory for the merging process.
        """,
        
        "Tower of Hanoi": """
        TOWER OF HANOI RECURSION
        =======================
        
        Definition:
        Tower of Hanoi is a mathematical puzzle where we have three rods and n disks.
        The objective is to move the entire stack of disks from one rod to another,
        obeying specific rules.
        
        Rules:
        1. Only one disk can be moved at a time.
        2. Each move consists of taking the top disk from one stack and placing it on top of another stack.
        3. No disk may be placed on top of a smaller disk.
        
        Recursive Implementation:
        ```python
        def hanoi(n, source, target, auxiliary):
            if n == 1:
                print(f"Move disk 1 from {source} to {target}")
                return
                
            # Move n-1 disks from source to auxiliary using target as the auxiliary
            hanoi(n-1, source, auxiliary, target)
            
            # Move the nth disk from source to target
            print(f"Move disk {n} from {source} to {target}")
            
            # Move n-1 disks from auxiliary to target using source as the auxiliary
            hanoi(n-1, auxiliary, target, source)
        ```
        
        Number of Moves:
        The minimum number of moves required to solve a Tower of Hanoi puzzle is 2^n - 1.
        
        Time Complexity: O(2^n) - Exponential
        Space Complexity: O(n) due to the recursion stack
        
        Recursive Pattern for hanoi(3, 'A', 'C', 'B'):
        
        hanoi(3, A, C, B)
        ├── hanoi(2, A, B, C)
        │   ├── hanoi(1, A, C, B)
        │   │   └── Move disk 1 from A to C
        │   ├── Move disk 2 from A to B
        │   └── hanoi(1, C, B, A)
        │       └── Move disk 1 from C to B
        ├── Move disk 3 from A to C
        └── hanoi(2, B, C, A)
            ├── hanoi(1, B, A, C)
            │   └── Move disk 1 from B to A
            ├── Move disk 2 from B to C
            └── hanoi(1, A, C, B)
                └── Move disk 1 from A to C
        
        Solution for n=3:
        1. Move disk 1 from A to C
        2. Move disk 2 from A to B
        3. Move disk 1 from C to B
        4. Move disk 3 from A to C
        5. Move disk 1 from B to A
        6. Move disk 2 from B to C
        7. Move disk 1 from A to C
        
        The Tower of Hanoi is a classic example of a problem that has an elegant
        recursive solution but an exponential time complexity, making it impractical
        for large numbers of disks.
        """,
        
        "GCD": """
        RECURSIVE GREATEST COMMON DIVISOR (GCD)
        ======================================
        
        Definition:
        The Greatest Common Divisor (GCD) of two integers is the largest positive
        integer that divides both numbers without a remainder.
        
        Euclidean Algorithm:
        The GCD of two numbers can be efficiently computed using the Euclidean algorithm,
        which is based on the principle that if a and b are two positive integers,
        gcd(a, b) = gcd(b, a mod b).
        
        Recursive Implementation:
        ```python
        def gcd(a, b):
            if b == 0:
                return a
            return gcd(b, a % b)
        ```
        
        Time Complexity: O(log(min(a, b)))
        Space Complexity: O(log(min(a, b))) due to the recursion stack
        
        Example Execution for gcd(48, 18):
        
        gcd(48, 18)
        └── gcd(18, 48 % 18 = 12)
            └── gcd(12, 18 % 12 = 6)
                └── gcd(6, 12 % 6 = 0)
                    └── return 6
                    
        Explanation:
        1. We start with gcd(48, 18)
        2. Since 18 is not 0, we compute 48 % 18 = 12 and call gcd(18, 12)
        3. Since 12 is not 0, we compute 18 % 12 = 6 and call gcd(12, 6)
        4. Since 6 is not 0, we compute 12 % 6 = 0 and call gcd(6, 0)
        5. Since the second argument is 0, we return the first argument: 6
        
        The Euclidean algorithm is remarkably efficient, with a logarithmic
        time complexity, making it suitable for computing GCDs of very large integers.
        
        Iterative Version:
        ```python
        def gcd_iterative(a, b):
            while b:
                a, b = b, a % b
            return a
        ```
        
        Applications:
        - Simplifying fractions
        - Modular arithmetic
        - Public key cryptography (e.g., RSA algorithm)
        - Solving linear Diophantine equations
        """,
        
        "Power Function": """
        RECURSIVE POWER FUNCTION
        ======================
        
        Definition:
        The power function calculates b^n (b raised to the power of n).
        
        Naive Recursive Implementation:
        ```python
        def power(base, exponent):
            if exponent == 0:
                return 1
            return base * power(base, exponent - 1)
        ```
        
        Time Complexity: O(n) - Linear
        Space Complexity: O(n) due to the recursion stack
        
        Optimized Implementation (Divide and Conquer):
        ```python
        def power_optimized(base, exponent):
            if exponent == 0:
                return 1
                
            # Calculate the result for half the exponent
            half = power_optimized(base, exponent // 2)
            
            # If exponent is even, return half * half
            if exponent % 2 == 0:
                return half * half
            # If exponent is odd, return base * half * half
            else:
                return base * half * half
        ```
        
        Time Complexity: O(log n) - Logarithmic
        Space Complexity: O(log n) due to the recursion stack
        
        Example Execution for power_optimized(2, 10):
        
        power_optimized(2, 10)
        └── half = power_optimized(2, 5)
            └── half = power_optimized(2, 2)
                └── half = power_optimized(2, 1)
                    └── half = power_optimized(2, 0)
                        └── return 1
                    └── return 2 * 1 * 1 = 2 (odd exponent)
                └── return 2 * 2 = 4 (even exponent)
            └── return 2 * 4 * 4 = 32 (odd exponent)
        └── return 32 * 32 = 1024 (even exponent)
        
        Explanation:
        1. To calculate 2^10, we first calculate 2^5
        2. To calculate 2^5, we first calculate 2^2, which is 4
        3. Since 5 is odd, we return 2 * 4 * 4 = 32
        4. Since 10 is even, we return 32 * 32 = 1024
        
        The optimized power function demonstrates how the divide-and-conquer approach
        can reduce the time complexity from O(n) to O(log n), making it much more
        efficient for large exponents.
        
        This technique, known as "binary exponentiation" or "exponentiation by squaring,"
        is widely used in cryptography, modular arithmetic, and other areas requiring
        efficient power calculations.
        """
    }
    
    if name in explanations:
        print(explanations[name])
    else:
        print(f"No detailed explanation available for {name}.")


def interactive_learning():
    """Interactive learning mode for recursive algorithms"""
    visualizer = RecursionVisualizer()
    
    print("\n" + "=" * 60)
    print("🧠 RECURSION ANALYZER - INTERACTIVE LEARNING")
    print("=" * 60)
    
    while True:
        print("\nWhat would you like to do?")
        print("1. Learn about a specific recursive algorithm")
        print("2. Analyze time complexity of an algorithm")
        print("3. Compare different implementations")
        print("4. Visualize recursive call patterns")
        print("5. Exit")
        
        try:
            choice = int(input("\nEnter your choice (1-5): "))
            
            if choice == 1:
                print("\nAvailable recursive algorithms:")
                algorithms = [
                    "Factorial", "Fibonacci", "Binary Search", "Merge Sort",
                    "Tower of Hanoi", "GCD", "Power Function"
                ]
                
                for i, algo in enumerate(algorithms, 1):
                    print(f"{i}. {algo}")
                
                algo_choice = int(input("\nWhich algorithm would you like to learn about? "))
                if 1 <= algo_choice <= len(algorithms):
                    explain_algorithm(algorithms[algo_choice - 1])
                else:
                    print("Invalid choice.")
            
            elif choice == 2:
                print("\nChoose an algorithm to analyze:")
                print("1. Factorial")
                print("2. Fibonacci (naive recursive)")
                print("3. Fibonacci (memoized)")
                print("4. Merge Sort")
                print("5. Tower of Hanoi")
                
                algo_choice = int(input("\nEnter your choice: "))
                
                if algo_choice == 1:
                    sizes = [5, 10, 50, 100, 500, 1000]
                    visualizer.plot_complexity_analysis(factorial, sizes, "Factorial", "factorial_analysis.png")
                
                elif algo_choice == 2:
                    # Use smaller sizes for naive Fibonacci due to exponential growth
                    sizes = [5, 10, 15, 20, 25, 30]
                    visualizer.plot_complexity_analysis(fibonacci_recursive, sizes, "Naive Fibonacci", "naive_fibonacci_analysis.png")
                
                elif algo_choice == 3:
                    sizes = [5, 10, 50, 100, 500, 1000]
                    visualizer.plot_complexity_analysis(fibonacci_memo, sizes, "Memoized Fibonacci", "memo_fibonacci_analysis.png")
                
                elif algo_choice == 4:
                    # Generate random arrays for merge sort
                    @measure_time
                    def merge_sort_wrapper(n):
                        arr = [random.randint(1, 1000) for _ in range(n)]
                        return merge_sort(arr)[0]
                    
                    sizes = [10, 100, 1000, 5000, 10000]
                    visualizer.plot_complexity_analysis(merge_sort_wrapper, sizes, "Merge Sort", "merge_sort_analysis.png")
                
                elif algo_choice == 5:
                    sizes = [5, 10, 15, 20, 25]
                    visualizer.plot_complexity_analysis(hanoi_moves, sizes, "Tower of Hanoi", "hanoi_analysis.png")
                
                else:
                    print("Invalid choice.")
            
            elif choice == 3:
                print("\nCompare different implementations:")
                print("1. Fibonacci implementations")
                print("2. Power function implementations")
                
                compare_choice = int(input("\nEnter your choice: "))
                
                if compare_choice == 1:
                    sizes = [5, 10, 20, 30, 35]
                    funcs = [fibonacci_recursive, fibonacci_memo, fibonacci_iterative]
                    names = ["Naive Recursive", "Memoized", "Iterative"]
                    visualizer.compare_implementations(funcs, sizes, names, "fibonacci_comparison.png")
                
                elif compare_choice == 2:
                    # For power function comparison
                    base = 2
                    exponents = [10, 100, 1000, 5000, 10000]
                    
                    @measure_time
                    def power_naive_wrapper(n):
                        return power_recursive(base, n)[0]
                    
                    @measure_time
                    def power_optimized_wrapper(n):
                        return power_optimized(base, n)[0]
                    
                    funcs = [power_naive_wrapper, power_optimized_wrapper]
                    names = ["Naive Recursive", "Optimized (D&C)"]
                    visualizer.compare_implementations(funcs, exponents, names, "power_comparison.png")
                
                else:
                    print("Invalid choice.")
            
            elif choice == 4:
                print("\nVisualize recursive call patterns:")
                print("1. Factorial")
                print("2. Fibonacci")
                print("3. Merge Sort")
                
                vis_choice = int(input("\nEnter your choice: "))
                
                # Reset stats
                visualizer.reset_stats()
                
                if vis_choice == 1:
                    # Wrap factorial with call tracking
                    n = int(input("Enter n for factorial (5-10 recommended): "))
                    n = max(1, min(n, 15))  # Limit for visualization
                    
                    tracked_factorial = visualizer.track_recursive_calls(factorial.__wrapped__)
                    result = tracked_factorial(n)
                    
                    print(f"\nFactorial({n}) = {result}")
                    visualizer.print_call_tree()
                    visualizer.plot_call_pattern("Factorial", "factorial_calls.png")
                
                elif vis_choice == 2:
                    n = int(input("Enter n for Fibonacci (5-10 recommended): "))
                    n = max(1, min(n, 10))  # Limit for visualization
                    
                    tracked_fibonacci = visualizer.track_recursive_calls(fibonacci_recursive.__wrapped__)
                    result = tracked_fibonacci(n)
                    
                    print(f"\nFibonacci({n}) = {result}")
                    visualizer.print_call_tree(200)  # Limit to 200 calls
                    visualizer.plot_call_pattern("Fibonacci", "fibonacci_calls.png")
                
                elif vis_choice == 3:
                    size = int(input("Enter array size for Merge Sort (5-10 recommended): "))
                    size = max(2, min(size, 15))  # Limit for visualization
                    
                    # Generate a random array
                    arr = [random.randint(1, 100) for _ in range(size)]
                    print(f"\nOriginal array: {arr}")
                    
                    # Create a tracked version of merge_sort
                    tracked_merge_sort = visualizer.track_recursive_calls(merge_sort.__wrapped__)
                    result = tracked_merge_sort(arr)
                    
                    print(f"Sorted array: {result}")
                    visualizer.print_call_tree()
                    visualizer.plot_call_pattern("Merge Sort", "merge_sort_calls.png")
                
                else:
                    print("Invalid choice.")
            
            elif choice == 5:
                print("\nThank you for learning about recursive algorithms!")
                break
            
            else:
                print("Invalid choice. Please select a number between 1 and 5.")
        
        except ValueError:
            print("Invalid input. Please enter a number.")
        except RecursionError:
            print("RecursionError! Input size too large for this recursive algorithm.")


def demo_mode():
    """Run a quick demonstration of recursion analysis"""
    print("\n" + "=" * 60)
    print("🔬 RECURSION ANALYZER - DEMONSTRATION MODE")
    print("=" * 60)
    
    visualizer = RecursionVisualizer()
    
    # Compare Fibonacci implementations
    print("\nComparing Fibonacci implementations:")
    sizes = [5, 10, 15, 20, 25, 30]
    funcs = [fibonacci_recursive, fibonacci_memo, fibonacci_iterative]
    names = ["Naive Recursive", "Memoized", "Iterative"]
    visualizer.compare_implementations(funcs, sizes, names, "fibonacci_comparison.png")
    
    # Analyze factorial time complexity
    print("\nAnalyzing factorial time complexity:")
    sizes = [5, 10, 50, 100, 500, 1000]
    visualizer.plot_complexity_analysis(factorial, sizes, "Factorial", "factorial_analysis.png")
    
    # Visualize a recursive call pattern
    print("\nVisualizing the Fibonacci recursive call pattern:")
    visualizer.reset_stats()
    tracked_fibonacci = visualizer.track_recursive_calls(fibonacci_recursive.__wrapped__)
    result = tracked_fibonacci(7)
    print(f"Fibonacci(7) = {result}")
    visualizer.print_call_tree()
    visualizer.plot_call_pattern("Fibonacci", "fibonacci_calls.png")
    
    print("\nDemonstration complete! Check the generated plots.")


def main():
    """Main function"""
    print("\nWelcome to Recursion Analyzer!")
    print("This tool helps you understand and visualize recursive algorithms.")
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_mode()
    else:
        print("\nChoose mode:")
        print("1. Demo (quick demonstration)")
        print("2. Interactive Learning")
        
        try:
            choice = int(input("> "))
            if choice == 1:
                demo_mode()
            else:
                interactive_learning()
        except ValueError:
            print("Invalid input. Running demo mode.")
            demo_mode()


if __name__ == "__main__":
    main()
