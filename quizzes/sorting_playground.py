#!/usr/bin/env python3
"""
Sorting Algorithms Playground

This educational tool demonstrates various sorting algorithms,
visualizes their performance, and provides interactive learning.
"""

import time
import random
import matplotlib.pyplot as plt
import numpy as np
from functools import wraps


def measure_time(func):
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        return result, end_time - start_time
    return wrapper


class SortingVisualizer:
    """Class for visualizing sorting algorithms"""
    
    def __init__(self):
        self.algorithms = {}
        self.results = {}
        
    def register_algorithm(self, name, func, complexity, category):
        """Register a sorting algorithm"""
        self.algorithms[name] = {
            'function': func,
            'complexity': complexity,
            'category': category
        }
    
    def generate_data(self, size, data_type='random'):
        """Generate test data for sorting"""
        if data_type == 'random':
            return [random.randint(1, 1000) for _ in range(size)]
        elif data_type == 'nearly_sorted':
            data = list(range(1, size + 1))
            # Swap ~5% of elements
            swaps = max(1, size // 20)
            for _ in range(swaps):
                i, j = random.sample(range(size), 2)
                data[i], data[j] = data[j], data[i]
            return data
        elif data_type == 'reversed':
            return list(range(size, 0, -1))
        elif data_type == 'few_unique':
            return [random.randint(1, 10) for _ in range(size)]
        else:
            return [random.randint(1, 1000) for _ in range(size)]
    
    def benchmark(self, sizes, data_types=['random'], runs=3):
        """Benchmark registered algorithms with different data sizes and types"""
        self.results = {'sizes': sizes}
        
        for data_type in data_types:
            print(f"\nBenchmarking with {data_type} data:")
            
            for name, algo_info in self.algorithms.items():
                print(f"  Running {name}...")
                times = []
                
                for size in sizes:
                    total_time = 0
                    for _ in range(runs):
                        data = self.generate_data(size, data_type)
                        _, execution_time = algo_info['function'](data.copy())
                        total_time += execution_time
                    
                    avg_time = total_time / runs
                    times.append(avg_time)
                    print(f"    Size {size}: {avg_time:.6f} seconds")
                
                result_key = f"{name} ({data_type})"
                self.results[result_key] = times
    
    def plot_results(self, log_scale=False, data_types=['random'], save_path=None):
        """Plot benchmark results"""
        plt.figure(figsize=(12, 8))
        
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple']
        
        for i, name in enumerate(self.algorithms.keys()):
            for j, data_type in enumerate(data_types):
                result_key = f"{name} ({data_type})"
                if result_key in self.results:
                    marker = markers[i % len(markers)]
                    color = colors[(i + j) % len(colors)]
                    
                    if log_scale:
                        plt.loglog(self.results['sizes'], self.results[result_key], 
                                  marker + '-', label=result_key, color=color)
                    else:
                        plt.plot(self.results['sizes'], self.results[result_key], 
                                marker + '-', label=result_key, color=color)
        
        plt.xlabel('Input Size')
        plt.ylabel('Time (seconds)')
        plt.title('Sorting Algorithm Performance Comparison' + 
                 (' (Log Scale)' if log_scale else ''))
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_dashboard(self, data_types=['random'], save_path=None):
        """Create a comprehensive dashboard of sorting performance"""
        if not self.results:
            print("No results to plot. Run benchmark first.")
            return
        
        # Create subplots based on data types
        fig, axes = plt.subplots(2, len(data_types), figsize=(15, 10))
        if len(data_types) == 1:
            axes = axes.reshape(2, 1)
        
        for i, data_type in enumerate(data_types):
            # Linear scale
            ax = axes[0, i]
            for name, algo_info in self.algorithms.items():
                result_key = f"{name} ({data_type})"
                if result_key in self.results:
                    ax.plot(self.results['sizes'], self.results[result_key], 'o-', label=name)
            
            ax.set_xlabel('Input Size')
            ax.set_ylabel('Time (seconds)')
            ax.set_title(f'{data_type.capitalize()} Data (Linear Scale)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Log scale
            ax = axes[1, i]
            for name, algo_info in self.algorithms.items():
                result_key = f"{name} ({data_type})"
                if result_key in self.results:
                    ax.loglog(self.results['sizes'], self.results[result_key], 'o-', label=name)
            
            ax.set_xlabel('Input Size')
            ax.set_ylabel('Time (seconds)')
            ax.set_title(f'{data_type.capitalize()} Data (Log Scale)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def animate_sort(self, algorithm_name, size=20, delay=0.1):
        """Animate a sorting algorithm (simple terminal visualization)"""
        if algorithm_name not in self.algorithms:
            print(f"Algorithm '{algorithm_name}' not found.")
            return
        
        # Generate data
        data = self.generate_data(size, 'random')
        print(f"Initial array: {data}")
        
        # Create a modified function to show steps
        original_func = self.algorithms[algorithm_name]['function'].__wrapped__
        
        # Create history of states
        history = [data.copy()]
        
        def track_steps(arr):
            """Track sorting steps by saving array states"""
            result = original_func(arr.copy())
            if hasattr(result, 'history'):
                return result
            return arr
        
        # Get the sorting steps
        if algorithm_name == "Bubble Sort":
            sorted_arr = self._bubble_sort_steps(data.copy())
        elif algorithm_name == "Selection Sort":
            sorted_arr = self._selection_sort_steps(data.copy())
        elif algorithm_name == "Insertion Sort":
            sorted_arr = self._insertion_sort_steps(data.copy())
        elif algorithm_name == "Merge Sort":
            # Merge sort is harder to visualize step by step in this simple format
            print("Merge Sort visualization not available in simple mode.")
            return
        elif algorithm_name == "Quick Sort":
            # Quick sort is harder to visualize step by step in this simple format
            print("Quick Sort visualization not available in simple mode.")
            return
        else:
            print(f"Step visualization for {algorithm_name} not implemented.")
            return
        
        # Display the animation
        print(f"\nAnimating {algorithm_name}:")
        for i, state in enumerate(sorted_arr):
            time.sleep(delay)
            print(f"Step {i+1}: {state}")
        
        print(f"Final sorted array: {sorted_arr[-1]}")
    
    def _bubble_sort_steps(self, arr):
        """Bubble sort with step tracking"""
        steps = [arr.copy()]
        n = len(arr)
        
        for i in range(n):
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    steps.append(arr.copy())
        
        return steps
    
    def _selection_sort_steps(self, arr):
        """Selection sort with step tracking"""
        steps = [arr.copy()]
        n = len(arr)
        
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                if arr[j] < arr[min_idx]:
                    min_idx = j
            
            if min_idx != i:
                arr[i], arr[min_idx] = arr[min_idx], arr[i]
                steps.append(arr.copy())
        
        return steps
    
    def _insertion_sort_steps(self, arr):
        """Insertion sort with step tracking"""
        steps = [arr.copy()]
        n = len(arr)
        
        for i in range(1, n):
            key = arr[i]
            j = i - 1
            while j >= 0 and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
            steps.append(arr.copy())
        
        return steps


# ===== Sorting Algorithm Implementations =====

@measure_time
def bubble_sort(arr):
    """
    Bubble Sort Implementation
    
    Time Complexity:
    - Best Case: O(n) with optimization
    - Average Case: O(n²)
    - Worst Case: O(n²)
    
    Space Complexity: O(1)
    """
    n = len(arr)
    
    # Optimization flag to detect if array is already sorted
    for i in range(n):
        swapped = False
        
        # Last i elements are already in place
        for j in range(0, n - i - 1):
            # Compare adjacent elements
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        
        # If no swapping occurred in this pass, array is sorted
        if not swapped:
            break
    
    return arr

@measure_time
def selection_sort(arr):
    """
    Selection Sort Implementation
    
    Time Complexity:
    - Best Case: O(n²)
    - Average Case: O(n²)
    - Worst Case: O(n²)
    
    Space Complexity: O(1)
    """
    n = len(arr)
    
    for i in range(n):
        # Find the minimum element in remaining unsorted array
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        
        # Swap the found minimum element with first element
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    
    return arr

@measure_time
def insertion_sort(arr):
    """
    Insertion Sort Implementation
    
    Time Complexity:
    - Best Case: O(n) when array is already sorted
    - Average Case: O(n²)
    - Worst Case: O(n²)
    
    Space Complexity: O(1)
    """
    n = len(arr)
    
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        
        # Move elements greater than key one position ahead
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        
        arr[j + 1] = key
    
    return arr

@measure_time
def merge_sort(arr):
    """
    Merge Sort Implementation
    
    Time Complexity:
    - Best Case: O(n log n)
    - Average Case: O(n log n)
    - Worst Case: O(n log n)
    
    Space Complexity: O(n)
    """
    if len(arr) <= 1:
        return arr
    
    # Divide array into two halves
    mid = len(arr) // 2
    left_half = arr[:mid]
    right_half = arr[mid:]
    
    # Recursive calls to sort both halves
    left_half = merge_sort(left_half)[0]  # Extract the sorted array from the tuple
    right_half = merge_sort(right_half)[0]
    
    # Merge the sorted halves
    return _merge(left_half, right_half)

def _merge(left, right):
    """Helper function for merge sort"""
    result = []
    i = j = 0
    
    # Merge the two sorted arrays
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

@measure_time
def quick_sort(arr):
    """
    Quick Sort Implementation
    
    Time Complexity:
    - Best Case: O(n log n)
    - Average Case: O(n log n)
    - Worst Case: O(n²) when array is already sorted
    
    Space Complexity: O(log n) due to recursion stack
    """
    # Create a copy to avoid modifying the original during timing
    arr_copy = arr.copy()
    return _quick_sort(arr_copy, 0, len(arr_copy) - 1)

def _quick_sort(arr, low, high):
    """Helper function for quick sort"""
    if low < high:
        # Partition the array and get the pivot index
        pivot_index = _partition(arr, low, high)
        
        # Sort elements before and after pivot
        _quick_sort(arr, low, pivot_index - 1)
        _quick_sort(arr, pivot_index + 1, high)
    
    return arr

def _partition(arr, low, high):
    """Helper function for partitioning in quick sort"""
    # Choose rightmost element as pivot
    pivot = arr[high]
    
    # Index of smaller element
    i = low - 1
    
    for j in range(low, high):
        # If current element is smaller than or equal to pivot
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    # Place pivot in its final position
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    
    return i + 1

@measure_time
def python_sort(arr):
    """
    Python's built-in sort (Timsort)
    
    Time Complexity:
    - Best Case: O(n)
    - Average Case: O(n log n)
    - Worst Case: O(n log n)
    
    Space Complexity: O(n)
    """
    return sorted(arr)

@measure_time
def heap_sort(arr):
    """
    Heap Sort Implementation
    
    Time Complexity:
    - Best Case: O(n log n)
    - Average Case: O(n log n)
    - Worst Case: O(n log n)
    
    Space Complexity: O(1)
    """
    n = len(arr)
    
    # Build a max heap
    for i in range(n // 2 - 1, -1, -1):
        _heapify(arr, n, i)
    
    # Extract elements one by one
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        _heapify(arr, i, 0)
    
    return arr

def _heapify(arr, n, i):
    """Helper function to maintain heap property"""
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    
    # Check if left child exists and is greater than root
    if left < n and arr[left] > arr[largest]:
        largest = left
    
    # Check if right child exists and is greater than root
    if right < n and arr[right] > arr[largest]:
        largest = right
    
    # Change root if needed
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        _heapify(arr, n, largest)


def explain_algorithm(name):
    """Print a detailed explanation of a sorting algorithm"""
    explanations = {
        "Bubble Sort": """
        BUBBLE SORT
        ===========
        
        Concept:
        Repeatedly step through the list, compare adjacent elements, and swap them if they are in the wrong order.
        The pass through the list is repeated until the list is sorted.
        
        How it works:
        1. Compare adjacent elements. If the first is greater than the second, swap them.
        2. Do this for each pair of adjacent elements, from the beginning to the end.
        3. After each pass, the largest element "bubbles up" to the end.
        4. Repeat the process for all elements except the last n elements that are already sorted.
        
        Time Complexity:
        - Best Case: O(n) when the array is already sorted (with optimization)
        - Average Case: O(n²)
        - Worst Case: O(n²)
        
        Space Complexity: O(1) - In-place sorting
        
        Characteristics:
        - Simple to understand and implement
        - Stable sorting algorithm (preserves the relative order of equal elements)
        - Poor performance on large lists
        - Adaptive: becomes faster when the list is nearly sorted (with optimization)
        
        Visualization:
        For array [5, 3, 8, 4, 2]:
        
        Initial: [5, 3, 8, 4, 2]
        
        First Pass:
        Compare 5 & 3: [3, 5, 8, 4, 2]
        Compare 5 & 8: [3, 5, 8, 4, 2] (no change)
        Compare 8 & 4: [3, 5, 4, 8, 2]
        Compare 8 & 2: [3, 5, 4, 2, 8]
        
        Second Pass:
        Compare 3 & 5: [3, 5, 4, 2, 8] (no change)
        Compare 5 & 4: [3, 4, 5, 2, 8]
        Compare 5 & 2: [3, 4, 2, 5, 8]
        
        Third Pass:
        Compare 3 & 4: [3, 4, 2, 5, 8] (no change)
        Compare 4 & 2: [3, 2, 4, 5, 8]
        
        Fourth Pass:
        Compare 3 & 2: [2, 3, 4, 5, 8]
        
        Final sorted array: [2, 3, 4, 5, 8]
        """,
        
        "Selection Sort": """
        SELECTION SORT
        ==============
        
        Concept:
        Repeatedly find the minimum element from the unsorted part of the array
        and put it at the beginning.
        
        How it works:
        1. Divide the array into sorted and unsorted parts (initially, sorted part is empty).
        2. Find the minimum element in the unsorted part.
        3. Swap it with the first element of the unsorted part.
        4. Expand the sorted part to include this newly placed element.
        5. Repeat until the entire array is sorted.
        
        Time Complexity:
        - Best Case: O(n²)
        - Average Case: O(n²)
        - Worst Case: O(n²)
        
        Space Complexity: O(1) - In-place sorting
        
        Characteristics:
        - Simple implementation
        - Performs well on small arrays
        - Unstable sorting algorithm (equal elements may change relative order)
        - Makes the minimum number of swaps (n-1 swaps) compared to other algorithms
        
        Visualization:
        For array [5, 3, 8, 4, 2]:
        
        Initial: [5, 3, 8, 4, 2]
        
        First iteration:
        Find minimum (2) and swap with first element: [2, 3, 8, 4, 5]
        
        Second iteration:
        Find minimum in remaining array (3) - already in position: [2, 3, 8, 4, 5]
        
        Third iteration:
        Find minimum in remaining array (4) and swap with third element: [2, 3, 4, 8, 5]
        
        Fourth iteration:
        Find minimum in remaining array (5) and swap with fourth element: [2, 3, 4, 5, 8]
        
        Final sorted array: [2, 3, 4, 5, 8]
        """,
        
        "Insertion Sort": """
        INSERTION SORT
        ==============
        
        Concept:
        Build the sorted array one element at a time by inserting each new element
        into its correct position within the already sorted part.
        
        How it works:
        1. Start with the second element (assume the first element is already sorted).
        2. Compare the current element with elements in the sorted part.
        3. Shift elements in the sorted part to make space for the current element.
        4. Insert the current element into its correct position in the sorted part.
        5. Repeat for all elements in the array.
        
        Time Complexity:
        - Best Case: O(n) when the array is already sorted
        - Average Case: O(n²)
        - Worst Case: O(n²) when the array is sorted in reverse order
        
        Space Complexity: O(1) - In-place sorting
        
        Characteristics:
        - Simple implementation
        - Efficient for small datasets
        - Stable sorting algorithm
        - Adaptive: becomes faster when the list is nearly sorted
        - Online algorithm: can sort a list as it receives it
        
        Visualization:
        For array [5, 3, 8, 4, 2]:
        
        Initial: [5, 3, 8, 4, 2]
        
        First iteration:
        Insert 3 into sorted part: [3, 5, 8, 4, 2]
        
        Second iteration:
        Insert 8 into sorted part (already in position): [3, 5, 8, 4, 2]
        
        Third iteration:
        Insert 4 into sorted part: [3, 4, 5, 8, 2]
        
        Fourth iteration:
        Insert 2 into sorted part: [2, 3, 4, 5, 8]
        
        Final sorted array: [2, 3, 4, 5, 8]
        """,
        
        "Merge Sort": """
        MERGE SORT
        ==========
        
        Concept:
        Divide the array into halves, sort each half, then merge the sorted halves.
        Uses the "divide and conquer" paradigm.
        
        How it works:
        1. Divide the unsorted array into n subarrays, each with one element.
        2. Repeatedly merge subarrays to produce new sorted subarrays.
        3. Continue until only one sorted array remains.
        
        Time Complexity:
        - Best Case: O(n log n)
        - Average Case: O(n log n)
        - Worst Case: O(n log n)
        
        Space Complexity: O(n) - Requires additional space for merging
        
        Characteristics:
        - Stable sorting algorithm
        - Guaranteed O(n log n) performance in all cases
        - Not in-place: requires extra memory
        - Efficient for large datasets
        - Parallelizable: can be efficiently implemented in parallel processing
        
        Visualization:
        For array [5, 3, 8, 4, 2]:
        
        Divide:
        [5, 3, 8, 4, 2] → [5, 3] and [8, 4, 2]
        [5, 3] → [5] and [3]
        [8, 4, 2] → [8] and [4, 2]
        [4, 2] → [4] and [2]
        
        Merge:
        [5] and [3] → [3, 5]
        [8] and [4, 2] → [4, 2, 8] (need to merge [4] and [2] first → [2, 4])
        [3, 5] and [2, 4, 8] → [2, 3, 4, 5, 8]
        
        Final sorted array: [2, 3, 4, 5, 8]
        """,
        
        "Quick Sort": """
        QUICK SORT
        ==========
        
        Concept:
        Choose a 'pivot' element and partition the array around it.
        Recursively sort the sub-arrays before and after the pivot.
        
        How it works:
        1. Choose a pivot element from the array.
        2. Rearrange elements: all elements less than the pivot go before it, 
           all elements greater go after it (equal elements can go either way).
        3. Recursively sort the sub-arrays created by the partitioning.
        
        Time Complexity:
        - Best Case: O(n log n)
        - Average Case: O(n log n)
        - Worst Case: O(n²) when the array is already sorted and pivot is chosen poorly
        
        Space Complexity: O(log n) - Due to recursion stack
        
        Characteristics:
        - Often faster in practice than other O(n log n) algorithms
        - Unstable sorting algorithm
        - In-place sorting with careful implementation
        - Poor pivot selection can lead to worst-case performance
        - Widely used in programming languages' standard libraries
        
        Optimization:
        - Pivot selection strategies (median-of-three, random selection)
        - Switch to insertion sort for small subarrays
        - Tail recursion elimination
        
        Visualization:
        For array [5, 3, 8, 4, 2]:
        
        Choose pivot (5):
        Partition: [3, 2, 4, 5, 8]
        
        Recursively sort [3, 2, 4]:
          Choose pivot (3):
          Partition: [2, 3, 4]
          
          Recursively sort [2]:
            Already sorted
            
          Recursively sort [4]:
            Already sorted
        
        Recursively sort [8]:
          Already sorted
        
        Final sorted array: [2, 3, 4, 5, 8]
        """,
        
        "Heap Sort": """
        HEAP SORT
        =========
        
        Concept:
        Build a max-heap from the array, then repeatedly extract the maximum element 
        and rebuild the heap until the array is sorted.
        
        How it works:
        1. Build a max-heap from the input array (where parent nodes are greater than children).
        2. Extract the root (maximum element) and place it at the end of the array.
        3. Reduce the heap size by 1 and heapify the root.
        4. Repeat steps 2-3 until the heap size is 1.
        
        Time Complexity:
        - Best Case: O(n log n)
        - Average Case: O(n log n)
        - Worst Case: O(n log n)
        
        Space Complexity: O(1) - In-place sorting
        
        Characteristics:
        - In-place sorting algorithm
        - Not stable (equal elements may change relative order)
        - Guaranteed O(n log n) time complexity in all cases
        - Efficient for large datasets
        - Uses the heap data structure
        
        Visualization:
        For array [5, 3, 8, 4, 2]:
        
        Build max-heap:
        [5, 3, 8, 4, 2] → [8, 5, 3, 4, 2]
        
        Extract maximum:
        [8, 5, 3, 4, 2] → [2, 5, 3, 4, 8]
        Heapify: [5, 4, 3, 2]
        
        Extract maximum:
        [5, 4, 3, 2] → [2, 4, 3, 5]
        Heapify: [4, 2, 3]
        
        Extract maximum:
        [4, 2, 3] → [3, 2, 4]
        Heapify: [3, 2]
        
        Extract maximum:
        [3, 2] → [2, 3]
        Heapify: [2]
        
        Final sorted array: [2, 3, 4, 5, 8]
        """,
        
        "Python Sort": """
        PYTHON'S BUILT-IN SORT (TIMSORT)
        ================================
        
        Concept:
        A hybrid sorting algorithm derived from merge sort and insertion sort,
        designed to perform well on many kinds of real-world data.
        
        How it works:
        1. Divide the array into small chunks (runs)
        2. Sort these runs using insertion sort
        3. Merge the sorted runs using a merge operation from merge sort
        4. Use galloping mode for efficient merging when patterns are detected
        
        Time Complexity:
        - Best Case: O(n) when the array is already sorted
        - Average Case: O(n log n)
        - Worst Case: O(n log n)
        
        Space Complexity: O(n)
        
        Characteristics:
        - Stable sorting algorithm
        - Adaptive: takes advantage of existing order
        - Complex but highly optimized implementation
        - Used in Python's built-in sort and Java's Arrays.sort for objects
        - Performs minimal comparisons in best cases
        
        Advantages:
        - Very fast on real-world data
        - Efficiently handles partially sorted arrays
        - Minimizes the number of comparisons and memory writes
        - Optimized for modern CPU architecture
        
        Note: Timsort is the standard sorting algorithm in Python, Java, and many other languages
        because it combines the best features of merge sort and insertion sort while avoiding
        their weaknesses.
        """
    }
    
    if name in explanations:
        print(explanations[name])
    else:
        print(f"No detailed explanation available for {name}.")


def interactive_learning():
    """Interactive learning mode with explanations and visualizations"""
    print("\n" + "=" * 60)
    print("🧠 SORTING ALGORITHMS INTERACTIVE LEARNING")
    print("=" * 60)
    
    visualizer = SortingVisualizer()
    
    # Register algorithms
    visualizer.register_algorithm("Bubble Sort", bubble_sort, "O(n²)", "Simple")
    visualizer.register_algorithm("Selection Sort", selection_sort, "O(n²)", "Simple")
    visualizer.register_algorithm("Insertion Sort", insertion_sort, "O(n²)", "Simple")
    visualizer.register_algorithm("Merge Sort", merge_sort, "O(n log n)", "Efficient")
    visualizer.register_algorithm("Quick Sort", quick_sort, "O(n log n)", "Efficient")
    visualizer.register_algorithm("Heap Sort", heap_sort, "O(n log n)", "Efficient")
    visualizer.register_algorithm("Python Sort", python_sort, "O(n log n)", "Built-in")
    
    while True:
        print("\nWhat would you like to do?")
        print("1. Learn about a specific sorting algorithm")
        print("2. Compare sorting algorithm performance")
        print("3. Visualize a sorting algorithm in action")
        print("4. Benchmark all algorithms")
        print("5. Exit")
        
        try:
            choice = int(input("\nEnter your choice (1-5): "))
            
            if choice == 1:
                print("\nAvailable sorting algorithms:")
                algorithms = [
                    "Bubble Sort", "Selection Sort", "Insertion Sort", 
                    "Merge Sort", "Quick Sort", "Heap Sort", "Python Sort"
                ]
                
                for i, algo in enumerate(algorithms, 1):
                    print(f"{i}. {algo}")
                
                algo_choice = int(input("\nWhich algorithm would you like to learn about? "))
                if 1 <= algo_choice <= len(algorithms):
                    explain_algorithm(algorithms[algo_choice - 1])
                else:
                    print("Invalid choice.")
            
            elif choice == 2:
                print("\nComparing sorting algorithms...")
                sizes = [100, 500, 1000, 5000]
                data_types = ['random', 'nearly_sorted', 'reversed']
                
                visualizer.benchmark(sizes, data_types, runs=1)
                visualizer.plot_results(log_scale=False, data_types=data_types)
                visualizer.plot_results(log_scale=True, data_types=data_types)
            
            elif choice == 3:
                print("\nAvailable sorting algorithms for visualization:")
                vis_algorithms = ["Bubble Sort", "Selection Sort", "Insertion Sort"]
                
                for i, algo in enumerate(vis_algorithms, 1):
                    print(f"{i}. {algo}")
                
                algo_choice = int(input("\nWhich algorithm would you like to visualize? "))
                if 1 <= algo_choice <= len(vis_algorithms):
                    size = int(input("Enter array size (5-30 recommended): "))
                    size = max(5, min(size, 30))  # Limit size for visualization
                    delay = float(input("Enter delay between steps (0.1-2.0 seconds): "))
                    delay = max(0.1, min(delay, 2.0))
                    
                    visualizer.animate_sort(vis_algorithms[algo_choice - 1], size, delay)
                else:
                    print("Invalid choice.")
            
            elif choice == 4:
                print("\nBenchmarking all algorithms...")
                sizes = [100, 500, 1000, 5000, 10000]
                
                visualizer.benchmark(sizes, ['random'], runs=3)
                visualizer.create_dashboard(['random'], 'sorting_benchmark.png')
            
            elif choice == 5:
                print("\nThank you for learning about sorting algorithms!")
                break
            
            else:
                print("Invalid choice. Please select a number between 1 and 5.")
        
        except ValueError:
            print("Invalid input. Please enter a number.")


def main():
    """Main function"""
    print("\nWelcome to the Sorting Algorithms Playground!")
    print("This interactive tool helps you learn and benchmark sorting algorithms.")
    
    interactive_learning()


if __name__ == "__main__":
    main()
