# Lesson 6: Real-World Performance Analysis

## Moving Beyond Toy Examples

While O(1), O(n), and O(n²) examples are great for learning, real-world performance analysis involves:
- **Real algorithms** with practical applications
- **Multiple factors** affecting performance
- **Trade-offs** between time and space complexity
- **System constraints** and optimization strategies

## Case Study 1: Search Algorithms

### Linear Search vs Binary Search vs Hash Table Lookup

```python
import time
import random
from functools import wraps
import matplotlib.pyplot as plt

def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        return result, end - start
    return wrapper

@measure_time
def linear_search(arr, target):
    """O(n) - Check every element"""
    for i, element in enumerate(arr):
        if element == target:
            return i
    return -1

@measure_time
def binary_search(arr, target):
    """O(log n) - Divide and conquer on sorted array"""
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

@measure_time
def hash_lookup(hash_table, target):
    """O(1) average - Direct access via hash"""
    return hash_table.get(target, -1)

def analyze_search_algorithms():
    """Compare search algorithms across different data sizes"""
    sizes = [100, 500, 1000, 5000, 10000, 50000, 100000]
    
    linear_times = []
    binary_times = []
    hash_times = []
    
    for size in sizes:
        # Create test data
        data = list(range(size))
        hash_table = {val: idx for idx, val in enumerate(data)}
        target = size - 1  # Worst case for linear search
        
        # Linear search
        _, linear_time = linear_search(data, target)
        linear_times.append(linear_time)
        
        # Binary search (requires sorted data)
        _, binary_time = binary_search(data, target)
        binary_times.append(binary_time)
        
        # Hash lookup
        _, hash_time = hash_lookup(hash_table, target)
        hash_times.append(hash_time)
        
        print(f"Size {size:>6}: Linear={linear_time:.6f}s, "
              f"Binary={binary_time:.6f}s, Hash={hash_time:.6f}s")
    
    # Visualize results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(sizes, linear_times, 'ro-', label='Linear O(n)')
    plt.plot(sizes, binary_times, 'go-', label='Binary O(log n)')
    plt.plot(sizes, hash_times, 'bo-', label='Hash O(1)')
    plt.xlabel('Array Size')
    plt.ylabel('Time (seconds)')
    plt.title('Search Algorithm Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.loglog(sizes, linear_times, 'ro-', label='Linear O(n)')
    plt.loglog(sizes, binary_times, 'go-', label='Binary O(log n)')
    plt.loglog(sizes, hash_times, 'bo-', label='Hash O(1)')
    plt.xlabel('Array Size')
    plt.ylabel('Time (seconds)')
    plt.title('Log-Log Scale')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    speedup_binary = [l/b for l, b in zip(linear_times, binary_times)]
    speedup_hash = [l/h for l, h in zip(linear_times, hash_times)]
    plt.plot(sizes, speedup_binary, 'go-', label='Binary vs Linear')
    plt.plot(sizes, speedup_hash, 'bo-', label='Hash vs Linear')
    plt.xlabel('Array Size')
    plt.ylabel('Speedup Factor')
    plt.title('Performance Improvement')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    # Space complexity comparison (theoretical)
    linear_space = [1] * len(sizes)  # O(1) extra space
    binary_space = [1] * len(sizes)  # O(1) extra space
    hash_space = sizes  # O(n) extra space for hash table
    
    plt.plot(sizes, linear_space, 'ro-', label='Linear Search')
    plt.plot(sizes, binary_space, 'go-', label='Binary Search')
    plt.plot(sizes, hash_space, 'bo-', label='Hash Table')
    plt.xlabel('Array Size')
    plt.ylabel('Extra Space Required')
    plt.title('Space Complexity Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('search_algorithm_analysis.png', dpi=300)
    plt.show()

# Run the analysis
analyze_search_algorithms()
```

## Case Study 2: Sorting Algorithm Shootout

### Comparing Real Sorting Algorithms

```python
import random
import time
from functools import wraps

@measure_time
def bubble_sort(arr):
    """O(n²) - Educational but inefficient"""
    arr = arr.copy()
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:  # Optimization: early termination
            break
    return arr

@measure_time
def quick_sort(arr):
    """O(n log n) average, O(n²) worst case"""
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort(left) + middle + quick_sort(right)

@measure_time
def merge_sort(arr):
    """O(n log n) guaranteed"""
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
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

@measure_time
def python_sort(arr):
    """Python's built-in Timsort - O(n log n)"""
    return sorted(arr)

def comprehensive_sort_analysis():
    """Analyze sorting algorithms with different data characteristics"""
    
    sizes = [100, 250, 500, 1000, 2500]
    data_types = {
        'Random': lambda n: [random.randint(1, 1000) for _ in range(n)],
        'Nearly Sorted': lambda n: list(range(n)) + [random.randint(1, 1000) for _ in range(n//10)],
        'Reverse Sorted': lambda n: list(range(n, 0, -1)),
        'All Same': lambda n: [42] * n
    }
    
    algorithms = [
        ('Bubble Sort', bubble_sort),
        ('Quick Sort', quick_sort),
        ('Merge Sort', merge_sort),
        ('Python Sort', python_sort)
    ]
    
    results = {data_type: {alg_name: [] for alg_name, _ in algorithms} for data_type in data_types}
    
    for data_type, data_generator in data_types.items():
        print(f"\n=== {data_type} Data ===")
        
        for size in sizes:
            print(f"Size {size}:")
            data = data_generator(size)
            
            for alg_name, alg_func in algorithms:
                try:
                    if alg_name == 'Bubble Sort' and size > 1000:
                        # Skip bubble sort for large sizes
                        results[data_type][alg_name].append(None)
                        print(f"  {alg_name}: Skipped (too slow)")
                        continue
                    
                    _, exec_time = alg_func(data.copy())
                    results[data_type][alg_name].append(exec_time)
                    print(f"  {alg_name}: {exec_time:.6f}s")
                    
                except RecursionError:
                    results[data_type][alg_name].append(None)
                    print(f"  {alg_name}: RecursionError")
    
    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Sorting Algorithm Performance Analysis', fontsize=16)
    
    for idx, (data_type, data_results) in enumerate(results.items()):
        ax = axes[idx // 2, idx % 2]
        
        for alg_name, times in data_results.items():
            # Filter out None values
            valid_times = [(size, time) for size, time in zip(sizes, times) if time is not None]
            if valid_times:
                valid_sizes, valid_times = zip(*valid_times)
                ax.plot(valid_sizes, valid_times, 'o-', label=alg_name, linewidth=2)
        
        ax.set_xlabel('Array Size')
        ax.set_ylabel('Time (seconds)')
        ax.set_title(f'{data_type} Data')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # Log scale for better visualization
    
    plt.tight_layout()
    plt.savefig('sorting_analysis.png', dpi=300)
    plt.show()
    
    return results

# Run the comprehensive analysis
sort_results = comprehensive_sort_analysis()
```

## Case Study 3: Data Structure Performance

### List vs Dictionary vs Set Operations

```python
import random
import string

@measure_time
def list_operations(data, operations):
    """Test list performance"""
    results = []
    for op_type, value in operations:
        if op_type == 'search':
            results.append(value in data)
        elif op_type == 'insert':
            data.append(value)
        elif op_type == 'delete' and value in data:
            data.remove(value)
    return len(results)

@measure_time
def dict_operations(data, operations):
    """Test dictionary performance"""
    results = []
    for op_type, value in operations:
        if op_type == 'search':
            results.append(value in data)
        elif op_type == 'insert':
            data[value] = True
        elif op_type == 'delete' and value in data:
            del data[value]
    return len(results)

@measure_time
def set_operations(data, operations):
    """Test set performance"""
    results = []
    for op_type, value in operations:
        if op_type == 'search':
            results.append(value in data)
        elif op_type == 'insert':
            data.add(value)
        elif op_type == 'delete' and value in data:
            data.discard(value)
    return len(results)

def data_structure_analysis():
    """Compare performance of different data structures"""
    
    sizes = [100, 500, 1000, 5000, 10000]
    
    list_times = []
    dict_times = []
    set_times = []
    
    for size in sizes:
        # Generate test data
        base_data = [random.randint(1, size * 2) for _ in range(size)]
        
        # Generate operations (70% search, 20% insert, 10% delete)
        operations = []
        for _ in range(size // 2):  # Fewer operations than data size
            op_type = random.choices(['search', 'insert', 'delete'], 
                                   weights=[70, 20, 10])[0]
            value = random.randint(1, size * 2)
            operations.append((op_type, value))
        
        # Test list
        test_list = base_data.copy()
        _, list_time = list_operations(test_list, operations.copy())
        list_times.append(list_time)
        
        # Test dictionary
        test_dict = {val: True for val in base_data}
        _, dict_time = dict_operations(test_dict, operations.copy())
        dict_times.append(dict_time)
        
        # Test set
        test_set = set(base_data)
        _, set_time = set_operations(test_set, operations.copy())
        set_times.append(set_time)
        
        print(f"Size {size}: List={list_time:.6f}s, Dict={dict_time:.6f}s, Set={set_time:.6f}s")
    
    # Visualize
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(sizes, list_times, 'ro-', label='List', linewidth=2)
    plt.plot(sizes, dict_times, 'go-', label='Dictionary', linewidth=2)
    plt.plot(sizes, set_times, 'bo-', label='Set', linewidth=2)
    plt.xlabel('Data Size')
    plt.ylabel('Time (seconds)')
    plt.title('Data Structure Operations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.loglog(sizes, list_times, 'ro-', label='List')
    plt.loglog(sizes, dict_times, 'go-', label='Dictionary')
    plt.loglog(sizes, set_times, 'bo-', label='Set')
    plt.xlabel('Data Size')
    plt.ylabel('Time (seconds)')
    plt.title('Log-Log Scale')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    # Relative performance
    dict_speedup = [l/d for l, d in zip(list_times, dict_times)]
    set_speedup = [l/s for l, s in zip(list_times, set_times)]
    plt.plot(sizes, dict_speedup, 'go-', label='Dict vs List')
    plt.plot(sizes, set_speedup, 'bo-', label='Set vs List')
    plt.xlabel('Data Size')
    plt.ylabel('Speedup Factor')
    plt.title('Performance Improvement over List')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    # Operation breakdown analysis
    search_ops = len([op for op in operations if op[0] == 'search'])
    insert_ops = len([op for op in operations if op[0] == 'insert'])
    delete_ops = len([op for op in operations if op[0] == 'delete'])
    
    ops = ['Search', 'Insert', 'Delete']
    counts = [search_ops, insert_ops, delete_ops]
    plt.bar(ops, counts, color=['red', 'green', 'blue'])
    plt.ylabel('Number of Operations')
    plt.title('Operation Mix in Test')
    
    plt.tight_layout()
    plt.savefig('data_structure_analysis.png', dpi=300)
    plt.show()

# Run the analysis
data_structure_analysis()
```

## Memory Profiling

### Understanding Space Complexity

```python
import sys
import tracemalloc

def memory_profile(func):
    """Decorator to profile memory usage"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        
        result = func(*args, **kwargs)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return result, current, peak
    return wrapper

@memory_profile
def memory_intensive_operation(n):
    """Create and manipulate large data structures"""
    # Create a list of lists (matrix)
    matrix = [[i * j for j in range(n)] for i in range(n)]
    
    # Create a dictionary mapping
    mapping = {i: [i] * n for i in range(n)}
    
    # Sum everything
    total = sum(sum(row) for row in matrix)
    total += sum(sum(values) for values in mapping.values())
    
    return total

def analyze_memory_usage():
    """Analyze memory usage patterns"""
    sizes = [10, 25, 50, 100, 200]
    memory_usage = []
    peak_memory = []
    
    for size in sizes:
        result, current, peak = memory_intensive_operation(size)
        memory_usage.append(current / 1024 / 1024)  # Convert to MB
        peak_memory.append(peak / 1024 / 1024)
        
        print(f"Size {size}: Current={current/1024/1024:.2f}MB, Peak={peak/1024/1024:.2f}MB")
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, memory_usage, 'ro-', label='Current Memory', linewidth=2)
    plt.plot(sizes, peak_memory, 'bo-', label='Peak Memory', linewidth=2)
    plt.xlabel('Problem Size')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('memory_analysis.png', dpi=300)
    plt.show()

# Run memory analysis
analyze_memory_usage()
```

## Practical Optimization Strategies

### 1. Algorithm Selection Based on Data Characteristics

```python
def smart_search(data, target, is_sorted=None):
    """Choose search algorithm based on data characteristics"""
    
    if is_sorted is None:
        # Quick check if data appears sorted
        is_sorted = all(data[i] <= data[i+1] for i in range(min(100, len(data)-1)))
    
    if len(data) < 50:
        # For small data, linear search is often fastest due to low overhead
        return linear_search(data, target)
    elif is_sorted:
        # For sorted data, binary search is optimal
        return binary_search(data, target)
    else:
        # For large unsorted data, consider building a hash table
        # if we'll do multiple searches
        return linear_search(data, target)
```

### 2. Caching and Memoization

```python
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci_cached(n):
    """Fibonacci with memoization - O(n) instead of O(2^n)"""
    if n <= 1:
        return n
    return fibonacci_cached(n-1) + fibonacci_cached(n-2)

@measure_time
def fibonacci_uncached(n):
    """Standard recursive fibonacci - O(2^n)"""
    if n <= 1:
        return n
    return fibonacci_uncached(n-1) + fibonacci_uncached(n-2)

def compare_fibonacci_implementations():
    """Compare cached vs uncached fibonacci"""
    sizes = range(10, 36, 5)
    cached_times = []
    uncached_times = []
    
    for n in sizes:
        # Cached version
        _, cached_time = fibonacci_cached(n)
        cached_times.append(cached_time)
        
        # Uncached version (careful with large n!)
        if n <= 30:  # Don't test very large numbers
            _, uncached_time = fibonacci_uncached(n)
            uncached_times.append(uncached_time)
        else:
            uncached_times.append(None)
        
        print(f"n={n}: Cached={cached_time:.6f}s, "
              f"Uncached={uncached_time:.6f}s" if uncached_time else f"Uncached=Too slow")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.semilogy(sizes, cached_times, 'go-', label='Cached O(n)', linewidth=2)
    
    valid_uncached = [(n, t) for n, t in zip(sizes, uncached_times) if t is not None]
    if valid_uncached:
        valid_n, valid_times = zip(*valid_uncached)
        plt.semilogy(valid_n, valid_times, 'ro-', label='Uncached O(2^n)', linewidth=2)
    
    plt.xlabel('Fibonacci Number (n)')
    plt.ylabel('Time (seconds)')
    plt.title('Impact of Memoization on Fibonacci')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('fibonacci_comparison.png', dpi=300)
    plt.show()

# Run fibonacci comparison
compare_fibonacci_implementations()
```

## Key Takeaways

- ✅ **Algorithm choice matters**: Different algorithms excel in different scenarios
- ✅ **Data characteristics matter**: Sorted vs unsorted, size, access patterns
- ✅ **Trade-offs are everywhere**: Time vs space, simplicity vs performance
- ✅ **Measurement is crucial**: Theory guides, but real performance varies
- ✅ **Optimization strategies**: Caching, choosing right data structures, preprocessing
- ✅ **Context is important**: Small datasets may favor simple algorithms
- ✅ **Memory matters too**: Space complexity can be as important as time complexity

## Next Steps

1. **Profile your own code** using these techniques
2. **Experiment with different algorithms** for your specific use cases  
3. **Consider the full system** - I/O, network, memory hierarchy
4. **Measure in production** - synthetic benchmarks don't always reflect reality
5. **Learn about advanced topics** - parallel algorithms, cache-friendly programming, etc.
