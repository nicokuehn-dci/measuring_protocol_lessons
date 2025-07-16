# Quick Reference: Performance Analysis Cheat Sheet

## 🕒 Time Complexity Quick Reference

| Notation | Name | Example | Growth Rate |
|----------|------|---------|-------------|
| O(1) | Constant | Array access, hash lookup | Best |
| O(log n) | Logarithmic | Binary search | Excellent |
| O(n) | Linear | Linear search, array sum | Good |
| O(n log n) | Log-linear | Merge sort, heap sort | Acceptable |
| O(n²) | Quadratic | Nested loops, bubble sort | Poor |
| O(2ⁿ) | Exponential | Naive fibonacci | Terrible |

## 🎯 Decorator Templates

### Basic Timer Decorator
```python
import time
from functools import wraps

def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        return result, end - start
    return wrapper
```

### Benchmarking Decorator
```python
def benchmark(iterations=10):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                end = time.perf_counter()
                times.append(end - start)
            
            return {
                'result': result,
                'mean_time': statistics.mean(times),
                'min_time': min(times),
                'max_time': max(times),
                'std_dev': statistics.stdev(times) if len(times) > 1 else 0
            }
        return wrapper
    return decorator
```

### Caching Decorator
```python
from functools import lru_cache

@lru_cache(maxsize=None)
def expensive_function(n):
    # Your expensive computation here
    pass
```

## 📊 Matplotlib Quick Start

### Basic Performance Plot
```python
import matplotlib.pyplot as plt

# Simple comparison
plt.plot(sizes, times1, 'bo-', label='Algorithm 1')
plt.plot(sizes, times2, 'ro-', label='Algorithm 2')
plt.xlabel('Input Size')
plt.ylabel('Runtime (seconds)')
plt.title('Performance Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Log-Log Plot
```python
plt.loglog(sizes, times1, 'bo-', label='O(n)')
plt.loglog(sizes, times2, 'ro-', label='O(n²)')
plt.xlabel('Input Size')
plt.ylabel('Runtime (seconds)')
plt.title('Log-Log Scale Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
```

### Multiple Subplots
```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Top left
axes[0, 0].plot(x, y1)
axes[0, 0].set_title('Linear Scale')

# Top right  
axes[0, 1].loglog(x, y1)
axes[0, 1].set_title('Log-Log Scale')

# Bottom left
axes[1, 0].plot(x, y2)
axes[1, 0].set_title('Growth Rate')

# Bottom right
axes[1, 1].bar(labels, values)
axes[1, 1].set_title('Comparison')

plt.tight_layout()
plt.show()
```

## ⚡ Performance Measurement Best Practices

### DO ✅
- Use `time.perf_counter()` for high precision
- Run multiple iterations and report statistics
- Warm up caches before measuring
- Separate setup from measurement
- Use appropriate time scales (seconds, milliseconds)
- Visualize results with plots
- Consider both time AND space complexity

### DON'T ❌
- Don't optimize before measuring
- Don't ignore constant factors for small datasets
- Don't assume theoretical complexity matches reality
- Don't forget about memory usage
- Don't test with unrealistic data
- Don't skip error bars or confidence intervals

## 🔍 Common Algorithm Patterns

### O(1) - Constant Time
```python
def constant_example(arr):
    return arr[0]  # Always one operation
```

### O(log n) - Logarithmic
```python
def binary_search(arr, target):
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
```

### O(n) - Linear
```python
def linear_example(arr):
    total = 0
    for item in arr:  # One pass through data
        total += item
    return total
```

### O(n²) - Quadratic
```python
def quadratic_example(arr):
    for i in range(len(arr)):      # Outer loop
        for j in range(len(arr)):  # Inner loop
            # Do something with arr[i] and arr[j]
            pass
```

## 🛠️ Useful Python Tools

### Built-in Timing
```python
import timeit

# Time a simple expression
time_taken = timeit.timeit('sum(range(100))', number=1000)
print(f"Average: {time_taken/1000:.9f} seconds")
```

### Memory Profiling
```python
import tracemalloc

tracemalloc.start()
# Your code here
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"Current: {current / 1024 / 1024:.2f} MB")
print(f"Peak: {peak / 1024 / 1024:.2f} MB")
```

### Context Manager Timer
```python
from contextlib import contextmanager
import time

@contextmanager
def timer():
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"Time: {end - start:.6f} seconds")

# Usage
with timer():
    # Your code here
    result = some_function()
```

## 📈 Growth Rate Recognition

### How to Identify Complexity

1. **Look for nested loops**: Usually O(n²) or worse
2. **Divide and conquer**: Often O(log n) or O(n log n)
3. **Single loop**: Usually O(n)
4. **Recursive with two calls**: Often O(2ⁿ) - watch out!
5. **Hash table operations**: Usually O(1) average case

### Complexity Comparison for n = 1,000,000

| Complexity | Operations | Human Time |
|------------|------------|------------|
| O(1) | 1 | Instant |
| O(log n) | ~20 | Instant |
| O(n) | 1,000,000 | ~1 second |
| O(n log n) | ~20,000,000 | ~20 seconds |
| O(n²) | 1,000,000,000,000 | ~31 years |
| O(2ⁿ) | 2^1,000,000 | Heat death of universe |

## 🎯 Quick Algorithm Selection Guide

### For Searching:
- **Small datasets (< 100)**: Linear search
- **Sorted data**: Binary search O(log n)
- **Frequent searches**: Hash table O(1)
- **Range queries**: Tree structures

### For Sorting:
- **Small datasets (< 50)**: Insertion sort
- **General purpose**: Python's built-in sort (Timsort)
- **Memory constrained**: Heap sort O(n log n)
- **Nearly sorted**: Insertion sort or Timsort

### For Data Structures:
- **Fast lookups**: Dictionary/Hash table
- **Ordered data**: List or sorted structures
- **Set operations**: Python sets
- **Priority queues**: heapq module

---

**Need more details?** Check out the complete [lesson series](lessons/README.md) for in-depth explanations and practice exercises!
