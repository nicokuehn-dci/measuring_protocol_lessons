# Lesson 4: Measuring Performance in Python

## Why Measure Performance?

- **Validate theoretical complexity**: Does O(n²) really grow quadratically?
- **Compare algorithms**: Which sorting algorithm is faster in practice?
- **Optimize bottlenecks**: Find the slowest parts of your code
- **Make data-driven decisions**: Choose the right algorithm for your use case

## Python's `time` Module

### Basic Timing
```python
import time

start_time = time.time()
# Your code here
end_time = time.time()

execution_time = end_time - start_time
print(f"Execution time: {execution_time:.6f} seconds")
```

### Common Timing Functions

| Function | Purpose | Precision |
|----------|---------|-----------|
| `time.time()` | Wall clock time | ~1 microsecond |
| `time.perf_counter()` | High precision | ~1 nanosecond |
| `time.process_time()` | CPU time only | ~1 microsecond |

### Best Practice: Use `time.perf_counter()`
```python
import time

start = time.perf_counter()
# Your code here
end = time.perf_counter()

print(f"Elapsed: {end - start:.6f} seconds")
```

## Building a Performance Measurement Framework

### Simple Timer Class
```python
import time

class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        self.start_time = time.perf_counter()
    
    def stop(self):
        self.end_time = time.perf_counter()
        return self.elapsed_time()
    
    def elapsed_time(self):
        if self.start_time is None or self.end_time is None:
            return None
        return self.end_time - self.start_time

# Usage
timer = Timer()
timer.start()
# Your code here
elapsed = timer.stop()
print(f"Time: {elapsed:.6f} seconds")
```

### Context Manager Timer
```python
import time
from contextlib import contextmanager

@contextmanager
def timer():
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"Elapsed: {end - start:.6f} seconds")

# Usage
with timer():
    # Your code here
    time.sleep(1)
```

## Creating a Timing Decorator

### Basic Version
```python
import time
from functools import wraps

def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        execution_time = end - start
        return result, execution_time
    return wrapper

@measure_time
def slow_function(n):
    return sum(i**2 for i in range(n))

result, time_taken = slow_function(10000)
print(f"Result: {result}, Time: {time_taken:.6f}s")
```

### Advanced Version with Statistics
```python
import time
import statistics
from functools import wraps

def benchmark(iterations=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            times = []
            results = []
            
            for _ in range(iterations):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                end = time.perf_counter()
                
                times.append(end - start)
                results.append(result)
            
            # Return statistics
            return {
                'result': results[0],  # Assuming consistent results
                'mean_time': statistics.mean(times),
                'min_time': min(times),
                'max_time': max(times),
                'std_dev': statistics.stdev(times) if len(times) > 1 else 0,
                'iterations': iterations
            }
        return wrapper
    return decorator

@benchmark(iterations=10)
def test_function(n):
    return sum(range(n))

stats = test_function(1000)
print(f"Mean time: {stats['mean_time']:.6f}s")
print(f"Std dev: {stats['std_dev']:.6f}s")
```

## Benchmarking Different Algorithms

### Example: Comparing Sorting Algorithms
```python
import random
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

@measure_time
def bubble_sort(arr):
    arr = arr.copy()
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

@measure_time
def python_sort(arr):
    return sorted(arr)

# Test with different sizes
sizes = [100, 500, 1000, 2000]
for size in sizes:
    # Generate random data
    data = [random.randint(1, 1000) for _ in range(size)]
    
    # Test bubble sort
    _, bubble_time = bubble_sort(data)
    
    # Test Python's built-in sort
    _, python_time = python_sort(data)
    
    print(f"Size {size}:")
    print(f"  Bubble sort: {bubble_time:.6f}s")
    print(f"  Python sort: {python_time:.6f}s")
    print(f"  Speedup: {bubble_time/python_time:.1f}x")
    print()
```

## Performance Testing Best Practices

### 1. Warm-up Runs
```python
def benchmark_with_warmup(func, args, warmup=3, iterations=10):
    # Warm-up runs (not counted)
    for _ in range(warmup):
        func(*args)
    
    # Actual timing runs
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args)
        end = time.perf_counter()
        times.append(end - start)
    
    return statistics.mean(times)
```

### 2. Multiple Iterations
```python
def reliable_timing(func, args, min_iterations=10):
    times = []
    
    # Run until we have stable results
    while len(times) < min_iterations:
        start = time.perf_counter()
        func(*args)
        end = time.perf_counter()
        times.append(end - start)
    
    # Check for stability (coefficient of variation < 10%)
    mean_time = statistics.mean(times)
    std_dev = statistics.stdev(times)
    cv = std_dev / mean_time
    
    if cv > 0.1:  # More than 10% variation
        print(f"Warning: High variation in timing (CV: {cv:.2%})")
    
    return mean_time
```

### 3. Isolate the Code Under Test
```python
# ❌ Bad: Includes setup time
@measure_time
def bad_test(n):
    data = list(range(n))  # Setup included in timing
    return sum(data)

# ✅ Good: Only measures the algorithm
@measure_time  
def good_test(data):
    return sum(data)

# Setup outside of timing
data = list(range(1000))
result, time_taken = good_test(data)
```

## Common Pitfalls

### 1. Python's Garbage Collector
```python
import gc

# Disable garbage collection during timing
gc.disable()
start = time.perf_counter()
# Your code here
end = time.perf_counter()
gc.enable()
```

### 2. System Load
- Close other applications
- Run multiple iterations
- Report average/median times
- Consider using `nice` or `taskset` on Unix systems

### 3. Cold vs Warm Cache
```python
# First run might be slower due to caching
def test_with_cache_warmup(func, data):
    # Warm up caches
    func(data)
    
    # Now measure
    start = time.perf_counter()
    result = func(data)
    end = time.perf_counter()
    
    return result, end - start
```

## Profiling Tools

### 1. Python's `timeit` Module
```python
import timeit

# Time a simple expression
time_taken = timeit.timeit('sum(range(100))', number=10000)
print(f"Average time: {time_taken/10000:.9f} seconds")

# Time a function
def test_function():
    return sum(range(100))

time_taken = timeit.timeit(test_function, number=10000)
```

### 2. Line Profiler (`line_profiler`)
```bash
pip install line_profiler

# Add @profile decorator to functions
# Run with: kernprof -l -v script.py
```

### 3. Memory Profiler (`memory_profiler`)
```bash
pip install memory_profiler

# Add @profile decorator to functions
# Run with: python -m memory_profiler script.py
```

## Practice Exercise

Create a comprehensive benchmarking suite that:
1. Tests the three time complexity functions (O(1), O(n), O(n²))
2. Runs multiple iterations for reliability
3. Calculates statistics (mean, std dev, min, max)
4. Creates a performance report

```python
def create_performance_report(functions, input_sizes, iterations=5):
    """
    Create a comprehensive performance report
    
    Args:
        functions: List of (name, function) tuples
        input_sizes: List of input sizes to test
        iterations: Number of iterations per test
    """
    # Your implementation here
    pass
```

## Key Takeaways

- ✅ Use `time.perf_counter()` for high-precision timing
- ✅ Run multiple iterations and report statistics
- ✅ Separate setup from the code being measured
- ✅ Be aware of system factors that affect timing
- ✅ Use decorators to make timing convenient and reusable
- ✅ Validate theoretical complexity with empirical measurements
