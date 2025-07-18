# Lesson 5: Data Visualization with Matplotlib

## Why Visualize Performance Data?

- **Spot patterns**: See how algorithms scale with input size
- **Compare algorithms**: Visual comparison is more intuitive than numbers
- **Communicate results**: Graphs tell a story better than tables
- **Validate theory**: Check if O(n²) really looks quadratic

## Matplotlib Basics

### Simple Line Plot
```python
import matplotlib.pyplot as plt

# Sample data
x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]  # y = x²

plt.plot(x, y)
plt.xlabel('Input Size')
plt.ylabel('Runtime (seconds)')
plt.title('Quadratic Growth')
plt.show()
```

### Customizing Plots
```python
import matplotlib.pyplot as plt

x = [10, 50, 100, 200, 500, 1000]
y1 = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]  # O(n)
y2 = [0.001, 0.025, 0.1, 0.4, 2.5, 10]      # O(n²)

plt.figure(figsize=(10, 6))

# Plot multiple lines
plt.plot(x, y1, 'bo-', label='O(n) Linear', linewidth=2, markersize=6)
plt.plot(x, y2, 'ro-', label='O(n²) Quadratic', linewidth=2, markersize=6)

# Customize the plot
plt.xlabel('Input Size (n)', fontsize=12)
plt.ylabel('Runtime (seconds)', fontsize=12)
plt.title('Time Complexity Comparison', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Save the plot
plt.savefig('time_complexity.png', dpi=300, bbox_inches='tight')
plt.show()
```

## Plotting Performance Data

### Complete Example: Benchmarking and Plotting
```python
import matplotlib.pyplot as plt
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
def linear_algorithm(n):
    return sum(range(n))

@measure_time
def quadratic_algorithm(n):
    total = 0
    for i in range(n):
        for j in range(n):
            total += 1
    return total

# Collect performance data
input_sizes = [10, 25, 50, 100, 200, 500]
linear_times = []
quadratic_times = []

for size in input_sizes:
    _, linear_time = linear_algorithm(size)
    linear_times.append(linear_time)
    
    _, quad_time = quadratic_algorithm(size)
    quadratic_times.append(quad_time)

# Create the plot
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)  # 2x2 grid, position 1
plt.plot(input_sizes, linear_times, 'go-', label='O(n)')
plt.plot(input_sizes, quadratic_times, 'ro-', label='O(n²)')
plt.xlabel('Input Size')
plt.ylabel('Time (seconds)')
plt.title('Linear Scale')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)  # 2x2 grid, position 2
plt.loglog(input_sizes, linear_times, 'go-', label='O(n)')
plt.loglog(input_sizes, quadratic_times, 'ro-', label='O(n²)')
plt.xlabel('Input Size')
plt.ylabel('Time (seconds)')
plt.title('Log-Log Scale')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)  # 2x2 grid, position 3
plt.plot(input_sizes, linear_times, 'go-', linewidth=2)
plt.xlabel('Input Size')
plt.ylabel('Time (seconds)')
plt.title('Linear O(n) Only')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)  # 2x2 grid, position 4
plt.plot(input_sizes, quadratic_times, 'ro-', linewidth=2)
plt.xlabel('Input Size')
plt.ylabel('Time (seconds)')
plt.title('Quadratic O(n²) Only')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comprehensive_analysis.png', dpi=300)
plt.show()
```

## Different Plot Types

### 1. Line Plots (Best for Time Series)
```python
plt.plot(x, y, 'bo-')  # Blue circles connected by lines
plt.plot(x, y, 'r--')  # Red dashed line
plt.plot(x, y, 'g^-')  # Green triangles connected by lines
```

### 2. Scatter Plots (Best for Relationships)
```python
plt.scatter(x, y, c='red', s=50, alpha=0.7)
```

### 3. Bar Charts (Best for Comparisons)
```python
algorithms = ['Bubble Sort', 'Quick Sort', 'Python Sort']
times = [2.5, 0.1, 0.05]

plt.bar(algorithms, times, color=['red', 'green', 'blue'])
plt.ylabel('Time (seconds)')
plt.title('Algorithm Comparison')
plt.xticks(rotation=45)
```

### 4. Multiple Y-Axes (Different Scales)
```python
fig, ax1 = plt.subplots()

# First y-axis
ax1.plot(x, linear_times, 'g-', label='Linear')
ax1.set_xlabel('Input Size')
ax1.set_ylabel('Linear Time (s)', color='g')

# Second y-axis
ax2 = ax1.twinx()
ax2.plot(x, quadratic_times, 'r-', label='Quadratic')
ax2.set_ylabel('Quadratic Time (s)', color='r')

plt.title('Dual Y-Axis Comparison')
plt.show()
```

## Log Scales for Performance Data

### When to Use Log Scales
- **Large range of values**: When your data spans several orders of magnitude
- **Exponential growth**: Makes exponential curves appear linear
- **Multiple algorithms**: Compare algorithms with very different performance

### Log-Log Plots
```python
# Normal scale - hard to see details
plt.subplot(1, 2, 1)
plt.plot(sizes, constant_times, label='O(1)')
plt.plot(sizes, linear_times, label='O(n)')
plt.plot(sizes, quadratic_times, label='O(n²)')
plt.title('Linear Scale')
plt.legend()

# Log-log scale - easier to compare
plt.subplot(1, 2, 2)
plt.loglog(sizes, constant_times, label='O(1)')
plt.loglog(sizes, linear_times, label='O(n)')
plt.loglog(sizes, quadratic_times, label='O(n²)')
plt.title('Log-Log Scale')
plt.legend()
```

### Semi-Log Plots
```python
# Log scale on y-axis only
plt.semilogy(x, exponential_data)

# Log scale on x-axis only  
plt.semilogx(x, y)
```

## Advanced Visualization Techniques

### 1. Error Bars (Show Variability)
```python
import numpy as np

# Sample data with error bars
x = [100, 200, 500, 1000]
mean_times = [0.01, 0.04, 0.25, 1.0]
std_devs = [0.001, 0.005, 0.02, 0.1]

plt.errorbar(x, mean_times, yerr=std_devs, 
             fmt='o-', capsize=5, capthick=2)
plt.xlabel('Input Size')
plt.ylabel('Mean Time ± Std Dev')
plt.title('Performance with Error Bars')
```

### 2. Fill Between (Show Ranges)
```python
# Show min/max range
plt.fill_between(x, min_times, max_times, alpha=0.3, label='Min-Max Range')
plt.plot(x, mean_times, 'o-', label='Mean Time')
plt.legend()
```

### 3. Annotations
```python
plt.plot(x, y, 'o-')

# Annotate specific points
plt.annotate('Inflection Point', 
             xy=(500, 0.25), 
             xytext=(600, 0.4),
             arrowprops=dict(arrowstyle='->'))
```

### 4. Subplots with Shared Axes
```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

ax1.plot(x, linear_times, 'g-')
ax1.set_title('Linear Algorithm')
ax1.set_xlabel('Input Size')
ax1.set_ylabel('Time (seconds)')

ax2.plot(x, quadratic_times, 'r-')
ax2.set_title('Quadratic Algorithm')
ax2.set_xlabel('Input Size')

plt.tight_layout()
```

## Creating Professional-Looking Plots

### Color Schemes
```python
# Professional color palette
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for i, (label, data) in enumerate(datasets):
    plt.plot(x, data, color=colors[i], label=label, linewidth=2)
```

### Styling
```python
# Use a professional style
plt.style.use('seaborn-v0_8')  # or 'ggplot', 'bmh', etc.

# Or customize manually
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 2
```

### Grid and Formatting
```python
plt.grid(True, alpha=0.3, linestyle='--')
plt.tick_params(axis='both', which='major', labelsize=10)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
```

## Interactive Elements

### Legends and Labels
```python
# Detailed legend
plt.legend(loc='upper left', 
          frameon=True, 
          fancybox=True, 
          shadow=True,
          fontsize=10)

# Detailed axis labels
plt.xlabel('Input Size (number of elements)', fontsize=12)
plt.ylabel('Execution Time (seconds)', fontsize=12)
plt.title('Algorithm Performance Comparison\n(Lower is Better)', fontsize=14)
```

## Complete Visualization Template

```python
import matplotlib.pyplot as plt
import numpy as np

def create_performance_plot(results_dict, save_path=None):
    """
    Create a comprehensive performance visualization
    
    Args:
        results_dict: Dict with 'sizes', 'algorithm_name': times
        save_path: Optional path to save the plot
    """
    
    # Set up the figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Performance Analysis Dashboard', fontsize=16)
    
    sizes = results_dict['sizes']
    
    # Plot 1: Linear scale comparison
    ax = axes[0, 0]
    for name, times in results_dict.items():
        if name != 'sizes':
            ax.plot(sizes, times, 'o-', label=name, linewidth=2, markersize=6)
    ax.set_xlabel('Input Size')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Linear Scale Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Log-log scale
    ax = axes[0, 1]
    for name, times in results_dict.items():
        if name != 'sizes':
            ax.loglog(sizes, times, 'o-', label=name, linewidth=2, markersize=6)
    ax.set_xlabel('Input Size')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Log-Log Scale')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Growth rate analysis
    ax = axes[1, 0]
    for name, times in results_dict.items():
        if name != 'sizes' and len(times) > 1:
            growth_rates = [times[i]/times[i-1] for i in range(1, len(times))]
            ax.plot(sizes[1:], growth_rates, 'o-', label=f'{name} Growth Rate')
    ax.set_xlabel('Input Size')
    ax.set_ylabel('Time Ratio (T(n)/T(n-1))')
    ax.set_title('Growth Rate Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Relative performance
    ax = axes[1, 1]
    baseline = next(iter([times for name, times in results_dict.items() if name != 'sizes']))
    for name, times in results_dict.items():
        if name != 'sizes':
            relative = [t/b for t, b in zip(times, baseline)]
            ax.plot(sizes, relative, 'o-', label=name, linewidth=2)
    ax.set_xlabel('Input Size')
    ax.set_ylabel('Relative Performance')
    ax.set_title('Relative to First Algorithm')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

# Usage example
results = {
    'sizes': [10, 50, 100, 200, 500],
    'Linear O(n)': [0.001, 0.005, 0.01, 0.02, 0.05],
    'Quadratic O(n²)': [0.001, 0.025, 0.1, 0.4, 2.5],
    'Constant O(1)': [0.001, 0.001, 0.001, 0.001, 0.001]
}

create_performance_plot(results, 'performance_dashboard.png')
```

## Key Takeaways

- ✅ Use line plots for showing how performance changes with input size
- ✅ Log-log plots help compare algorithms with very different scales
- ✅ Error bars show the reliability of your measurements
- ✅ Multiple subplots tell a complete story
- ✅ Professional styling makes your results more credible
- ✅ Save high-resolution plots for reports and presentations
- ✅ Always label axes and include legends
- ✅ Use color and markers to distinguish between algorithms
