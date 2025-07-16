# Performance Analysis & Time Complexity Learning Curriculum

A comprehensive educational package for learning performance analysis, Python decorators, and time complexity through hands-on examples and measurements.

## 🎯 Project Overview

This project demonstrates how to measure function execution times using Python decorators and analyze different time complexities. It includes a complete learning curriculum with 6 detailed lessons, practical examples, and real-world algorithm analysis.

## 📁 Files

1. **`complete_solution.py`** - Complete exercise solution with all features
2. **`time_complexity_analysis.py`** - Detailed analysis script with plots
3. **`decorator_example.py`** - Simple decorator usage example
4. **`advanced_examples.py`** - More complex decorator examples
5. **`requirements.txt`** - Required Python packages
6. **`lessons/`** - Complete learning curriculum with 6 comprehensive lessons
7. **`quick_reference.md`** - Handy cheat sheet for quick reference

## 🔧 The `measure_time` Decorator

The core `measure_time` decorator is designed to measure the execution time of any function:

```python
@measure_time
def my_function(n):
    # function implementation
    return result

# Usage
result, execution_time = my_function(100)
print(f"Result: {result}, Time: {execution_time:.6f} seconds")
```

## 📊 Time Complexities Demonstrated

### O(1) - Constant Time
- **Function**: `constant_time_function(n)`
- **Behavior**: Performs simple arithmetic operations
- **Characteristic**: Runtime stays approximately the same regardless of input size

### O(n) - Linear Time
- **Function**: `linear_time_function(n)`
- **Behavior**: Sums numbers from 1 to n using a loop
- **Characteristic**: Runtime grows linearly with input size

### O(n²) - Quadratic Time
- **Function**: `quadratic_time_function(n)`
- **Behavior**: Uses nested loops to perform operations
- **Characteristic**: Runtime grows quadratically with input size

## 🚀 Quick Start

Run the complete solution:
```bash
python complete_solution.py
```

Or try individual examples:
```bash
# Simple decorator demonstration
python decorator_example.py

# Detailed analysis with multiple plots
python time_complexity_analysis.py

# Advanced algorithm examples
python advanced_examples.py
```

## 🧪 Running the Analysis

1. Run the simple decorator example:
```bash
python decorator_example.py
```

2. Run the full time complexity analysis:
```bash
python time_complexity_analysis.py
```

This will:
- Test each function with different input sizes
- Measure execution times
- Generate plots showing the relationship between input size and runtime
- Save a detailed analysis plot as `time_complexity_analysis.png`

## 📈 Expected Results

- **O(1)**: Flat line (constant time)
- **O(n)**: Straight diagonal line (linear growth)
- **O(n²)**: Curved line that grows rapidly (quadratic growth)

The log-log plot makes it easier to see the different growth rates on the same scale.

## 📦 Requirements

- Python 3.x
- matplotlib
- numpy

Install requirements:
```bash
pip install matplotlib numpy
```

## 📚 Learning Materials

This project includes a comprehensive learning curriculum in the `lessons/` directory:

- **Lesson 1**: Introduction to Python Decorators
- **Lesson 2**: Advanced Decorators with Arguments and Return Values  
- **Lesson 3**: Understanding Time Complexity (Big O Notation)
- **Lesson 4**: Measuring Performance in Python
- **Lesson 5**: Data Visualization with Matplotlib
- **Lesson 6**: Real-World Performance Analysis

Each lesson includes theory, examples, practice exercises, and real-world applications. See [`lessons/README.md`](lessons/README.md) for the complete learning path.
