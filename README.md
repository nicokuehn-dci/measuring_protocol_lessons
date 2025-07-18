# Performance Analysis & Time Complexity Learning Curriculum

A comprehensive educational package for learning performance analysis, Python decorators, and time complexity through hands-on examples and measurements.

## �� Project Overview

This project demonstrates how to measure function execution times using Python decorators and analyze different time complexities. It includes a complete learning curriculum with 6 detailed lessons, interactive quizzes, practical examples, and real-world algorithm analysis.

## �� Files

1. __`complete_solution.py`__ - Complete exercise solution with all features
2. __`time_complexity_analysis.py`__ - Detailed analysis script with plots
3. __`decorator_example.py`__ - Simple decorator usage example
4. __`advanced_examples.py`__ - More complex decorator examples
5. __`binary_search_complete.py`__ - Complete binary search implementation
6. __`binary_search_lesson.md`__ - Binary search lesson material
7. __`binary_search_solutions.py`__ - Solutions for binary search exercises
8. **`requirements.txt`** - Required Python packages
9. **`quizzes/`** - Interactive tools for algorithm performance analysis and visualization
10. __`quick_reference.md`__ - Handy cheat sheet for quick reference
11. __`time_complexity_guide.md`__ - Guide to understanding time complexity
12. __`exercise.md`__ - Project exercises
13. __`bubble_search/`__ - Bubble search algorithm implementation

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

- __Function__: `constant_time_function(n)`
- **Behavior**: Performs simple arithmetic operations
- **Characteristic**: Runtime stays approximately the same regardless of input size

### O(n) - Linear Time

- __Function__: `linear_time_function(n)`
- **Behavior**: Sums numbers from 1 to n using a loop
- **Characteristic**: Runtime grows linearly with input size

### O(n²) - Quadratic Time

- __Function__: `quadratic_time_function(n)`
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

# Binary search implementation
python binary_search_complete.py
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
- matplotlib >= 3.5.0
- numpy >= 1.21.0
- pandas (for quizzes)
- plotly (for quizzes)

Install requirements:

```bash
pip install matplotlib numpy pandas plotly
```

## �� Learning Materials

This project includes a comprehensive learning curriculum with detailed markdown lessons:

- **Lesson 1**: Introduction to Python Decorators (`lesson1_decorators_basics.md`)
- **Lesson 2**: Advanced Decorators with Arguments and Return Values (`lesson2_advanced_decorators.md`)
- **Lesson 3**: Understanding Time Complexity (Big O Notation) (`lesson3_time_complexity.md`)
- **Lesson 4**: Measuring Performance in Python (`lesson4_performance_measurement.md`)
- **Lesson 5**: Data Visualization with Matplotlib (`lesson5_data_visualization.md`)
- **Lesson 6**: Real-World Performance Analysis (`lesson6_real_world_analysis.md`)
- **Additional**: Binary Search Implementation and Analysis (`binary_search_lesson.md`)

Each lesson includes theory, examples, practice exercises, and real-world applications. The `quick_reference.md` file provides a handy cheat sheet for quick reference.

## 🧮 Interactive Quizzes

The project includes interactive tools in the `quizzes/` directory to help students practice and visualize algorithm performance:

- **`algorithm_analyzer.py`** - Analyze and visualize various algorithm time complexities
- **`recursion_analyzer.py`** - Visualize recursive function calls and analyze stack behavior
- **`sorting_playground.py`** - Compare different sorting algorithms and their performance characteristics
- **`time_complexity_quiz.py`** - Interactive quiz to test understanding of time complexity

Run any quiz tool to get an interactive learning experience:

```bash
python quizzes/algorithm_analyzer.py
python quizzes/recursion_analyzer.py
python quizzes/sorting_playground.py
python quizzes/time_complexity_quiz.py
```
