# Lesson 3: Understanding Time Complexity (Big O Notation)

## What is Time Complexity?

**Time complexity** describes how the runtime of an algorithm changes as the input size grows. It's expressed using **Big O notation** (O), which describes the upper bound of an algorithm's growth rate.

## Why Does Time Complexity Matter?

Consider these scenarios:
- **Small dataset (n=100)**: All algorithms seem fast
- **Large dataset (n=1,000,000)**: Efficiency differences become critical
- **Real-world applications**: Can mean the difference between seconds and hours

## Common Time Complexities (Best to Worst)

### 1. O(1) - Constant Time ⚡
**Runtime stays the same regardless of input size**

```python
def get_first_element(arr):
    return arr[0]  # Always one operation

# Examples:
# - Accessing array element by index
# - Hash table lookup
# - Simple arithmetic operations
```

**Real-world analogy**: Looking up a word in a dictionary if you know the exact page number.

### 2. O(log n) - Logarithmic Time 📈
**Runtime grows slowly as input increases**

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

# Each step eliminates half the remaining elements
```

**Real-world analogy**: Finding a word in a dictionary by repeatedly opening to the middle and eliminating half the pages.

### 3. O(n) - Linear Time 📊
**Runtime grows proportionally with input size**

```python
def find_maximum(arr):
    max_val = arr[0]
    for element in arr:  # Check each element once
        if element > max_val:
            max_val = element
    return max_val

# Examples:
# - Searching unsorted array
# - Printing all elements
# - Summing all numbers
```

**Real-world analogy**: Reading every page of a book to find a specific quote.

### 4. O(n log n) - Log-Linear Time 📈📊
**Common in efficient sorting algorithms**

```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])    # Divide
    right = merge_sort(arr[mid:])   # Divide
    
    return merge(left, right)       # Conquer

# Examples:
# - Merge sort, heap sort, quick sort (average case)
# - Many divide-and-conquer algorithms
```

### 5. O(n²) - Quadratic Time ⚠️
**Runtime grows quadratically with input size**

```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):          # Outer loop: n times
        for j in range(n-1):    # Inner loop: n times
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

# Examples:
# - Nested loops over the same data
# - Bubble sort, insertion sort
# - Comparing all pairs of elements
```

**Real-world analogy**: Comparing every person in a room with every other person.

### 6. O(2ⁿ) - Exponential Time 🚨
**Runtime doubles with each additional input**

```python
def fibonacci_recursive(n):
    if n <= 1:
        return n
    return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)

# Each call spawns two more calls
# Very inefficient for large n
```

**Real-world analogy**: A chain letter where each person sends it to 2 more people.

## Visual Comparison

For n = 1000:
- **O(1)**: 1 operation
- **O(log n)**: ~10 operations
- **O(n)**: 1,000 operations  
- **O(n log n)**: ~10,000 operations
- **O(n²)**: 1,000,000 operations
- **O(2ⁿ)**: 2^1000 operations (impossible!)

## Time Complexity Rules

### 1. Drop Constants
- O(2n) → O(n)
- O(100) → O(1)
- O(n/2) → O(n)

### 2. Drop Lower Order Terms
- O(n² + n) → O(n²)
- O(n + log n) → O(n)
- O(n³ + n² + n) → O(n³)

### 3. Consider Worst Case
Big O describes the worst-case scenario, not average case.

## Space Complexity

Similar to time complexity, but measures memory usage:

```python
# O(1) space - only uses a few variables
def sum_array(arr):
    total = 0  # One variable
    for num in arr:
        total += num
    return total

# O(n) space - creates new array
def double_array(arr):
    return [x * 2 for x in arr]  # New array same size as input
```

## Practice: Analyze These Functions

### Function A
```python
def mystery_function_a(n):
    for i in range(n):
        print(i)
```
**Time Complexity**: ?

### Function B  
```python
def mystery_function_b(n):
    for i in range(n):
        for j in range(n):
            print(i, j)
```
**Time Complexity**: ?

### Function C
```python
def mystery_function_c(arr):
    return len(arr) > 0 and arr[0] == "target"
```
**Time Complexity**: ?

### Function D
```python
def mystery_function_d(n):
    i = 1
    while i < n:
        print(i)
        i *= 2
```
**Time Complexity**: ?

## Answers
- **Function A**: O(n) - single loop
- **Function B**: O(n²) - nested loops  
- **Function C**: O(1) - constant operations
- **Function D**: O(log n) - dividing problem in half each time

## Key Takeaways

- ✅ Big O describes growth rate, not exact runtime
- ✅ Focus on the dominant term for large inputs
- ✅ O(1) < O(log n) < O(n) < O(n log n) < O(n²) < O(2ⁿ)
- ✅ Nested loops often indicate O(n²) or worse
- ✅ Divide-and-conquer algorithms are often O(log n) or O(n log n)
- ✅ Consider both time AND space complexity

## Next Steps

In the next lesson, we'll learn how to measure actual runtime and see these complexities in action!
