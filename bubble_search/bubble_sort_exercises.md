# 🫧 Bubble Sort Exercises

## Exercise 1: Product Price Sorting

### Problem Statement
Given a list of product prices, implement a Python function that uses the Bubble Sort algorithm to sort the prices in ascending order. Additionally, calculate and return the number of swaps performed during the sorting process.

**Dataset:** `[59.99, 29.99, 89.50, 45.75, 12.50, 75.30, 39.99, 65.40, 9.99, 55.00]`

### Requirements
1. Define a function named `sort_product_prices` that takes a list of product prices as input
2. Implement the Bubble Sort algorithm within the function
3. Initialize a counter variable to track the number of swaps
4. Update the counter whenever a swap is performed
5. Return both the sorted list and the total number of swaps

### Expected Usage
```python
product_prices = [59.99, 29.99, 89.50, 45.75, 12.50, 75.30, 39.99, 65.40, 9.99, 55.00]
sorted_prices, swaps = sort_product_prices(product_prices)
print("Original product prices:", product_prices)
print("Sorted product prices:", sorted_prices)
print("Number of swaps:", swaps)
```

---

## Exercise 2: Bubble Sort with Invariant Checking

### Problem Statement
Implement the Bubble Sort algorithm in Python and add a debugging function that checks the invariant of the algorithm. The invariant in Bubble Sort is the sorted sublist at the end of the list.

### Requirements
1. Define a function named `bubble_sort_with_invariant` that takes a list of numbers as input
2. Implement the Bubble Sort algorithm inside the function
3. After each iteration, call a debugging function to check the invariant
4. Implement the `check_invariant` function to verify the loop invariant
5. The `check_invariant` function should take the array and the index of the last element in the sorted sublist
6. Verify that elements in the sorted sublist are in ascending order
7. If the invariant is violated, raise an exception or print an error message
8. Return the sorted list

### Loop Invariant
After the i-th pass of bubble sort:
- The largest i elements are in their correct positions at the end of the array
- These elements are sorted among themselves

---

## Exercise 3: Enhanced Bubble Sort Analysis

### Problem Statement
Create an enhanced version of bubble sort that provides detailed analysis including:
- Number of comparisons
- Number of swaps
- Pass-by-pass breakdown
- Early termination detection

### Requirements
1. Function should return a dictionary with analysis data
2. Track comparisons and swaps separately
3. Show which pass achieved early termination (if any)
4. Display the state of the array after each pass

---

## Exercise 4: Bubble Sort Variants

### Problem Statement
Implement different variants of bubble sort and compare their performance:

1. **Standard Bubble Sort**: Basic implementation
2. **Optimized Bubble Sort**: With early termination
3. **Cocktail Shaker Sort**: Bidirectional bubble sort
4. **Odd-Even Sort**: Parallel-friendly variant

### Requirements
- Implement all four variants
- Test with the same dataset
- Compare number of operations
- Analyze best and worst-case scenarios

---

## Exercise 5: Custom Comparator Bubble Sort

### Problem Statement
Implement a bubble sort that accepts a custom comparison function, allowing sorting by different criteria.

### Requirements
1. Function should accept a comparator function parameter
2. Test with different comparison criteria:
   - Sort numbers in descending order
   - Sort strings by length
   - Sort tuples by second element
   - Sort objects by custom attribute

### Test Cases
```python
# Sort by length
words = ["bubble", "sort", "algorithm", "data", "structure"]

# Sort tuples by second element
coordinates = [(1, 3), (2, 1), (5, 2), (3, 4)]

# Sort in descending order
numbers = [5, 2, 8, 1, 9, 3]
```

---

## Exercise 6: Bubble Sort Performance Analysis

### Problem Statement
Create a comprehensive performance analysis tool for bubble sort that:
- Tests different array sizes
- Measures execution time
- Counts operations
- Compares with Python's built-in sort
- Generates performance graphs (optional)

### Requirements
1. Test with arrays of sizes: 10, 50, 100, 500, 1000
2. Generate random, sorted, and reverse-sorted test data
3. Measure and compare performance metrics
4. Create a summary report

---

## Exercise 7: Stable Sorting Verification

### Problem Statement
Demonstrate that bubble sort is a stable sorting algorithm by creating test cases with duplicate elements and verifying that their relative order is preserved.

### Requirements
1. Create test data with duplicate elements that have additional identifying information
2. Sort the data and verify stability
3. Implement a function to check if a sorting algorithm is stable
4. Compare with an unstable sort to show the difference

### Test Data Example
```python
students = [
    ("Alice", 85, 1),
    ("Bob", 92, 2), 
    ("Charlie", 85, 3),
    ("Diana", 78, 4),
    ("Eve", 85, 5)
]
# Sort by grade, verify order of students with same grade is preserved
```

---

## Bonus Exercise: Interactive Bubble Sort Visualizer

### Problem Statement
Create an interactive text-based visualizer that shows the bubble sort algorithm in action step by step.

### Requirements
1. Display the array at each step
2. Highlight elements being compared
3. Show swaps with visual indicators
4. Allow user to step through or run automatically
5. Display statistics during execution

### Example Output
```
Step 1: [64, 34, 25, 12, 22, 11, 90]
        Comparing 64 and 34
        [64, 34, 25, 12, 22, 11, 90] → SWAP!
        [34, 64, 25, 12, 22, 11, 90]

Step 2: [34, 64, 25, 12, 22, 11, 90]
        Comparing 64 and 25
        [34, 64, 25, 12, 22, 11, 90] → SWAP!
        [34, 25, 64, 12, 22, 11, 90]
```

---

## Getting Started

1. Start with Exercise 1 (Product Price Sorting) as it's the most straightforward
2. Move to Exercise 2 (Invariant Checking) to understand algorithm correctness
3. Progress through the remaining exercises based on your interest and skill level
4. Each exercise builds upon concepts from previous ones

## Tips for Success

- **Start Simple**: Begin with the basic implementation before adding complexity
- **Test Thoroughly**: Use various test cases including edge cases
- **Understand the Algorithm**: Make sure you understand why bubble sort works before optimizing
- **Document Your Code**: Add comments explaining your logic
- **Compare Results**: Verify your implementations produce correct results

Good luck with your bubble sort journey! 🚀
