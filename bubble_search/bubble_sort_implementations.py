"""
Complete Bubble Sort Implementations
Multiple versions of bubble sort for educational purposes
"""

import time
import random


def bubble_sort_basic(arr):
    """
    Basic bubble sort implementation.
    Time Complexity: O(n²)
    Space Complexity: O(1)
    """
    n = len(arr)
    
    # Traverse through all array elements
    for i in range(n):
        # Last i elements are already in place
        for j in range(0, n - i - 1):
            # Traverse the array from 0 to n-i-1
            # Swap if the element found is greater than the next element
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    
    return arr


def bubble_sort_optimized(arr):
    """
    Optimized bubble sort with early termination.
    If no swaps occur in a pass, the array is already sorted.
    Best case: O(n) when array is already sorted
    Worst case: O(n²)
    """
    n = len(arr)
    
    for i in range(n):
        # Flag to track if any swap happened
        swapped = False
        
        # Last i elements are already sorted
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                # Swap elements
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        
        # If no swapping happened, array is sorted
        if not swapped:
            print(f"Array sorted early! Stopped after {i + 1} passes.")
            break
    
    return arr


def bubble_sort_detailed(arr):
    """
    Bubble sort with detailed step-by-step visualization.
    Shows each comparison and swap operation.
    """
    n = len(arr)
    total_comparisons = 0
    total_swaps = 0
    
    print(f"Starting Bubble Sort on: {arr}")
    print("=" * 50)
    
    for i in range(n):
        print(f"\n🔄 PASS {i + 1}:")
        swapped = False
        pass_swaps = 0
        
        for j in range(0, n - i - 1):
            total_comparisons += 1
            print(f"  Compare arr[{j}]={arr[j]} with arr[{j+1}]={arr[j+1]}", end="")
            
            if arr[j] > arr[j + 1]:
                # Swap needed
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                total_swaps += 1
                pass_swaps += 1
                swapped = True
                print(f" → SWAP! New array: {arr}")
            else:
                print(f" → No swap needed")
        
        print(f"  📊 Pass {i + 1} complete: {pass_swaps} swaps, Array: {arr}")
        
        # Early termination if no swaps occurred
        if not swapped:
            print(f"  ✅ No swaps in this pass - array is sorted!")
            break
    
    print("\n" + "=" * 50)
    print(f"🎉 BUBBLE SORT COMPLETE!")
    print(f"📊 Statistics:")
    print(f"   Total comparisons: {total_comparisons}")
    print(f"   Total swaps: {total_swaps}")
    print(f"   Final sorted array: {arr}")
    
    return arr


def bubble_sort_count_operations(arr):
    """
    Bubble sort that counts and returns operation statistics.
    Returns: (sorted_array, comparisons, swaps)
    """
    n = len(arr)
    comparisons = 0
    swaps = 0
    
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            comparisons += 1
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swaps += 1
                swapped = True
        
        if not swapped:
            break
    
    return arr, comparisons, swaps


def bubble_sort_descending(arr):
    """
    Bubble sort implementation for descending order.
    """
    n = len(arr)
    
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            # Change comparison for descending order
            if arr[j] < arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        
        if not swapped:
            break
    
    return arr


def cocktail_shaker_sort(arr):
    """
    Cocktail Shaker Sort (bidirectional bubble sort).
    Sorts in both directions in each pass.
    """
    n = len(arr)
    left = 0
    right = n - 1
    
    while left < right:
        # Forward pass (left to right)
        swapped = False
        for i in range(left, right):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                swapped = True
        
        if not swapped:
            break
        
        right -= 1
        
        # Backward pass (right to left)
        swapped = False
        for i in range(right, left, -1):
            if arr[i] < arr[i - 1]:
                arr[i], arr[i - 1] = arr[i - 1], arr[i]
                swapped = True
        
        if not swapped:
            break
        
        left += 1
    
    return arr


def bubble_sort_strings_by_length(arr):
    """
    Bubble sort for strings, sorted by length.
    """
    n = len(arr)
    
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if len(arr[j]) > len(arr[j + 1]):
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        
        if not swapped:
            break
    
    return arr


def demonstrate_all_implementations():
    """Demonstrate all bubble sort implementations"""
    print("🫧 BUBBLE SORT IMPLEMENTATIONS DEMO")
    print("=" * 50)
    
    # Test data
    test_data = [64, 34, 25, 12, 22, 11, 90]
    print(f"Original array: {test_data}")
    print()
    
    # Basic implementation
    print("1. BASIC BUBBLE SORT:")
    result = bubble_sort_basic(test_data.copy())
    print(f"Result: {result}")
    print()
    
    # Optimized with early termination
    print("2. OPTIMIZED BUBBLE SORT (with already sorted array):")
    sorted_data = [1, 2, 3, 4, 5]
    print(f"Input: {sorted_data}")
    bubble_sort_optimized(sorted_data.copy())
    print()
    
    # Detailed visualization
    print("3. DETAILED BUBBLE SORT:")
    small_data = [5, 2, 8, 1]
    bubble_sort_detailed(small_data)
    print()
    
    # Count operations
    print("4. BUBBLE SORT WITH STATISTICS:")
    test_copy = test_data.copy()
    sorted_arr, comps, swaps = bubble_sort_count_operations(test_copy)
    print(f"Sorted: {sorted_arr}")
    print(f"Comparisons: {comps}, Swaps: {swaps}")
    print()
    
    # Descending order
    print("5. DESCENDING ORDER BUBBLE SORT:")
    desc_result = bubble_sort_descending(test_data.copy())
    print(f"Descending: {desc_result}")
    print()
    
    # Cocktail shaker sort
    print("6. COCKTAIL SHAKER SORT:")
    cocktail_result = cocktail_shaker_sort(test_data.copy())
    print(f"Result: {cocktail_result}")
    print()
    
    # String sorting by length
    print("7. SORT STRINGS BY LENGTH:")
    strings = ["hello", "a", "algorithm", "sort", "bubble", "implementation"]
    print(f"Original: {strings}")
    string_result = bubble_sort_strings_by_length(strings.copy())
    print(f"By length: {string_result}")


if __name__ == "__main__":
    demonstrate_all_implementations()
