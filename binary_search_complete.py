"""
Complete Binary Search Implementation and Examples

This file contains comprehensive binary search implementations,
performance analysis, and advanced examples for educational purposes.
"""

import math
import random
import time


# ============================================================================
# BASIC BINARY SEARCH IMPLEMENTATIONS
# ============================================================================

def binary_search_iterative(arr, target):
    """
    Iterative binary search implementation.
    
    Args:
        arr: Sorted list of comparable elements
        target: Element to search for
    
    Returns:
        int: Index of target if found, -1 if not found
    """
    left = 0
    right = len(arr) - 1
    
    while left <= right:
        # Calculate middle index (prevents overflow)
        mid = left + (right - left) // 2
        
        # Check if target is at mid
        if arr[mid] == target:
            return mid
        
        # If target is smaller, search left half
        elif arr[mid] > target:
            right = mid - 1
        
        # If target is larger, search right half
        else:
            left = mid + 1
    
    # Target not found
    return -1


def binary_search_recursive(arr, target, left=0, right=None):
    """
    Recursive binary search implementation.
    
    Args:
        arr: Sorted list
        target: Element to find
        left: Left boundary (inclusive)
        right: Right boundary (inclusive)
    
    Returns:
        int: Index of target if found, -1 if not found
    """
    if right is None:
        right = len(arr) - 1
    
    # Base case: search space is empty
    if left > right:
        return -1
    
    # Calculate middle index
    mid = left + (right - left) // 2
    
    # Check if target is at mid
    if arr[mid] == target:
        return mid
    
    # Search left half
    elif arr[mid] > target:
        return binary_search_recursive(arr, target, left, mid - 1)
    
    # Search right half
    else:
        return binary_search_recursive(arr, target, mid + 1, right)


def binary_search_with_details(arr, target):
    """
    Binary search with detailed step-by-step output for learning.
    """
    left = 0
    right = len(arr) - 1
    step = 0
    
    print(f"Searching for {target} in array: {arr}")
    print(f"Array indices: {list(range(len(arr)))}")
    print("-" * 50)
    
    while left <= right:
        step += 1
        mid = left + (right - left) // 2
        
        print(f"Step {step}:")
        print(f"  Search range: [{left}, {right}]")
        print(f"  Middle index: {mid}, value: {arr[mid]}")
        
        if arr[mid] == target:
            print(f"  🎉 Found {target} at index {mid}!")
            return mid
        elif arr[mid] > target:
            print(f"  {arr[mid]} > {target}, search left half")
            right = mid - 1
        else:
            print(f"  {arr[mid]} < {target}, search right half")
            left = mid + 1
        
        print()
    
    print(f"❌ {target} not found in array")
    return -1


# ============================================================================
# PHONE DIRECTORY IMPLEMENTATION
# ============================================================================

class Phone:
    """Phone object for directory search example"""
    def __init__(self, number):
        self.number = number
    
    def get_number(self):
        return self.number
    
    def __str__(self):
        return f"Phone({self.number})"
    
    def __repr__(self):
        return self.__str__()


def generate_phone_directory(size=100000):
    """
    Generate a sorted list of phone numbers as Phone objects.
    
    Args:
        size: Number of phone numbers to generate
    
    Returns:
        list: Sorted list of Phone objects
    """
    print(f"Generating {size:,} phone numbers...")
    
    # Use set to avoid duplicates
    phone_numbers = set()
    
    while len(phone_numbers) < size:
        # Generate realistic 10-digit phone number
        area_code = random.randint(200, 999)
        exchange = random.randint(200, 999)
        number = random.randint(0, 9999)
        phone_num = f"{area_code}{exchange}{number:04d}"
        phone_numbers.add(phone_num)
    
    # Convert to sorted list
    phone_list = sorted(list(phone_numbers))
    
    # Create Phone objects
    phone_objects = [Phone(num) for num in phone_list]
    
    print(f"✅ Generated {len(phone_objects):,} unique phone numbers")
    return phone_objects


def binary_search_phone_directory(phone_list, target_number):
    """
    Binary search implementation for Phone objects.
    
    Args:
        phone_list: Sorted list of Phone objects
        target_number: Phone number string to search for
    
    Returns:
        tuple: (index, comparisons) or (-1, comparisons) if not found
    """
    left = 0
    right = len(phone_list) - 1
    comparisons = 0
    
    while left <= right:
        comparisons += 1
        mid = left + (right - left) // 2
        current_number = phone_list[mid].get_number()
        
        if current_number == target_number:
            return mid, comparisons
        elif current_number < target_number:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1, comparisons


# ============================================================================
# PERFORMANCE ANALYSIS
# ============================================================================

def linear_search_with_counter(arr, target):
    """Linear search with comparison counter"""
    for i, element in enumerate(arr):
        if element == target:
            return i, i + 1  # Index, comparisons
    return -1, len(arr)


def compare_search_algorithms():
    """
    Compare performance of linear vs binary search.
    """
    print("📊 SEARCH ALGORITHM PERFORMANCE COMPARISON")
    print("=" * 60)
    
    sizes = [100, 1000, 10000, 100000]
    
    print(f"{'Size':<10} {'Linear (avg)':<12} {'Binary (max)':<12} {'Speedup':<10}")
    print("-" * 60)
    
    for size in sizes:
        # Generate sorted test data
        data = list(range(size))
        
        # Run multiple tests for average performance
        total_linear_comparisons = 0
        max_binary_comparisons = 0
        num_tests = 100
        
        for _ in range(num_tests):
            target = random.randint(0, size - 1)
            
            # Linear search
            _, linear_comps = linear_search_with_counter(data, target)
            total_linear_comparisons += linear_comps
            
            # Binary search
            _, binary_comps = binary_search_iterative_with_counter(data, target)
            max_binary_comparisons = max(max_binary_comparisons, binary_comps)
        
        avg_linear = total_linear_comparisons / num_tests
        theoretical_binary = math.ceil(math.log2(size))
        speedup = avg_linear / theoretical_binary if theoretical_binary > 0 else 0
        
        print(f"{size:<10} {avg_linear:<12.1f} {theoretical_binary:<12} {speedup:<10.1f}x")


def binary_search_iterative_with_counter(arr, target):
    """Binary search with comparison counter"""
    left = 0
    right = len(arr) - 1
    comparisons = 0
    
    while left <= right:
        comparisons += 1
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            return mid, comparisons
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1, comparisons


def time_search_algorithms():
    """
    Time-based performance comparison of search algorithms.
    """
    print("\n⏱️  TIMING COMPARISON")
    print("=" * 40)
    
    sizes = [1000, 10000, 100000]
    
    for size in sizes:
        data = list(range(size))
        target = size // 2  # Middle element
        
        # Time linear search
        start_time = time.perf_counter()
        for _ in range(1000):  # Multiple runs for better measurement
            linear_search_with_counter(data, target)
        linear_time = time.perf_counter() - start_time
        
        # Time binary search
        start_time = time.perf_counter()
        for _ in range(1000):  # Multiple runs for better measurement
            binary_search_iterative_with_counter(data, target)
        binary_time = time.perf_counter() - start_time
        
        speedup = linear_time / binary_time if binary_time > 0 else float('inf')
        
        print(f"Size {size:,}:")
        print(f"  Linear search:  {linear_time:.6f} seconds")
        print(f"  Binary search:  {binary_time:.6f} seconds")
        print(f"  Speedup: {speedup:.1f}x")
        print()


# ============================================================================
# ADVANCED BINARY SEARCH VARIANTS
# ============================================================================

def find_first_occurrence(arr, target):
    """Find the first occurrence of target in sorted array with duplicates"""
    left, right = 0, len(arr) - 1
    result = -1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            result = mid
            right = mid - 1  # Continue searching left for first occurrence
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return result


def find_last_occurrence(arr, target):
    """Find the last occurrence of target in sorted array with duplicates"""
    left, right = 0, len(arr) - 1
    result = -1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            result = mid
            left = mid + 1  # Continue searching right for last occurrence
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return result


def find_insert_position(arr, target):
    """Find position where target should be inserted to maintain sorted order"""
    left, right = 0, len(arr)
    
    while left < right:
        mid = left + (right - left) // 2
        
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid
    
    return left


def binary_search_range(arr, target):
    """Find the range [first, last] of target in sorted array"""
    first = find_first_occurrence(arr, target)
    if first == -1:
        return [-1, -1]
    
    last = find_last_occurrence(arr, target)
    return [first, last]


# ============================================================================
# BINARY SEARCH ON ANSWER
# ============================================================================

def sqrt_binary_search(x, precision=1e-6):
    """
    Find square root using binary search on the answer.
    Example of "binary search on answer" technique.
    """
    if x < 0:
        return None
    if x == 0:
        return 0
    
    left = 0
    right = x if x >= 1 else 1
    
    while right - left > precision:
        mid = (left + right) / 2
        
        if mid * mid <= x:
            left = mid
        else:
            right = mid
    
    return (left + right) / 2


def find_peak_element(arr):
    """
    Find a peak element in an array using binary search.
    A peak element is greater than its neighbors.
    """
    left, right = 0, len(arr) - 1
    
    while left < right:
        mid = left + (right - left) // 2
        
        if arr[mid] < arr[mid + 1]:
            left = mid + 1
        else:
            right = mid
    
    return left


# ============================================================================
# DEMONSTRATION FUNCTIONS
# ============================================================================

def demonstrate_basic_binary_search():
    """Demonstrate basic binary search functionality"""
    print("🔍 BASIC BINARY SEARCH DEMONSTRATION")
    print("=" * 50)
    
    # Test array
    test_array = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    
    # Test successful search
    target = 7
    print(f"Array: {test_array}")
    result = binary_search_with_details(test_array, target)
    print(f"Result: {result}\n")
    
    # Test unsuccessful search
    target = 8
    print(f"Searching for non-existent element:")
    result = binary_search_with_details(test_array, target)


def demonstrate_phone_directory():
    """Demonstrate phone directory search"""
    print("\n📞 PHONE DIRECTORY SEARCH DEMONSTRATION")
    print("=" * 50)
    
    # Generate smaller directory for demonstration
    directory = generate_phone_directory(1000)
    
    print(f"\nDirectory created with {len(directory)} entries")
    print(f"First entry: {directory[0].get_number()}")
    print(f"Last entry: {directory[-1].get_number()}")
    
    # Test searches
    test_numbers = [
        directory[0].get_number(),      # First
        directory[len(directory)//2].get_number(),  # Middle
        directory[-1].get_number(),     # Last
        "0000000000"                    # Non-existent
    ]
    
    for number in test_numbers:
        result, comparisons = binary_search_phone_directory(directory, number)
        status = "Found" if result != -1 else "Not found"
        print(f"Search for {number}: {status} (index: {result}, comparisons: {comparisons})")


def demonstrate_advanced_variants():
    """Demonstrate advanced binary search variants"""
    print("\n🚀 ADVANCED BINARY SEARCH VARIANTS")
    print("=" * 50)
    
    # Array with duplicates
    arr_with_dups = [1, 2, 2, 2, 3, 4, 4, 5, 5, 5]
    target = 2
    
    print(f"Array with duplicates: {arr_with_dups}")
    print(f"Target: {target}")
    
    first = find_first_occurrence(arr_with_dups, target)
    last = find_last_occurrence(arr_with_dups, target)
    range_result = binary_search_range(arr_with_dups, target)
    
    print(f"First occurrence: index {first}")
    print(f"Last occurrence: index {last}")
    print(f"Range: {range_result}")
    
    # Insert position
    numbers = [1, 3, 5, 7, 9]
    insert_target = 6
    insert_pos = find_insert_position(numbers, insert_target)
    print(f"\nInsert {insert_target} into {numbers} at position: {insert_pos}")
    
    # Square root using binary search
    sqrt_target = 25
    sqrt_result = sqrt_binary_search(sqrt_target)
    print(f"Square root of {sqrt_target} ≈ {sqrt_result}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all demonstrations and examples"""
    print("🎯 COMPLETE BINARY SEARCH IMPLEMENTATION AND EXAMPLES")
    print("=" * 60)
    
    # Basic demonstrations
    demonstrate_basic_binary_search()
    
    # Phone directory example
    demonstrate_phone_directory()
    
    # Performance comparison
    compare_search_algorithms()
    time_search_algorithms()
    
    # Advanced variants
    demonstrate_advanced_variants()
    
    print("\n" + "=" * 60)
    print("✅ All demonstrations completed!")
    print("📚 Check binary_search_lesson.md for detailed explanations")
    print("💡 Run binary_search_solutions.py for exercise solutions")


if __name__ == "__main__":
    main()
