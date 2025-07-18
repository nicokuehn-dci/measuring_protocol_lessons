"""
Solutions to Binary Search Exercises

This file contains complete solutions to the exercises from the binary search lesson.
"""

import math
import random
import time


# ============================================================================
# SOLUTION 1: Count Divisions Exercise
# ============================================================================

def count_divisions(n):
    """Return the number of times n can be divided by 2 before reaching 1."""
    count = 0
    while n > 1:
        n //= 2  # Integer division by 2
        count += 1
    return count


def solution_1_count_divisions():
    """Complete solution for Exercise 1"""
    print("=== Exercise 1: Count Divisions ===")
    print("Understanding log₂(n) through division\n")
    
    # Test the function with provided values
    test_values = [2, 4, 8, 16, 32, 64, 128, 256]
    
    print(f"{'n':<4} {'count_divisions':<15} {'math.log2(n)':<12} {'Match?'}")
    print("-" * 45)
    
    for n in test_values:
        manual_log = count_divisions(n)
        math_log = math.log2(n)
        match = manual_log == int(math_log)
        
        print(f"{n:<4} {manual_log:<15} {math_log:<12.0f} {'✓' if match else '✗'}")
    
    print("\nConclusion: count_divisions(n) = ⌊log₂(n)⌋ for powers of 2")
    print("This is exactly what binary search does - halve the search space!")


# ============================================================================
# SOLUTION 2: Plotting Exercise
# ============================================================================

def solution_2_plotting():
    """Complete solution for Exercise 2 - Plotting"""
    print("\n=== Exercise 2: Plotting log-scaled data ===")
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Generate data for n = 1 to 1000
        n_values = np.arange(1, 1001)
        linear_values = n_values  # f(n) = n
        log_values = np.log2(n_values)  # g(n) = log₂(n)
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Linear x-axis
        ax1.plot(n_values, linear_values, 'b-', label='f(n) = n', linewidth=2)
        ax1.plot(n_values, log_values, 'r-', label='g(n) = log₂(n)', linewidth=2)
        ax1.set_xlabel('n')
        ax1.set_ylabel('Function Value')
        ax1.set_title('Linear X-Axis')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Logarithmic x-axis
        ax2.semilogx(n_values, linear_values, 'b-', label='f(n) = n', linewidth=2)
        ax2.semilogx(n_values, log_values, 'r-', label='g(n) = log₂(n)', linewidth=2)
        ax2.set_xlabel('n (log scale)')
        ax2.set_ylabel('Function Value')
        ax2.set_title('Logarithmic X-Axis')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/nico-kuehn-dci/Desktop/lectures/measuring/log_scale_comparison.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Plots created successfully!")
        print("\n🔍 Observations:")
        print("• Linear scale: log(n) grows very slowly compared to n")
        print("• Log scale: log(n) becomes more linear-looking")
        print("• The log function appears straighter on log-scaled axes because")
        print("  logarithmic growth is fundamentally about multiplicative relationships")
        
    except ImportError:
        print("⚠️  matplotlib/numpy not available. Here's the conceptual solution:")
        print("• On linear x-axis: log(n) curve flattens quickly")
        print("• On log x-axis: log(n) appears more linear")
        print("• This is because log transforms multiplicative to additive relationships")


# ============================================================================
# SOLUTION 3: Phone Number Search Exercise
# ============================================================================

class Phone:
    """Phone object with get_number() method as specified"""
    def __init__(self, number):
        self.number = number
    
    def get_number(self):
        return self.number
    
    def __str__(self):
        return f"Phone({self.number})"


def generate_sorted_phone_list(size=100000):
    """
    Generate a sorted list of 100,000 phone numbers as Phone objects.
    Uses random.randint and sort as hinted.
    """
    print(f"Generating {size:,} phone numbers...")
    
    # Generate random phone numbers using random.randint
    phone_numbers = set()  # Use set to avoid duplicates
    
    while len(phone_numbers) < size:
        # Generate 10-digit phone number
        # Area code: 200-999, Exchange: 200-999, Number: 0000-9999
        area_code = random.randint(200, 999)
        exchange = random.randint(200, 999)
        number = random.randint(0, 9999)
        phone_num = f"{area_code}{exchange}{number:04d}"
        phone_numbers.add(phone_num)
    
    # Convert to list and sort using sort()
    phone_list = list(phone_numbers)
    phone_list.sort()  # Sort the phone numbers
    
    # Create Phone objects using lambda for sorting key if needed
    phone_objects = [Phone(num) for num in phone_list]
    
    print(f"Generated and sorted {len(phone_objects):,} unique phone numbers")
    return phone_objects


def binary_search_phone(phone_list, target_number):
    """
    Binary search implementation for Phone objects.
    Returns index if found, -1 if not found.
    """
    left = 0
    right = len(phone_list) - 1
    comparisons = 0
    
    while left <= right:
        comparisons += 1
        mid = (left + right) // 2
        current_number = phone_list[mid].get_number()
        
        if current_number == target_number:
            return mid, comparisons
        elif current_number < target_number:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1, comparisons


def solution_3_phone_search():
    """Complete solution for Exercise 3 - Phone Number Search"""
    print("\n=== Exercise 3: Phone Number Search ===")
    
    # Generate smaller list for demonstration (1000 instead of 100,000)
    phone_list = generate_sorted_phone_list(1000)
    
    # Test with randomly generated phone numbers
    print("\n🔍 Testing binary search with random phone numbers:")
    
    # Test 1: Search for existing numbers
    test_indices = [0, len(phone_list)//4, len(phone_list)//2, 3*len(phone_list)//4, -1]
    
    for i, idx in enumerate(test_indices):
        target = phone_list[idx].get_number()
        result, comparisons = binary_search_phone(phone_list, target)
        
        print(f"Test {i+1}: Searching for {target}")
        print(f"  Found at index {result} (expected {idx % len(phone_list)}) in {comparisons} comparisons")
    
    # Test 2: Search for non-existent number
    fake_number = "0000000000"  # Likely doesn't exist
    result, comparisons = binary_search_phone(phone_list, fake_number)
    print(f"\nSearching for non-existent number {fake_number}:")
    print(f"  Result: {result} (not found) after {comparisons} comparisons")
    
    # Performance comparison with theoretical
    theoretical_max = math.ceil(math.log2(len(phone_list)))
    print(f"\n📊 Performance Analysis:")
    print(f"  List size: {len(phone_list):,}")
    print(f"  Theoretical max comparisons: {theoretical_max}")
    print(f"  Actual max comparisons in tests: {comparisons}")
    print(f"  Efficiency: O(log n) = O(log {len(phone_list)}) ≈ {theoretical_max} steps")


# ============================================================================
# BONUS: Complete Binary Search with Performance Analysis
# ============================================================================

def basic_binary_search(arr, target):
    """Basic binary search implementation for comparison"""
    left, right = 0, len(arr) - 1
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


def linear_search(arr, target):
    """Linear search for performance comparison"""
    for i, element in enumerate(arr):
        if element == target:
            return i, i + 1  # Found at index i, took i+1 comparisons
    return -1, len(arr)  # Not found, checked all elements


def performance_comparison():
    """Compare binary search vs linear search performance"""
    print("\n=== BONUS: Performance Comparison ===")
    
    sizes = [100, 1000, 10000, 100000]
    
    print(f"{'Size':<10} {'Linear (avg)':<12} {'Binary (max)':<12} {'Speedup':<10}")
    print("-" * 50)
    
    for size in sizes:
        # Generate sorted test data
        data = list(range(size))
        
        # Test multiple searches for average performance
        total_linear_comparisons = 0
        max_binary_comparisons = 0
        
        # Test 100 random searches
        for _ in range(100):
            target = random.randint(0, size - 1)
            
            # Linear search
            _, linear_comps = linear_search(data, target)
            total_linear_comparisons += linear_comps
            
            # Binary search
            _, binary_comps = basic_binary_search(data, target)
            max_binary_comparisons = max(max_binary_comparisons, binary_comps)
        
        avg_linear = total_linear_comparisons / 100
        theoretical_binary = math.ceil(math.log2(size))
        speedup = avg_linear / theoretical_binary
        
        print(f"{size:<10} {avg_linear:<12.1f} {theoretical_binary:<12} {speedup:<10.1f}x")


# ============================================================================
# ADVANCED: Binary Search Variants
# ============================================================================

def find_first_occurrence(arr, target):
    """Find the first occurrence of target in sorted array with duplicates"""
    left, right = 0, len(arr) - 1
    result = -1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            result = mid
            right = mid - 1  # Continue searching left
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
            left = mid + 1  # Continue searching right
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


def advanced_binary_search_examples():
    """Demonstrate advanced binary search variants"""
    print("\n=== Advanced Binary Search Examples ===")
    
    # Array with duplicates
    arr_with_dups = [1, 2, 2, 2, 3, 4, 4, 5, 5, 5]
    target = 2
    
    print(f"Array: {arr_with_dups}")
    print(f"Target: {target}")
    
    first = find_first_occurrence(arr_with_dups, target)
    last = find_last_occurrence(arr_with_dups, target)
    
    print(f"First occurrence of {target}: index {first}")
    print(f"Last occurrence of {target}: index {last}")
    
    # Insert position
    numbers = [1, 3, 5, 7, 9]
    insert_target = 6
    insert_pos = find_insert_position(numbers, insert_target)
    print(f"\nInsert {insert_target} into {numbers} at position: {insert_pos}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all exercise solutions"""
    print("🔍 BINARY SEARCH EXERCISE SOLUTIONS")
    print("=" * 50)
    
    # Solution 1: Count divisions
    solution_1_count_divisions()
    
    # Solution 2: Plotting (if libraries available)
    solution_2_plotting()
    
    # Solution 3: Phone number search
    solution_3_phone_search()
    
    # Bonus: Performance comparison
    performance_comparison()
    
    # Advanced examples
    advanced_binary_search_examples()
    
    print("\n" + "=" * 50)
    print("✅ All exercises completed!")
    print("📚 See binary_search_lesson.md for complete theory")


if __name__ == "__main__":
    main()
