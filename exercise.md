##  Exercise: Count the number of times you can divide n by 2 before reaching 1
#### Objective:
* Understand the meaning of log₂(n) (base-2 logarithm) as "the number of times you can halve n before reaching 1".
* Write a Python program that computes this and compares it to math.log2(n).
---
### Instructions
Write a function count_divisions(n) that:
* Starts with a positive integer n > 0
* Keeps dividing n by 2 (integer division) as long as n > 1
* Counts how many times you performed the division
* Returns that count
 Use math.log2(n) to compare your result.
 Run your function with different values of n to see how it grows.
---
###  Starter Code
python
import math

def count_divisions(n):
    """Return the number of times n can be divided by 2 before reaching 1."""
??????

# Test the function
for n in [2, 4, 8, 16, 32, 64, 128, 256]:
    manual_log = count_divisions(n)
    math_log = math.log2(n)
    print(f"n = {n:<3} → count_divisions = {manual_log}, math.log2(n) = {math_log:.0f}")
---
### Expected Output
n = 2   → count_divisions = 1, math.log2(n) = 1
n = 4   → count_divisions = 2, math.log2(n) = 2
n = 8   → count_divisions = 3, math.log2(n) = 3
n = 16  → count_divisions = 4, math.log2(n) = 4
n = 32  → count_divisions = 5, math.log2(n) = 5
n = 64  → count_divisions = 6, math.log2(n) = 6
n = 128 → count_divisions = 7, math.log2(n) = 7
n = 256 → count_divisions = 8, math.log2(n) = 8

---
## 🔍 LÖSUNG & ERKLÄRUNG - Exercise 1

### Vollständige Lösung:

```python
import math

def count_divisions(n):
    """Return the number of times n can be divided by 2 before reaching 1."""
    count = 0
    while n > 1:
        n //= 2  # Integer division by 2
        count += 1
    return count

# Test the function
for n in [2, 4, 8, 16, 32, 64, 128, 256]:
    manual_log = count_divisions(n)
    math_log = math.log2(n)
    print(f"n = {n:<3} → count_divisions = {manual_log}, math.log2(n) = {math_log:.0f}")
```

### 📝 Erklärung der Lösung:

**Algorithmus-Schritte:**
1. **Initialisierung**: `count = 0` - Zähler für die Anzahl der Divisionen
2. **While-Schleife**: Solange `n > 1` ist, führe folgende Schritte aus:
   - Teile `n` durch 2 (Integer-Division `//`)
   - Erhöhe den Zähler um 1
3. **Rückgabe**: Die Anzahl der durchgeführten Divisionen

**Beispiel für n = 16:**
- Start: n = 16, count = 0
- 1. Division: n = 16 // 2 = 8, count = 1
- 2. Division: n = 8 // 2 = 4, count = 2
- 3. Division: n = 4 // 2 = 2, count = 3
- 4. Division: n = 2 // 2 = 1, count = 4
- Ende: n = 1, return count = 4

**Warum ist das log₂(n)?**
- Jede Division halbiert die Zahl
- Die Frage ist: "Wie oft kann ich n halbieren bis ich bei 1 ankomme?"
- Das ist genau die Definition von log₂(n)
- Für Zweierpotenzen: count_divisions(2ᵏ) = k = log₂(2ᵏ)

**Verbindung zu Binary Search:**
- Binary Search halbiert bei jedem Schritt den Suchbereich
- Maximale Anzahl Schritte = log₂(Array-Größe)
- Das macht Binary Search so effizient: O(log n) statt O(n)


12:13 Uhr
##  Exercise: Plotting log-scaled data
###  Objective
* See how logarithmic growth and exponential growth look on linear and log-scaled plots.
* Understand why a logarithmic algorithm (`O(log n)`) becomes a straight line on a logarithmic x-axis.
---
###  Task
Write a Python script that plots the following functions over `n = 1 … 1000`:
* Linear: $f(n) = n$
* Logarithmic: $g(n) = \log_2(n)$
 Plot these on two plots:
* Linear x-axis (normal)
* Logarithmic x-axis
Observe and explain:
* Why does `log(n)` look like a straight line when plotted against a log-scaled x-axis?

---
## 🔍 LÖSUNG & ERKLÄRUNG - Exercise 2: Plotting

### Vollständige Lösung:

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate data for n = 1 to 1000
n_values = np.arange(1, 1001)
linear_values = n_values          # f(n) = n
log_values = np.log2(n_values)    # g(n) = log₂(n)

# Create plots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Linear x-axis (normal)
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
plt.show()
```

### 📊 Erklärung der Beobachtungen:

**1. Linearer X-Achsen-Plot:**
- **f(n) = n**: Gerade Linie (45° Steigung)
- **g(n) = log₂(n)**: Kurve, die schnell flacher wird
- **Beobachtung**: log₂(n) wächst sehr langsam im Vergleich zu n

**2. Logarithmischer X-Achsen-Plot:**
- **f(n) = n**: Wird zu einer Kurve nach oben
- **g(n) = log₂(n)**: Wird zu einer geraden Linie!

**🤔 Warum wird log(n) zu einer geraden Linie auf log-Achse?**

**Mathematische Erklärung:**
- Logarithmische x-Achse bedeutet: x-Koordinaten werden durch log(x) ersetzt
- Wir plotten dann: log₂(n) gegen log₁₀(n) (oder ln(n))
- Basiswechsel-Formel: log₂(n) = ln(n)/ln(2) = konstante × ln(n)
- Das ergibt: y = konstante × x → **gerade Linie!**

**Intuitive Erklärung:**
- Logarithmus transformiert multiplikative Beziehungen in additive
- log(a×b) = log(a) + log(b)
- Auf log-Skala werden exponentielle Beziehungen linear
- Deshalb erscheinen logarithmische Funktionen als Geraden

**Praktische Bedeutung:**
- **Binary Search Komplexität**: O(log n) ist auf großen Skalen praktisch konstant
- **Performance**: Verdopplung der Datenmenge = nur +1 Schritt mehr
- **Skalierbarkeit**: Logarithmische Algorithmen skalieren ausgezeichnet
Bonus:
### Exercise:
1. Generate a sorted list of 100,000 phone numbers represented as Phone objects. Each Phone object should have a get_number() method that returns the phone number as a string.
Hint:
- use random.randint
- use sort
- use lambda lambda
2. Implement the binary search algorithm to efficiently find whether a given phone number exists in the list or not.
3. Test your implementation by searching for randomly generated phone numbers within the list.

---
## 🔍 LÖSUNG & ERKLÄRUNG - Bonus Exercise: Phone Number Search

### Vollständige Lösung:

```python
import random
import math

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
    Generate a sorted list of phone numbers as Phone objects.
    Uses random.randint and sort as requested.
    """
    print(f"Generiere {size:,} Telefonnummern...")
    
    # Generate unique phone numbers using random.randint
    phone_numbers = set()  # Set verhindert Duplikate
    
    while len(phone_numbers) < size:
        # 10-stellige Telefonnummer generieren
        # Vorwahl: 200-999, Zentrale: 200-999, Nummer: 0000-9999
        area_code = random.randint(200, 999)
        exchange = random.randint(200, 999)
        number = random.randint(0, 9999)
        phone_num = f"{area_code}{exchange}{number:04d}"
        phone_numbers.add(phone_num)
    
    # In Liste konvertieren und sortieren
    phone_list = list(phone_numbers)
    phone_list.sort()  # Sortierung wie verlangt
    
    # Phone-Objekte erstellen
    phone_objects = [Phone(num) for num in phone_list]
    
    print(f"✅ {len(phone_objects):,} eindeutige Telefonnummern generiert und sortiert")
    return phone_objects

def binary_search_phone(phone_list, target_number):
    """
    Binary search implementation für Phone objects.
    Returns: (index, anzahl_vergleiche) oder (-1, anzahl_vergleiche)
    """
    left = 0
    right = len(phone_list) - 1
    comparisons = 0
    
    while left <= right:
        comparisons += 1
        mid = (left + right) // 2
        current_number = phone_list[mid].get_number()
        
        # Debug-Output für erste paar Iterationen
        if comparisons <= 3:
            print(f"  Schritt {comparisons}: Prüfe Index {mid} → {current_number}")
        
        if current_number == target_number:
            return mid, comparisons
        elif current_number < target_number:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1, comparisons

def test_phone_search():
    """Teste die Phone Number Search Implementation"""
    print("🔍 PHONE NUMBER SEARCH TEST")
    print("=" * 50)
    
    # Kleinere Liste für Demo (1000 statt 100,000)
    phone_list = generate_sorted_phone_list(1000)
    
    print(f"\n📋 Liste erstellt:")
    print(f"   Erste Nummer: {phone_list[0].get_number()}")
    print(f"   Letzte Nummer: {phone_list[-1].get_number()}")
    print(f"   Größe: {len(phone_list):,}")
    
    # Test 1: Suche nach existierenden Nummern
    print(f"\n🎯 Test 1: Suche nach existierenden Nummern")
    test_indices = [0, len(phone_list)//4, len(phone_list)//2, 3*len(phone_list)//4, -1]
    
    for i, idx in enumerate(test_indices):
        target = phone_list[idx].get_number()
        print(f"\nSuche #{i+1}: {target}")
        result, comparisons = binary_search_phone(phone_list, target)
        
        print(f"  ✅ Gefunden bei Index {result} (erwartet: {idx % len(phone_list)})")
        print(f"  📊 Benötigte Vergleiche: {comparisons}")
    
    # Test 2: Suche nach nicht-existierender Nummer
    print(f"\n❌ Test 2: Suche nach nicht-existierender Nummer")
    fake_number = "0000000000"
    print(f"Suche: {fake_number}")
    result, comparisons = binary_search_phone(phone_list, fake_number)
    print(f"  Ergebnis: {result} (nicht gefunden)")
    print(f"  📊 Benötigte Vergleiche: {comparisons}")
    
    # Performance-Analyse
    theoretical_max = math.ceil(math.log2(len(phone_list)))
    print(f"\n📊 PERFORMANCE ANALYSE:")
    print(f"   Array-Größe: {len(phone_list):,}")
    print(f"   Theoretisches Maximum: ⌈log₂({len(phone_list)})⌉ = {theoretical_max}")
    print(f"   Tatsächliche Vergleiche: ≤ {comparisons}")
    print(f"   Komplexität: O(log n)")
    
    # Vergleich mit linearer Suche
    linear_worst_case = len(phone_list)
    speedup = linear_worst_case / theoretical_max
    print(f"\n⚡ SPEEDUP vs. Linear Search:")
    print(f"   Linear Search (worst case): {linear_worst_case:,} Vergleiche")
    print(f"   Binary Search (worst case): {theoretical_max} Vergleiche")
    print(f"   Geschwindigkeitsvorteil: {speedup:.0f}x schneller!")

# Ausführung
if __name__ == "__main__":
    test_phone_search()
```

### 📚 Detaillierte Erklärung:

**1. Phone Class Implementation:**
```python
class Phone:
    def __init__(self, number):
        self.number = number
    
    def get_number(self):
        return self.number
```
- **Einfache Kapselung**: Telefonnummer wird in Objekt gekapselt
- **get_number() Method**: Wie in der Aufgabe verlangt
- **Erweiterbar**: Könnte später um Name, Adresse etc. erweitert werden

**2. Sortierte Liste generieren:**
```python
phone_numbers = set()  # Verhindert Duplikate
while len(phone_numbers) < size:
    # 10-stellige Nummer: AAABBBCCCC
    area_code = random.randint(200, 999)     # 3 Stellen
    exchange = random.randint(200, 999)      # 3 Stellen  
    number = random.randint(0, 9999)         # 4 Stellen
    phone_num = f"{area_code}{exchange}{number:04d}"
    phone_numbers.add(phone_num)

phone_list.sort()  # Sortierung ist ESSENTIELL für Binary Search!
```

**3. Binary Search Algorithm:**
```python
def binary_search_phone(phone_list, target_number):
    left = 0
    right = len(phone_list) - 1
    
    while left <= right:
        mid = (left + right) // 2
        current_number = phone_list[mid].get_number()
        
        if current_number == target_number:
            return mid    # GEFUNDEN!
        elif current_number < target_number:
            left = mid + 1    # Suche in rechter Hälfte
        else:
            right = mid - 1   # Suche in linker Hälfte
    
    return -1  # NICHT GEFUNDEN
```

**🔄 Algorithm Walkthrough (Beispiel mit 8 Elementen):**
```
Array: [A, B, C, D, E, F, G, H]  (sortiert)
Suche: F

Schritt 1: left=0, right=7, mid=3 → Element D
         F > D → left = 4 (suche rechts)

Schritt 2: left=4, right=7, mid=5 → Element F  
         F == F → GEFUNDEN bei Index 5!
```

**📊 Komplexitäts-Analyse:**

| Array-Größe | Linear Search | Binary Search | Speedup |
|-------------|---------------|---------------|---------|
| 1,000       | 1,000         | 10           | 100x    |
| 10,000      | 10,000        | 14           | 714x    |
| 100,000     | 100,000       | 17           | 5,882x  |
| 1,000,000   | 1,000,000     | 20           | 50,000x |

**🎯 Warum ist Binary Search so effizient?**
- **Halbierung**: Jeder Schritt eliminiert 50% der verbleibenden Möglichkeiten
- **Logarithmisches Wachstum**: Verdopplung der Daten → nur +1 Schritt mehr
- **Worst Case**: ⌈log₂(n)⌉ Vergleiche
- **Best Case**: 1 Vergleich (Element in der Mitte)

**⚠️ Wichtige Voraussetzung:**
- **Array MUSS sortiert sein!**
- Ohne Sortierung funktioniert Binary Search nicht
- Sortierung: O(n log n), aber einmalige Kosten
- Danach: Beliebig viele Suchen in O(log n)

---

# 📊 Common Algorithm Time Complexity Examples (With Explanations)

## O(1) – Constant Time

**Example:** Accessing the first element of an array.

```python
def get_first_element(arr):
    return arr[0]  # Always takes the same time, regardless of array size
```

**Explanation:** No matter how large the array is, retrieving `arr[0]` always takes the same amount of time. The time required does not depend on the input size.

---

## O(n) – Linear Time

**Example:** Finding the maximum value in a list.

```python
def find_maximum(arr):
    max_val = arr[0]
    for element in arr:  # Must check each element once
        if element > max_val:
            max_val = element
    return max_val
```

**Explanation:** You must look at each element in the list once. If the list has `n` elements, the number of steps will be proportional to `n`.

---

## O(n²) – Quadratic Time

**Example:** Comparing every pair of elements in a list (e.g., in bubble sort).

```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):        # Outer loop: n iterations
        for j in range(n-1):  # Inner loop: n-1 iterations
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]  # Swap
    return arr
```

**Explanation:** Two nested loops — for every element, you compare it to every other element. For a list of `n` elements, this results in `n × n = n²` operations.

---

## O(log n) – Logarithmic Time

**Example:** Binary search in a sorted array.

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2  # Halve the search space
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

**Explanation:** Each step reduces the problem size by half. If the array has `n` elements, it will take about `log₂(n)` steps to find the target or determine it isn't present.

---

## O(n log n) – Linearithmic Time

**Example:** Merge sort or heapsort.

```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])    # Divide: log n levels
    right = merge_sort(arr[mid:])   # Divide: log n levels
    
    return merge(left, right)       # Conquer: n operations per level

def merge(left, right):
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):  # O(n) merge operation
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result
```

**Explanation:** The array is divided in half repeatedly (logarithmic divisions), but merging (or heapifying) requires checking all `n` items on each level of division. This produces `O(n log n)` complexity and is common in efficient sorting algorithms.

---

## O(2ⁿ) – Exponential Time

**Example:** Calculating the nth Fibonacci number with the naïve recursive algorithm.

```python
def fibonacci_naive(n):
    if n <= 1:
        return n
    return fibonacci_naive(n-1) + fibonacci_naive(n-2)  # Two recursive calls!
```

**Explanation:** The function calls itself twice for each input, and the number of function calls doubles with each increase of `n`. This leads to a very rapid increase in execution steps as the input grows.

**Call tree for fibonacci_naive(5):**
```
                    fib(5)
                  /        \
              fib(4)        fib(3)
            /      \      /      \
        fib(3)   fib(2) fib(2)  fib(1)
       /    \    /   \   /   \
   fib(2) fib(1) ...  ... ... ...
```

---

# 📋 Table: Summary of Example Algorithms and Complexities

| Complexity | Typical Example | Description |
|------------|----------------|-------------|
| **O(1)** | `arr[0]` | Constant-time access |
| **O(n)** | Linear search | Scan each element once |
| **O(n²)** | Bubble sort | Compare all pairs with nested loops |
| **O(log n)** | Binary search | Halve the problem at each step |
| **O(n log n)** | Merge sort, Heapsort | Divide and process all items at each log-level |
| **O(2ⁿ)** | Recursive Fibonacci | Operations double with every single element added |

---

# 📚 Additional Notes

## O(m × n) – Multiple Input Sizes
```python
def find_common_elements(list1, list2):
    common = []
    for item1 in list1:        # m iterations
        for item2 in list2:    # n iterations
            if item1 == item2:
                common.append(item1)
    return common
```

If you have two different input sizes (`m` and `n`), like nested loops over two lists, the complexity is `O(m × n)`.

## Key Principles

- **Asymptotic notation** focuses on how the steps grow with very large input sizes, ignoring lower order terms and constant factors.

- **Worst-case analysis** is typically used unless otherwise specified.

- **Space complexity** follows similar patterns but measures memory usage instead of time.

## 📈 Growth Rate Comparison

For `n = 1,000,000`:

| Complexity | Operations | Practical Meaning |
|------------|------------|------------------|
| O(1) | 1 | Instant |
| O(log n) | ~20 | Effectively instant |
| O(n) | 1,000,000 | Very fast |
| O(n log n) | ~20,000,000 | Fast |
| O(n²) | 1,000,000,000,000 | Impractical |
| O(2ⁿ) | 2^1,000,000 | Universe's lifetime insufficient |

These examples cover the most commonly discussed time complexities in computer science and algorithm analysis.

---

## 🔗 Related Topics

- **How do nested loops lead to O(n²) time complexity in algorithms**
- **Why is binary search classified as an O(log n) algorithm**
- **What makes merge sort's O(n log n) complexity optimal for sorting**
- **How does exponential growth in Fibonacci calculations affect performance**
- **Why do divide-and-conquer algorithms often have logarithmic or linearithmic complexities**

---

# 🫧 Bubble Sort: Complete Lesson with Detailed Examples

## 📚 What is Bubble Sort?

**Bubble Sort** is one of the simplest sorting algorithms to understand and implement. It works by repeatedly stepping through the list, comparing adjacent elements and swapping them if they are in the wrong order. The algorithm gets its name because smaller elements "bubble" to the top of the list, just like air bubbles rising to the surface of water.

### 🎯 Key Characteristics:
- **Simple to understand and implement**
- **Stable**: Equal elements maintain their relative order
- **In-place**: Only requires O(1) extra memory
- **Inefficient**: O(n²) time complexity makes it impractical for large datasets

---

## 🔍 How Bubble Sort Works

### Basic Algorithm Steps:
1. **Compare** adjacent elements in the array
2. **Swap** them if they are in the wrong order (left > right for ascending sort)
3. **Repeat** for the entire array
4. **Continue** passes until no swaps are needed

### 📊 Visual Example - Sorting [64, 34, 25, 12, 22, 11, 90]

```
Initial Array: [64, 34, 25, 12, 22, 11, 90]

Pass 1:
[64, 34, 25, 12, 22, 11, 90] → Compare 64 & 34 → Swap
[34, 64, 25, 12, 22, 11, 90] → Compare 64 & 25 → Swap
[34, 25, 64, 12, 22, 11, 90] → Compare 64 & 12 → Swap
[34, 25, 12, 64, 22, 11, 90] → Compare 64 & 22 → Swap
[34, 25, 12, 22, 64, 11, 90] → Compare 64 & 11 → Swap
[34, 25, 12, 22, 11, 64, 90] → Compare 64 & 90 → No swap
Result: [34, 25, 12, 22, 11, 64, 90] ✅ Largest element (90) is now in correct position

Pass 2:
[34, 25, 12, 22, 11, 64, 90] → Compare 34 & 25 → Swap
[25, 34, 12, 22, 11, 64, 90] → Compare 34 & 12 → Swap
[25, 12, 34, 22, 11, 64, 90] → Compare 34 & 22 → Swap
[25, 12, 22, 34, 11, 64, 90] → Compare 34 & 11 → Swap
[25, 12, 22, 11, 34, 64, 90] → Compare 34 & 64 → No swap
Result: [25, 12, 22, 11, 34, 64, 90] ✅ Second largest (64) is now in position

... (continue until sorted)

Final Result: [11, 12, 22, 25, 34, 64, 90]
```

---

## 💻 Complete Implementation with Detailed Comments

### Basic Bubble Sort Implementation:

```python
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

# Test the basic implementation
test_array = [64, 34, 25, 12, 22, 11, 90]
print("Original array:", test_array)
sorted_array = bubble_sort_basic(test_array.copy())
print("Sorted array:  ", sorted_array)
```

### Optimized Bubble Sort with Early Termination:

```python
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

# Test with already sorted array
sorted_test = [1, 2, 3, 4, 5]
print("Testing with sorted array:", sorted_test)
bubble_sort_optimized(sorted_test.copy())
```

### Detailed Bubble Sort with Step-by-Step Visualization:

```python
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

# Detailed example
print("DETAILED BUBBLE SORT EXAMPLE:")
test_data = [5, 2, 8, 1, 9]
bubble_sort_detailed(test_data)
```

---

## 📊 Time and Space Complexity Analysis

### Time Complexity:

| Case | Complexity | Explanation |
|------|------------|-------------|
| **Best Case** | O(n) | Array already sorted (with optimization) |
| **Average Case** | O(n²) | Random order elements |
| **Worst Case** | O(n²) | Array sorted in reverse order |

### Detailed Complexity Breakdown:

```python
def analyze_bubble_sort_complexity():
    """
    Analyze the mathematical complexity of bubble sort.
    """
    print("📊 BUBBLE SORT COMPLEXITY ANALYSIS")
    print("=" * 40)
    
    # For array of size n:
    n = 10  # Example size
    
    print(f"For array size n = {n}:")
    print(f"  Pass 1: {n-1} comparisons")
    print(f"  Pass 2: {n-2} comparisons") 
    print(f"  Pass 3: {n-3} comparisons")
    print(f"  ...")
    print(f"  Pass {n-1}: 1 comparison")
    
    total_comparisons = sum(range(1, n))
    print(f"\n  Total comparisons = 1 + 2 + 3 + ... + {n-1}")
    print(f"                    = {total_comparisons}")
    print(f"                    = (n-1) × n / 2")
    print(f"                    = n² / 2 - n / 2")
    print(f"                    = O(n²)  (ignoring lower-order terms)")
    
    # Space complexity
    print(f"\n🧠 Space Complexity:")
    print(f"   Only uses a constant amount of extra space")
    print(f"   → O(1) space complexity")

analyze_bubble_sort_complexity()
```

---

## ⚡ Performance Comparison with Other Algorithms

```python
import time
import random

def performance_comparison():
    """
    Compare bubble sort performance with other sorting algorithms.
    """
    def bubble_sort(arr):
        n = len(arr)
        for i in range(n):
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return arr

    def python_built_in_sort(arr):
        return sorted(arr)
    
    # Test with different array sizes
    sizes = [100, 500, 1000, 2000]
    
    print("🏁 PERFORMANCE COMPARISON")
    print("=" * 60)
    print(f"{'Size':<8} {'Bubble Sort':<15} {'Python sorted()':<15} {'Speedup':<10}")
    print("-" * 60)
    
    for size in sizes:
        # Generate random data
        data = [random.randint(1, 1000) for _ in range(size)]
        
        # Time bubble sort
        start_time = time.perf_counter()
        bubble_sort(data.copy())
        bubble_time = time.perf_counter() - start_time
        
        # Time Python's built-in sort
        start_time = time.perf_counter()
        python_built_in_sort(data.copy())
        python_time = time.perf_counter() - start_time
        
        speedup = bubble_time / python_time if python_time > 0 else float('inf')
        
        print(f"{size:<8} {bubble_time:<15.4f} {python_time:<15.6f} {speedup:<10.1f}x")

# Run performance comparison
performance_comparison()
```

---

## 🎯 When to Use Bubble Sort

### ✅ Good for:
- **Educational purposes**: Easy to understand and implement
- **Small datasets**: When n < 50, performance difference is negligible
- **Nearly sorted data**: With optimization, can be O(n) for sorted arrays
- **Memory-constrained environments**: Only uses O(1) extra space

### ❌ Avoid for:
- **Large datasets**: O(n²) complexity becomes prohibitive
- **Production systems**: Much better algorithms available
- **Performance-critical applications**: Use quicksort, mergesort, or heapsort instead

---

## 🧪 Practice Exercises

### Exercise 1: Implement Bubble Sort Variants

```python
def bubble_sort_descending(arr):
    """
    Exercise: Modify bubble sort to sort in descending order.
    TODO: Implement this function
    """
    # Your code here
    pass

def bubble_sort_count_operations(arr):
    """
    Exercise: Count the number of comparisons and swaps.
    Return: (sorted_array, comparisons, swaps)
    """
    # Your code here
    pass

def cocktail_shaker_sort(arr):
    """
    Exercise: Implement cocktail shaker sort (bidirectional bubble sort).
    This variant sorts in both directions in each pass.
    """
    # Your code here
    pass
```

### Exercise 2: Bubble Sort Analysis

**Question 1:** How many swaps are needed to sort the array [5, 4, 3, 2, 1] using bubble sort?

**Question 2:** What is the best-case input for bubble sort with early termination optimization?

**Question 3:** Modify bubble sort to sort an array of strings by length.

---

## 🔍 Common Mistakes and Debugging Tips

### Mistake 1: Wrong Loop Bounds
```python
# ❌ WRONG - May cause index out of bounds
for i in range(n):
    for j in range(n - 1):  # Should be n - i - 1
        if arr[j] > arr[j + 1]:
            arr[j], arr[j + 1] = arr[j + 1], arr[j]

# ✅ CORRECT
for i in range(n):
    for j in range(n - i - 1):  # Correctly excludes sorted elements
        if arr[j] > arr[j + 1]:
            arr[j], arr[j + 1] = arr[j + 1], arr[j]
```

### Mistake 2: Forgetting Early Termination
```python
# ❌ INEFFICIENT - Always does n passes
def bubble_sort_inefficient(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

# ✅ OPTIMIZED - Stops early if sorted
def bubble_sort_optimized(arr):
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
```

---

## 📈 Bubble Sort vs Other O(n²) Algorithms

```python
def compare_quadratic_sorts():
    """
    Compare bubble sort with other O(n²) sorting algorithms.
    """
    import random
    
    def selection_sort(arr):
        n = len(arr)
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                if arr[j] < arr[min_idx]:
                    min_idx = j
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
        return arr
    
    def insertion_sort(arr):
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            while j >= 0 and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
        return arr
    
    # Performance comparison
    size = 1000
    data = [random.randint(1, 1000) for _ in range(size)]
    
    algorithms = {
        "Bubble Sort": bubble_sort_basic,
        "Selection Sort": selection_sort,
        "Insertion Sort": insertion_sort
    }
    
    print("🏆 O(n²) ALGORITHM COMPARISON")
    print("=" * 40)
    
    for name, algorithm in algorithms.items():
        start_time = time.perf_counter()
        algorithm(data.copy())
        elapsed_time = time.perf_counter() - start_time
        print(f"{name:<15}: {elapsed_time:.4f} seconds")

# Run the comparison
compare_quadratic_sorts()
```

---

## 🎉 Summary

**Bubble Sort** is a fundamental sorting algorithm that, while inefficient for large datasets, serves as an excellent introduction to:

- **Algorithm design principles**
- **Loop optimization techniques**
- **Time complexity analysis**
- **The importance of choosing the right algorithm**

### Key Takeaways:
1. **Simple but slow**: Easy to understand but O(n²) makes it impractical for large data
2. **Educational value**: Perfect for learning algorithm concepts
3. **Optimization opportunities**: Early termination can improve best-case performance
4. **Real-world lesson**: Sometimes simple solutions aren't the best solutions

**Next Steps**: Learn more efficient sorting algorithms like Quicksort (O(n log n) average), Mergesort (O(n log n) guaranteed), or explore specialized sorts for specific data types!