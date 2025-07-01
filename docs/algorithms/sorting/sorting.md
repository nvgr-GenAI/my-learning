
# Sorting Algorithms

Master the art of arranging data in order - from basic algorithms to advanced techniques.

## Overview

Sorting is one of the most fundamental operations in computer science. Understanding different sorting algorithms helps you choose the right approach based on your data characteristics and constraints.

## 📊 Algorithm Comparison

| Algorithm | Best Case | Average Case | Worst Case | Space | Stable | In-Place | Notes |
|-----------|-----------|--------------|------------|-------|--------|----------|-------|
| **[Bubble Sort](bubble-sort.md)** | O(n) | O(n²) | O(n²) | O(1) | ✅ | ✅ | Simple, educational |
| **[Selection Sort](selection-sort.md)** | O(n²) | O(n²) | O(n²) | O(1) | ❌ | ✅ | Minimizes swaps |
| **[Insertion Sort](insertion-sort.md)** | O(n) | O(n²) | O(n²) | O(1) | ✅ | ✅ | Good for small/nearly sorted |
| **[Merge Sort](merge-sort.md)** | O(n log n) | O(n log n) | O(n log n) | O(n) | ✅ | ❌ | Guaranteed performance |
| **[Quick Sort](quick-sort.md)** | O(n log n) | O(n log n) | O(n²) | O(log n) | ❌ | ✅ | Fast in practice |
| **[Heap Sort](heap-sort.md)** | O(n log n) | O(n log n) | O(n log n) | O(1) | ❌ | ✅ | Guaranteed, not adaptive |

## 🎯 Algorithm Categories

<div class="grid cards" markdown>

-   :material-school: **Simple Sorting (O(n²))**
    
    ---
    
    Easy to understand and implement
    
    [:material-arrow-right: Bubble Sort](bubble-sort.md)
    [:material-arrow-right: Selection Sort](selection-sort.md)
    [:material-arrow-right: Insertion Sort](insertion-sort.md)

-   :material-lightning-bolt: **Efficient Sorting (O(n log n))**
    
    ---
    
    Divide and conquer algorithms
    
    [:material-arrow-right: Quick Sort](quick-sort.md)
    [:material-arrow-right: Merge Sort](merge-sort.md)
    [:material-arrow-right: Heap Sort](heap-sort.md)

-   :material-chart-line: **Specialized Sorting**
    
    ---
    
    Non-comparison based algorithms
    
    Coming soon: Counting Sort, Radix Sort, Bucket Sort

-   :material-cog: **Hybrid Algorithms**
    
    ---
    
    Real-world optimized sorting
    
    Coming soon: Tim Sort, Intro Sort

</div>

## 🚀 Quick Start

Choose your learning path based on your needs:

### For Beginners

Start with simple O(n²) algorithms to understand the fundamentals:

1. **[Bubble Sort](bubble-sort.md)** - Easiest to understand
2. **[Selection Sort](selection-sort.md)** - Simple selection process  
3. **[Insertion Sort](insertion-sort.md)** - Good for small arrays

### For Interviews

Master the efficient O(n log n) algorithms:

1. **[Quick Sort](quick-sort.md)** - Most commonly asked
2. **[Merge Sort](merge-sort.md)** - Stable and predictable
3. **[Heap Sort](heap-sort.md)** - Guaranteed performance

### For Advanced Users

Explore specialized and hybrid algorithms for specific use cases.

## 🎯 When to Use Which Algorithm?

### Data Size

- **Small (< 50 elements)**: Insertion Sort
- **Medium (50-10,000)**: Quick Sort
- **Large (> 10,000)**: Merge Sort or Quick Sort

### Data Characteristics

- **Nearly sorted**: Insertion Sort
- **Random data**: Quick Sort
- **Worst-case matters**: Merge Sort or Heap Sort
- **Memory constrained**: Heap Sort
- **Stability required**: Merge Sort

### Special Cases

- **Integer range known**: Counting Sort
- **Strings/Multi-key**: Radix Sort
- **Real-world data**: Tim Sort (Python's default)

---

*Ready to dive deep? Click on any algorithm above to see detailed implementations, complexity analysis, and practical examples!*
    
    return arr

def selection_sort_stable(arr):
    """Stable version of selection sort."""
    n = len(arr)
    
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        
        # Instead of swapping, insert minimum at correct position
        min_val = arr[min_idx]
        for k in range(min_idx, i, -1):
            arr[k] = arr[k - 1]
        arr[i] = min_val
    
    return arr

# Example usage
data = [64, 25, 12, 22, 11]
print(f"Selection sort: {selection_sort(data.copy())}")
```

### Insertion Sort

Builds the sorted array one element at a time by inserting each element into its correct position.

```python
def insertion_sort(arr):
    """
    Insertion Sort - O(n²) time, O(1) space
    Stable, in-place, adaptive
    """
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        
        # Move elements greater than key one position ahead
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        
        # Insert key at correct position
        arr[j + 1] = key
    
    return arr

def insertion_sort_binary(arr):
    """Insertion sort with binary search for position."""
    def binary_search(arr, val, start, end):
        """Find insertion position using binary search."""
        if start == end:
            return start if arr[start] > val else start + 1
        
        if start > end:
            return start
        
        mid = (start + end) // 2
        
        if arr[mid] < val:
            return binary_search(arr, val, mid + 1, end)
        elif arr[mid] > val:
            return binary_search(arr, val, start, mid - 1)
        else:
            return mid
    
    for i in range(1, len(arr)):
        val = arr[i]
        j = binary_search(arr, val, 0, i - 1)
        
        # Shift elements and insert
        arr[j + 1:i + 1] = arr[j:i]
        arr[j] = val
    
    return arr

# Example usage
data = [5, 2, 4, 6, 1, 3]
print(f"Insertion sort: {insertion_sort(data.copy())}")
```

## ⚡ Efficient Sorting Algorithms

### Merge Sort

Divide-and-conquer algorithm that divides the array into halves, sorts them, and merges the results.

```python
def merge_sort(arr):
    """
    Merge Sort - O(n log n) time, O(n) space
    Stable, not in-place, not adaptive
    """
    if len(arr) <= 1:
        return arr
    
    # Divide
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    # Conquer (merge)
    return merge(left, right)

def merge(left, right):
    """Merge two sorted arrays."""
    result = []
    i = j = 0
    
    # Merge elements in sorted order
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:  # <= for stability
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # Add remaining elements
    result.extend(left[i:])
    result.extend(right[j:])
    
    return result

def merge_sort_inplace(arr, temp_arr, left, right):
    """In-place merge sort using auxiliary array."""
    if left < right:
        mid = (left + right) // 2
        
        merge_sort_inplace(arr, temp_arr, left, mid)
        merge_sort_inplace(arr, temp_arr, mid + 1, right)
        merge_inplace(arr, temp_arr, left, mid, right)

def merge_inplace(arr, temp_arr, left, mid, right):
    """In-place merge using temporary array."""
    # Copy data to temp array
    for i in range(left, right + 1):
        temp_arr[i] = arr[i]
    
    i, j, k = left, mid + 1, left
    
    # Merge temp arrays back into arr
    while i <= mid and j <= right:
        if temp_arr[i] <= temp_arr[j]:
            arr[k] = temp_arr[i]
            i += 1
        else:
            arr[k] = temp_arr[j]
            j += 1
        k += 1
    
    # Copy remaining elements
    while i <= mid:
        arr[k] = temp_arr[i]
        i += 1
        k += 1
    
    while j <= right:
        arr[k] = temp_arr[j]
        j += 1
        k += 1

# Example usage
data = [38, 27, 43, 3, 9, 82, 10]
print(f"Merge sort: {merge_sort(data)}")
```

### Quick Sort

Efficient divide-and-conquer algorithm that picks a pivot and partitions the array around it.

```python
def quick_sort(arr):
    """
    Quick Sort - O(n log n) average, O(n²) worst, O(log n) space
    Not stable, in-place, not adaptive
    """
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort(left) + middle + quick_sort(right)

def quick_sort_inplace(arr, low=0, high=None):
    """In-place quick sort implementation."""
    if high is None:
        high = len(arr) - 1
    
    if low < high:
        # Partition and get pivot index
        pi = partition(arr, low, high)
        
        # Recursively sort elements before and after partition
        quick_sort_inplace(arr, low, pi - 1)
        quick_sort_inplace(arr, pi + 1, high)
    
    return arr

def partition(arr, low, high):
    """Lomuto partition scheme."""
    # Choose rightmost element as pivot
    pivot = arr[high]
    
    # Index of smaller element
    i = low - 1
    
    for j in range(low, high):
        # If current element is smaller than or equal to pivot
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    # Place pivot in correct position
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

def quick_sort_hoare(arr, low=0, high=None):
    """Quick sort with Hoare partition scheme."""
    if high is None:
        high = len(arr) - 1
    
    if low < high:
        pi = hoare_partition(arr, low, high)
        quick_sort_hoare(arr, low, pi)
        quick_sort_hoare(arr, pi + 1, high)
    
    return arr

def hoare_partition(arr, low, high):
    """Hoare partition scheme - more efficient."""
    pivot = arr[low]
    i = low - 1
    j = high + 1
    
    while True:
        i += 1
        while arr[i] < pivot:
            i += 1
        
        j -= 1
        while arr[j] > pivot:
            j -= 1
        
        if i >= j:
            return j
        
        arr[i], arr[j] = arr[j], arr[i]

def quick_sort_random(arr, low=0, high=None):
    """Randomized quick sort to avoid worst case."""
    import random
    
    if high is None:
        high = len(arr) - 1
    
    if low < high:
        # Random pivot selection
        random_index = random.randint(low, high)
        arr[random_index], arr[high] = arr[high], arr[random_index]
        
        pi = partition(arr, low, high)
        quick_sort_random(arr, low, pi - 1)
        quick_sort_random(arr, pi + 1, high)
    
    return arr

# Example usage
data = [10, 7, 8, 9, 1, 5]
print(f"Quick sort: {quick_sort_inplace(data.copy())}")
```

### Heap Sort

Uses a binary heap data structure to sort the array.

```python
def heap_sort(arr):
    """
    Heap Sort - O(n log n) time, O(1) space
    Not stable, in-place, not adaptive
    """
    n = len(arr)
    
    # Build max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    
    # Extract elements from heap one by one
    for i in range(n - 1, 0, -1):
        # Move current root to end
        arr[0], arr[i] = arr[i], arr[0]
        
        # Call heapify on reduced heap
        heapify(arr, i, 0)
    
    return arr

def heapify(arr, n, i):
    """Heapify subtree rooted at index i."""
    largest = i  # Initialize largest as root
    left = 2 * i + 1
    right = 2 * i + 2
    
    # If left child exists and is greater than root
    if left < n and arr[left] > arr[largest]:
        largest = left
    
    # If right child exists and is greater than largest so far
    if right < n and arr[right] > arr[largest]:
        largest = right
    
    # If largest is not root
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        
        # Recursively heapify affected subtree
        heapify(arr, n, largest)

class MaxHeap:
    """Max heap implementation for heap sort."""
    def __init__(self):
        self.heap = []
    
    def parent(self, i):
        return (i - 1) // 2
    
    def left_child(self, i):
        return 2 * i + 1
    
    def right_child(self, i):
        return 2 * i + 2
    
    def insert(self, val):
        """Insert value into heap."""
        self.heap.append(val)
        self._heapify_up(len(self.heap) - 1)
    
    def extract_max(self):
        """Remove and return maximum element."""
        if not self.heap:
            return None
        
        if len(self.heap) == 1:
            return self.heap.pop()
        
        max_val = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._heapify_down(0)
        
        return max_val
    
    def _heapify_up(self, i):
        """Heapify up from index i."""
        while i > 0 and self.heap[i] > self.heap[self.parent(i)]:
            self.heap[i], self.heap[self.parent(i)] = self.heap[self.parent(i)], self.heap[i]
            i = self.parent(i)
    
    def _heapify_down(self, i):
        """Heapify down from index i."""
        largest = i
        left = self.left_child(i)
        right = self.right_child(i)
        
        if left < len(self.heap) and self.heap[left] > self.heap[largest]:
            largest = left
        
        if right < len(self.heap) and self.heap[right] > self.heap[largest]:
            largest = right
        
        if largest != i:
            self.heap[i], self.heap[largest] = self.heap[largest], self.heap[i]
            self._heapify_down(largest)

# Example usage
data = [12, 11, 13, 5, 6, 7]
print(f"Heap sort: {heap_sort(data.copy())}")
```

## 🔢 Non-Comparison Based Sorting

### Counting Sort

Efficient for sorting integers within a small range.

```python
def counting_sort(arr, max_val=None):
    """
    Counting Sort - O(n + k) time, O(k) space
    Stable, not in-place
    """
    if not arr:
        return arr
    
    if max_val is None:
        max_val = max(arr)
    
    min_val = min(arr)
    range_val = max_val - min_val + 1
    
    # Count occurrences
    count = [0] * range_val
    for num in arr:
        count[num - min_val] += 1
    
    # Cumulative count for stable sorting
    for i in range(1, len(count)):
        count[i] += count[i - 1]
    
    # Build output array
    output = [0] * len(arr)
    for i in range(len(arr) - 1, -1, -1):
        output[count[arr[i] - min_val] - 1] = arr[i]
        count[arr[i] - min_val] -= 1
    
    return output

def counting_sort_simple(arr):
    """Simple counting sort for positive integers."""
    if not arr:
        return arr
    
    max_val = max(arr)
    count = [0] * (max_val + 1)
    
    # Count occurrences
    for num in arr:
        count[num] += 1
    
    # Reconstruct array
    result = []
    for i, freq in enumerate(count):
        result.extend([i] * freq)
    
    return result

# Example usage
data = [4, 2, 2, 8, 3, 3, 1]
print(f"Counting sort: {counting_sort(data)}")
```

### Radix Sort

Sorts by processing digits from least significant to most significant.

```python
def radix_sort(arr):
    """
    Radix Sort - O(d(n + k)) time
    Stable, not in-place
    """
    if not arr:
        return arr
    
    # Find maximum number to know number of digits
    max_num = max(arr)
    
    # Do counting sort for every digit
    exp = 1
    while max_num // exp > 0:
        counting_sort_for_radix(arr, exp)
        exp *= 10
    
    return arr

def counting_sort_for_radix(arr, exp):
    """Counting sort for radix sort."""
    n = len(arr)
    output = [0] * n
    count = [0] * 10
    
    # Count occurrences of each digit
    for i in range(n):
        index = arr[i] // exp
        count[index % 10] += 1
    
    # Cumulative count
    for i in range(1, 10):
        count[i] += count[i - 1]
    
    # Build output array
    i = n - 1
    while i >= 0:
        index = arr[i] // exp
        output[count[index % 10] - 1] = arr[i]
        count[index % 10] -= 1
        i -= 1
    
    # Copy output array to arr
    for i in range(n):
        arr[i] = output[i]

def radix_sort_strings(arr):
    """Radix sort for strings of equal length."""
    if not arr or not arr[0]:
        return arr
    
    max_len = len(arr[0])
    
    # Process from rightmost character to leftmost
    for i in range(max_len - 1, -1, -1):
        counting_sort_chars(arr, i)
    
    return arr

def counting_sort_chars(arr, char_index):
    """Counting sort for characters at specific position."""
    n = len(arr)
    output = [''] * n
    count = [0] * 256  # ASCII characters
    
    # Count occurrences
    for string in arr:
        index = ord(string[char_index])
        count[index] += 1
    
    # Cumulative count
    for i in range(1, 256):
        count[i] += count[i - 1]
    
    # Build output array
    for i in range(n - 1, -1, -1):
        index = ord(arr[i][char_index])
        output[count[index] - 1] = arr[i]
        count[index] -= 1
    
    # Copy back to original array
    for i in range(n):
        arr[i] = output[i]

# Example usage
data = [170, 45, 75, 90, 2, 802, 24, 66]
print(f"Radix sort: {radix_sort(data.copy())}")

string_data = ["abc", "def", "ghi", "aaa", "zzz"]
print(f"String radix sort: {radix_sort_strings(string_data.copy())}")
```

## 🎛️ Hybrid and Advanced Algorithms

### Tim Sort (Python's Built-in Sort)

A hybrid stable sorting algorithm combining merge sort and insertion sort.

```python
def insertion_sort_for_timsort(arr, left, right):
    """Insertion sort for small subarrays in Timsort."""
    for i in range(left + 1, right + 1):
        key_item = arr[i]
        j = i - 1
        while j >= left and arr[j] > key_item:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key_item

def merge_for_timsort(arr, left, mid, right):
    """Merge function for Timsort."""
    left_part = arr[left:mid + 1]
    right_part = arr[mid + 1:right + 1]
    
    i = j = 0
    k = left
    
    while i < len(left_part) and j < len(right_part):
        if left_part[i] <= right_part[j]:
            arr[k] = left_part[i]
            i += 1
        else:
            arr[k] = right_part[j]
            j += 1
        k += 1
    
    while i < len(left_part):
        arr[k] = left_part[i]
        i += 1
        k += 1
    
    while j < len(right_part):
        arr[k] = right_part[j]
        j += 1
        k += 1

def tim_sort(arr):
    """Simplified Timsort implementation."""
    min_merge = 32
    n = len(arr)
    
    # Sort individual subarrays of size min_merge
    for start in range(0, n, min_merge):
        end = min(start + min_merge - 1, n - 1)
        insertion_sort_for_timsort(arr, start, end)
    
    # Start merging from size min_merge
    size = min_merge
    while size < n:
        for start in range(0, n, size * 2):
            mid = start + size - 1
            end = min(start + size * 2 - 1, n - 1)
            
            if mid < end:
                merge_for_timsort(arr, start, mid, end)
        
        size *= 2
    
    return arr

# Example usage
data = [5, 21, 7, 23, 19, 10, 4, 15, 2, 18]
print(f"Tim sort: {tim_sort(data.copy())}")
```

## 🔍 Sorting Algorithm Selection Guide

### Choose Based on Your Needs

**Small datasets (n < 50):**
- **Insertion Sort** - Simple, efficient for small arrays
- **Selection Sort** - Minimizes memory writes

**Large datasets:**
- **Merge Sort** - Guaranteed O(n log n), stable
- **Quick Sort** - Fast average case, in-place
- **Heap Sort** - Guaranteed O(n log n), in-place

**Special cases:**
- **Nearly sorted data** - Insertion Sort (adaptive)
- **Limited memory** - Heap Sort (O(1) space)
- **Stability required** - Merge Sort or Tim Sort
- **Integer range known** - Counting Sort or Radix Sort

### Performance Visualization

```python
import time
import random

def benchmark_sorts():
    """Benchmark different sorting algorithms."""
    sizes = [100, 1000, 5000]
    algorithms = {
        'Bubble Sort': bubble_sort,
        'Selection Sort': selection_sort,
        'Insertion Sort': insertion_sort,
        'Merge Sort': merge_sort,
        'Quick Sort': lambda x: quick_sort_inplace(x.copy()),
        'Heap Sort': heap_sort,
        'Tim Sort': tim_sort
    }
    
    for size in sizes:
        print(f"\nArray size: {size}")
        data = [random.randint(1, 1000) for _ in range(size)]
        
        for name, algorithm in algorithms.items():
            if size > 1000 and name in ['Bubble Sort', 'Selection Sort']:
                continue  # Skip slow algorithms for large datasets
            
            test_data = data.copy()
            start_time = time.time()
            algorithm(test_data)
            end_time = time.time()
            
            print(f"{name}: {end_time - start_time:.4f} seconds")

# Uncomment to run benchmark
# benchmark_sorts()
```

---

**Sort wisely, code efficiently! 🚀**
